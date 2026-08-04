"""src/evaluation.py — Análisis estadístico de señales de incertidumbre (formato largo).

`results_full.csv` está en formato largo: una fila por imagen × prompt × variante
de señal, con columnas `signal_type, layer, tau, pooling, value`. Las columnas de
nivel de observación (logits, p_yes, correct, baselines entropy/msp/energy) están
repetidas en cada fila de la misma observación.

Implementa:
    - Normalización del formato largo (incluye fix de filas `kl_prompt_L34`).
    - Selección de variante ganadora SOLO con train (protocolo congelado).
    - Bootstrap CI (BCa) para AUROC y AUPRC (9.999 remuestreos).
    - Mann-Whitney U + effect size r = |Z|/√N.
    - Spearman para H4 (correlación con severidad CDR) con permutación.
    - Sensitivity @ 80% specificity (métrica clínica) y TPR a FPR fijos
      (rejilla 5%/10%/20%, protocolo FUSE §5.2).
    - Curvas accuracy-coverage.
    - Calibración estilo Guo et al. 2017 / FUSE §5.2: Platt scaling ajustado
      SOLO en train, bins equiprobables, ECE, correlaciones de calibración
      (Pearson/Spearman) y Brier score, con IC bootstrap percentil.
    - Baselines de igual costo (entropy, 1-MSP, energy) y señal combinada
      rank(KL) + rank(1-MSP), que no requiere ajuste de parámetros.

Uso:
    python -m src.evaluation                     # protocolo completo (P1 y P4)
    python -m src.evaluation --signal kl_t_v_L34_tau1.0_max --prompt P1
    python -m src.evaluation --all-signals       # tabla AUROC de todas las variantes
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from src.config import Config

# Poolings no deployables (oracle): se excluyen de la SELECCIÓN de la ganadora
POOLINGS_ORACLE = {"roi"}


# ------------------------------------------------------------------------------
# Carga y normalización del formato largo
# ------------------------------------------------------------------------------
def load_results(cfg: Config, filename: str = "results_full.csv") -> pd.DataFrame:
    """Carga el CSV largo y añade el nombre canónico de cada variante.

    Nombre canónico: ``{signal_type}_L{layer}_tau{tau}_{pooling}``
    (p.ej. ``kl_t_v_L34_tau1.0_max``). Repara las filas ``kl_prompt_L34``
    guardadas con columnas desalineadas (layer='tau1', tau='mean').
    """
    path = Path(cfg.paths.results) / filename
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Correr src.inference primero.")
    df = pd.read_csv(path)
    df["tau"] = df["tau"].astype(str)
    df["layer"] = df["layer"].astype(str)

    # Fix filas kl_prompt_L34 desalineadas: (layer='tau1', tau='mean', pooling='mean')
    mask = df["signal_type"] == "kl_prompt_L34"
    if mask.any():
        df.loc[mask, "signal_type"] = "kl_prompt"
        df.loc[mask, "layer"] = "34"
        df.loc[mask, "tau"] = "1.0"
        df.loc[mask, "pooling"] = "mean"

    df["signal"] = (
        df["signal_type"] + "_L" + df["layer"] + "_tau" + df["tau"] + "_" + df["pooling"]
    )
    return df


def load_master(cfg: Config) -> pd.DataFrame:
    """Carga master_table.csv para metadatos (CDR, artefactos)."""
    path = Path(cfg.paths.master_table)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Correr src.data primero.")
    return pd.read_csv(path)


def signal_frame(df: pd.DataFrame, prompt: str, signal: str) -> pd.DataFrame:
    """Devuelve una fila por imagen con la señal pedida + metadatos de observación."""
    obs_cols = [
        "image_filename", "split", "label", "correct",
        "p_yes", "entropy_answer", "msp_answer", "energy_answer",
    ]
    if signal in BASELINE_SIGNALS:
        sub = df[df["prompt_id"] == prompt].drop_duplicates("image_filename")[obs_cols].copy()
        sub["value"] = baseline_values(sub, signal)
        return sub[["image_filename", "split", "label", "correct", "value"]]

    sub = df[(df["prompt_id"] == prompt) & (df["signal"] == signal)]
    if sub.empty:
        return sub
    return sub[["image_filename", "split", "label", "correct", "value"]].copy()


# ------------------------------------------------------------------------------
# Señales derivadas: baselines de igual costo y combinación por ranks
# ------------------------------------------------------------------------------
BASELINE_SIGNALS = {
    "entropy": "Entropía de la respuesta (mayor = más incierto)",
    "one_minus_msp": "1 - MSP (mayor = más incierto)",
    "energy": "Energy = -logsumexp(logits) (mayor = más incierto)",
}


def baseline_values(obs: pd.DataFrame, name: str) -> np.ndarray:
    if name == "entropy":
        return obs["entropy_answer"].values
    if name == "one_minus_msp":
        return 1.0 - obs["msp_answer"].values
    if name == "energy":
        return obs["energy_answer"].values
    raise ValueError(f"Baseline desconocido: {name}")


def rank_combination_frame(df: pd.DataFrame, prompt: str, kl_signal: str) -> pd.DataFrame:
    """Señal combinada u(x) = rank(KL) + rank(1-MSP), sin parámetros ajustables.

    Usa ranks fraccionarios en [0, 1]; no requiere estandarización ni ajuste en
    train, por lo que no introduce riesgo de sobreajuste.
    """
    kl = signal_frame(df, prompt, kl_signal)
    if kl.empty:
        return kl
    obs = df[df["prompt_id"] == prompt].drop_duplicates("image_filename")[
        ["image_filename", "msp_answer"]
    ]
    m = kl.merge(obs, on="image_filename")
    n = len(m)
    m["value"] = m["value"].rank() / n + (1 - m["msp_answer"]).rank() / n
    return m[["image_filename", "split", "label", "correct", "value"]]


# ------------------------------------------------------------------------------
# Bootstrap CI (BCa)
# ------------------------------------------------------------------------------
def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "auroc",
    n_bootstrap: int = 9999,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Intervalo de confianza bootstrap BCa para AUROC o AUPRC."""
    if len(y_true) != len(y_score):
        warnings.warn(f"Longitudes inconsistentes: y_true={len(y_true)}, y_score={len(y_score)}")
        return {"mean": np.nan, "point": np.nan, "ci_low": np.nan, "ci_high": np.nan, "std": np.nan}

    if len(np.unique(y_true)) < 2:
        return {"mean": np.nan, "point": np.nan, "ci_low": np.nan, "ci_high": np.nan, "std": np.nan}

    def statistic(y, x):
        if metric == "auroc":
            return roc_auc_score(y, x)
        elif metric == "auprc":
            return average_precision_score(y, x)
        raise ValueError(f"Métrica desconocida: {metric}")

    point = statistic(y_true, y_score)

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boot_stats = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_boot = y_true[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        try:
            boot_stats.append(statistic(y_boot, y_score[idx]))
        except Exception:
            continue

    if len(boot_stats) < 100:
        warnings.warn(f"Bootstrap: solo {len(boot_stats)} remuestreos válidos")
        return {"mean": point, "point": point, "ci_low": np.nan, "ci_high": np.nan, "std": np.nan}

    boot_stats = np.array(boot_stats)

    alpha = 1 - confidence
    z0 = stats.norm.ppf(np.mean(boot_stats < point))

    jack_stats = []
    for i in range(n):
        idx = np.arange(n) != i
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            jack_stats.append(statistic(y_true[idx], y_score[idx]))
        except Exception:
            continue

    if len(jack_stats) < 2:
        ci_low = np.percentile(boot_stats, 100 * alpha / 2)
        ci_high = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    else:
        jack_stats = np.array(jack_stats)
        jack_mean = np.mean(jack_stats)
        num = np.sum((jack_mean - jack_stats) ** 3)
        den = 6 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
        acc = num / den if den > 0 else 0.0

        z_alpha_2 = stats.norm.ppf(alpha / 2)
        z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)

        p_low = stats.norm.cdf(z0 + (z0 + z_alpha_2) / (1 - acc * (z0 + z_alpha_2)))
        p_high = stats.norm.cdf(z0 + (z0 + z_1_alpha_2) / (1 - acc * (z0 + z_1_alpha_2)))

        ci_low = np.percentile(boot_stats, 100 * p_low)
        ci_high = np.percentile(boot_stats, 100 * p_high)

    return {
        "mean": float(np.mean(boot_stats)),
        "point": float(point),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "std": float(np.std(boot_stats)),
    }


# ------------------------------------------------------------------------------
# Mann-Whitney U + effect size
# ------------------------------------------------------------------------------
def mann_whitney_effect(y_correct: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    """Mann-Whitney U (errores > aciertos), con effect size r = |Z|/√N."""
    correct_vals = signal[y_correct == 1]
    error_vals = signal[y_correct == 0]

    if len(correct_vals) == 0 or len(error_vals) == 0:
        return {"U": np.nan, "p_value": np.nan, "z": np.nan, "r": np.nan,
                "median_correct": np.nan, "median_error": np.nan}

    try:
        res = stats.mannwhitneyu(error_vals, correct_vals, alternative="greater", method="asymptotic")
        n1, n2 = len(error_vals), len(correct_vals)
        U = res.statistic
        mu_U = n1 * n2 / 2
        sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (U - mu_U) / sigma_U if sigma_U > 0 else 0.0
        r = abs(z) / np.sqrt(n1 + n2)
        return {
            "U": float(U),
            "p_value": float(res.pvalue),
            "z": float(z),
            "r": float(r),
            "median_correct": float(np.median(correct_vals)),
            "median_error": float(np.median(error_vals)),
        }
    except Exception as exc:
        warnings.warn(f"Mann-Whitney falló: {exc}")
        return {"U": np.nan, "p_value": np.nan, "z": np.nan, "r": np.nan,
                "median_correct": np.nan, "median_error": np.nan}


# ------------------------------------------------------------------------------
# Spearman para H4 (correlación con severidad CDR)
# ------------------------------------------------------------------------------
def spearman_h4(df: pd.DataFrame, value_col: str = "value", cdr_col: str = "cdr_grade") -> dict[str, float]:
    """Spearman entre u(x) y severidad CDR en patológicos, p por permutación."""
    path = df[df["label"] == 1].copy()
    if len(path) < 2:
        return {"rho": np.nan, "p_value": np.nan, "n": 0}

    signal = path[value_col].values.astype(float)
    cdr = path[cdr_col].values.astype(float)

    mask = ~np.isnan(signal) & ~np.isnan(cdr)
    if mask.sum() < 2:
        return {"rho": np.nan, "p_value": np.nan, "n": int(mask.sum())}

    signal, cdr = signal[mask], cdr[mask]

    try:
        res = stats.permutation_test(
            (signal, cdr),
            lambda x, y: stats.spearmanr(x, y).statistic,
            permutation_type="pairings",
            n_resamples=9999,
            random_state=42,
        )
        rho = stats.spearmanr(signal, cdr).statistic
        return {"rho": float(rho), "p_value": float(res.pvalue), "n": int(len(signal))}
    except Exception as exc:
        warnings.warn(f"Spearman H4 falló: {exc}")
        rho, p = stats.spearmanr(signal, cdr)
        return {"rho": float(rho), "p_value": float(p), "n": int(len(signal))}


# ------------------------------------------------------------------------------
# Sensitivity @ 80% specificity
# ------------------------------------------------------------------------------
def sensitivity_at_specificity(y_correct: np.ndarray, signal: np.ndarray, target_spec: float = 0.80) -> dict[str, float]:
    """Sensitivity de detección de errores fijando specificity = target_spec."""
    y_error = 1 - y_correct
    if len(np.unique(y_error)) < 2:
        return {"threshold": np.nan, "sensitivity": np.nan, "specificity": np.nan}

    n_correct = (y_correct == 1).sum()
    n_correct_below = int(np.ceil(n_correct * target_spec))

    correct_signals_sorted = np.sort(signal[y_correct == 1])
    if n_correct_below >= len(correct_signals_sorted):
        threshold = correct_signals_sorted[-1]
    else:
        threshold = correct_signals_sorted[n_correct_below]

    sensitivity = (signal[y_error == 1] > threshold).mean() if (y_error == 1).sum() > 0 else np.nan
    specificity = (signal[y_correct == 1] <= threshold).mean()

    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


# ------------------------------------------------------------------------------
# Curvas accuracy-coverage
# ------------------------------------------------------------------------------
def accuracy_coverage(y_correct: np.ndarray, signal: np.ndarray, steps: list[float] | None = None) -> pd.DataFrame:
    """Accuracy en el subconjunto retenido al abstenerse en el c% más incierto."""
    steps = steps or [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    order = np.argsort(-signal)  # mayor incertidumbre primero
    n = len(signal)

    rows = []
    for step in steps:
        n_abstain = int(np.ceil(n * step))
        keep = order[n_abstain:]
        if len(keep) == 0:
            continue
        rows.append({
            "coverage": 1 - step,
            "abstention": step,
            "accuracy": float(y_correct[keep].mean()),
            "n_kept": int(len(keep)),
            "n_abstained": int(n_abstain),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# AURC / Excess-AURC (métrica de selective prediction, Geifman & El-Yaniv 2017)
# ------------------------------------------------------------------------------
def aurc(y_correct: np.ndarray, signal: np.ndarray) -> float:
    """Área bajo la curva riesgo-cobertura (menor = mejor).

    Ordena de menor a mayor u(x) (el modelo responde primero a los más seguros)
    y promedia la tasa de error del subconjunto retenido para cada cobertura.
    """
    order = np.argsort(signal)  # menos incierto primero
    y_err_sorted = 1 - y_correct[order]
    n = len(y_err_sorted)
    risks = [y_err_sorted[:k].mean() for k in range(1, n + 1)]
    return float(np.mean(risks))


def aurc_oracle(y_correct: np.ndarray) -> float:
    """AURC del ranking perfecto (errores derivados primero)."""
    y_err_sorted = np.sort(1 - y_correct)  # aciertos primero
    n = len(y_err_sorted)
    risks = [y_err_sorted[:k].mean() for k in range(1, n + 1)]
    return float(np.mean(risks))


def excess_aurc(y_correct: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    """Excess-AURC normalizado: 0 = oracle, 1 = azar (menor = mejor)."""
    a = aurc(y_correct, signal)
    a_oracle = aurc_oracle(y_correct)
    a_random = float((1 - y_correct).mean())
    norm = (a - a_oracle) / (a_random - a_oracle) if a_random > a_oracle else np.nan
    return {"aurc": a, "aurc_oracle": a_oracle, "aurc_random": a_random,
            "excess_aurc_norm": float(norm)}


# ------------------------------------------------------------------------------
# TPR a FPR fijos (protocolo FUSE §5.2: discriminación en puntos operativos)
# ------------------------------------------------------------------------------
def tpr_at_fpr(y_correct: np.ndarray, signal: np.ndarray,
               fprs: tuple[float, ...] = (0.05, 0.10, 0.20)) -> dict[str, float]:
    """TPR de detección de errores interpolado a FPR fijos (5%, 10%, 20%).

    Invariante monotónica: se computa sobre la señal cruda (no calibrada).
    """
    y_error = 1 - y_correct
    if len(np.unique(y_error)) < 2:
        return {f"tpr_fpr{int(f * 100):02d}": np.nan for f in fprs}
    fpr_curve, tpr_curve, _ = roc_curve(y_error, signal)
    out = {}
    for f in fprs:
        # np.interp exige xp creciente; roc_curve devuelve fpr no decreciente
        out[f"tpr_fpr{int(f * 100):02d}"] = float(np.interp(f, fpr_curve, tpr_curve))
    return out


# ------------------------------------------------------------------------------
# Calibración (Guo et al. 2017; protocolo FUSE §5.2)
#
# La señal cruda (KL en nats) no vive en [0, 1]: para el test P(correct|u*) ≈ 1-u*
# se ajusta un Platt scaling (sigmoide 1-feature u -> P(error)) SOLO con el split
# train y se aplica al resto. Advertencias:
#   - N=129 -> 10 bins ≈ 13 obs/bin: ECE y correlaciones son ruidosas; reportar
#     SIEMPRE con IC bootstrap y lenguaje honesto.
#   - Platt es monótona por construcción: correlaciones altas no prueban
#     calibración por sí solas. ECE + reliability diagram son la evidencia
#     principal; las correlaciones son secundarias.
# ------------------------------------------------------------------------------
def platt_calibrate(u_train: np.ndarray, y_err_train: np.ndarray,
                    u_all: np.ndarray) -> dict[str, Any] | None:
    """Platt scaling: sigmoide 1-feature u -> P(error), ajustada solo en train.

    Devuelve dict con ``u_cal`` (probabilidades para ``u_all``) y los coeficientes
    (a, b) para trazabilidad. None si el ajuste no es posible (una sola clase o
    señal constante en train).
    """
    u_train = np.asarray(u_train, dtype=float)
    y_err_train = np.asarray(y_err_train, dtype=float)
    u_all = np.asarray(u_all, dtype=float)
    if len(u_train) < 10 or len(np.unique(y_err_train)) < 2 or len(np.unique(u_train)) < 2:
        return None
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(u_train.reshape(-1, 1), y_err_train)
    return {
        "u_cal": lr.predict_proba(u_all.reshape(-1, 1))[:, 1],
        "a": float(lr.coef_[0, 0]),
        "b": float(lr.intercept_[0]),
    }


def calibration_bins(u_cal: np.ndarray, y_error: np.ndarray,
                     n_bins: int = 10) -> pd.DataFrame:
    """Bins equiprobables (quantiles) de u_calibrada vs. tasa de error empírica.

    Columnas: ``bin, mean_u, empirical_error, n``. Bins con <2 valores únicos de
    u se fusionan (``duplicates='drop'``), por lo que puede haber menos de
    ``n_bins`` filas.
    """
    u_cal = np.asarray(u_cal, dtype=float)
    y_error = np.asarray(y_error, dtype=float)
    n_bins_eff = min(n_bins, len(u_cal))
    if n_bins_eff < 2 or len(np.unique(u_cal)) < 2:
        return pd.DataFrame({
            "bin": [0], "mean_u": [float(np.mean(u_cal))],
            "empirical_error": [float(np.mean(y_error))], "n": [int(len(u_cal))],
        })
    asign = pd.qcut(pd.Series(u_cal), q=n_bins_eff, labels=False, duplicates="drop")
    df = pd.DataFrame({"bin": asign, "u": u_cal, "err": y_error})
    return (
        df.groupby("bin")
        .agg(mean_u=("u", "mean"), empirical_error=("err", "mean"), n=("err", "size"))
        .reset_index()
    )


def expected_calibration_error(bins: pd.DataFrame) -> float:
    """ECE (versión incertidumbre): media ponderada |u_media - error_empírico|."""
    if bins.empty or bins["n"].sum() == 0:
        return np.nan
    w = bins["n"] / bins["n"].sum()
    return float((w * (bins["mean_u"] - bins["empirical_error"]).abs()).sum())


def calibration_correlations(bins: pd.DataFrame) -> dict[str, float]:
    """Pearson y Spearman entre u media por bin y error empírico por bin."""
    if len(bins) < 3 or bins["mean_u"].nunique() < 2 or bins["empirical_error"].nunique() < 2:
        return {"pearson": np.nan, "spearman": np.nan}
    return {
        "pearson": float(stats.pearsonr(bins["mean_u"], bins["empirical_error"]).statistic),
        "spearman": float(stats.spearmanr(bins["mean_u"], bins["empirical_error"]).statistic),
    }


def calibration_analysis(frame_eval: pd.DataFrame, frame_fit: pd.DataFrame | None = None,
                         n_bins: int = 10, n_bootstrap: int = 1999,
                         random_state: int = 42) -> dict[str, Any]:
    """Análisis de calibración completo de una señal.

    Ajusta Platt con las filas ``split == 'train'`` de ``frame_fit`` (por defecto
    el propio ``frame_eval``) y evalúa la calibración sobre ``frame_eval``.
    Devuelve ECE, correlaciones Pearson/Spearman y Brier con IC bootstrap
    percentil 95% (Platt fijo, remuestreo de observaciones), la tabla de bins,
    la u calibrada, ECE con 5 bins (sensibilidad) y flags de trazabilidad.
    """
    empty = {"ece": np.nan, "ece_ci_low": np.nan, "ece_ci_high": np.nan,
             "pearson": np.nan, "pearson_ci_low": np.nan, "pearson_ci_high": np.nan,
             "spearman": np.nan, "spearman_ci_low": np.nan, "spearman_ci_high": np.nan,
             "brier": np.nan, "ece_bins5": np.nan, "n_bins": 0,
             "platt_a": np.nan, "platt_b": np.nan, "in_sample": True,
             "bins": pd.DataFrame(), "u_cal": np.array([])}
    if frame_fit is None:
        frame_fit = frame_eval
    ev = frame_eval.dropna(subset=["value"])
    if len(ev) < 20 or len(np.unique(ev["correct"])) < 2:
        return empty
    fit = frame_fit.dropna(subset=["value"])
    fit_train = fit[fit["split"] == "train"] if (fit["split"] == "train").any() else fit
    in_sample = bool(ev["image_filename"].isin(set(fit_train["image_filename"])).all())

    platt = platt_calibrate(fit_train["value"].values, 1 - fit_train["correct"].values,
                            ev["value"].values)
    if platt is None:
        return empty

    u_cal = platt["u_cal"]
    y_error = (1 - ev["correct"].values).astype(float)

    bins = calibration_bins(u_cal, y_error, n_bins=n_bins)
    ece = expected_calibration_error(bins)
    corr = calibration_correlations(bins)
    brier = float(brier_score_loss(y_error, u_cal))
    ece5 = expected_calibration_error(calibration_bins(u_cal, y_error, n_bins=5))

    out = {
        "ece": ece, "ece_ci_low": np.nan, "ece_ci_high": np.nan,
        "pearson": corr["pearson"], "pearson_ci_low": np.nan, "pearson_ci_high": np.nan,
        "spearman": corr["spearman"], "spearman_ci_low": np.nan, "spearman_ci_high": np.nan,
        "brier": brier, "ece_bins5": ece5, "n_bins": int(len(bins)),
        "platt_a": platt["a"], "platt_b": platt["b"], "in_sample": in_sample,
        "bins": bins, "u_cal": u_cal,
    }

    if n_bootstrap and n_bootstrap > 0:
        rng = np.random.default_rng(random_state)
        n = len(u_cal)
        boot = {"ece": [], "pearson": [], "spearman": []}
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y_error[idx])) < 2:
                continue
            b = calibration_bins(u_cal[idx], y_error[idx], n_bins=n_bins)
            boot["ece"].append(expected_calibration_error(b))
            c = calibration_correlations(b)
            boot["pearson"].append(c["pearson"])
            boot["spearman"].append(c["spearman"])
        for key in boot:
            vals = np.array([v for v in boot[key] if not np.isnan(v)])
            if len(vals) >= 100:
                out[f"{key}_ci_low"] = float(np.percentile(vals, 2.5))
                out[f"{key}_ci_high"] = float(np.percentile(vals, 97.5))
    return out


# ------------------------------------------------------------------------------
# Análisis completo de una señal (ya como frame imagen × valor)
# ------------------------------------------------------------------------------
def analyze_frame(
    frame: pd.DataFrame,
    signal_name: str,
    split: str = "all",
    master: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Análisis completo: AUROC/AUPRC con BCa, MWU, sens@80spec, TPR@FPR, H4,
    calibración (Platt ajustado en train, ECE/correlaciones/Brier con IC)."""
    frame_fit = frame  # copia sin filtrar: de aquí salen las filas train para Platt
    if split == "val+test":
        frame = frame[frame["split"].isin(["validation", "test"])].copy()
    elif split != "all":
        frame = frame[frame["split"] == split].copy()

    signal = frame["value"].values.astype(float)
    y_correct = frame["correct"].values
    y_error = 1 - y_correct

    mask = ~np.isnan(signal)
    if mask.sum() < 2 or len(np.unique(y_error[mask])) < 2:
        return {"signal": signal_name, "split": split, "n": int(mask.sum()),
                "n_errors": int(y_error[mask].sum()), "error": "insufficient data"}

    frame = frame[mask]
    y_correct, y_error, signal = y_correct[mask], y_error[mask], signal[mask]

    result = {
        "signal": signal_name,
        "split": split,
        "n": int(len(signal)),
        "n_errors": int(y_error.sum()),
        "accuracy": float(y_correct.mean()),
    }

    result["auroc"] = bootstrap_ci(y_error, signal, metric="auroc")
    result["auprc"] = bootstrap_ci(y_error, signal, metric="auprc")
    result["mann_whitney"] = mann_whitney_effect(y_correct, signal)
    result["sens_80spec"] = sensitivity_at_specificity(y_correct, signal, target_spec=0.80)
    result["tpr_fpr"] = tpr_at_fpr(y_correct, signal)
    result["acc_cov"] = accuracy_coverage(y_correct, signal)
    result["aurc"] = excess_aurc(y_correct, signal)
    result["calibration"] = calibration_analysis(frame, frame_fit=frame_fit)

    if master is not None and "cdr_grade" in master.columns:
        merged = frame.merge(master[["image_filename", "cdr_grade"]], on="image_filename", how="left")
        result["h4_spearman"] = spearman_h4(merged, "value", "cdr_grade")

    return result


# ------------------------------------------------------------------------------
# Reporte
# ------------------------------------------------------------------------------
def print_signal_report(result: dict[str, Any]) -> None:
    """Imprime el reporte de una señal de forma legible."""
    print(f"\n{'=' * 70}")
    print(f"SEÑAL: {result['signal']} | Split: {result['split']} | n={result['n']} | errores={result['n_errors']}")
    print(f"{'=' * 70}")

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    print(f"Accuracy base: {result['accuracy']:.3f}")

    auroc = result["auroc"]
    print(f"AUROC: {auroc['point']:.3f} [{auroc['ci_low']:.3f}, {auroc['ci_high']:.3f}] (BCa 95%)")
    auprc = result["auprc"]
    print(f"AUPRC: {auprc['point']:.3f} [{auprc['ci_low']:.3f}, {auprc['ci_high']:.3f}] (BCa 95%)")

    mw = result["mann_whitney"]
    print(f"Mann-Whitney: U={mw['U']:.1f}, p={mw['p_value']:.4f}, r={mw['r']:.3f}")
    print(f"  (medianas: correctos={mw['median_correct']:.4f}, errores={mw['median_error']:.4f})")

    s80 = result["sens_80spec"]
    print(f"Sensitivity@80%Spec: {s80['sensitivity']:.3f} (umbral={s80['threshold']:.4f})")

    tf = result["tpr_fpr"]
    print(f"TPR@FPR: 5%={tf['tpr_fpr05']:.3f} | 10%={tf['tpr_fpr10']:.3f} | 20%={tf['tpr_fpr20']:.3f}")

    arc = result["aurc"]
    print(f"AURC: {arc['aurc']:.4f} (oracle={arc['aurc_oracle']:.4f}, azar={arc['aurc_random']:.4f})"
          f" | Excess-AURC norm: {arc['excess_aurc_norm']:.3f} (0=oracle, 1=azar)")

    cal = result["calibration"]
    if not np.isnan(cal["ece"]):
        tag = " [IN-SAMPLE: Platt ajustado en este mismo split]" if cal["in_sample"] else ""
        print(f"Calibración (Platt en train, {cal['n_bins']} bins){tag}:")
        print(f"  ECE: {cal['ece']:.3f} [{cal['ece_ci_low']:.3f}, {cal['ece_ci_high']:.3f}]"
              f" | ECE(5 bins): {cal['ece_bins5']:.3f} | Brier: {cal['brier']:.3f}")
        print(f"  Corr. calibración: Pearson={cal['pearson']:.3f} [{cal['pearson_ci_low']:.3f},"
              f" {cal['pearson_ci_high']:.3f}] | Spearman={cal['spearman']:.3f}"
              f" [{cal['spearman_ci_low']:.3f}, {cal['spearman_ci_high']:.3f}]")
    else:
        print("Calibración: n/a (datos insuficientes para Platt)")

    if "h4_spearman" in result:
        h4 = result["h4_spearman"]
        print(f"H4 Spearman (CDR): rho={h4['rho']:.3f}, p={h4['p_value']:.4f} (n={h4['n']})")


def _summary_row(r: dict[str, Any], prompt: str) -> dict[str, Any]:
    cal = r["calibration"]
    tf = r["tpr_fpr"]
    return {
        "prompt": prompt,
        "signal": r["signal"],
        "split": r["split"],
        "n": r["n"],
        "n_errors": r["n_errors"],
        "auroc": r["auroc"]["point"],
        "auroc_ci_low": r["auroc"]["ci_low"],
        "auroc_ci_high": r["auroc"]["ci_high"],
        "auprc": r["auprc"]["point"],
        "auprc_ci_low": r["auprc"]["ci_low"],
        "auprc_ci_high": r["auprc"]["ci_high"],
        "aurc": r["aurc"]["aurc"],
        "excess_aurc_norm": r["aurc"]["excess_aurc_norm"],
        "mannwhitney_p": r["mann_whitney"]["p_value"],
        "effect_size_r": r["mann_whitney"]["r"],
        "sens_80spec": r["sens_80spec"]["sensitivity"],
        "tpr_fpr05": tf["tpr_fpr05"],
        "tpr_fpr10": tf["tpr_fpr10"],
        "tpr_fpr20": tf["tpr_fpr20"],
        "ece": cal["ece"],
        "ece_ci_low": cal["ece_ci_low"],
        "ece_ci_high": cal["ece_ci_high"],
        "cal_pearson": cal["pearson"],
        "cal_spearman": cal["spearman"],
        "brier": cal["brier"],
        "calibration_in_sample": cal["in_sample"],
    }


# ------------------------------------------------------------------------------
# Selección de la variante ganadora (SOLO con train, protocolo congelado)
# ------------------------------------------------------------------------------
def select_winner(df: pd.DataFrame, prompt: str) -> str | None:
    """Elige la variante con mayor AUROC en train, excluyendo poolings oracle."""
    train = df[(df["prompt_id"] == prompt) & (df["split"] == "train")]
    best_signal, best_auroc = None, -np.inf
    for signal, sub in train.groupby("signal"):
        pooling = sub["pooling"].iloc[0]
        if pooling in POOLINGS_ORACLE:
            continue
        v = sub["value"].values.astype(float)
        m = ~np.isnan(v)
        y = 1 - sub["correct"].values[m]
        if m.sum() < 10 or len(np.unique(y)) < 2:
            continue
        a = roc_auc_score(y, v[m])
        if a > best_auroc:
            best_signal, best_auroc = signal, a
    return best_signal


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis estadístico de señales UQ (formato largo)")
    parser.add_argument("--signal", default=None,
                        help="Nombre canónico de la señal (p.ej. kl_t_v_L34_tau1.0_max)")
    parser.add_argument("--prompt", default="both", choices=["P1", "P4", "both"])
    parser.add_argument("--all-signals", action="store_true",
                        help="Tabla de AUROC (sin CI) de todas las variantes")
    parser.add_argument("--split", default="all", choices=["all", "train", "validation", "test"])
    parser.add_argument("--filename", default="results_full.csv")
    args = parser.parse_args()

    cfg = Config()
    df = load_results(cfg, filename=args.filename)
    master = load_master(cfg)
    prompts = ["P1", "P4"] if args.prompt == "both" else [args.prompt]

    print(f"[eval] {len(df)} filas cargadas | {df['signal'].nunique()} variantes | prompts={prompts}")

    summary_rows: list[dict[str, Any]] = []

    if args.all_signals:
        # Tabla rápida de AUROC por variante (sin CI) en el split pedido
        sub_all = df if args.split == "all" else df[df["split"] == args.split]
        for prompt in prompts:
            rows = []
            for signal, sub in sub_all[sub_all["prompt_id"] == prompt].groupby("signal"):
                v = sub["value"].values.astype(float)
                m = ~np.isnan(v)
                y = 1 - sub["correct"].values[m]
                if m.sum() < 10 or len(np.unique(y)) < 2:
                    continue
                rows.append({"prompt": prompt, "signal": signal,
                             "auroc": roc_auc_score(y, v[m]), "n": int(m.sum())})
            tabla = pd.DataFrame(rows).sort_values("auroc", ascending=False)
            print(f"\n=== AUROC por variante ({prompt}, split={args.split}) ===")
            print(tabla.to_string(index=False))
            summary_rows.extend(rows)
    else:
        for prompt in prompts:
            winner = args.signal or select_winner(df, prompt)
            if winner is None:
                warnings.warn(f"No se pudo seleccionar ganadora para {prompt}")
                continue
            print(f"\n[eval] Ganadora {prompt} (seleccionada en train): {winner}")

            # 1) Ganadora: análisis en el split pedido + confirmación val+test
            frame = signal_frame(df, prompt, winner)
            for split in ([args.split] if args.split != "all" else ["all"]):
                r = analyze_frame(frame, winner, split=split, master=master)
                print_signal_report(r)
                if "error" not in r:
                    summary_rows.append(_summary_row(r, prompt))
                    r["acc_cov"].to_csv(Path(cfg.paths.results) / f"acc_cov_{prompt}_winner.csv", index=False)
            if args.split == "all":
                r = analyze_frame(frame, winner, split="val+test", master=None)
                print_signal_report(r)
                if "error" not in r:
                    summary_rows.append(_summary_row(r, prompt))

            # 2) Dirección espejo de la ganadora (misma capa/τ/pooling) — la
            #    elección de dirección se reporta SIEMPRE con datos de ambas
            mirror = None
            if winner.startswith("kl_t_v"):
                mirror = "kl_v_t" + winner[len("kl_t_v"):]
            elif winner.startswith("kl_v_t"):
                mirror = "kl_t_v" + winner[len("kl_v_t"):]
            if mirror is not None:
                frame_m = signal_frame(df, prompt, mirror)
                if not frame_m.empty:
                    r = analyze_frame(frame_m, f"{mirror} (espejo)", split=args.split, master=None)
                    print_signal_report(r)
                    if "error" not in r:
                        summary_rows.append(_summary_row(r, prompt))

            # 3) Baselines de igual costo
            for bl in BASELINE_SIGNALS:
                frame_bl = signal_frame(df, prompt, bl)
                r = analyze_frame(frame_bl, bl, split=args.split, master=None)
                print_signal_report(r)
                if "error" not in r:
                    summary_rows.append(_summary_row(r, prompt))

            # 4) Combinación por ranks (sin parámetros)
            frame_rc = rank_combination_frame(df, prompt, winner)
            if not frame_rc.empty:
                r = analyze_frame(frame_rc, f"rank({winner})+rank(1-MSP)", split=args.split, master=master)
                print_signal_report(r)
                if "error" not in r:
                    summary_rows.append(_summary_row(r, prompt))
                    r["acc_cov"].to_csv(Path(cfg.paths.results) / f"acc_cov_{prompt}_rankcombo.csv", index=False)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        out_path = Path(cfg.paths.results) / "evaluation_summary.csv"
        summary_df.to_csv(out_path, index=False)
        print(f"\n[eval] Resumen guardado: {out_path}")


if __name__ == "__main__":
    main()
