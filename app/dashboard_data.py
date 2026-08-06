"""Carga de datos y métricas para el dashboard BIP 2026.

Todo se calcula con numpy/pandas (sin scipy ni scikit-learn) para que el
Space arranque en CPU básico. El AUROC se computa por rangos (estadístico
de Mann-Whitney con corrección de empates), equivalente a roc_auc_score.

Los datos viven en ``app/assets/`` (generados por ``prepare_assets.py``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = Path(__file__).resolve().parent / "assets"

# Variante ganadora congelada (seleccionada SOLO en train, ver AGENTS.md §6.3)
WINNER = {"family": "kl_t_v", "pooling": "max", "tau": "1.0"}
WINNER_NAME = "kl_t_v_L34_tau1.0_max"

FAMILIAS_KL = ["kl_t_v", "kl_v_t", "jsd", "cosine", "kl_prompt"]
FAMILIAS_BASELINE = ["entropy", "1-msp", "energy", "rankcombo"]
POOLINGS = ["mean", "max", "topk", "normw", "attn", "rollout", "headspec", "roi (oracle)"]
TAUS = ["1.0", "2.0", "4.0"]

NOMBRES_FAMILIA = {
    "kl_t_v": "KL texto→visión (ganadora)",
    "kl_v_t": "KL visión→texto (espejo)",
    "jsd": "Divergencia Jensen-Shannon",
    "cosine": "Distancia coseno",
    "kl_prompt": "KL prompt→texto (control)",
    "entropy": "Entropía de la respuesta (baseline)",
    "1-msp": "1 − MSP (baseline)",
    "energy": "Energía (baseline)",
    "rankcombo": "rank(KL) + rank(1−MSP) (combinada)",
}


# ---------------------------------------------------------------------------
# Carga cacheada
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_long() -> pd.DataFrame:
    """CSV largo normalizado (misma reparación que src.evaluation.load_results)."""
    df = pd.read_csv(ASSETS / "results_full.csv")
    df["tau"] = df["tau"].astype(str)
    df["layer"] = df["layer"].astype(str)
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


@lru_cache(maxsize=1)
def load_master() -> pd.DataFrame:
    return pd.read_csv(ASSETS / "master_table.csv")


@lru_cache(maxsize=1)
def load_summary() -> pd.DataFrame:
    return pd.read_csv(ASSETS / "evaluation_summary.csv")


@lru_cache(maxsize=1)
def load_tabla(nombre: str) -> pd.DataFrame:
    return pd.read_csv(ASSETS / nombre)


# ---------------------------------------------------------------------------
# Extracción de u(x) por imagen
# ---------------------------------------------------------------------------

_OBS_COLS = [
    "image_filename", "split", "label", "correct",
    "p_yes", "entropy_answer", "msp_answer", "energy_answer",
]


def _observations(df: pd.DataFrame, prompt: str) -> pd.DataFrame:
    """Una fila por imagen con las columnas de observación."""
    sub = df[df["prompt_id"] == prompt]
    return sub[_OBS_COLS].drop_duplicates(subset=["image_filename"]).reset_index(drop=True)


def get_u(
    prompt: str = "P1",
    family: str = "kl_t_v",
    pooling: str = "max",
    tau: str = "1.0",
    split: str = "all",
) -> pd.DataFrame:
    """Devuelve una fila por imagen con la señal de incertidumbre ``u``.

    Columnas: image_filename, split, label, correct, p_yes, u.
    Para ``roi (oracle)`` solo hay filas patológicas (69 por prompt).
    """
    df = load_long()
    obs = _observations(df, prompt)

    if family in FAMILIAS_BASELINE:
        out = obs.copy()
        if family == "entropy":
            out["u"] = out["entropy_answer"]
        elif family == "1-msp":
            out["u"] = 1.0 - out["msp_answer"]
        elif family == "energy":
            out["u"] = out["energy_answer"]
        elif family == "rankcombo":
            kl = get_u(prompt, "kl_t_v", "max", "1.0", "all")
            out = out.merge(kl[["image_filename", "u"]], on="image_filename", how="left")
            u_kl = out["u"].rank()
            u_msp = (1.0 - out["msp_answer"]).rank()
            out["u"] = u_kl + u_msp
    else:
        pooling_csv = "roi" if pooling == "roi (oracle)" else pooling
        sig = f"{family}_L34_tau{tau}_{pooling_csv}"
        sub = df[(df["prompt_id"] == prompt) & (df["signal"] == sig)]
        out = obs.merge(
            sub[["image_filename", "value"]].rename(columns={"value": "u"}),
            on="image_filename", how="inner",
        )

    if split != "all":
        out = out[out["split"] == split].reset_index(drop=True)
    return out.dropna(subset=["u"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Métricas (numpy puro)
# ---------------------------------------------------------------------------

def auroc(u: np.ndarray, y: np.ndarray) -> float:
    """AUROC por rangos (Mann-Whitney, corrección de empates). y=1 es "error"."""
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=int)
    pos, neg = u[y == 1], u[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(u).rank(method="average").to_numpy()
    r_pos = ranks[y == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def roc_curve(u: np.ndarray, y: np.ndarray):
    """Devuelve (fpr, tpr, thresholds) ordenados."""
    order = np.argsort(-np.asarray(u, dtype=float))
    y = np.asarray(y, dtype=int)[order]
    u_sorted = np.asarray(u, dtype=float)[order]
    P, N = y.sum(), len(y) - y.sum()
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = np.concatenate([[0.0], tps / max(P, 1)])
    fpr = np.concatenate([[0.0], fps / max(N, 1)])
    thr = np.concatenate([[np.inf], u_sorted])
    return fpr, tpr, thr


def pr_curve(u: np.ndarray, y: np.ndarray):
    """Devuelve (recall, precision) y AUPRC (average precision)."""
    order = np.argsort(-np.asarray(u, dtype=float))
    y = np.asarray(y, dtype=int)[order]
    P = max(y.sum(), 1)
    tps = np.cumsum(y)
    recall = tps / P
    precision = tps / np.arange(1, len(y) + 1)
    ap = float((precision * np.diff(np.concatenate([[0.0], recall]))).sum())
    return recall, precision, ap


def sens_at_spec(u: np.ndarray, y: np.ndarray, spec: float = 0.80) -> float:
    """Sensibilidad (TPR para detectar errores) al nivel de especificidad dado."""
    fpr, tpr, _ = roc_curve(u, y)
    ok = fpr <= (1 - spec) + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def accuracy_coverage(u: np.ndarray, correct: np.ndarray, steps=None):
    """Curva accuracy-coverage: se retienen los casos de MENOR incertidumbre."""
    if steps is None:
        steps = np.linspace(0.50, 1.0, 26)
    order = np.argsort(np.asarray(u, dtype=float))  # menor u = más confiado
    correct = np.asarray(correct, dtype=int)[order]
    n = len(correct)
    rows = []
    for cov in steps:
        k = max(int(round(cov * n)), 1)
        rows.append({
            "coverage": float(k / n),
            "accuracy": float(correct[:k].mean()),
            "n_kept": int(k),
            "n_referred": int(n - k),
            "errors_captured": int((correct[k:] == 0).sum()),
        })
    df = pd.DataFrame(rows)
    df["accuracy_random"] = float(correct.mean())
    return df


def triage_at_coverage(u: np.ndarray, correct: np.ndarray, coverage: float) -> dict:
    """Métricas del triage para una cobertura concreta."""
    order = np.argsort(np.asarray(u, dtype=float))
    correct = np.asarray(correct, dtype=int)[order]
    n = len(correct)
    k = max(int(round(coverage * n)), 1)
    kept, referred = correct[:k], correct[k:]
    return {
        "n_total": n,
        "n_kept": int(k),
        "n_referred": int(n - k),
        "accuracy_kept": float(kept.mean()),
        "accuracy_overall": float(correct.mean()),
        "errors_total": int((correct == 0).sum()),
        "errors_captured": int((referred == 0).sum()),
        "errors_expected_random": float((correct == 0).sum() * (n - k) / n),
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Coeficiente de correlación de Spearman (por rangos, sin p-valor)."""
    rx = pd.Series(x).rank(method="average")
    ry = pd.Series(y).rank(method="average")
    return float(rx.corr(ry))


def percentile_of(values: np.ndarray, v: float) -> float:
    """Percentil de v dentro de la cohorte (0-100)."""
    values = np.asarray(values, dtype=float)
    return float((values <= v).mean() * 100)


def platt_fit(u: np.ndarray, y: np.ndarray, iters: int = 200) -> tuple[float, float]:
    """Platt scaling 1-feature por IRLS: P(error) = sigmoid(a·u + b).

    Devuelve (a, b). Se ajusta SOLO con el split train (protocolo FUSE §5.2).
    """
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([u, np.ones_like(u)])
    beta = np.zeros(2)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        w = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * w[:, None])
        g = X.T @ (p - y)
        step = np.linalg.solve(H, g)
        beta -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def platt_predict(u: np.ndarray, a: float, b: float) -> np.ndarray:
    z = a * np.asarray(u, dtype=float) + b
    return 1.0 / (1.0 + np.exp(-z))


def metricas_basicas(frame: pd.DataFrame) -> dict:
    """Resumen de discriminación para un frame de get_u()."""
    y_err = 1 - frame["correct"].astype(int).to_numpy()
    u = frame["u"].to_numpy()
    _, _, ap = pr_curve(u, y_err)
    return {
        "n": len(frame),
        "n_errors": int(y_err.sum()),
        "accuracy_modelo": float(frame["correct"].mean()),
        "auroc": auroc(u, y_err),
        "auprc": ap,
        "sens_80spec": sens_at_spec(u, y_err, 0.80),
    }


def h4_frame(prompt: str = "P1") -> pd.DataFrame:
    """u(x) vs. cdr_grade en los 69 patológicos (hipótesis H4)."""
    u = get_u(prompt, WINNER["family"], WINNER["pooling"], WINNER["tau"], "all")
    master = load_master()
    pat = master[master["label"] == 1][["image_filename", "cdr_grade"]]
    out = u.merge(pat, on="image_filename", how="inner").dropna(subset=["cdr_grade"])
    return out


# ---------------------------------------------------------------------------
# Mapas de pooling (results/pooling_maps.csv — generado en Colab/GPU)
# ---------------------------------------------------------------------------

POOLING_MAPS_ORDEN = ["mean", "max", "topk", "normw", "attn", "rollout", "headspec", "roi"]


@lru_cache(maxsize=1)
def load_pooling_maps() -> pd.DataFrame | None:
    """CSV largo: image_filename, prompt_id, pooling, token_idx, weight.
    Devuelve None si aún no se ha generado (ver src/extract_pooling_maps.py)."""
    path = ASSETS / "pooling_maps.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def pooling_maps_grid(image_filename: str, prompt_id: str = "P1") -> dict[str, np.ndarray]:
    """Devuelve {pooling: array(16,16)} para una imagen, listo para heatmap."""
    df = load_pooling_maps()
    if df is None:
        return {}
    sub = df[(df["image_filename"] == image_filename) & (df["prompt_id"] == prompt_id)]
    grids: dict[str, np.ndarray] = {}
    for pooling, g in sub.groupby("pooling"):
        g = g.sort_values("token_idx")
        grids[pooling] = g["weight"].to_numpy().reshape(16, 16)
    return grids
