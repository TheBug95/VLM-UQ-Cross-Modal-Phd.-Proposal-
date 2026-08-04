"""src/figures.py — Figures and tables for the BIP 2026 paper (long format).

All figures are generated in English with a consistent seaborn style
(colorblind-safe palette, one color per signal across all figures).

Generates:
    - Fig 2: boxplots of u(x) in correct vs. incorrect cases (one per signal).
    - Fig 3: ROC and PR curves (winner, baselines, rank combination).
    - Fig 4: accuracy-coverage curves with AURC/Excess-AURC annotations.
    - Fig 5: quadrant examples with ophthalmologist transcriptions.
    - Fig 6: Spearman correlation heatmap between UQ signals.
    - Fig 7: UQ performance vs. computational cost (+ Table T4).
    - Fig 8: verbalized confidence degeneracy and miscalibration.
    - Fig 9: self-consistency signal boxplots.
    - Fig 10: reliability diagram (calibration, Platt fit on train only).
    - Tables T1-T5 (T5: discrimination + calibration, FUSE §5.2 style).

Usage:
    python -m src.figures                      # P1 prompt, winner chosen on train
    python -m src.figures --prompt P4
    python -m src.figures --signal kl_t_v_L34_tau1.0_max
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.config import Config
from src.evaluation import (
    load_master,
    load_results,
    rank_combination_frame,
    select_winner,
    sensitivity_at_specificity,
    calibration_analysis,
    excess_aurc,
    signal_frame,
    tpr_at_fpr,
)

# ------------------------------------------------------------------------------
# Estilo global consistente
# ------------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Un color por señal, consistente en TODAS las figuras
C_KL = "#0072B2"      # azul
C_KLVT = "#56B4E9"    # azul claro (dirección espejo v→t)
C_COMBO = "#CC79A7"   # púrpura rosado
C_MSP = "#009E73"     # verde
C_ENTROPY = "#E69F00" # naranja
C_ENERGY = "#D55E00"  # bermellón
C_GRAY = "#7F8C8D"    # gris para baselines de costo

SIGNAL_COLORS = {
    "kl": C_KL,
    "klvt": C_KLVT,
    "rank(KL) + rank(1-MSP)": C_COMBO,
    "1 - MSP": C_MSP,
    "Entropy": C_ENTROPY,
    "Energy": C_ENERGY,
}
BOX_COLORS = {"Correct": "#009E73", "Incorrect": "#D55E00"}

LABELS = {
    "entropy": "Entropy",
    "one_minus_msp": "1 - MSP",
    "energy": "Energy",
}


def _signal_key(name: str) -> str:
    if name.startswith("KL cross-modal"):
        return "kl"
    if name.startswith(("KL v→t", "KL t→v")):
        return "klvt"
    return name


def _y_v(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(y_error, valor) sin NaN."""
    v = frame["value"].values.astype(float)
    mask = ~np.isnan(v)
    return 1 - frame["correct"].values[mask], v[mask]


# ------------------------------------------------------------------------------
# Conjunto de señales a comparar
# ------------------------------------------------------------------------------
def build_signal_set(df: pd.DataFrame, prompt: str, winner: str) -> dict[str, pd.DataFrame]:
    """Frames (image_filename, split, label, correct, value) de las señales del paper.

    Incluye SIEMPRE la dirección espejo de la ganadora (kl_v_t <-> kl_t_v con
    la misma capa/τ/pooling): la justificación de la dirección se reporta con
    datos de ambas, no solo de la ganadora.
    """
    signals: dict[str, pd.DataFrame] = {}
    signals[f"KL cross-modal ({winner})"] = signal_frame(df, prompt, winner)

    # Dirección espejo con idéntica configuración
    if winner.startswith("kl_t_v"):
        mirror = "kl_v_t" + winner[len("kl_t_v"):]
        m = signal_frame(df, prompt, mirror)
        if not m.empty:
            signals[f"KL v→t (mirror, {mirror})"] = m
    elif winner.startswith("kl_v_t"):
        mirror = "kl_t_v" + winner[len("kl_v_t"):]
        m = signal_frame(df, prompt, mirror)
        if not m.empty:
            signals[f"KL t→v (mirror, {mirror})"] = m

    for bl, label in LABELS.items():
        signals[label] = signal_frame(df, prompt, bl)
    rc = rank_combination_frame(df, prompt, winner)
    if not rc.empty:
        signals["rank(KL) + rank(1-MSP)"] = rc
    return {k: v.sort_values("image_filename").reset_index(drop=True) for k, v in signals.items()}


# ------------------------------------------------------------------------------
# Fig 2: Boxplot de u(x) en correctos vs. incorrectos
# ------------------------------------------------------------------------------
def fig2_boxplot(frame: pd.DataFrame, signal_name: str, out_path: Path) -> None:
    """Boxplot + stripplot de u(x) en correctos vs. incorrectos (English)."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    data = frame[["correct", "value"]].dropna().copy()
    data["correct"] = data["correct"].map({1: "Correct", 0: "Incorrect"})
    data["correct"] = pd.Categorical(data["correct"], ["Correct", "Incorrect"])

    sns.boxplot(data=data, x="correct", y="value", ax=ax, showfliers=False,
                palette=BOX_COLORS, width=0.55, linewidth=1.4,
                medianprops={"color": "black", "linewidth": 2})
    sns.stripplot(data=data, x="correct", y="value", ax=ax, color="black",
                  alpha=0.30, size=2.5, jitter=0.18)

    correct_vals = data[data["correct"] == "Correct"]["value"]
    error_vals = data[data["correct"] == "Incorrect"]["value"]

    try:
        mw = stats.mannwhitneyu(error_vals, correct_vals, alternative="greater", method="asymptotic")
        n1, n2 = len(error_vals), len(correct_vals)
        U = mw.statistic
        mu_U = n1 * n2 / 2
        sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (U - mu_U) / sigma_U if sigma_U > 0 else 0
        r = abs(z) / np.sqrt(n1 + n2)
        p_text = f"Mann-Whitney p = {mw.pvalue:.4f}   |   effect size r = {r:.3f}"
    except Exception:
        p_text = "Mann-Whitney: n/a"

    data_all = frame[["correct", "value"]].dropna()
    y_c = data_all["correct"].values
    v = data_all["value"].values.astype(float)
    try:
        auroc_val = roc_auc_score(1 - y_c, v)
        exc_val = excess_aurc(y_c, v)["excess_aurc_norm"]
        p_text += f"\nAUROC = {auroc_val:.3f}   |   Excess-AURC = {exc_val:.3f}"
    except Exception:
        pass

    ax.text(0.03, 0.97, p_text, transform=ax.transAxes, fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.8", alpha=0.9))

    ax.set_xlabel("Model outcome")
    ax.set_ylabel(f"u(x): {signal_name}")
    ax.set_title("Uncertainty signal in correct vs. incorrect cases")
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[fig] Saved: {out_path}")


# ------------------------------------------------------------------------------
# Fig 3: Curvas ROC y PR
# ------------------------------------------------------------------------------
def fig3_roc_pr(signals: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Curvas ROC y PR de la señal ganadora, baselines y combinación (English)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    for name, frame in signals.items():
        y_error, vals = _y_v(frame)
        if len(np.unique(vals)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_error, vals)
        auroc = roc_auc_score(y_error, vals)
        ax.plot(fpr, tpr, label=f"{name} ({auroc:.3f})", linewidth=2,
                color=SIGNAL_COLORS.get(_signal_key(name)))

    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", label="Chance (0.500)", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("(a) ROC curves — error detection")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    ax = axes[1]
    y_error_ref = None
    for name, frame in signals.items():
        y_error, vals = _y_v(frame)
        y_error_ref = y_error
        if len(np.unique(vals)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_error, vals)
        auprc = average_precision_score(y_error, vals)
        ax.plot(recall, precision, label=f"{name} ({auprc:.3f})", linewidth=2,
                color=SIGNAL_COLORS.get(_signal_key(name)))

    if y_error_ref is not None:
        baseline = y_error_ref.mean()
        ax.axhline(baseline, color="0.5", linestyle="--",
                   label=f"Chance ({baseline:.3f})", linewidth=1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(b) Precision-Recall curves")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[fig] Saved: {out_path}")


# ------------------------------------------------------------------------------
# Fig 4: Curvas accuracy-coverage
# ------------------------------------------------------------------------------
def fig4_accuracy_coverage(signals: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Accuracy del subconjunto retenido al derivar el X% más incierto (English)."""
    fig, ax = plt.subplots(figsize=(7, 4.8))

    base_acc = None
    for name, frame in signals.items():
        y_error, vals = _y_v(frame)
        y_correct = 1 - y_error
        base_acc = y_correct.mean()
        n_valid = len(vals)

        arc = excess_aurc(y_correct, vals)
        label = f"{name}  (AURC={arc['aurc']:.3f}, Excess={arc['excess_aurc_norm']:.3f})"

        order = np.argsort(-vals)
        coverages, accuracies = [], []
        for coverage in np.linspace(0.0, 1.0, 26):
            n_keep = int(np.ceil(n_valid * coverage))
            keep = order[n_valid - n_keep:]
            if len(keep) == 0:
                continue
            coverages.append(coverage)
            accuracies.append(y_correct[keep].mean())

        ax.plot(coverages, accuracies, label=label, linewidth=2, marker="o",
                markersize=3, color=SIGNAL_COLORS.get(_signal_key(name)))

    if base_acc is not None:
        ax.axhline(base_acc, color="0.5", linestyle="--",
                   label=f"Base accuracy ({base_acc:.3f})", linewidth=1)

    ax.set_xlabel("Coverage (fraction of cases answered by the model)")
    ax.set_ylabel("Model accuracy")
    ax.set_xlim(0, 1)
    ax.set_title("Accuracy vs. coverage — referring the X% most uncertain cases")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[fig] Saved: {out_path}")


# ------------------------------------------------------------------------------
# Fig 5: Ejemplos de cuadrantes
# ------------------------------------------------------------------------------
def fig5_quadrants(df: pd.DataFrame, frame: pd.DataFrame, master: pd.DataFrame,
                   signal_name: str, out_path: Path) -> None:
    """Ejemplos de cuadrantes: correcto/incorrecto × u(x) alta/baja (English)."""
    obs = df.drop_duplicates("image_filename")[["image_filename", "p_yes"]]
    df_merged = (
        frame.merge(obs, on="image_filename")
        .merge(master[["image_filename", "transcription"]], on="image_filename", how="left")
    )

    median_u = df_merged["value"].median()
    df_merged["u_high"] = df_merged["value"] > median_u
    df_merged["quadrant"] = df_merged.apply(
        lambda r: (
            "Correct + high u(x)" if r["correct"] == 1 and r["u_high"]
            else "Correct + low u(x)" if r["correct"] == 1
            else "Error + high u(x)" if r["u_high"]
            else "Error + low u(x)"
        ),
        axis=1,
    )

    examples = []
    for quad in ["Correct + low u(x)", "Correct + high u(x)", "Error + high u(x)", "Error + low u(x)"]:
        subset = df_merged[df_merged["quadrant"] == quad].head(2)
        for _, row in subset.iterrows():
            examples.append({
                "image_filename": row["image_filename"],
                "quadrant": quad,
                "u": row["value"],
                "p_yes": row["p_yes"],
                "label": row["label"],
                "transcription": str(row["transcription"]),
            })

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    y_pos = 0.97
    for ex in examples:
        text = (
            f"{ex['quadrant']}\n"
            f"Image: {ex['image_filename']}\n"
            f"u(x): {ex['u']:.4f} | P(yes): {ex['p_yes']:.3f} | Label: {ex['label']}\n"
            f"Transcription: {ex['transcription'][:200]}...\n"
        )
        ax.text(0.03, y_pos, text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0EAD6", edgecolor="0.8", alpha=0.9))
        y_pos -= 0.22

    ax.set_title(f"Quadrant examples ({signal_name})", fontsize=12, pad=16)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[fig] Saved: {out_path}")


# ------------------------------------------------------------------------------
# Fig 6: heatmap de correlación entre señales
# ------------------------------------------------------------------------------
def fig6_correlation(signals: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Matriz de Spearman entre rankings de las señales principales (English)."""
    frames = []
    for name, frame in signals.items():
        s = frame[["image_filename", "value"]].rename(columns={"value": name})
        frames.append(s)
    m = frames[0]
    for f in frames[1:]:
        m = m.merge(f, on="image_filename")

    names = [c for c in m.columns if c != "image_filename"]
    mat = np.ones((len(names), len(names)))
    for i, s1 in enumerate(names):
        for j, s2 in enumerate(names):
            if i < j:
                mat[i, j] = mat[j, i] = stats.spearmanr(m[s1], m[s2]).statistic
    mat_df = pd.DataFrame(mat, index=names, columns=names)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    sns.heatmap(mat_df, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, ax=ax, fmt=".2f", linewidths=0.5,
                cbar_kws={"label": "Spearman ρ"})
    ax.set_title("Spearman correlation between UQ signals (P1)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[fig] Saved: {out_path}")


# ------------------------------------------------------------------------------
# Tabla T1: Resultados principales
# ------------------------------------------------------------------------------
def table_t1(signals: dict[str, pd.DataFrame], out_path: Path) -> pd.DataFrame:
    """Tabla de resultados principales (costo 1×: todas son single-pass)."""
    rows = []

    for name, frame in signals.items():
        y_error, vals = _y_v(frame)
        if len(np.unique(vals)) < 2:
            continue
        auroc = roc_auc_score(y_error, vals)
        auprc = average_precision_score(y_error, vals)
        sens = sensitivity_at_specificity(1 - y_error, vals, target_spec=0.80)
        arc = excess_aurc(1 - y_error, vals)

        rows.append({
            "Signal": name,
            "AUROC": f"{auroc:.3f}",
            "AUPRC": f"{auprc:.3f}",
            "AURC": f"{arc['aurc']:.4f}",
            "Excess-AURC": f"{arc['excess_aurc_norm']:.3f}",
            "Sens@80%Spec": f"{sens['sensitivity']:.3f}",
            "Cost": "1×",
        })

    if signals:
        first = next(iter(signals.values()))
        base_acc = first["correct"].mean()
        rows.insert(0, {
            "Signal": f"Model base accuracy ({base_acc:.3f})",
            "AUROC": "—", "AUPRC": "—", "AURC": "—", "Excess-AURC": "—",
            "Sens@80%Spec": "—", "Cost": "—",
        })

    t1 = pd.DataFrame(rows)
    t1.to_csv(out_path, index=False)
    print(f"[table] Saved: {out_path}")
    return t1


# ------------------------------------------------------------------------------
# Tabla T2: Ablaciones
# ------------------------------------------------------------------------------
def table_t2(df: pd.DataFrame, prompt: str, out_path: Path) -> pd.DataFrame:
    """Ablaciones: dirección × τ × pooling (capa 34), AUROC en ALL y train."""
    sub = df[df["prompt_id"] == prompt]
    rows = []
    for (stype, layer, tau, pooling), g in sub.groupby(["signal_type", "layer", "tau", "pooling"]):
        v = g["value"].values.astype(float)
        m = ~np.isnan(v)
        y = 1 - g["correct"].values[m]
        if m.sum() < 10 or len(np.unique(y)) < 2:
            continue
        g_tr = g[g["split"] == "train"]
        v_tr = g_tr["value"].values.astype(float)
        m_tr = ~np.isnan(v_tr)
        y_tr = 1 - g_tr["correct"].values[m_tr]
        auroc_train = roc_auc_score(y_tr, v_tr[m_tr]) if len(np.unique(y_tr)) == 2 else np.nan
        rows.append({
            "Type": stype,
            "Layer": layer,
            "τ": tau,
            "Pooling": pooling,
            "AUROC_all": round(float(roc_auc_score(y, v[m])), 3),
            "AUROC_train": round(float(auroc_train), 3) if not np.isnan(auroc_train) else "—",
            "n": int(m.sum()),
        })

    t2 = pd.DataFrame(rows).sort_values("AUROC_all", ascending=False)
    t2.to_csv(out_path, index=False)
    print(f"[table] Saved: {out_path}")
    return t2


# ------------------------------------------------------------------------------
# Tabla T3: Comparativa de propiedades
# ------------------------------------------------------------------------------
def table_t3(out_path: Path) -> pd.DataFrame:
    """Tabla comparativa de propiedades vs. métodos de la literatura."""
    rows = [
        {"Method": "MC-Dropout", "Single-pass": "No", "Training-free": "No",
         "Cross-modal": "No", "Cost": "10–100×"},
        {"Method": "Semantic Entropy", "Single-pass": "No", "Training-free": "Yes",
         "Cross-modal": "No", "Cost": "10×"},
        {"Method": "UMPIRE", "Single-pass": "No", "Training-free": "Yes",
         "Cross-modal": "No", "Cost": "Multi-sample"},
        {"Method": "VIG-TUQ", "Single-pass": "Attention only", "Training-free": "Yes",
         "Cross-modal": "Yes", "Cost": "1× (attention), 2× (JSD)"},
        {"Method": "SAPLMA (probes)", "Single-pass": "Yes", "Training-free": "No",
         "Cross-modal": "No", "Cost": "Supervised"},
        {"Method": "Ours (cross-modal KL)", "Single-pass": "Yes", "Training-free": "Yes",
         "Cross-modal": "Yes", "Cost": "1×"},
    ]
    t3 = pd.DataFrame(rows)
    t3.to_csv(out_path, index=False)
    print(f"[table] Saved: {out_path}")
    return t3


# ------------------------------------------------------------------------------
# Fig 7 + Tabla T4: costo computacional vs. desempeño UQ
# ------------------------------------------------------------------------------
def _auroc_on(df: pd.DataFrame, prompt: str, images, signal_col: str | None,
              baseline: str | None = None) -> float:
    """AUROC de una señal restringida a un subconjunto de imágenes."""
    sub = df[(df["prompt_id"] == prompt) & df["image_filename"].isin(images)]
    if signal_col is not None:
        sub = sub[sub["signal"] == signal_col]
    sub = sub.drop_duplicates("image_filename")
    y = 1 - sub["correct"].values
    if baseline == "one_minus_msp":
        v = (1 - sub["msp_answer"]).values
    else:
        v = sub["value"].values.astype(float)
    m = ~np.isnan(v)
    return float(roc_auc_score(y[m], v[m]))


def fig7_tabla_t4_costo(df: pd.DataFrame, cfg: Config, out_dir: Path) -> pd.DataFrame | None:
    """Fig 7 (costo vs AUROC, 2 paneles) y Tabla T4 (costo-beneficio)."""
    results_dir = Path(cfg.paths.results)
    verb_path = results_dir / "results_verbalized.csv"
    sc_path = results_dir / "results_self_consistency.csv"
    if not verb_path.exists() and not sc_path.exists():
        print("[fig7] No results_verbalized.csv / results_self_consistency.csv — skipped")
        return None

    obs = df[df["prompt_id"] == "P1"].drop_duplicates("image_filename")
    y_all = 1 - obs["correct"].values
    winner = select_winner(df, "P1")
    n = len(obs)
    kl = df[(df["prompt_id"] == "P1") & (df["signal"] == winner)]
    kl = kl.set_index("image_filename")["value"]
    kl_v = kl[obs["image_filename"]].values
    combo_v = (kl[obs["image_filename"]].rank().values / n
               + (1 - obs["msp_answer"]).rank().values / n)

    filas_129 = [
        ("Energy (1×)", 1, roc_auc_score(y_all, obs["energy_answer"].values), "129", C_ENERGY),
        ("1-MSP (1×)", 1, roc_auc_score(y_all, 1 - obs["msp_answer"].values), "129", C_MSP),
        ("KL (1×)", 1, roc_auc_score(y_all, kl_v), "129", C_KL),
        ("rank(KL)+rank(1-MSP) (1×)", 1, roc_auc_score(y_all, combo_v), "129", C_COMBO),
    ]
    if verb_path.exists():
        v = pd.read_csv(verb_path)
        yv = 1 - v["correct"].values
        filas_129.append(("Verbalized conf (2×)", 2,
                          roc_auc_score(yv, v["u_verbalized"].values), "129", C_GRAY))

    filas_50 = []
    if sc_path.exists():
        sc = pd.read_csv(sc_path)
        d = sc[sc["prompt_id"] == "P1"].copy()
        d["pred"] = (d["self_consistency_frac_yes"] > d["sc_frac_no"]).astype(int)
        d["correct"] = (d["pred"] == d["label"]).astype(int)
        imgs = d["image_filename"].tolist()
        dsc = d.set_index("image_filename")
        y50 = 1 - dsc["correct"].values
        filas_50 = [
            ("SC 3-way entropy (10×)", 10, roc_auc_score(y50, dsc["self_consistency_entropy"].values), "50 (SC)", C_GRAY),
            ("SC binary entropy (10×)", 10, roc_auc_score(y50, dsc["sc_entropy_binary"].values), "50 (SC)", C_GRAY),
            ("SC frac_other (10×)", 10, roc_auc_score(y50, dsc["sc_frac_other"].values), "50 (SC)", C_GRAY),
            ("1-MSP (1×)", 1, _auroc_on(df, "P1", imgs, None, baseline="one_minus_msp"), "50 (SC)", C_MSP),
            ("KL (1×)", 1, _auroc_on(df, "P1", imgs, winner), "50 (SC)", C_KL),
        ]
        obs50 = df[(df["prompt_id"] == "P1") & df["image_filename"].isin(imgs)].drop_duplicates("image_filename")
        obs50 = obs50.set_index("image_filename").loc[imgs]
        kl50 = kl[imgs].values
        n50 = len(imgs)
        combo50 = (pd.Series(kl50).rank().values / n50
                   + pd.Series((1 - obs50["msp_answer"]).values).rank().values / n50)
        filas_50.append(("rank(KL)+rank(1-MSP) (1×)", 1, roc_auc_score(y50, combo50), "50 (SC)", C_COMBO))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, filas, titulo in [
        (axes[0], filas_129, "(a) Full cohort (129 images)"),
        (axes[1], filas_50, "(b) Self-consistency subset (50 images)"),
    ]:
        if not filas:
            ax.axis("off")
            continue
        for nombre, costo, auroc, _, color in filas:
            es_nuestro = nombre.startswith(("KL", "rank"))
            ax.scatter(costo, auroc, s=95, zorder=3, color=color,
                       marker="D" if es_nuestro else "o", edgecolor="white", linewidth=0.8)
            ax.annotate(nombre.replace(" (", "\n("), (costo, auroc),
                        textcoords="offset points", xytext=(9, -2), fontsize=7.5)
        ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 10])
        ax.set_xticklabels(["1×", "2×", "10×"])
        ax.set_xlabel("Computational cost (model passes)")
        ax.set_title(titulo, fontsize=10)
        ax.set_xlim(0.7, 16)
        sns.despine(ax=ax)
    axes[0].set_ylabel("AUROC (error detection)")
    axes[0].set_ylim(0.48, 0.82)
    fig.suptitle("UQ performance vs. computational cost (P1)", y=1.03, fontsize=12)
    plt.tight_layout()
    out_fig = out_dir / "fig7_costo_vs_auroc.png"
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()
    print(f"[fig] Saved: {out_fig}")

    t4 = pd.DataFrame(
        [(n_, c, f"{a:.3f}", s) for n_, c, a, s, _ in filas_129 + filas_50],
        columns=["Method", "Cost", "AUROC", "Evaluation set"],
    ).sort_values(["Evaluation set", "AUROC"], ascending=[True, False])
    out_t4 = out_dir / "tabla_t4_costo_beneficio.csv"
    t4.to_csv(out_t4, index=False)
    print(f"[table] Saved: {out_t4}")
    return t4


# ------------------------------------------------------------------------------
# Fig 8: degeneración de la confianza verbalizada
# ------------------------------------------------------------------------------
def fig8_verbalized(cfg: Config, out_dir: Path) -> None:
    """Dos paneles: distribución de valores declarados + declarado vs. real."""
    verb_path = Path(cfg.paths.results) / "results_verbalized.csv"
    if not verb_path.exists():
        print("[fig8] No results_verbalized.csv — skipped")
        return
    v = pd.read_csv(verb_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    counts = v["verbalized_conf"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color=C_ENERGY, alpha=0.9, width=0.5)
    for i, val in enumerate(counts.values):
        ax.text(i, val + 1.5, str(val), ha="center", fontsize=10)
    ax.set_xlabel("Confidence stated by the model (0–100)")
    ax.set_ylabel("Number of images")
    ax.set_title(f"(a) The model states only {len(counts)} distinct values\nin {len(v)} images",
                 fontsize=10)
    sns.despine(ax=ax)

    ax = axes[1]
    real = v.groupby("verbalized_conf")["correct"].mean() * 100
    ax.scatter(real.index, real.values, s=95, color=C_KL, zorder=3, edgecolor="white")
    for x, yv in real.items():
        n_c = int((v["verbalized_conf"] == x).sum())
        ax.annotate(f"states {x}%, scores {yv:.1f}%\n(n={n_c})", (x, yv),
                    textcoords="offset points", xytext=(10, -14), fontsize=8)
    ax.plot([0, 100], [0, 100], color="0.5", linestyle="--", label="Perfect calibration", linewidth=1)
    ax.set_xlabel("Stated confidence (%)")
    ax.set_ylabel("Actual accuracy (%)")
    ax.set_xlim(80, 100)
    ax.set_ylim(60, 100)
    ax.set_title("(b) Stated vs. actual (overconfidence)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    sns.despine(ax=ax)

    plt.tight_layout()
    out_fig = out_dir / "fig8_verbalized.png"
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()
    print(f"[fig] Saved: {out_fig}")


# ------------------------------------------------------------------------------
# Fig 9: boxplots de las señales self-consistency
# ------------------------------------------------------------------------------
def fig9_sc_boxplots(cfg: Config, out_dir: Path) -> None:
    """Boxplots correcto/incorrecto de frac_other y entropía 3-vías (P1)."""
    sc_path = Path(cfg.paths.results) / "results_self_consistency.csv"
    if not sc_path.exists():
        print("[fig9] No results_self_consistency.csv — skipped")
        return
    sc = pd.read_csv(sc_path)
    d = sc[sc["prompt_id"] == "P1"].copy()
    d["pred"] = (d["self_consistency_frac_yes"] > d["sc_frac_no"]).astype(int)
    d["correct"] = (d["pred"] == d["label"]).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, col, titulo in [
        (axes[0], "sc_frac_other", "(a) frac_other (out-of-format drift)"),
        (axes[1], "self_consistency_entropy", "(b) 3-way vote entropy"),
    ]:
        data = d[["correct", col]].copy()
        data["correct"] = data["correct"].map({1: "Correct", 0: "Incorrect"})
        data["correct"] = pd.Categorical(data["correct"], ["Correct", "Incorrect"])
        sns.boxplot(data=data, x="correct", y=col, ax=ax, showfliers=False,
                    palette=BOX_COLORS, width=0.55, linewidth=1.4,
                    medianprops={"color": "black", "linewidth": 2})
        sns.stripplot(data=data, x="correct", y=col, ax=ax, color="black",
                      alpha=0.35, size=2.5, jitter=0.18)
        corr = data[data["correct"] == "Correct"][col]
        err = data[data["correct"] == "Incorrect"][col]
        mw = stats.mannwhitneyu(err, corr, alternative="greater")
        y_err = (data["correct"] == "Incorrect").astype(int).values
        auroc = roc_auc_score(y_err, d[col].values)
        ax.text(0.03, 0.97, f"AUROC = {auroc:.3f}\nMWU p = {mw.pvalue:.4f}",
                transform=ax.transAxes, fontsize=8.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.8", alpha=0.9))
        ax.set_title(f"{titulo}\n(self-consistency 10×, P1, n={len(d)})", fontsize=10)
        ax.set_xlabel("Model outcome")
        ax.set_ylabel(col)
        sns.despine(ax=ax)

    plt.tight_layout()
    out_fig = out_dir / "fig9_sc_boxplots.png"
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()
    print(f"[fig] Saved: {out_fig}")


# ------------------------------------------------------------------------------
# Fig 10: reliability diagram (calibración estilo Guo et al. 2017 / FUSE §5.2)
# ------------------------------------------------------------------------------
def fig10_reliability(signals: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Error empírico por bin vs. u calibrada media (Platt en train) (English).

    Cada señal se calibra con su propio Platt ajustado SOLO en train; los bins
    son equiprobables (10 bins ≈ 13 obs/bin con N=129). La diagonal es la
    calibración perfecta P(error) = u*.
    """
    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    for name, frame in signals.items():
        cal = calibration_analysis(frame, n_bootstrap=0)
        bins = cal["bins"]
        if bins.empty or np.isnan(cal["ece"]):
            continue
        ax.plot(bins["mean_u"], bins["empirical_error"], marker="o", markersize=5,
                linewidth=2, label=f"{name} (ECE={cal['ece']:.3f})",
                color=SIGNAL_COLORS.get(_signal_key(name)))

    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--",
            label="Perfect calibration", linewidth=1)
    ax.set_xlabel("Mean calibrated uncertainty u* per bin (Platt fit on train)")
    ax.set_ylabel("Empirical error rate per bin")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Reliability diagram — calibrated uncertainty vs. empirical error (P1)\n"
                 "(10 equal-mass bins; Platt scaling fit on train split only)", fontsize=10)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[fig] Saved: {out_path}")


# ------------------------------------------------------------------------------
# Tabla T5: discriminación + calibración lado a lado (espejo de FUSE Table 1)
# ------------------------------------------------------------------------------
def table_t5_calibracion(signals: dict[str, pd.DataFrame], out_path: Path) -> pd.DataFrame:
    """Tabla de discriminación (AUROC, TPR@FPR) y calibración (ECE, corr, Brier)."""
    rows = []
    for name, frame in signals.items():
        y_error, vals = _y_v(frame)
        if len(np.unique(vals)) < 2:
            continue
        auroc = roc_auc_score(y_error, vals)
        tf = tpr_at_fpr(1 - y_error, vals)
        cal = calibration_analysis(frame, n_bootstrap=0)
        rows.append({
            "Signal": name,
            "AUROC": f"{auroc:.3f}",
            "TPR@FPR5%": f"{tf['tpr_fpr05']:.3f}",
            "TPR@FPR10%": f"{tf['tpr_fpr10']:.3f}",
            "TPR@FPR20%": f"{tf['tpr_fpr20']:.3f}",
            "ECE": f"{cal['ece']:.3f}" if not np.isnan(cal["ece"]) else "—",
            "Cal. Pearson": f"{cal['pearson']:.3f}" if not np.isnan(cal["pearson"]) else "—",
            "Cal. Spearman": f"{cal['spearman']:.3f}" if not np.isnan(cal["spearman"]) else "—",
            "Brier": f"{cal['brier']:.3f}" if not np.isnan(cal["brier"]) else "—",
            "Cost": "1×",
        })
    t5 = pd.DataFrame(rows)
    t5.to_csv(out_path, index=False)
    print(f"[table] Saved: {out_path}")
    return t5


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Paper figures and tables (long format, English)")
    parser.add_argument("--prompt", default="P1", choices=["P1", "P4"])
    parser.add_argument("--signal", default=None,
                        help="Canonical signal name (default: winner chosen on train)")
    parser.add_argument("--filename", default="results_full.csv")
    args = parser.parse_args()

    cfg = Config()
    cfg.ensure_paths()
    df = load_results(cfg, filename=args.filename)
    master = load_master(cfg)

    out_dir = Path(cfg.paths.figures)
    out_dir.mkdir(parents=True, exist_ok=True)

    winner = args.signal or select_winner(df, args.prompt)
    if winner is None:
        raise RuntimeError(f"Could not select a winner for {args.prompt}")
    print(f"[figures] Prompt: {args.prompt} | Main signal: {winner}")

    signals = build_signal_set(df, args.prompt, winner)
    frame_winner = signals[f"KL cross-modal ({winner})"]
    obs_prompt = df[df["prompt_id"] == args.prompt]

    # Fig 2: un boxplot por señal
    slugs = {
        f"KL cross-modal ({winner})": "kl",
        "Entropy": "entropy",
        "1 - MSP": "1msp",
        "Energy": "energy",
        "rank(KL) + rank(1-MSP)": "rankcombo",
    }
    for name, frame in signals.items():
        slug = slugs.get(name, "klvt" if name.startswith(("KL v→t", "KL t→v")) else "signal")
        suffix = "" if slug == "kl" else f"_{slug}"
        fig2_boxplot(frame, name, out_dir / f"fig2_boxplot{suffix}.png")

    fig3_roc_pr(signals, out_dir / "fig3_roc_pr.png")
    fig4_accuracy_coverage(signals, out_dir / "fig4_accuracy_coverage.png")
    fig5_quadrants(obs_prompt, frame_winner, master, winner, out_dir / "fig5_quadrants.png")
    fig6_correlation(signals, out_dir / "fig6_correlacion_senales.png")

    table_t1(signals, out_dir / "tabla_t1_resultados.csv")
    table_t2(df, args.prompt, out_dir / "tabla_t2_ablaciones.csv")
    table_t3(out_dir / "tabla_t3_comparativa.csv")

    # Baselines de costo (verbalized 2×, self-consistency 10×) — si los CSV existen
    fig7_tabla_t4_costo(df, cfg, out_dir)
    fig8_verbalized(cfg, out_dir)
    fig9_sc_boxplots(cfg, out_dir)

    # Calibración (protocolo FUSE §5.2): reliability diagram + tabla discriminación/calibración
    fig10_reliability(signals, out_dir / "fig10_reliability.png")
    table_t5_calibracion(signals, out_dir / "tabla_t5_calibracion.csv")

    print(f"\n[figures] All figures and tables saved to {out_dir}")


if __name__ == "__main__":
    main()
