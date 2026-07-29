"""src/figures.py — Generación de figuras y tablas del paper BIP 2026 (formato largo).

Trabaja sobre `results_full.csv` en formato largo (una fila por imagen × prompt ×
variante), reutilizando los helpers de `src.evaluation`.

Genera:
    - Fig 2: Boxplot de u(x) en correctos vs. incorrectos.
    - Fig 3: Curvas ROC y PR de la señal ganadora, baselines y combinación por ranks.
    - Fig 4: Curvas accuracy-coverage de las mismas señales.
    - Fig 5: Ejemplos de cuadrantes con transcripciones del oftalmólogo.
    - Tabla T1: Resultados principales (AUROC, AUPRC, Sens@80%Esp, costo).
    - Tabla T2: Ablaciones (dirección × τ × pooling; capa fija en 34).
    - Tabla T3: Comparativa de propiedades vs. métodos de la literatura.

Uso:
    python -m src.figures                      # prompt P1, ganadora elegida en train
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
    excess_aurc,
    signal_frame,
)

# Estilo de figuras
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

# Etiquetas legibles para las señales derivadas
LABELS = {
    "entropy": "Entropy",
    "one_minus_msp": "1 - MSP",
    "energy": "Energy",
}


# ------------------------------------------------------------------------------
# Conjunto de señales a comparar
# ------------------------------------------------------------------------------
def build_signal_set(df: pd.DataFrame, prompt: str, winner: str) -> dict[str, pd.DataFrame]:
    """Frames (image_filename, split, label, correct, value) de las señales del paper."""
    signals: dict[str, pd.DataFrame] = {}
    signals[f"KL cross-modal ({winner})"] = signal_frame(df, prompt, winner)
    for bl, label in LABELS.items():
        signals[label] = signal_frame(df, prompt, bl)
    rc = rank_combination_frame(df, prompt, winner)
    if not rc.empty:
        signals["rank(KL) + rank(1-MSP)"] = rc
    return {k: v.sort_values("image_filename").reset_index(drop=True) for k, v in signals.items()}


def _y_v(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(y_error, valor) sin NaN."""
    v = frame["value"].values.astype(float)
    mask = ~np.isnan(v)
    return 1 - frame["correct"].values[mask], v[mask]


# ------------------------------------------------------------------------------
# Fig 2: Boxplot de u(x) en correctos vs. incorrectos
# ------------------------------------------------------------------------------
def fig2_boxplot(frame: pd.DataFrame, signal_name: str, out_path: Path) -> None:
    """Boxplot + stripplot de u(x) en correctos vs. incorrectos."""
    fig, ax = plt.subplots(figsize=(6, 5))

    data = frame[["correct", "value"]].dropna().copy()
    data["correct"] = data["correct"].map({1: "Correcto", 0: "Incorrecto"})

    sns.boxplot(data=data, x="correct", y="value", ax=ax, palette=["#2ecc71", "#e74c3c"])
    sns.stripplot(data=data, x="correct", y="value", ax=ax, color="black", alpha=0.3, size=3)

    correct_vals = data[data["correct"] == "Correcto"]["value"]
    error_vals = data[data["correct"] == "Incorrecto"]["value"]

    try:
        mw = stats.mannwhitneyu(error_vals, correct_vals, alternative="greater", method="asymptotic")
        n1, n2 = len(error_vals), len(correct_vals)
        U = mw.statistic
        mu_U = n1 * n2 / 2
        sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (U - mu_U) / sigma_U if sigma_U > 0 else 0
        r = abs(z) / np.sqrt(n1 + n2)
        p_text = f"Mann-Whitney: p = {mw.pvalue:.4f}\nEffect size r = {r:.3f}"
    except Exception:
        p_text = "Mann-Whitney: no disponible"

    # Métricas de la señal mostrada (AUROC y Excess-AURC)
    data_all = frame[["correct", "value"]].dropna()
    y_c = data_all["correct"].values
    v = data_all["value"].values.astype(float)
    try:
        auroc_val = roc_auc_score(1 - y_c, v)
        exc_val = excess_aurc(y_c, v)["excess_aurc_norm"]
        p_text += f"\nAUROC = {auroc_val:.3f} | Excess-AURC = {exc_val:.3f}"
    except Exception:
        pass

    ax.text(0.05, 0.95, p_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Resultado del modelo")
    ax.set_ylabel(f"u(x) = {signal_name}")
    ax.set_title("Fig 2: Distribución de la señal de incertidumbre\nen casos correctos vs. incorrectos")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[fig] Guardada: {out_path}")


# ------------------------------------------------------------------------------
# Fig 3: Curvas ROC y PR
# ------------------------------------------------------------------------------
def fig3_roc_pr(signals: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Curvas ROC y PR de la señal ganadora, baselines y combinación."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for name, frame in signals.items():
        y_error, vals = _y_v(frame)
        if len(np.unique(vals)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_error, vals)
        auroc = roc_auc_score(y_error, vals)
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", label="Azar (AUROC=0.500)", linewidth=1)
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title("Fig 3a: Curvas ROC (detección de errores)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    y_error_ref = None
    for name, frame in signals.items():
        y_error, vals = _y_v(frame)
        y_error_ref = y_error
        if len(np.unique(vals)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_error, vals)
        auprc = average_precision_score(y_error, vals)
        ax.plot(recall, precision, label=f"{name} (AUPRC={auprc:.3f})", linewidth=2)

    if y_error_ref is not None:
        baseline = y_error_ref.mean()
        ax.axhline(baseline, color="k", linestyle="--", label=f"Azar (AUPRC={baseline:.3f})", linewidth=1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Fig 3b: Curvas Precision-Recall")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[fig] Guardada: {out_path}")


# ------------------------------------------------------------------------------
# Fig 4: Curvas accuracy-coverage
# ------------------------------------------------------------------------------
def fig4_accuracy_coverage(signals: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Accuracy del subconjunto retenido al derivar el X% más incierto."""
    fig, ax = plt.subplots(figsize=(7.5, 5))

    base_acc = None
    for name, frame in signals.items():
        y_error, vals = _y_v(frame)
        y_correct = 1 - y_error
        base_acc = y_correct.mean()
        n_valid = len(vals)

        # AURC de la señal (área bajo la curva riesgo-cobertura completa)
        arc = excess_aurc(y_correct, vals)
        label = f"{name} (AURC={arc['aurc']:.3f}, Excess={arc['excess_aurc_norm']:.3f})"

        order = np.argsort(-vals)  # más incierto primero
        coverages, accuracies = [], []
        for coverage in np.linspace(0.5, 1.0, 20):
            n_keep = int(np.ceil(n_valid * coverage))
            # retenemos los MENOS inciertos: los últimos del orden descendente
            keep = order[n_valid - n_keep:]
            if len(keep) == 0:
                continue
            coverages.append(coverage)
            accuracies.append(y_correct[keep].mean())

        ax.plot(coverages, accuracies, label=label, linewidth=2, marker="o", markersize=3)

    if base_acc is not None:
        ax.axhline(base_acc, color="k", linestyle="--",
                   label=f"Accuracy base ({base_acc:.3f})", linewidth=1)

    ax.set_xlabel("Cobertura (fracción de casos respondidos)")
    ax.set_ylabel("Accuracy del modelo")
    ax.set_title("Fig 4: Accuracy vs. Coverage\n(derivando el X% más incierto; AURC = área bajo 1−accuracy)")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[fig] Guardada: {out_path}")


# ------------------------------------------------------------------------------
# Fig 5: Ejemplos de cuadrantes
# ------------------------------------------------------------------------------
def fig5_quadrants(df: pd.DataFrame, frame: pd.DataFrame, master: pd.DataFrame,
                   signal_name: str, out_path: Path) -> None:
    """Ejemplos de cuadrantes: correcto/incorrecto × u(x) alta/baja."""
    obs = df.drop_duplicates("image_filename")[["image_filename", "p_yes"]]
    df_merged = (
        frame.merge(obs, on="image_filename")
        .merge(master[["image_filename", "transcription"]], on="image_filename", how="left")
    )

    median_u = df_merged["value"].median()
    df_merged["u_high"] = df_merged["value"] > median_u
    df_merged["quadrant"] = df_merged.apply(
        lambda r: (
            "Correcto + u(x) alta" if r["correct"] == 1 and r["u_high"]
            else "Correcto + u(x) baja" if r["correct"] == 1
            else "Error + u(x) alta" if r["u_high"]
            else "Error + u(x) baja"
        ),
        axis=1,
    )

    examples = []
    for quad in ["Correcto + u(x) baja", "Correcto + u(x) alta", "Error + u(x) alta", "Error + u(x) baja"]:
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

    y_pos = 0.95
    for ex in examples:
        text = (
            f"{ex['quadrant']}\n"
            f"Imagen: {ex['image_filename']}\n"
            f"u(x): {ex['u']:.4f} | P(yes): {ex['p_yes']:.3f} | Label: {ex['label']}\n"
            f"Transcripción: {ex['transcription'][:200]}...\n"
        )
        ax.text(0.05, y_pos, text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        y_pos -= 0.22

    ax.set_title(f"Fig 5: Ejemplos de cuadrantes ({signal_name})", fontsize=12, pad=20)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[fig] Guardada: {out_path}")


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
            "Señal": name,
            "AUROC": f"{auroc:.3f}",
            "AUPRC": f"{auprc:.3f}",
            "AURC": f"{arc['aurc']:.4f}",
            "Excess-AURC": f"{arc['excess_aurc_norm']:.3f}",
            "Sens@80%Esp": f"{sens['sensitivity']:.3f}",
            "Costo": "1×",
        })

    if signals:
        first = next(iter(signals.values()))
        base_acc = first["correct"].mean()
        rows.insert(0, {
            "Señal": f"Accuracy base del modelo ({base_acc:.3f})",
            "AUROC": "—", "AUPRC": "—", "AURC": "—", "Excess-AURC": "—",
            "Sens@80%Esp": "—", "Costo": "—",
        })

    t1 = pd.DataFrame(rows)
    t1.to_csv(out_path, index=False)
    print(f"[tabla] Guardada: {out_path}")
    return t1


# ------------------------------------------------------------------------------
# Tabla T2: Ablaciones
# ------------------------------------------------------------------------------
def table_t2(df: pd.DataFrame, prompt: str, out_path: Path) -> pd.DataFrame:
    """Ablaciones: dirección × τ × pooling (capa 34), AUROC en ALL y train.

    En formato largo no hace falta parsear nombres: las dimensiones ya vienen
    en columnas (signal_type, layer, tau, pooling).
    """
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
            "Tipo": stype,
            "Capa": layer,
            "τ": tau,
            "Pooling": pooling,
            "AUROC_all": round(float(roc_auc_score(y, v[m])), 3),
            "AUROC_train": round(float(auroc_train), 3) if not np.isnan(auroc_train) else "—",
            "n": int(m.sum()),
        })

    t2 = pd.DataFrame(rows).sort_values("AUROC_all", ascending=False)
    t2.to_csv(out_path, index=False)
    print(f"[tabla] Guardada: {out_path}")
    return t2


# ------------------------------------------------------------------------------
# Tabla T3: Comparativa de propiedades
# ------------------------------------------------------------------------------
def table_t3(out_path: Path) -> pd.DataFrame:
    """Tabla comparativa de propiedades vs. métodos de la literatura."""
    rows = [
        {"Método": "MC-Dropout", "Single-pass": "No", "Training-free": "No",
         "Cross-modal": "No", "Costo": "10–100×"},
        {"Método": "Semantic Entropy", "Single-pass": "No", "Training-free": "Sí",
         "Cross-modal": "No", "Costo": "10×"},
        {"Método": "UMPIRE", "Single-pass": "No", "Training-free": "Sí",
         "Cross-modal": "No", "Costo": "Multi-sample"},
        {"Método": "VIG-TUQ", "Single-pass": "Solo atención", "Training-free": "Sí",
         "Cross-modal": "Sí", "Costo": "1× (atención), 2× (JSD)"},
        {"Método": "SAPLMA (probes)", "Single-pass": "Sí", "Training-free": "No",
         "Cross-modal": "No", "Costo": "Supervisado"},
        {"Método": "Nuestro (KL cross-modal)", "Single-pass": "Sí", "Training-free": "Sí",
         "Cross-modal": "Sí", "Costo": "1×"},
    ]
    t3 = pd.DataFrame(rows)
    t3.to_csv(out_path, index=False)
    print(f"[tabla] Guardada: {out_path}")
    return t3


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generación de figuras y tablas del paper (formato largo)")
    parser.add_argument("--prompt", default="P1", choices=["P1", "P4"])
    parser.add_argument("--signal", default=None,
                        help="Nombre canónico de la señal principal (default: ganadora en train)")
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
        raise RuntimeError(f"No se pudo seleccionar ganadora para {args.prompt}")
    print(f"[figures] Prompt: {args.prompt} | Señal principal: {winner}")

    signals = build_signal_set(df, args.prompt, winner)
    frame_winner = signals[f"KL cross-modal ({winner})"]
    obs_prompt = df[df["prompt_id"] == args.prompt]

    fig2_boxplot(frame_winner, winner, out_dir / "fig2_boxplot.png")
    rc_key = "rank(KL) + rank(1-MSP)"
    if rc_key in signals:
        fig2_boxplot(signals[rc_key], rc_key, out_dir / "fig2_boxplot_rankcombo.png")
    fig3_roc_pr(signals, out_dir / "fig3_roc_pr.png")
    fig4_accuracy_coverage(signals, out_dir / "fig4_accuracy_coverage.png")
    fig5_quadrants(obs_prompt, frame_winner, master, winner, out_dir / "fig5_quadrants.png")

    table_t1(signals, out_dir / "tabla_t1_resultados.csv")
    table_t2(df, args.prompt, out_dir / "tabla_t2_ablaciones.csv")
    table_t3(out_dir / "tabla_t3_comparativa.csv")

    print(f"\n[figures] Todas las figuras y tablas guardadas en {out_dir}")


if __name__ == "__main__":
    main()
