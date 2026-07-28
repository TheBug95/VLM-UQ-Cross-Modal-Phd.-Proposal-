"""src/figures.py — Generación de figuras y tablas del paper BIP 2026.

Genera:
    - Fig 2: Boxplot de u(x) en correctos vs. incorrectos.
    - Fig 3: Curvas ROC y PR de la señal principal y baselines.
    - Fig 4: Curvas accuracy-coverage de todas las señales.
    - Fig 5: Ejemplos de cuadrantes con transcripciones del oftalmólogo.
    - Tabla T1: Resultados principales (AUROC, AUPRC, Sens@80%Esp, costo).
    - Tabla T2: Ablaciones (capa × dirección × τ × pooling).
    - Tabla T3: Comparativa de propiedades vs. métodos de la literatura.

Uso:
    python -m src.figures [--signal kl_t_v_L26_tau2.0_rollout]
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.config import Config

# Estilo de figuras
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


# ------------------------------------------------------------------------------
# Carga de datos
# ------------------------------------------------------------------------------
def load_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga results_full.csv y master_table.csv."""
    results_path = Path(cfg.paths.results) / "results_full.csv"
    master_path = Path(cfg.paths.master_table)

    if not results_path.exists():
        raise FileNotFoundError(f"No existe {results_path}. Correr src.inference primero.")
    if not master_path.exists():
        raise FileNotFoundError(f"No existe {master_path}. Correr src.data primero.")

    df = pd.read_csv(results_path)
    master = pd.read_csv(master_path)
    return df, master


# ------------------------------------------------------------------------------
# Fig 2: Boxplot de u(x) en correctos vs. incorrectos
# ------------------------------------------------------------------------------
def fig2_boxplot(df: pd.DataFrame, signal: str, out_path: Path) -> None:
    """Boxplot + stripplot de u(x) en correctos vs. incorrectos."""
    fig, ax = plt.subplots(figsize=(6, 5))

    data = df[["correct", signal]].dropna()
    data["correct"] = data["correct"].map({1: "Correcto", 0: "Incorrecto"})

    # Boxplot
    sns.boxplot(data=data, x="correct", y=signal, ax=ax, palette=["#2ecc71", "#e74c3c"])
    # Stripplot (puntos individuales)
    sns.stripplot(data=data, x="correct", y=signal, ax=ax, color="black", alpha=0.3, size=3)

    # Estadísticas
    correct_vals = data[data["correct"] == "Correcto"][signal]
    error_vals = data[data["correct"] == "Incorrecto"][signal]

    from scipy import stats
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

    ax.text(0.05, 0.95, p_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Resultado del modelo")
    ax.set_ylabel(f"u(x) = {signal}")
    ax.set_title("Fig 2: Distribución de la señal de incertidumbre\nen casos correctos vs. incorrectos")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[fig] Guardada: {out_path}")


# ------------------------------------------------------------------------------
# Fig 3: Curvas ROC y PR
# ------------------------------------------------------------------------------
def fig3_roc_pr(df: pd.DataFrame, signals: dict[str, str], out_path: Path) -> None:
    """Curvas ROC y PR de la señal principal y baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    y_error = 1 - df["correct"].values

    # ROC
    ax = axes[0]
    for name, col in signals.items():
        if col not in df.columns:
            continue
        vals = df[col].values
        mask = ~np.isnan(vals)
        if mask.sum() < 2 or len(np.unique(vals[mask])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_error[mask], vals[mask])
        auroc = roc_auc_score(y_error[mask], vals[mask])
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", label="Azar (AUROC=0.500)", linewidth=1)
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title("Fig 3a: Curvas ROC")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # PR
    ax = axes[1]
    for name, col in signals.items():
        if col not in df.columns:
            continue
        vals = df[col].values
        mask = ~np.isnan(vals)
        if mask.sum() < 2 or len(np.unique(vals[mask])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_error[mask], vals[mask])
        auprc = average_precision_score(y_error[mask], vals[mask])
        ax.plot(recall, precision, label=f"{name} (AUPRC={auprc:.3f})", linewidth=2)

    baseline = y_error.mean()
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
def fig4_accuracy_coverage(df: pd.DataFrame, signals: dict[str, str], out_path: Path) -> None:
    """Curvas accuracy-coverage de todas las señales.

    Nota: para señales de confianza (como MSP), se convierte a incertidumbre
    como 1 - confianza antes de ordenar. Para señales de incertidumbre
    (KL, entropy, energy), se usa directamente.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    y_correct = df["correct"].values
    n = len(y_correct)

    # Señales que son de confianza (mayor = más seguro, no más incierto)
    CONFIDENCE_SIGNALS = {"msp_answer"}

    for name, col in signals.items():
        if col not in df.columns:
            continue
        vals = df[col].values
        mask = ~np.isnan(vals)
        if mask.sum() < 2:
            continue

        y_c = y_correct[mask]
        s = vals[mask]
        n_valid = len(y_c)

        # Convertir a incertidumbre: mayor = más incierto
        if col in CONFIDENCE_SIGNALS:
            s_uncertainty = 1.0 - s  # MSP -> 1 - MSP
        else:
            s_uncertainty = s

        # Ordenar por incertidumbre descendente (más incierto primero)
        order = np.argsort(-s_uncertainty)
        coverages = []
        accuracies = []

        for coverage in np.linspace(0.5, 1.0, 20):
            n_keep = int(np.ceil(n_valid * coverage))
            keep = order[:n_keep]
            if len(keep) == 0:
                continue
            coverages.append(coverage)
            accuracies.append(y_c[keep].mean())

        ax.plot(coverages, accuracies, label=name, linewidth=2, marker="o", markersize=3)

    # Línea de accuracy base
    base_acc = y_correct.mean()
    ax.axhline(base_acc, color="k", linestyle="--", label=f"Accuracy base ({base_acc:.3f})", linewidth=1)

    ax.set_xlabel("Cobertura (fracción de casos respondidos)")
    ax.set_ylabel("Accuracy del modelo")
    ax.set_title("Fig 4: Accuracy vs. Coverage\n(derivando el X% más incierto al oftalmólogo)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[fig] Guardada: {out_path}")


# ------------------------------------------------------------------------------
# Fig 5: Ejemplos de cuadrantes
# ------------------------------------------------------------------------------
def fig5_quadrants(df: pd.DataFrame, master: pd.DataFrame, signal: str, out_path: Path) -> None:
    """Ejemplos de cuadrantes: correcto/incorrecto × KL alta/baja."""
    # Merge para obtener transcripciones
    df_merged = df.merge(
        master[["image_filename", "transcription", "label"]],
        on="image_filename",
        how="left",
        suffixes=("", "_master"),
    )

    # Calcular cuadrantes
    median_kl = df_merged[signal].median()
    df_merged["kl_high"] = df_merged[signal] > median_kl
    df_merged["quadrant"] = df_merged.apply(
        lambda r: (
            "Correcto + KL alta" if r["correct"] == 1 and r["kl_high"]
            else "Correcto + KL baja" if r["correct"] == 1
            else "Error + KL alta" if r["kl_high"]
            else "Error + KL baja"
        ),
        axis=1,
    )

    # Seleccionar 2 ejemplos por cuadrante
    examples = []
    for quad in ["Correcto + KL baja", "Correcto + KL alta", "Error + KL alta", "Error + KL baja"]:
        subset = df_merged[df_merged["quadrant"] == quad].head(2)
        for _, row in subset.iterrows():
            examples.append({
                "image_filename": row["image_filename"],
                "quadrant": quad,
                "kl": row[signal],
                "p_yes": row["p_yes"],
                "label": row["label"],
                "transcription": row["transcription"],
            })

    # Crear figura con texto (no imágenes, porque no podemos mostrarlas aquí)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    y_pos = 0.95
    for ex in examples:
        text = (
            f"{ex['quadrant']}\n"
            f"Imagen: {ex['image_filename']}\n"
            f"KL: {ex['kl']:.4f} | P(yes): {ex['p_yes']:.3f} | Label: {ex['label']}\n"
            f"Transcripción: {ex['transcription'][:200]}...\n"
        )
        ax.text(0.05, y_pos, text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        y_pos -= 0.22

    ax.set_title("Fig 5: Ejemplos de cuadrantes (correcto/incorrecto × KL alta/baja)", fontsize=12, pad=20)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[fig] Guardada: {out_path}")


# ------------------------------------------------------------------------------
# Tabla T1: Resultados principales
# ------------------------------------------------------------------------------
def table_t1(df: pd.DataFrame, signals: dict[str, str], out_path: Path) -> pd.DataFrame:
    """Tabla de resultados principales."""
    y_error = 1 - df["correct"].values
    rows = []

    for name, col in signals.items():
        if col not in df.columns:
            continue
        vals = df[col].values
        mask = ~np.isnan(vals)
        if mask.sum() < 2 or len(np.unique(vals[mask])) < 2:
            continue

        auroc = roc_auc_score(y_error[mask], vals[mask])
        auprc = average_precision_score(y_error[mask], vals[mask])

        # Sensitivity @ 80% specificity
        from src.evaluation import sensitivity_at_specificity
        sens = sensitivity_at_specificity(df["correct"].values[mask], vals[mask], target_spec=0.80)

        rows.append({
            "Señal": name,
            "AUROC": f"{auroc:.3f}",
            "AUPRC": f"{auprc:.3f}",
            "Sens@80%Esp": f"{sens['sensitivity']:.3f}",
            "Costo": "1×",
        })

    # Añadir accuracy base
    base_acc = df["correct"].mean()
    rows.insert(0, {
        "Señal": "Accuracy base del modelo",
        "AUROC": "—",
        "AUPRC": "—",
        "Sens@80%Esp": "—",
        "Costo": "—",
    })

    t1 = pd.DataFrame(rows)
    t1.to_csv(out_path, index=False)
    print(f"[tabla] Guardada: {out_path}")
    return t1


# ------------------------------------------------------------------------------
# Tabla T2: Ablaciones
# ------------------------------------------------------------------------------
def table_t2(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Tabla de ablaciones: capa × dirección × τ × pooling."""
    y_error = 1 - df["correct"].values
    signal_cols = [c for c in df.columns if any(k in c for k in ["kl_", "jsd_"])]

    rows = []
    for col in signal_cols:
        vals = df[col].values
        mask = ~np.isnan(vals)
        if mask.sum() < 2 or len(np.unique(vals[mask])) < 2:
            continue

        try:
            auroc = roc_auc_score(y_error[mask], vals[mask])
            # Parsear columna
            parts = col.split("_")
            if col.startswith("kl_"):
                direction = "_".join(parts[:3])
                layer = parts[3].replace("L", "")
                tau = parts[4].replace("tau", "")
                pooling = parts[5] if len(parts) > 5 else "mean"
            elif col.startswith("jsd_"):
                direction = "jsd"
                layer = parts[1].replace("L", "")
                tau = parts[2].replace("tau", "")
                pooling = parts[3] if len(parts) > 3 else "mean"
            else:
                continue

            rows.append({
                "Capa": layer,
                "Dirección": direction,
                "τ": tau,
                "Pooling": pooling,
                "AUROC": f"{auroc:.3f}",
                "n": mask.sum(),
            })
        except Exception:
            continue

    t2 = pd.DataFrame(rows)
    t2 = t2.sort_values("AUROC", ascending=False)
    t2.to_csv(out_path, index=False)
    print(f"[tabla] Guardada: {out_path}")
    return t2


# ------------------------------------------------------------------------------
# Tabla T3: Comparativa de propiedades
# ------------------------------------------------------------------------------
def table_t3(out_path: Path) -> pd.DataFrame:
    """Tabla comparativa de propiedades vs. métodos de la literatura."""
    rows = [
        {
            "Método": "MC-Dropout",
            "Single-pass": "No",
            "Training-free": "No",
            "Cross-modal": "No",
            "Costo": "10–100×",
        },
        {
            "Método": "Semantic Entropy",
            "Single-pass": "No",
            "Training-free": "Sí",
            "Cross-modal": "No",
            "Costo": "10×",
        },
        {
            "Método": "UMPIRE",
            "Single-pass": "No",
            "Training-free": "Sí",
            "Cross-modal": "No",
            "Costo": "Multi-sample",
        },
        {
            "Método": "VIG-TUQ",
            "Single-pass": "Solo atención",
            "Training-free": "Sí",
            "Cross-modal": "Sí",
            "Costo": "1× (atención), 2× (JSD)",
        },
        {
            "Método": "SAPLMA (probes)",
            "Single-pass": "Sí",
            "Training-free": "No",
            "Cross-modal": "No",
            "Costo": "Supervisado",
        },
        {
            "Método": "Nuestro (KL cross-modal)",
            "Single-pass": "Sí",
            "Training-free": "Sí",
            "Cross-modal": "Sí",
            "Costo": "1×",
        },
    ]
    t3 = pd.DataFrame(rows)
    t3.to_csv(out_path, index=False)
    print(f"[tabla] Guardada: {out_path}")
    return t3


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generación de figuras y tablas del paper")
    parser.add_argument("--signal", default="kl_t_v_L26_tau2.0_rollout",
                        help="Columna de señal principal para Fig 2 y Fig 5")
    args = parser.parse_args()

    cfg = Config()
    cfg.ensure_paths()
    df, master = load_data(cfg)

    out_dir = Path(cfg.paths.figures)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Señales a comparar
    signals = {
        "KL (rollout)": "kl_t_v_L26_tau2.0_rollout",
        "KL (mean)": "kl_t_v_L26_tau2.0_mean",
        "Entropy": "entropy_answer",
        "MSP": "msp_answer",
        "Energy": "energy_answer",
    }

    # Fig 2: Boxplot
    fig2_boxplot(df, args.signal, out_dir / "fig2_boxplot.png")

    # Fig 3: ROC y PR
    fig3_roc_pr(df, signals, out_dir / "fig3_roc_pr.png")

    # Fig 4: Accuracy-coverage
    fig4_accuracy_coverage(df, signals, out_dir / "fig4_accuracy_coverage.png")

    # Fig 5: Cuadrantes
    fig5_quadrants(df, master, args.signal, out_dir / "fig5_quadrants.png")

    # Tablas
    table_t1(df, signals, out_dir / "tabla_t1_resultados.csv")
    table_t2(df, out_dir / "tabla_t2_ablaciones.csv")
    table_t3(out_dir / "tabla_t3_comparativa.csv")

    print(f"\n[figures] Todas las figuras y tablas guardadas en {out_dir}")


if __name__ == "__main__":
    main()
