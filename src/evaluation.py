"""src/evaluation.py — Métricas y comparación de variantes de señal de incertidumbre.

Carga results_full.csv y evalúa qué tan bien cada señal separa errores de
aciertos del modelo (detección de errores). Implementa AUROC, AUPRC y
comparaciones entre variantes (mean vs. roi_weighted).

Uso:
    python -m src.evaluation [--split train] [--compare mean roi]
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import Config


# ------------------------------------------------------------------------------
# Carga de resultados
# ------------------------------------------------------------------------------
def load_results(cfg: Config, filename: str = "results_full.csv") -> pd.DataFrame:
    """Carga el CSV de resultados de inferencia."""
    path = Path(cfg.paths.results) / filename
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Correr src.inference primero.")
    return pd.read_csv(path)


# ------------------------------------------------------------------------------
# Métricas de detección de errores
# ------------------------------------------------------------------------------
def error_detection_metrics(y_correct: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    """Calcula AUROC y AUPRC para detectar errores del modelo.

    Args:
        y_correct: array binario (1 = acierto, 0 = error).
        signal: señal de incertidumbre (mayor = más incierto).

    Returns:
        Dict con AUROC y AUPRC. Valores NaN si no hay errores o solo una clase.
    """
    y_error = 1 - y_correct
    if len(np.unique(y_error)) < 2:
        return {"auroc": np.nan, "auprc": np.nan}

    auroc = roc_auc_score(y_error, signal)
    auprc = average_precision_score(y_error, signal)
    return {"auroc": float(auroc), "auprc": float(auprc)}


# ------------------------------------------------------------------------------
# Comparación de variantes
# ------------------------------------------------------------------------------
def compare_pooling_variants(
    df: pd.DataFrame,
    split: str = "train",
    directions: list[str] | None = None,
) -> pd.DataFrame:
    """Compara mean/max vs. roi_weighted para cada capa, tau y dirección.

    Devuelve un DataFrame con columnas:
        variant, layer, tau, direction, pooling, auroc, auprc, n_errors
    """
    directions = directions or ["kl_v_t", "kl_t_v", "jsd"]
    rows = []

    df_split = df[df["split"] == split].copy()
    if len(df_split) == 0:
        warnings.warn(f"No hay datos para split={split}")
        return pd.DataFrame()

    y_correct = df_split["correct"].values
    n_errors = int((y_correct == 0).sum())

    for direction in directions:
        for layer in [17, 26, 34]:
            for tau in [1.0, 2.0, 4.0]:
                for pooling in ["mean", "max", "roi"]:
                    col = f"{direction}_L{layer}_tau{tau}_{pooling}"
                    if col not in df_split.columns:
                        continue
                    signal = df_split[col].values
                    metrics = error_detection_metrics(y_correct, signal)
                    rows.append({
                        "variant": col,
                        "layer": layer,
                        "tau": tau,
                        "direction": direction,
                        "pooling": pooling,
                        "auroc": metrics["auroc"],
                        "auprc": metrics["auprc"],
                        "n_errors": n_errors,
                    })

    return pd.DataFrame(rows)


def summarize_comparison(comp_df: pd.DataFrame) -> pd.DataFrame:
    """Resumen por pooling: media y desviación estándar de AUROC/AUPRC."""
    if comp_df.empty:
        return comp_df
    summary = (
        comp_df.groupby("pooling")[["auroc", "auprc"]]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    return summary


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación de señales de incertidumbre")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test", "all"])
    parser.add_argument("--compare", nargs="+", default=["mean", "roi"],
                        help="Poolings a comparar")
    parser.add_argument("--filename", default="results_full.csv")
    args = parser.parse_args()

    cfg = Config()
    df = load_results(cfg, filename=args.filename)

    if args.split != "all":
        df = df[df["split"] == args.split]

    print(f"[eval] {len(df)} filas cargadas (split={args.split})")
    print(f"[eval] Accuracy base del modelo: {df['correct'].mean():.3f}")

    comp = compare_pooling_variants(df, split=args.split if args.split != "all" else "train")
    if comp.empty:
        print("[eval] No se encontraron columnas de variantes para comparar.")
        return

    print("\n" + "=" * 70)
    print("COMPARACIÓN DE POOLING — detección de errores (AUROC / AUPRC)")
    print("=" * 70)
    print(comp.to_string(index=False))

    summary = summarize_comparison(comp)
    print("\n" + "=" * 70)
    print("RESUMEN POR POOLING")
    print("=" * 70)
    print(summary)

    # Guardar comparación
    out_path = Path(cfg.paths.results) / "evaluation_pooling_comparison.csv"
    comp.to_csv(out_path, index=False)
    print(f"\n[eval] Guardado: {out_path}")


if __name__ == "__main__":
    main()
