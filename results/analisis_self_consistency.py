"""Análisis de results_self_consistency.csv (baseline 10×).

IMPORTANTE: el baseline corrió sobre 50 imágenes estratificadas. Toda
comparación de AUROC/AURC con nuestras señales se hace EN EL MISMO
subconjunto de 50, no en las 129.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats

sc = pd.read_csv("results/results_self_consistency.csv")
df = pd.read_csv("results/results_full.csv")
df["tau_s"] = df["tau"].astype(str)

print("=== 1. Qué hay en el archivo ===")
print(f"filas: {len(sc)} ({sc.image_filename.nunique()} imágenes × prompts {sorted(sc.prompt_id.unique())})")
print(f"labels: {sc.label.value_counts().to_dict()}")
print(f"frac_yes: min={sc.self_consistency_frac_yes.min():.2f} mediana={sc.self_consistency_frac_yes.median():.2f} max={sc.self_consistency_frac_yes.max():.2f}")
sat = ((sc.self_consistency_frac_yes == 0) | (sc.self_consistency_frac_yes == 1)).mean()
print(f"muestras saturadas (10/10 de acuerdo): {sat:.1%}")
print(f"entropy == ~0 (respuesta unánime): {(sc.self_consistency_entropy.abs() < 1e-6).mean():.1%}")

for prompt in ["P1", "P4"]:
    print(f"\n=== 2. Prompt {prompt} ===")
    d = sc[sc.prompt_id == prompt].copy()
    # predicción por voto mayoritario
    d["pred"] = (d.self_consistency_frac_yes > 0.5).astype(int)
    d["correct"] = (d.pred == d.label).astype(int)
    n_err = int((1 - d.correct).sum())
    print(f"accuracy (voto mayoritario) = {d.correct.mean():.3f}  ({d.correct.sum()}/{len(d)}, errores={n_err})")

    # --- Comparación JUSTA: nuestras señales en las mismas 50 imágenes ---
    imgs = d.image_filename
    obs = df[(df.prompt_id == prompt) & df.image_filename.isin(imgs)].drop_duplicates("image_filename")
    obs = obs.set_index("image_filename").loc[imgs]
    kl = df[(df.prompt_id == prompt) & (df.signal_type == "kl_t_v")
            & (df.pooling == "max") & (df.tau_s == "1.0")]
    kl = kl.set_index("image_filename").loc[imgs, "value"]

    # OJO: 'correct' debe ser el MISMO para todas las señales → usar el del voto
    y_err = 1 - d.set_index("image_filename").loc[imgs, "correct"].values
    n = len(imgs)
    candidatas = {
        "SC entropy (10×)": d.set_index("image_filename").loc[imgs, "self_consistency_entropy"].values,
        "KL ganadora (1×)": kl.values,
        "1-MSP (1×)": (1 - obs.msp_answer).values,
        "rank(KL)+rank(1-MSP) (1×)": (kl.rank() / n + (1 - obs.msp_answer).rank() / n).values,
    }
    print(f"{'señal':30s} {'AUROC':>7s}")
    for nombre, u in candidatas.items():
        if y_err.sum() == 0:
            print(f"{nombre:30s}   n/a (sin errores en el subconjunto)")
            continue
        print(f"{nombre:30s} {roc_auc_score(y_err, u):7.3f}")

    # ¿Coinciden las predicciones del voto con la corrida greedy principal?
    obs_main = df[(df.prompt_id == prompt)].drop_duplicates("image_filename").set_index("image_filename")
    pred_main = obs_main.loc[imgs, "pred"].values
    pred_vote = d.set_index("image_filename").loc[imgs, "pred"].values
    print(f"predicciones voto == greedy principal: {(pred_main == pred_vote).mean():.1%}")

print("\n=== 3. Distribución de frac_yes (P1) ===")
d1 = sc[sc.prompt_id == "P1"]
print(d1.self_consistency_frac_yes.value_counts().sort_index().to_string())
