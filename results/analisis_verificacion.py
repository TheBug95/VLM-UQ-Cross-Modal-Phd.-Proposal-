"""Analisis de verificacion_manual_kl.csv: robustez cross-GPU completa."""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

m = pd.read_csv("results/verificacion_manual_kl.csv")
df = pd.read_csv("results/results_full.csv")

obs = df[df.prompt_id == "P1"].drop_duplicates("image_filename")[
    ["image_filename", "logit_yes", "logit_no", "p_yes", "msp_answer", "label", "correct"]]
obs = obs.rename(columns={"logit_yes": "ly_orig", "logit_no": "ln_orig",
                          "correct": "correct_orig"})
mm = m.merge(obs, on="image_filename")
assert (mm.correct == mm.correct_orig).all(), "desalineacion de filas!"

y = 1 - mm.correct.values

print("=== Confirmacion numeros reportados ===")
print(f"Spearman KL: {stats.spearmanr(mm.kl_manual, mm.kl_csv).statistic:.4f}")
print(f"AUROC manual={roc_auc_score(y, mm.kl_manual):.4f}  csv={roc_auc_score(y, mm.kl_csv):.4f}")
dabs = np.abs(mm.kl_manual - mm.kl_csv)
print(f"diff |KL|: media={dabs.mean():.2f}  mediana={dabs.median():.2f}  max={dabs.max():.2f}")

print("\n=== Estabilidad de logits entre GPUs ===")
dly = np.abs(mm.logit_yes_manual - mm.ly_orig)
dln = np.abs(mm.logit_no_manual - mm.ln_orig)
print(f"|diff logit_yes|: media={dly.mean():.3f}  max={dly.max():.3f}")
print(f"|diff logit_no |: media={dln.mean():.3f}  max={dln.max():.3f}")
pred_new = (mm.logit_yes_manual > mm.logit_no_manual).astype(int)
pred_orig = (mm.ly_orig > mm.ln_orig).astype(int)
n_flip = int((pred_new != pred_orig).sum())
print(f"predicciones que cambian de GPU: {n_flip} de {len(mm)}")

print("\n=== Distribucion de las diferencias de KL ===")
d = mm.kl_manual - mm.kl_csv
print(f"media={d.mean():+.2f} (corrimiento comun)   std={d.std():.2f} (ruido por imagen)")
print(f"percentiles |diff|: p50={dabs.quantile(.5):.2f}  p90={dabs.quantile(.9):.2f}  p99={dabs.quantile(.99):.2f}")

print("\n=== Combinacion por ranks con KL de la GPU nueva ===")
u_msp = 1 - mm.msp_answer.values
n = len(mm)
combo_new = pd.Series(mm.kl_manual).rank().values / n + pd.Series(u_msp).rank().values / n
combo_csv = pd.Series(mm.kl_csv).rank().values / n + pd.Series(u_msp).rank().values / n
print(f"AUROC combo (KL GPU nueva) = {roc_auc_score(y, combo_new):.4f}")
print(f"AUROC combo (KL original)  = {roc_auc_score(y, combo_csv):.4f}")

print("\n=== MSP/entropy estables? (salen de logits) ===")
p_new = 1.0 / (1.0 + np.exp(mm.logit_no_manual - mm.logit_yes_manual))
u_msp_new = 1 - np.maximum(p_new, 1 - p_new)
print(f"AUROC 1-MSP (GPU nueva) = {roc_auc_score(y, u_msp_new):.4f}")
print(f"AUROC 1-MSP (original)  = {roc_auc_score(y, u_msp):.4f}")

print("\n=== Errores: mantienen ranks altos en la GPU nueva? ===")
rank_new = mm.kl_manual.rank() / n
rank_csv = mm.kl_csv.rank() / n
err = mm.correct == 0
print(f"rank medio de errores: csv={rank_csv[err].mean():.3f}  nueva GPU={rank_new[err].mean():.3f}")
print(f"rank medio de aciertos: csv={rank_csv[~err].mean():.3f}  nueva GPU={rank_new[~err].mean():.3f}")
