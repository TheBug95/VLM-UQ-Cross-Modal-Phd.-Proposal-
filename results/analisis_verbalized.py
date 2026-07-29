"""Análisis de results_verbalized.csv (baseline P5, confianza verbalizada 2×)."""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats

v = pd.read_csv("results/results_verbalized.csv")
df = pd.read_csv("results/results_full.csv")
df["tau_s"] = df["tau"].astype(str)

print("=== 1. Integridad ===")
print(f"filas: {len(v)} | parse_ok: {v.parse_ok.sum()}/{len(v)}")
print(f"accuracy turno 1 (verbalized) = {v.correct.mean():.3f}")
obs_main = df[df.prompt_id == "P1"].drop_duplicates("image_filename").set_index("image_filename")
match = (obs_main.loc[v.image_filename, "correct"].values == v.correct.values).mean()
print(f"¿correct coincide con la corrida principal? {match:.1%}")

print("\n=== 2. Distribución de la confianza verbalizada ===")
print(f"min={v.verbalized_conf.min()} mediana={v.verbalized_conf.median()} max={v.verbalized_conf.max()}")
print(v.verbalized_conf.value_counts().sort_index().head(15).to_string())
print(f"¿cuántos valores distintos? {v.verbalized_conf.nunique()}")

print("\n=== 3. AUROC detección de errores (u = 1 - conf/100) ===")
y_err = 1 - v.correct.values
mask = ~np.isnan(v.u_verbalized.values)
auroc_verb = roc_auc_score(y_err[mask], v.u_verbalized.values[mask])
print(f"verbalized (2×): AUROC = {auroc_verb:.3f}")

kl = df[(df.prompt_id == "P1") & (df.signal_type == "kl_t_v")
        & (df.pooling == "max") & (df.tau_s == "1.0")]
kl = kl.set_index("image_filename").loc[v.image_filename, "value"]
n = len(v)
cand = {
    "verbalized (2×)": v.u_verbalized.values,
    "KL ganadora (1×)": kl.values,
    "1-MSP (1×)": (1 - obs_main.loc[v.image_filename, "msp_answer"]).values,
    "rank(KL)+rank(1-MSP) (1×)": (kl.rank() / n + (1 - obs_main.loc[v.image_filename, "msp_answer"]).rank() / n).values,
}
for nombre, u in cand.items():
    m2 = ~np.isnan(u)
    print(f"  {nombre:28s} AUROC = {roc_auc_score(y_err[m2], u[m2]):.3f}")

print("\n=== 4. ¿Se correlaciona con nuestras señales? ===")
for nombre, u in [("KL", kl.values), ("1-MSP", (1 - obs_main.loc[v.image_filename, "msp_answer"]).values)]:
    rho = stats.spearmanr(v.u_verbalized.values[mask], u[mask]).statistic
    print(f"Spearman(verbalized, {nombre}) = {rho:+.3f}")

print("\n=== 5. ¿Y la combinación con verbalized? ===")
combo_verb = (v.u_verbalized.rank() / n + kl.rank() / n).values
print(f"rank(verb)+rank(KL):        AUROC = {roc_auc_score(y_err, combo_verb):.3f}")
combo_3 = (v.u_verbalized.rank() / n + kl.rank() / n
           + (1 - obs_main.loc[v.image_filename, "msp_answer"]).rank() / n).values
print(f"rank(verb)+rank(KL)+rank(MSP): AUROC = {roc_auc_score(y_err, combo_3):.3f}")

print("\n=== 6. Calibración bruta: ¿dice el modelo la verdad? ===")
for lo, hi in [(0, 60), (60, 80), (80, 90), (90, 95), (95, 100.01)]:
    sub = v[(v.verbalized_conf >= lo) & (v.verbalized_conf < hi)]
    if len(sub) > 0:
        print(f"conf [{lo:3.0f},{hi:3.0f}): n={len(sub):3d}  accuracy real = {sub.correct.mean():.3f}")
