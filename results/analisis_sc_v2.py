"""Análisis de results_self_consistency.csv (baseline 10×, voto 3-vías, T=1.5).

Comparación JUSTA: todas las señales evaluadas en las mismas 50 imágenes.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats

sc = pd.read_csv("results/results_self_consistency.csv")
df = pd.read_csv("results/results_full.csv")
df["tau_s"] = df["tau"].astype(str)

print("=== 1. Panorama del voto 3-vías ===")
print(f"frac_other: mediana={sc.sc_frac_other.median():.2f}  max={sc.sc_frac_other.max():.2f}")
print(f"imágenes sin deriva (frac_other=0): {(sc.sc_frac_other == 0).mean():.1%}")
print(f"imágenes con deriva total (frac_other=1): {(sc.sc_frac_other == 1).mean():.1%}")
unanim = (sc.self_consistency_entropy < 1e-9).mean()
print(f"votos unánimes (entropía 3-vías = 0): {unanim:.1%}")

for prompt in ["P1", "P4"]:
    print(f"\n=== 2. Prompt {prompt} ===")
    d = sc[sc.prompt_id == prompt].copy()
    d["pred"] = (d.self_consistency_frac_yes > d.sc_frac_no).astype(int)
    d["correct"] = (d.pred == d.label).astype(int)
    n_err = int((1 - d.correct).sum())
    print(f"accuracy (voto mayoritario yes/no) = {d.correct.mean():.3f} (errores={n_err})")

    imgs = d.image_filename
    obs = df[(df.prompt_id == prompt) & df.image_filename.isin(imgs)].drop_duplicates("image_filename")
    obs = obs.set_index("image_filename").loc[imgs]
    kl = df[(df.prompt_id == prompt) & (df.signal_type == "kl_t_v")
            & (df.pooling == "max") & (df.tau_s == "1.0")]
    kl = kl.set_index("image_filename").loc[imgs, "value"]

    y_err = 1 - d.set_index("image_filename").loc[imgs, "correct"].values
    n = len(imgs)
    cand = {
        "SC entropía 3-vías (10×)": d.set_index("image_filename").loc[imgs, "self_consistency_entropy"].values,
        "SC frac_other (10×)": d.set_index("image_filename").loc[imgs, "sc_frac_other"].values,
        "SC entropía binaria (10×)": d.set_index("image_filename").loc[imgs, "sc_entropy_binary"].values,
        "KL ganadora (1×)": kl.values,
        "1-MSP (1×)": (1 - obs.msp_answer).values,
        "rank(KL)+rank(1-MSP) (1×)": (kl.rank() / n + (1 - obs.msp_answer).rank() / n).values,
    }
    print(f"{'señal':32s} {'AUROC':>7s}")
    for nombre, u in cand.items():
        if y_err.sum() == 0 or np.isnan(u).all():
            print(f"{nombre:32s}   n/a")
            continue
        print(f"{nombre:32s} {roc_auc_score(y_err, u):7.3f}")

    # ¿El voto cambia predicciones vs greedy?
    obs_main = df[df.prompt_id == prompt].drop_duplicates("image_filename").set_index("image_filename")
    pred_main = obs_main.loc[imgs, "pred"].values
    pred_vote = d.set_index("image_filename").loc[imgs, "pred"].values
    print(f"predicciones voto == greedy: {(pred_main == pred_vote).mean():.1%}")

    # correlación frac_other con KL
    rho = stats.spearmanr(d.set_index("image_filename").loc[imgs, "sc_frac_other"], kl.values).statistic
    print(f"Spearman(frac_other, KL) = {rho:+.3f}")

print("\n=== 3. ¿Qué imágenes tienen más deriva (frac_other) en P1? ===")
d1 = sc[sc.prompt_id == "P1"].copy()
d1["pred"] = (d1.self_consistency_frac_yes > d1.sc_frac_no).astype(int)
d1["correct"] = (d1.pred == d1.label).astype(int)
top = d1.nlargest(6, "sc_frac_other")
print(top[["image_filename", "sc_frac_other", "sc_frac_no", "self_consistency_frac_yes",
           "label", "correct", "sc_samples"]].to_string(index=False))
