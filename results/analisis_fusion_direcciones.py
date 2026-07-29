"""Fusión de las dos direcciones de KL como señal UQ (tabla, sin GPU).

Pregunta: kl_t_v y kl_v_t ya están calculadas; ¿fusionarlas da una señal
más fuerte que kl_t_v sola? Se prueban suma, max, diff y suma de ranks,
para pooling max y mean, tau=1, P1, sobre las 129 imágenes.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats

df = pd.read_csv("results/results_full.csv")
df["tau_s"] = df["tau"].astype(str)

for pooling in ["max", "mean"]:
    print(f"\n{'='*60}\nPOOLING = {pooling}  (P1, tau=1, N=129)\n{'='*60}")
    tv = df[(df.prompt_id == "P1") & (df.signal_type == "kl_t_v")
            & (df.pooling == pooling) & (df.tau_s == "1.0")]
    tv = tv[["image_filename", "correct", "value"]].rename(columns={"value": "kl_tv"})
    vt = df[(df.prompt_id == "P1") & (df.signal_type == "kl_v_t")
            & (df.pooling == pooling) & (df.tau_s == "1.0")]
    vt = vt[["image_filename", "value"]].rename(columns={"value": "kl_vt"})
    m = tv.merge(vt, on="image_filename")
    y = 1 - m.correct.values
    n = len(m)

    rho = stats.spearmanr(m.kl_tv, m.kl_vt).statistic
    print(f"Correlación entre direcciones: Spearman = {rho:+.3f}")

    candidatas = {
        "kl_t_v sola (ganadora actual)": m.kl_tv.values,
        "kl_v_t sola": m.kl_vt.values,
        "suma (KL simétrica)": (m.kl_tv + m.kl_vt).values,
        "max(t_v, v_t)": np.maximum(m.kl_tv, m.kl_vt).values,
        "min(t_v, v_t)": np.minimum(m.kl_tv, m.kl_vt).values,
        "t_v - v_t (asimetría)": (m.kl_tv - m.kl_vt).values,
        "rank(t_v) + rank(v_t)": (m.kl_tv.rank() / n + m.kl_vt.rank() / n).values,
    }
    for nombre, v in candidatas.items():
        a = roc_auc_score(y, v)
        print(f"  {nombre:32s} AUROC = {a:.3f}")

    # JSD ya calculada en la tabla (la fusión "oficial")
    j = df[(df.prompt_id == "P1") & (df.signal_type == "jsd")
           & (df.pooling == pooling) & (df.tau_s == "1.0")]
    j = j.set_index("image_filename").loc[m.image_filename, "value"].values
    print(f"  {'JSD (fusión con mezcla, saturada)':32s} AUROC = {roc_auc_score(y, j):.3f}")

    # ¿Y si además le sumamos el MSP? (la combinación estrella + direcciones)
    obs = df[df.prompt_id == "P1"].drop_duplicates("image_filename")
    obs = obs.set_index("image_filename").loc[m.image_filename]
    u_msp = 1 - obs.msp_answer.values
    combo3 = (m.kl_tv.rank() / n + m.kl_vt.rank() / n + pd.Series(u_msp).rank() / n).values
    combo2 = (m.kl_tv.rank() / n + pd.Series(u_msp).rank() / n).values
    print(f"  {'rank(t_v)+rank(v_t)+rank(1-MSP)':32s} AUROC = {roc_auc_score(y, combo3):.3f}")
    print(f"  {'rank(t_v)+rank(1-MSP) [actual]':32s} AUROC = {roc_auc_score(y, combo2):.3f}")
