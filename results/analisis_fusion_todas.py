"""Fusión de direcciones KL en TODAS las variantes: 8 poolings x 3 taus.

Para cada (pooling, tau): AUROC de kl_t_v sola, kl_v_t sola, fusiones
(suma, max, min, asimetría, rank-sum), JSD, y combinaciones con 1-MSP.
P1, N=129 (roi: N=69, solo patológicos — no comparable directamente).
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats

df = pd.read_csv("results/results_full.csv")
df["tau_s"] = df["tau"].astype(str)
obs = df[df.prompt_id == "P1"].drop_duplicates("image_filename").set_index("image_filename")

POOLINGS = ["mean", "max", "topk", "normw", "attn", "rollout", "headspec", "roi"]
TAUS = ["1.0", "2.0", "4.0"]

def get(st, pooling, tau):
    sub = df[(df.prompt_id == "P1") & (df.signal_type == st)
             & (df.pooling == pooling) & (df.tau_s == tau)]
    return sub.set_index("image_filename")["value"]

rows = []
for pooling in POOLINGS:
    for tau in TAUS:
        tv = get("kl_t_v", pooling, tau)
        vt = get("kl_v_t", pooling, tau)
        common = tv.dropna().index.intersection(vt.dropna().index)
        if len(common) < 20:
            continue
        m = pd.DataFrame({"tv": tv[common], "vt": vt[common]})
        y = 1 - obs.loc[common, "correct"].values
        n = len(m)
        u_msp = 1 - obs.loc[common, "msp_answer"].values

        rk_tv = m.tv.rank().values / n
        rk_vt = m.vt.rank().values / n
        rk_msp = pd.Series(u_msp).rank().values / n
        cand = {
            "t_v": m.tv.values,
            "v_t": m.vt.values,
            "suma": (m.tv + m.vt).values,
            "max": np.maximum(m.tv, m.vt).values,
            "min": np.minimum(m.tv, m.vt).values,
            "asim": (m.tv - m.vt).values,
            "rk_sum": rk_tv + rk_vt,
            "rk_tv+msp": rk_tv + rk_msp,
            "rk_tv+vt+msp": rk_tv + rk_vt + rk_msp,
        }
        rho = stats.spearmanr(m.tv, m.vt).statistic
        row = {"pooling": pooling, "tau": tau, "n": n, "rho_dir": round(rho, 2)}
        for k, v in cand.items():
            mask = ~np.isnan(v)
            row[k] = round(roc_auc_score(y[mask], v[mask]), 3) if mask.sum() > 20 else np.nan
        rows.append(row)

res = pd.DataFrame(rows)
res["mejor_fusion"] = res[["suma", "max", "min", "asim", "rk_sum"]].max(axis=1)
res["fusion_gana?"] = np.where(res.mejor_fusion > res.t_v, "SÍ", "no")
res["3rank_gana?"] = np.where(res["rk_tv+vt+msp"] > res["rk_tv+msp"], "SÍ", "no")

pd.set_option("display.width", 200)
print(res[["pooling", "tau", "n", "rho_dir", "t_v", "v_t", "suma", "max", "min",
           "asim", "rk_sum", "mejor_fusion", "fusion_gana?",
           "rk_tv+msp", "rk_tv+vt+msp", "3rank_gana?"]].to_string(index=False))

print("\n=== Mejor AUROC global por columna ===")
for c in ["t_v", "suma", "max", "min", "asim", "rk_sum", "rk_tv+msp", "rk_tv+vt+msp"]:
    i = res[c].idxmax()
    print(f"{c:14s} {res.loc[i,c]:.3f}  ({res.loc[i,'pooling']}, tau={res.loc[i,'tau']})")
