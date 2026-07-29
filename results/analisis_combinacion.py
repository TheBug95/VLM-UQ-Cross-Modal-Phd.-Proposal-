"""Punto 1: combinacion de senales KL + 1-MSP sobre results_full.csv (formato largo).

Protocolo: pesos/estandarizacion ajustados SOLO en train; evaluacion en las 129
y confirmacion en val+test. Compara AUROC de cada combinacion contra las senales
individuales, con bootstrap 9999 para IC del 95%.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy import stats

rng = np.random.default_rng(42)
B = 9999

df = pd.read_csv("results/results_full.csv")
df["tau_s"] = df["tau"].astype(str)

# --- Tabla base: una fila por imagen x prompt, con KL ganadora y baselines ---
kl = df[(df.signal_type == "kl_t_v") & (df.pooling == "max") & (df.tau_s == "1.0")]
kl = kl[["image_filename", "prompt_id", "split", "label", "correct", "value"]]
kl = kl.rename(columns={"value": "kl"})
base = df.drop_duplicates(subset=["image_filename", "prompt_id"])
base = base[["image_filename", "prompt_id", "msp_answer", "entropy_answer", "p_yes"]]
m = kl.merge(base, on=["image_filename", "prompt_id"])
m["u_msp"] = 1 - m["msp_answer"]

def boot_ci(y, v):
    obs = roc_auc_score(y, v)
    n = len(y)
    idx = np.arange(n)
    boot = []
    for _ in range(B):
        b = rng.choice(idx, n, replace=True)
        if y[b].sum() == 0 or y[b].sum() == n:
            continue
        boot.append(roc_auc_score(y[b], v[b]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return obs, lo, hi

for prompt in ["P1", "P4"]:
    print(f"\n{'='*70}\nPROMPT {prompt}\n{'='*70}")
    d = m[m.prompt_id == prompt].reset_index(drop=True)
    y = (1 - d["correct"]).values
    tr = (d.split == "train").values
    vt = ~tr

    # Estandarizacion con estadisticos de TRAIN solamente
    kl_mu, kl_sd = d.loc[tr, "kl"].mean(), d.loc[tr, "kl"].std()
    msp_mu, msp_sd = d.loc[tr, "u_msp"].mean(), d.loc[tr, "u_msp"].std()
    d["z_kl"] = (d["kl"] - kl_mu) / kl_sd
    d["z_msp"] = (d["u_msp"] - msp_mu) / msp_sd

    # Ranks (sobre todo el set; ranks son ad-hoc pero estandar en UQ)
    d["r_kl"] = d["kl"].rank() / len(d)
    d["r_msp"] = d["u_msp"].rank() / len(d)

    # Logistic regression ajustada en train (2 features)
    lr = LogisticRegression()
    lr.fit(d.loc[tr, ["z_kl", "z_msp"]], y[tr])
    d["u_lr"] = lr.predict_proba(d[["z_kl", "z_msp"]])[:, 1]
    w = lr.coef_[0]
    print(f"LR (train): w_kl={w[0]:+.3f} w_msp={w[1]:+.3f}  bias={lr.intercept_[0]:+.3f}")

    # Correlacion entre senales: coinciden en los errores que marcan?
    rho = stats.spearmanr(d["kl"], d["u_msp"]).statistic
    print(f"Correlacion Spearman KL vs 1-MSP: rho={rho:+.3f}")

    combinaciones = {
        "KL sola": d["kl"].values,
        "1-MSP solo": d["u_msp"].values,
        "z(KL)+z(MSP)": (d["z_kl"] + d["z_msp"]).values,
        "max(z_kl, z_msp)": np.maximum(d["z_kl"], d["z_msp"]).values,
        "rank(KL)+rank(MSP)": (d["r_kl"] + d["r_msp"]).values,
        "LR(train)": d["u_lr"].values,
    }
    print(f"\n{'senal':22s} {'AUROC_all':>9s} {'IC95%':>17s} {'train':>7s} {'val+test':>8s}")
    for nombre, v in combinaciones.items():
        obs, lo, hi = boot_ci(y, v)
        a_tr = roc_auc_score(y[tr], v[tr])
        a_vt = roc_auc_score(y[vt], v[vt]) if y[vt].sum() > 0 else np.nan
        print(f"{nombre:22s} {obs:9.3f} [{lo:6.3f},{hi:6.3f}] {a_tr:7.3f} {a_vt:8.3f}")

    # Accuracy-coverage de la mejor combinacion vs KL sola vs MSP solo
    print("\nAccuracy-coverage (cobertura -> accuracy):")
    print(f"{'senal':22s} " + " ".join(f"{c:>6.1f}" for c in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    for nombre in ["KL sola", "1-MSP solo", "z(KL)+z(MSP)"]:
        v = combinaciones[nombre]
        orden = np.argsort(v)  # menor incertidumbre primero
        accs = []
        for cov in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            k = int(round(len(v) * cov))
            accs.append((1 - y[orden[:k]]).mean())
        print(f"{nombre:22s} " + " ".join(f"{a:6.3f}" for a in accs))
