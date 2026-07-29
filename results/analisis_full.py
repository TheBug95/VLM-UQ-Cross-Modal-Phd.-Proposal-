"""Analisis rapido de results_full.csv (formato largo)."""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats

rng = np.random.default_rng(42)
df = pd.read_csv("results/results_full.csv")

df["tau_s"] = df["tau"].astype(str)

def get_vec(prompt, st, pl, tau):
    sub = df[(df.prompt_id == prompt) & (df.signal_type == st) &
             (df.pooling == pl) & (df.tau_s == str(tau))].sort_values("image_filename")
    return 1 - sub["correct"].values, sub["value"].values

def boot_ci(y, v, B=9999):
    n = len(y)
    obs = roc_auc_score(y, v)
    boot = []
    idx = np.arange(n)
    for _ in range(B):
        b = rng.choice(idx, n, replace=True)
        if y[b].sum() == 0 or y[b].sum() == n:
            continue
        boot.append(roc_auc_score(y[b], v[b]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return obs, lo, hi

def reporte(nombre, prompt, st, pl, tau):
    y, v = get_vec(prompt, st, pl, tau)
    obs, lo, hi = boot_ci(y, v)
    u = stats.mannwhitneyu(v[y == 1], v[y == 0], alternative="greater")
    n1, n0 = int(y.sum()), int((y == 0).sum())
    r_rb = 2 * u.statistic / (n1 * n0) - 1
    print(f"{nombre:28s} AUROC={obs:.3f} [{lo:.3f},{hi:.3f}]  p(MWU)={u.pvalue:.4f}  r_rb={r_rb:+.3f}")

print("=== SENAL GANADORA vs BASELINES (P1, ALL 129, bootstrap 9999) ===")
reporte("KL(t||v) max tau1", "P1", "kl_t_v", "max", 1.0)
base = df[df.prompt_id == "P1"].drop_duplicates("image_filename").sort_values("image_filename")
y = 1 - base["correct"].values
for nombre, vec in [("1-MSP (baseline)", 1 - base["msp_answer"].values),
                    ("entropy (baseline)", base["entropy_answer"].values)]:
    obs, lo, hi = boot_ci(y, vec)
    u = stats.mannwhitneyu(vec[y == 1], vec[y == 0], alternative="greater")
    print(f"{nombre:28s} AUROC={obs:.3f} [{lo:.3f},{hi:.3f}]  p(MWU)={u.pvalue:.4f}")

print("\n=== P4 ===")
reporte("KL(t||v) max tau1", "P4", "kl_t_v", "max", 1.0)
base4 = df[df.prompt_id == "P4"].drop_duplicates("image_filename").sort_values("image_filename")
y4 = 1 - base4["correct"].values
obs, lo, hi = boot_ci(y4, 1 - base4["msp_answer"].values)
print(f"{'1-MSP (baseline)':28s} AUROC={obs:.3f} [{lo:.3f},{hi:.3f}]")

# --- H4: Spearman u(x) vs cdr_grade en patologicos ---
print("\n=== H4: Spearman u(x) vs cdr_grade (69 patologicos) ===")
mt = pd.read_csv("data/master_table.csv")
cdr = mt[["image_filename", "cdr_grade", "label"]].copy()
sub = df[(df.prompt_id == "P1") & (df.signal_type == "kl_t_v") &
         (df.pooling == "max") & (df.tau_s == "1.0")][["image_filename", "value"]]
m = sub.merge(cdr, on="image_filename")
pat = m[(m.label == 1) & m.cdr_grade.notna()]
rho, p = stats.spearmanr(pat.value, pat.cdr_grade)
print(f"KL(t||v) max tau1: n={len(pat)}  Spearman rho={rho:+.3f}  p={p:.4f}")

# --- Accuracy-coverage para la senal ganadora P1 ---
print("\n=== Accuracy-Coverage (P1, KL(t||v) max tau1) ===")
sub = df[(df.prompt_id == "P1") & (df.signal_type == "kl_t_v") &
         (df.pooling == "max") & (df.tau_s == "1.0")].copy()
sub = sub.sort_values("value")  # menor incertidumbre primero
n = len(sub)
for cov in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    k = int(round(n * cov))
    acc = sub.head(k)["correct"].mean()
    print(f"cobertura={cov:.1f}  accuracy={acc:.3f}  (n={k})")

# --- Accuracy-coverage para baseline 1-MSP ---
print("\n=== Accuracy-Coverage (P1, baseline 1-MSP) ===")
b = base.copy()
b["u"] = 1 - b["msp_answer"]
b = b.sort_values("u")
for cov in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    k = int(round(len(b) * cov))
    acc = b.head(k)["correct"].mean()
    print(f"cobertura={cov:.1f}  accuracy={acc:.3f}  (n={k})")
