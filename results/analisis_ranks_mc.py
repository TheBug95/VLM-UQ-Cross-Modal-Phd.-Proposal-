"""Monte Carlo CV de la combinacion por ranks + estadisticos de valores por variante."""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

df = pd.read_csv("results/results_full.csv")
df["tau_s"] = df["tau"].astype(str)
m = df.signal_type == "kl_prompt_L34"
df.loc[m, ["signal_type", "layer", "tau", "pooling"]] = ["kl_prompt", "34", "1.0", "mean"]
df["signal"] = (df.signal_type + "_L" + df.layer.astype(str)
                + "_tau" + df.tau_s + "_" + df.pooling)

d = df[df.prompt_id == "P1"]
piv = d.pivot_table(index="image_filename", columns="signal",
                    values="value", aggfunc="first")
meta = d.drop_duplicates("image_filename").set_index("image_filename")
piv["label"] = meta["label"]
piv["correct"] = meta["correct"]
piv["u_msp"] = 1 - meta["msp_answer"]

FROZEN = "kl_t_v_L34_tau1.0_max"
deployable = [c for c in piv.columns
              if c not in ("label", "correct", "u_msp") and not c.endswith("_roi")]

# --- MC CV de la combinacion por ranks (200 splits estratificados) ---
# La combinacion no tiene parametros: en cada split se evalua rank(KL)+rank(1-MSP)
# en la parte test. Dos sabores: KL congelada vs. KL re-seleccionada en cada split.
rng = np.random.default_rng(42)
lab = piv.label.values
idx_pos, idx_neg = np.where(lab == 1)[0], np.where(lab == 0)[0]

auc_combo_frozen, auc_combo_nested = [], []
for rep in range(200):
    tr = np.concatenate([rng.choice(idx_pos, int(0.6 * len(idx_pos)), replace=False),
                         rng.choice(idx_neg, int(0.6 * len(idx_neg)), replace=False)])
    te = np.setdiff1d(np.arange(len(piv)), tr)
    tr_df, te_df = piv.iloc[tr], piv.iloc[te]
    y_te = 1 - te_df.correct.values
    if y_te.sum() in (0, len(y_te)):
        continue

    # (a) KL congelada
    u = te_df[FROZEN].rank() + te_df["u_msp"].rank()
    auc_combo_frozen.append(roc_auc_score(y_te, u.values))

    # (b) KL re-seleccionada en ESTE split (solo con su train)
    y_tr = 1 - tr_df.correct.values
    best, best_a = None, -1
    for c in deployable:
        a = roc_auc_score(y_tr, tr_df[c].values)
        if a > best_a:
            best, best_a = c, a
    u = te_df[best].rank() + te_df["u_msp"].rank()
    auc_combo_nested.append(roc_auc_score(y_te, u.values))

auc_combo_frozen = np.array(auc_combo_frozen)
auc_combo_nested = np.array(auc_combo_nested)
print("=== Monte Carlo CV (200 splits) — combinacion rank(KL)+rank(1-MSP), P1 ===")
print(f"(a) KL congelada:      AUROC = {auc_combo_frozen.mean():.3f} +- {auc_combo_frozen.std():.3f}")
print(f"(b) KL re-seleccionada: AUROC = {auc_combo_nested.mean():.3f} +- {auc_combo_nested.std():.3f}")

# --- Estadisticos de valores por variante (P1, todas las imagenes) ---
print("\n=== Valores por familia de senal (P1, N=129; roi N=69) ===")
print(f"{'variante':34s} {'min':>9s} {'mediana':>9s} {'max':>9s} {'n':>4s}")
variantes = [c for c in piv.columns if c not in ("label", "correct", "u_msp")]
for c in sorted(variantes):
    v = piv[c].dropna()
    print(f"{c:34s} {v.min():9.3f} {v.median():9.3f} {v.max():9.3f} {len(v):4d}")

print("\n=== Baselines (P1) ===")
base = d.drop_duplicates("image_filename")
for nombre, col in [("entropy_answer", "entropy_answer"),
                    ("msp_answer", "msp_answer"),
                    ("energy_answer", "energy_answer")]:
    v = base[col]
    print(f"{nombre:34s} {v.min():9.4f} {v.median():9.4f} {v.max():9.4f} {len(v):4d}")
