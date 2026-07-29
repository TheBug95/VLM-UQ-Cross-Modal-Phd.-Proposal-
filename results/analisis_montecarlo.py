"""Monte Carlo CV con seleccion anidada + media naive de variantes.

1) Media +- std de AUROC entre TODAS las variantes (lo que propone el usuario):
   muestra por que ese numero no responde "funciona el metodo?".
2) Monte Carlo CV: 200 splits aleatorios estratificados ~60/40; en cada rep
   se SELECCIONA la variante solo con la parte train y se evalua en la parte
   test. Media +- std honesta del procedimiento completo (seleccion + senal).
3) Referencia: la variante congelada (kl_t_v_L34_tau1.0_max) evaluada en los
   mismos 200 test folds, sin re-seleccionar.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

df = pd.read_csv("results/results_full.csv")
df["tau_s"] = df["tau"].astype(str)
m = df.signal_type == "kl_prompt_L34"
df.loc[m, "signal_type"] = "kl_prompt"
df.loc[m, "layer"] = "34"
df.loc[m, "tau"] = "1.0"
df.loc[m, "pooling"] = "mean"
df["signal"] = (df.signal_type + "_L" + df.layer.astype(str)
                + "_tau" + df.tau_s + "_" + df.pooling)

d = df[df.prompt_id == "P1"]
piv = d.pivot_table(index="image_filename", columns="signal",
                    values="value", aggfunc="first")
meta = d.drop_duplicates("image_filename").set_index("image_filename")
piv["label"] = meta["label"]
piv["correct"] = meta["correct"]

variantes = [c for c in piv.columns if c not in ("label", "correct")]
deployable = [c for c in variantes if not c.endswith("_roi")]

# --- 1) Media naive entre variantes (TODAS las imagenes) ---
aucs_all = []
for c in variantes:
    v = piv[c]
    mm = ~v.isna()
    y = 1 - piv.correct[mm]
    if y.sum() in (0, len(y)):
        continue
    aucs_all.append(roc_auc_score(y, v[mm]))
aucs_all = np.array(aucs_all)
print(f"[1] Media entre {len(aucs_all)} variantes (todas las imagenes): "
      f"AUROC = {aucs_all.mean():.3f} +- {aucs_all.std():.3f}")
print(f"    (incluye variantes malas: roi-oracle, tau=4, capas pobres, etc.)")

# --- 2) Monte Carlo CV con seleccion anidada ---
rng = np.random.default_rng(42)
REPS = 200
lab = piv.label.values
idx_pos = np.where(lab == 1)[0]
idx_neg = np.where(lab == 0)[0]

auc_nested, auc_frozen = [], []
picks = {}
FROZEN = "kl_t_v_L34_tau1.0_max"

for rep in range(REPS):
    tr = np.concatenate([rng.choice(idx_pos, size=int(0.6 * len(idx_pos)), replace=False),
                         rng.choice(idx_neg, size=int(0.6 * len(idx_neg)), replace=False)])
    te = np.setdiff1d(np.arange(len(piv)), tr)
    tr_df, te_df = piv.iloc[tr], piv.iloc[te]
    y_tr = (1 - tr_df.correct.values)
    y_te = (1 - te_df.correct.values)
    if y_te.sum() == 0 or y_te.sum() == len(y_te):
        continue

    # seleccion SOLO con la parte train de esta repeticion
    best, best_a = None, -1
    for c in deployable:
        a = roc_auc_score(y_tr, tr_df[c].values)
        if a > best_a:
            best, best_a = c, a
    picks[best] = picks.get(best, 0) + 1
    auc_nested.append(roc_auc_score(y_te, te_df[best].values))

    # la congelada, sin re-seleccionar
    auc_frozen.append(roc_auc_score(y_te, te_df[FROZEN].values))

auc_nested = np.array(auc_nested)
auc_frozen = np.array(auc_frozen)
print(f"\n[2] Monte Carlo CV ({len(auc_nested)} reps, seleccion anidada en cada split):")
print(f"    AUROC = {auc_nested.mean():.3f} +- {auc_nested.std():.3f}")
print(f"    rango observado: [{auc_nested.min():.3f}, {auc_nested.max():.3f}]")
print(f"\n[3] Variante congelada ({FROZEN}) en los mismos {len(auc_frozen)} test folds:")
print(f"    AUROC = {auc_frozen.mean():.3f} +- {auc_frozen.std():.3f}")

print("\n    Variantes elegidas en los 200 splits (top 5):")
for sig, n in sorted(picks.items(), key=lambda x: -x[1])[:5]:
    print(f"      {n:3d}/200  {sig}")
