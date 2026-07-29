"""AURC / Excess-AURC como segunda métrica UQ + correlaciones entre métricas y señales.

Definiciones (Geifman & El-Yaniv 2017, selective prediction):
    - Ordenar por u(x) descendente (más incierto = se deriva primero).
    - risk(c) = tasa de error en el c·N MENOS incierto (lo que el modelo responde).
    - AURC = área bajo risk(coverage), discreta: mean_k risk(k/N). Menor = mejor.
    - Excess-AURC = AURC - AURC_oracle (oracle: errores al final, risk cae a 0).
    - Excess normalizado = (AURC - AURC_oracle) / (AURC_random - AURC_oracle):
      0 = oracle, 1 = azar.

Se calcula para todas las variantes deployable y se responde:
    1. ¿AUROC y AURC ordenan igual las variantes? (Spearman entre métricas)
    2. ¿Las señales principales ordenan igual las imágenes? (matriz Spearman)
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats

df = pd.read_csv("results/results_full.csv")
df["tau_s"] = df["tau"].astype(str)
obs = df[df.prompt_id == "P1"].drop_duplicates("image_filename").set_index("image_filename")
y_all = 1 - obs.correct.values  # 1 = error


def aurc(y_error: np.ndarray, u: np.ndarray) -> float:
    """Área bajo la curva riesgo-cobertura (menor = mejor)."""
    order = np.argsort(u)  # menos incierto primero
    y_sorted = y_error[order]
    n = len(y_sorted)
    risks = [y_sorted[:k].mean() for k in range(1, n + 1)]
    return float(np.mean(risks))


def aurc_oracle(y_error: np.ndarray) -> float:
    """AURC del ranking perfecto (errores derivados primero)."""
    y_sorted = np.sort(y_error)  # aciertos primero: el oracle retiene correctos al responder
    n = len(y_sorted)
    risks = [y_sorted[:k].mean() for k in range(1, n + 1)]
    return float(np.mean(risks))


A_ORACLE = aurc_oracle(y_all)
A_RANDOM = float(y_all.mean())  # riesgo constante del azar
print(f"Base: error rate = {y_all.mean():.3f} | AURC oracle = {A_ORACLE:.4f} | AURC azar = {A_RANDOM:.4f}")

# --------------------------------------------------------------------------
# 1) AURC y Excess-AURC para todas las variantes deployable + comparación AUROC
# --------------------------------------------------------------------------
print("\n=== Variantes: AUROC vs AURC (top 15 por Excess-AURC normalizado) ===")
rows = []
for (st, layer, tau, pooling), g in df[df.prompt_id == "P1"].groupby(
        ["signal_type", "layer", "tau_s", "pooling"]):
    if pooling == "roi":
        continue
    g = g.set_index("image_filename")
    common = obs.index.intersection(g.index)
    u = g.loc[common, "value"].values.astype(float)
    y = 1 - obs.loc[common, "correct"].values
    m = ~np.isnan(u)
    if m.sum() < 30:
        continue
    a = roc_auc_score(y[m], u[m])
    arc = aurc(y[m], u[m])
    exc = (arc - A_ORACLE) / (A_RANDOM - A_ORACLE)
    rows.append({"variante": f"{st}_L{layer}_tau{tau}_{pooling}",
                 "AUROC": a, "AURC": arc, "ExcessN": exc})
res = pd.DataFrame(rows)
res["AUROC"] = res.AUROC.round(3)
res["AURC"] = res.AURC.round(4)
res["ExcessN"] = res.ExcessN.round(3)
print(res.nsmallest(15, "ExcessN").to_string(index=False))

rho_m, p_m = stats.spearmanr(res.AUROC, res.ExcessN)
print(f"\n(a) ¿AUROC y Excess-AURC ordenan igual las variantes?")
print(f"    Spearman(AUROC, ExcessN) = {rho_m:+.3f} (p={p_m:.2e}) — nota: signo negativo esperado")
print(f"    (AUROC alto = bueno; Excess bajo = bueno -> correlación negativa = métricas de acuerdo)")

# --------------------------------------------------------------------------
# 2) Señales principales: tabla comparativa AUROC / AURC / ExcessN
# --------------------------------------------------------------------------
print("\n=== Señales principales (P1, N=129) ===")
obs_df = df[df.prompt_id == "P1"].drop_duplicates("image_filename").set_index("image_filename")
win = df[(df.prompt_id == "P1") & (df.signal_type == "kl_t_v") & (df.pooling == "max")
         & (df.tau_s == "1.0")].set_index("image_filename")["value"]
n = len(obs_df)
senales = {
    "KL(t||v) max tau1": win[obs_df.index].values,
    "1-MSP": (1 - obs_df.msp_answer).values,
    "entropy": obs_df.entropy_answer.values,
    "energy": obs_df.energy_answer.values,
    "rank(KL)+rank(1-MSP)": (win[obs_df.index].rank() / n
                             + (1 - obs_df.msp_answer).rank() / n).values,
}
print(f"{'señal':24s} {'AUROC':>7s} {'AURC':>8s} {'ExcessN':>8s}")
main_res = {}
for nombre, u in senales.items():
    a = roc_auc_score(y_all, u)
    arc = aurc(y_all, u)
    exc = (arc - A_ORACLE) / (A_RANDOM - A_ORACLE)
    main_res[nombre] = (a, arc, exc)
    print(f"{nombre:24s} {a:7.3f} {arc:8.4f} {exc:8.3f}")

# --------------------------------------------------------------------------
# 3) Matriz de correlación entre señales (¿ordenan igual las imágenes?)
# --------------------------------------------------------------------------
print("\n=== (b) Matriz Spearman entre señales (rankings de imágenes) ===")
nombres = list(senales.keys())
mat = np.ones((len(nombres), len(nombres)))
for i, s1 in enumerate(nombres):
    for j, s2 in enumerate(nombres):
        if i < j:
            mat[i, j] = mat[j, i] = stats.spearmanr(senales[s1], senales[s2]).statistic
mat_df = pd.DataFrame(mat, index=nombres, columns=nombres).round(2)
print(mat_df.to_string())

# Guardar heatmap
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(mat_df, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, ax=ax, fmt=".2f")
    ax.set_title("Correlación (Spearman) entre señales UQ — P1")
    plt.tight_layout()
    out = "figures/fig6_correlacion_senales.png"
    plt.savefig(out, dpi=300)
    print(f"\n[fig] Guardada: {out}")
except Exception as e:
    print(f"\n[fig] No se pudo generar heatmap: {e}")
