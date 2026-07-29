"""val_08_resultados.py — Verificación INDEPENDIENTE de los resultados.

No importa nada de src/ (salvo una comprobación puntual de kl_div contra un
cálculo a mano): todo se re-computa desde cero sobre los CSV. La idea es que
si el pipeline tuviera un bug, este script lo detecte en vez de repetirlo.

Nivel 1 — Invariantes del CSV: reglas matemáticas que deben cumplirse sí o sí.
Nivel 2 — Re-cómputo de métricas sin sklearn (AUROC por fórmula de ranks,
           AUPRC por suma manual, combinación por ranks a mano) y controles
           positivo/negativo.
Nivel 3 — (manual, en Colab) recomputar el KL de UNA imagen desde el modelo.

Uso:  python validacion/val_08_resultados.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "results_full.csv"
SUMMARY = ROOT / "results" / "evaluation_summary.csv"

n_pass, n_fail = 0, 0

def check(nombre: str, ok: bool, detalle: str = "") -> None:
    global n_pass, n_fail
    tag = "PASS" if ok else "FAIL"
    if ok:
        n_pass += 1
    else:
        n_fail += 1
    print(f"[{tag}] {nombre}" + (f" — {detalle}" if detalle else ""))


# ------------------------------------------------------------------------------
# AUROC y AUPRC implementados a mano (sin sklearn)
# ------------------------------------------------------------------------------
def auroc_manual(y_error: np.ndarray, score: np.ndarray) -> float:
    """AUROC = P(score de un error > score de un acierto), vía ranks (Mann-Whitney)."""
    n_e = int(y_error.sum())
    n_c = int((1 - y_error).sum())
    ranks = pd.Series(score).rank().values  # rank medio en empates
    sum_ranks_errors = ranks[y_error == 1].sum()
    U = sum_ranks_errors - n_e * (n_e + 1) / 2
    return U / (n_e * n_c)


def auprc_manual(y_error: np.ndarray, score: np.ndarray) -> float:
    """Average precision: suma de precision en cada error, ponderada por recall."""
    order = np.argsort(-score)
    y = y_error[order]
    n_errors = y.sum()
    if n_errors == 0:
        return np.nan
    precision_at_k = np.cumsum(y) / (np.arange(len(y)) + 1)
    return float((precision_at_k * y).sum() / n_errors)


# ------------------------------------------------------------------------------
# Nivel 1 — Invariantes del CSV
# ------------------------------------------------------------------------------
print("=" * 70)
print("NIVEL 1 — Invariantes matemáticos del CSV")
print("=" * 70)

df = pd.read_csv(RESULTS)
obs = df.drop_duplicates(subset=["image_filename", "prompt_id"]).copy()

# 1. p_yes debe ser la softmax de los logits yes/no
p_recalc = 1.0 / (1.0 + np.exp(obs.logit_no - obs.logit_yes))
check("p_yes == softmax(logit_yes, logit_no)",
      np.allclose(obs.p_yes, p_recalc, atol=1e-4),
      f"max diff = {np.abs(obs.p_yes - p_recalc).max():.2e}")

# 2. p_yes + p_no = 1 y pred = (p_yes > 0.5); en empate exacto (0.5) el
#    desempate lo hace el argmax del vocabulario completo (caso 1307_right P1)
tie = obs.p_yes == 0.5
ok_non_tie = bool((obs.loc[~tie, "pred"] == (obs.loc[~tie, "p_yes"] > 0.5).astype(int)).all())
check("pred == (p_yes > 0.5) fuera de empates",
      ok_non_tie, f"empates exactos: {int(tie.sum())} (desempate por argmax del modelo)")

# 3. correct == (pred == label)
check("correct == (pred == label)", bool((obs.correct == (obs.pred == obs.label).astype(int)).all()))

# 4. entropy binaria a mano: -p ln p - q ln q (nats, máximo ln 2 = 0.693)
p = obs.p_yes.clip(1e-12, 1 - 1e-12)
ent_recalc = -p * np.log(p) - (1 - p) * np.log(1 - p)
check("entropy == entropía binaria de p_yes",
      np.allclose(obs.entropy_answer, ent_recalc, atol=1e-4),
      f"max diff = {np.abs(obs.entropy_answer - ent_recalc).max():.2e}")

# 5. MSP = max(p_yes, 1 - p_yes)
msp_recalc = np.maximum(obs.p_yes, 1 - obs.p_yes)
check("msp == max(p_yes, 1-p_yes)",
      np.allclose(obs.msp_answer, msp_recalc, atol=1e-6))

# 6. energy == -logsumexp(logit_yes, logit_no)
mx = np.maximum(obs.logit_yes, obs.logit_no)
lse = mx + np.log(np.exp(obs.logit_yes - mx) + np.exp(obs.logit_no - mx))
check("energy == -logsumexp(logits)",
      np.allclose(obs.energy_answer, -lse, atol=1e-3),
      f"max diff = {np.abs(obs.energy_answer + lse).max():.2e}")

# 7. Consistencia del formato largo: mismos datos de observación en todas las
#    filas de la misma (imagen, prompt)
grp = df.groupby(["image_filename", "prompt_id"])[["p_yes", "label", "correct"]].nunique()
check("formato largo consistente (obs repetidas idénticas)", bool((grp == 1).all().all()))

# 8. Rangos válidos
check("JSD <= ln(2) + eps", bool(df[df.signal_type == "jsd"].value.max() <= 0.6932))
check("cosine en [0, 2]", bool(df[df.signal_type == "cosine"].value.between(0, 2).all()))
check("sin NaN en value", bool(df.value.notna().all()))

# 9. Conteos del diseño
n_img = obs[obs.prompt_id == "P1"]
check("129 imágenes, 60 Normal / 69 Pathological",
      len(n_img) == 129 and int((n_img.label == 0).sum()) == 60 and int((n_img.label == 1).sum()) == 69,
      f"{len(n_img)} imgs, {(n_img.label==0).sum()}N/{(n_img.label==1).sum()}P")

# ------------------------------------------------------------------------------
# Nivel 2 — Re-cómputo independiente de las métricas
# ------------------------------------------------------------------------------
print()
print("=" * 70)
print("NIVEL 2 — Métricas re-computadas a mano (sin sklearn, sin src/)")
print("=" * 70)

summary = pd.read_csv(SUMMARY)
WINNER = ("kl_t_v", "max", "1.0")
df["tau_s"] = df["tau"].astype(str)
win = df[(df.prompt_id == "P1") & (df.signal_type == "kl_t_v")
         & (df.pooling == "max") & (df.tau_s == "1.0")].sort_values("image_filename")
y_err = 1 - win.correct.values

# 10. Controles del AUROC manual: señal perfecta = 1.0, aleatoria ~= 0.5
rng = np.random.default_rng(0)
check("control positivo: AUROC(senal=y_error) == 1.0",
      abs(auroc_manual(1 - win.correct.values, (1 - win.correct.values) + rng.normal(0, 1e-6, len(win))) - 1.0) < 1e-3)
auroc_rand = auroc_manual(y_err, rng.normal(size=len(y_err)))
check("control negativo: AUROC(azar) ~= 0.5", abs(auroc_rand - 0.5) < 0.1, f"{auroc_rand:.3f}")

# 11. AUROC de la ganadora: manual vs reportado
auroc_win = auroc_manual(y_err, win.value.values)
rep = summary[(summary.prompt == "P1") & (summary.signal.str.contains("kl_t_v")) & (summary.split == "all")]
check("AUROC ganadora (manual) == reportado",
      abs(auroc_win - rep.auroc.iloc[0]) < 1e-6,
      f"manual = {auroc_win:.6f}, reportado = {rep.auroc.iloc[0]:.6f}")

# 12. Consistencia AUROC <-> Mann-Whitney: AUROC debe ser EXACTAMENTE U/(n_e·n_c)
from scipy import stats as _stats  # scipy solo para el test, no para el AUROC
U = _stats.mannwhitneyu(win.value.values[y_err == 1], win.value.values[y_err == 0],
                        alternative="greater").statistic
n_e, n_c = int(y_err.sum()), int((1 - y_err).sum())
check("AUROC == U/(n_errores·n_aciertos)",
      abs(auroc_win - U / (n_e * n_c)) < 1e-9,
      f"U/(·) = {U/(n_e*n_c):.6f}")

# 13. AUPRC manual vs reportado
auprc_win = auprc_manual(y_err, win.value.values)
check("AUPRC ganadora (manual) == reportado",
      abs(auprc_win - rep.auprc.iloc[0]) < 1e-6,
      f"manual = {auprc_win:.6f}, reportado = {rep.auprc.iloc[0]:.6f}")

# 14. Combinación por ranks a mano vs reportada
base_p1 = df[df.prompt_id == "P1"].drop_duplicates("image_filename").sort_values("image_filename")
u_msp = 1 - base_p1.msp_answer.values
kl_v = win.set_index("image_filename").loc[base_p1.image_filename, "value"].values
n = len(base_p1)
combo = pd.Series(kl_v).rank().values / n + pd.Series(u_msp).rank().values / n
y_err_b = 1 - base_p1.correct.values
auroc_combo = auroc_manual(y_err_b, combo)
rep_c = summary[summary.signal.str.contains("rank", regex=False)]
check("AUROC combinación (a mano) == reportado",
      abs(auroc_combo - rep_c.auroc.iloc[0]) < 1e-6,
      f"manual = {auroc_combo:.6f}, reportado = {rep_c.auroc.iloc[0]:.6f}")

# 15. Bootstrap rápido con OTRA semilla: el IC debe ser parecido al reportado
rng2 = np.random.default_rng(12345)
boots = []
for _ in range(2000):
    idx = rng2.integers(0, len(y_err), size=len(y_err))
    if y_err[idx].sum() in (0, len(y_err)):
        continue
    boots.append(auroc_manual(y_err[idx], win.value.values[idx]))
lo, hi = np.percentile(boots, [2.5, 97.5])
check("IC bootstrap (semilla distinta) compatible con el reportado",
      abs(lo - rep.auroc_ci_low.iloc[0]) < 0.06 and abs(hi - rep.auroc_ci_high.iloc[0]) < 0.06,
      f"re-computado [{lo:.3f}, {hi:.3f}] vs reportado [{rep.auroc_ci_low.iloc[0]:.3f}, {rep.auroc_ci_high.iloc[0]:.3f}]")

# 16. kl_div del pipeline contra un cálculo a mano (verifica la DIRECCIÓN de la KL)
sys.path.insert(0, str(ROOT))
try:
    import torch
    from src.inference import kl_div
    p_ = np.array([0.7, 0.2, 0.1])
    q_ = np.array([0.1, 0.2, 0.7])
    manual = float(np.sum(p_ * np.log(p_ / q_)))
    pipe = float(kl_div(torch.tensor(p_), torch.tensor(q_), 1e-12))
    check("kl_div(p, q) == suma_ p·ln(p/q) a mano", abs(manual - pipe) < 1e-5,
          f"manual = {manual:.6f}, pipeline = {pipe:.6f}")
except Exception as exc:
    check("kl_div(p, q) == suma_ p·ln(p/q) a mano", False, f"no se pudo verificar: {exc}")

# ------------------------------------------------------------------------------
print()
print("=" * 70)
print(f"RESULTADO: {n_pass} PASS, {n_fail} FAIL")
print("NIVEL 3 (manual, Colab): recomputar el KL de UNA imagen desde el modelo")
print("  y compararlo con results_full.csv — ver docstring del script.")
print("=" * 70)
sys.exit(1 if n_fail else 0)
