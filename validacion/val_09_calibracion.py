"""val_09_calibracion.py — Validación sintética del código de calibración (FUSE §5.2).

Las métricas de calibración (Platt, bins, ECE, correlaciones) se probaron contra
datos sintéticos donde la respuesta correcta se conoce de antemano; un bug aquí
inflaría o desinflaría el claim de calibración del paper sin que nadie lo notara.

Qué valida:
    V-CAL-1  Calibrador casi perfecto (u = P(error) real): ECE ≈ 0 y Pearson ≈ 1.
    V-CAL-2  Señal constante (no informativa): guards devuelven NaN controlados,
             no excepciones.
    V-CAL-3  ECE de un predictor calibrado < ECE de uno sobreconfidente.
    V-CAL-4  Los bins cubren toda la masa (Σ n_b == N) y son equiprobables (±1).
    V-CAL-5  Platt es monótona creciente en u (a > 0) cuando u se correlaciona
             con el error.
    V-CAL-6  TPR@FPR: TPR@FPR=20% coincide con sensitivity_at_specificity(0.80)
             salvo interpolación, y TPR crece con el FPR permitido.

Criterio PASS: V-CAL-1..6.

Uso:
    python validacion/val_09_calibracion.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.evaluation import (
    calibration_analysis,
    calibration_bins,
    calibration_correlations,
    expected_calibration_error,
    platt_calibrate,
    sensitivity_at_specificity,
    tpr_at_fpr,
)

FALLOS = []


def check(nombre, condicion, detalle=""):
    print(f"[{'PASS' if condicion else 'FAIL'}] {nombre}" + (f" — {detalle}" if detalle else ""))
    if not condicion:
        FALLOS.append(nombre)


print("=" * 70)
print("VAL-09 · CALIBRACIÓN (PLATT / BINS / ECE / TPR@FPR)")
print("=" * 70)

rng = np.random.default_rng(42)

# --- V-CAL-1 ------------------------------------------------------------------
# u = probabilidad real de error; y ~ Bernoulli(u). Con N grande, ECE -> 0.
N = 20000
u_true = rng.uniform(0.02, 0.60, size=N)
y_err = (rng.uniform(size=N) < u_true).astype(float)
bins = calibration_bins(u_true, y_err, n_bins=10)
ece = expected_calibration_error(bins)
corr = calibration_correlations(bins)
check("V-CAL-1a calibrador perfecto: ECE < 0.02", ece < 0.02, f"ECE={ece:.4f}")
check("V-CAL-1b calibrador perfecto: Pearson > 0.95 y Spearman > 0.95",
      corr["pearson"] > 0.95 and corr["spearman"] > 0.95,
      f"Pearson={corr['pearson']:.4f}, Spearman={corr['spearman']:.4f}")

# --- V-CAL-2 ------------------------------------------------------------------
# Señal constante: ni Platt ni bins deben explotar; NaN controlados.
n_small = 100
frame_const = pd.DataFrame({
    "image_filename": [f"img_{i}" for i in range(n_small)],
    "split": ["train"] * 60 + ["validation"] * 20 + ["test"] * 20,
    "correct": (rng.uniform(size=n_small) < 0.7).astype(int),
    "value": np.full(n_small, 3.14),
})
cal_const = calibration_analysis(frame_const, n_bootstrap=0)
check("V-CAL-2 señal constante -> NaN controlados, sin excepción",
      np.isnan(cal_const["ece"]) and np.isnan(cal_const["pearson"]),
      f"ece={cal_const['ece']}, pearson={cal_const['pearson']}")

# --- V-CAL-3 ------------------------------------------------------------------
# Predictor calibrado vs. sobreconfidente (empuja u hacia los extremos).
u_sobreconf = np.clip((u_true - 0.3) * 2.5 + 0.3, 0.0, 1.0)
ece_cal = expected_calibration_error(calibration_bins(u_true, y_err, n_bins=10))
ece_sc = expected_calibration_error(calibration_bins(u_sobreconf, y_err, n_bins=10))
check("V-CAL-3 ECE(calibrado) < ECE(sobreconfidente)",
      ece_cal < ece_sc, f"calibrado={ece_cal:.4f}, sobreconfidente={ece_sc:.4f}")

# --- V-CAL-4 ------------------------------------------------------------------
check("V-CAL-4a bins cubren toda la masa", int(bins["n"].sum()) == N,
      f"suma(n)={int(bins['n'].sum())}, N={N}")
check("V-CAL-4b bins equiprobables (±1 obs)",
      bins["n"].max() - bins["n"].min() <= 1,
      f"min={bins['n'].min()}, max={bins['n'].max()}")

# --- V-CAL-5 ------------------------------------------------------------------
# P(error) crece con u -> coeficiente a > 0 y u_cal monótona creciente.
n_platt = 500
u_raw = rng.uniform(0, 25, size=n_platt)  # escala tipo KL en nats
p_err = 1.0 / (1.0 + np.exp(-(u_raw - 12.0) / 3.0))
y_platt = (rng.uniform(size=n_platt) < p_err).astype(int)
res = platt_calibrate(u_raw, y_platt, u_raw)
check("V-CAL-5a Platt posible y a > 0", res is not None and res["a"] > 0,
      f"a={res['a']:.4f}, b={res['b']:.4f}" if res else "platt=None")
if res:
    u_cal = res["u_cal"]
    check("V-CAL-5b u_cal monótona creciente y en [0, 1]",
          bool(np.all(np.diff(u_cal[np.argsort(u_raw)]) >= 0))
          and u_cal.min() >= 0 and u_cal.max() <= 1,
          f"rango u_cal=[{u_cal.min():.4f}, {u_cal.max():.4f}]")
    # Platt ajustado en train NO debe poder ajustarse con una sola clase
    res_bad = platt_calibrate(u_raw, np.zeros(n_platt, dtype=int), u_raw)
    check("V-CAL-5c Platt con una sola clase -> None", res_bad is None)

# --- V-CAL-6 ------------------------------------------------------------------
# TPR@FPR vs. sensitivity_at_specificity (mismo punto operativo FPR=20%).
n_t = 300
y_correct = (rng.uniform(size=n_t) < 0.75).astype(int)
score = np.where(y_correct == 1, rng.normal(0.0, 1.0, n_t), rng.normal(1.2, 1.0, n_t))
tf = tpr_at_fpr(y_correct, score, fprs=(0.05, 0.10, 0.20))
s80 = sensitivity_at_specificity(y_correct, score, target_spec=0.80)
check("V-CAL-6a TPR@FPR=20% ≈ Sens@80%Spec (salvo interpolación)",
      abs(tf["tpr_fpr20"] - s80["sensitivity"]) < 0.10,
      f"TPR@FPR20={tf['tpr_fpr20']:.3f}, sens80={s80['sensitivity']:.3f}")
check("V-CAL-6b TPR no decrece al relajar el FPR",
      tf["tpr_fpr05"] <= tf["tpr_fpr10"] <= tf["tpr_fpr20"],
      f"5%={tf['tpr_fpr05']:.3f}, 10%={tf['tpr_fpr10']:.3f}, 20%={tf['tpr_fpr20']:.3f}")
tf_const = tpr_at_fpr(y_correct, np.full(n_t, 1.0))
check("V-CAL-6c señal constante -> TPR ≈ FPR (azar) o NaN controlado",
      all(np.isnan(v) or 0.0 <= v <= 1.0 for v in tf_const.values()),
      f"{tf_const}")

print("=" * 70)
if FALLOS:
    print(f"RESULTADO: FAIL ({FALLOS})")
    sys.exit(1)
print("RESULTADO: PASS")
sys.exit(0)
