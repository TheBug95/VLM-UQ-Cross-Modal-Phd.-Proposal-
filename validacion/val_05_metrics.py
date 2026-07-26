"""val_05_metrics.py — Validación de las convenciones matemáticas y estadísticas.

Cada métrica del paper depende de una convención de librería; un error aquí
invierte resultados (p.ej. el orden de argumentos de F.kl_div decide si se
calcula KL(p||q) o KL(q||p)).

Qué valida:
    V-MET-1  F.kl_div(log_q, p, reduction='batchmean') == KL(p || q)
             (primer argumento = log de la distribución MODELO q; el target p
             va SIN logaritmo). Se verifica contra el cálculo manual.
    V-MET-2  scipy.spatial.distance.jensenshannon devuelve la DISTANCIA
             (raíz de la JSD), no la divergencia: hay que elevar al cuadrado.
             Cotas verificadas: dist([1,0],[0,1]) = sqrt(ln2) ≈ 0.8326;
             JSD máx = ln2 ≈ 0.6931.
    V-MET-3  KL y JSD son 0 cuando p == q (sanity de simetría en el piloto:
             subconjunto con acuerdo forzado debe dar u(x) ≈ 0).
    V-MET-4  roc_auc_score / average_precision_score aceptan scores continuos
             (la KL cruda sirve directamente como score de ranking).
    V-MET-5  brier_score_loss: firma según versión de sklearn (y_prob <1.7 /
             y_proba >=1.7) y exige PROBABILIDADES calibradas (no scores KL).

Criterio PASS: V-MET-1..5.

Uso:
    python val_05_metrics.py
"""
import sys

FALLOS = []


def check(nombre, condicion, detalle=""):
    print(f"[{'PASS' if condicion else 'FAIL'}] {nombre}" + (f" — {detalle}" if detalle else ""))
    if not condicion:
        FALLOS.append(nombre)


import numpy as np
import torch
import torch.nn.functional as F

print("=" * 70)
print("VAL-05 · CONVENCIONES DE MÉTRICAS")
print("=" * 70)

# Distribuciones binarias de prueba (p = "verdad", q = "modelo")
p = torch.tensor([[0.65, 0.35]], dtype=torch.float64)
q = torch.tensor([[0.50, 0.50]], dtype=torch.float64)

# --- V-MET-1 ---------------------------------------------------------------------
kl_manual = float((p * (p / q).log()).sum())
kl_lib = float(F.kl_div(q.log(), p, reduction="batchmean"))
check("V-MET-1a F.kl_div(log_q, p) == KL(p||q) manual",
      abs(kl_lib - kl_manual) < 1e-9, f"manual={kl_manual:.6f}, lib={kl_lib:.6f}")
kl_invertida = float(F.kl_div(p.log(), q, reduction="batchmean"))
check("V-MET-1b la KL es ASIMÉTRICA (orden importa)",
      abs(kl_invertida - kl_manual) > 1e-3,
      f"KL(q||p)={kl_invertida:.6f} ≠ KL(p||q)={kl_manual:.6f} → documentar dirección en el código")

# --- V-MET-2 ---------------------------------------------------------------------
from scipy.spatial.distance import jensenshannon
d_max = float(jensenshannon([1.0, 0.0], [0.0, 1.0]))
check("V-MET-2a jensenshannon extremo == sqrt(ln2) ≈ 0.8326 (es DISTANCIA)",
      abs(d_max - np.sqrt(np.log(2))) < 1e-9, f"obtenido: {d_max:.6f}")
jsd_nuestra = float(jensenshannon(p[0].numpy(), q[0].numpy())) ** 2
m = ((p + q) / 2)
jsd_manual = 0.5 * float((p * (p / m).log()).sum()) + 0.5 * float((q * (q / m).log()).sum())
check("V-MET-2b jensenshannon² == JSD manual (base e)",
      abs(jsd_nuestra - jsd_manual) < 1e-9, f"manual={jsd_manual:.6f}, lib²={jsd_nuestra:.6f}")

# --- V-MET-3 ---------------------------------------------------------------------
check("V-MET-3a KL(p||p) == 0", abs(float(F.kl_div(p.log(), p, reduction="batchmean"))) < 1e-12)
check("V-MET-3b JSD(p,p) == 0", abs(float(jensenshannon(p[0].numpy(), p[0].numpy()))) < 1e-12)

# --- V-MET-4 ---------------------------------------------------------------------
from sklearn.metrics import roc_auc_score, average_precision_score
# Ejemplo canónico de la documentación de sklearn
auc = roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
check("V-MET-4a roc_auc_score con scores continuos", abs(auc - 0.75) < 1e-9, f"AUC={auc}")
ap = average_precision_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
check("V-MET-4b average_precision_score con scores continuos", 0.0 < ap <= 1.0, f"AP={ap:.4f}")

# --- V-MET-5 ---------------------------------------------------------------------
import sklearn
from sklearn.metrics import brier_score_loss
y_true = [0, 1, 1, 0]
y_prob = [0.1, 0.9, 0.6, 0.4]
try:
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 7):
        brier = brier_score_loss(y_true, y_prob, y_proba=None) if False else brier_score_loss(y_true, y_prob)
    else:
        brier = brier_score_loss(y_true, y_prob)
    manual = float(np.mean([(t - pr) ** 2 for t, pr in zip(y_true, y_prob)]))
    check("V-MET-5 brier_score_loss == error cuadrático medio de probabilidades",
          abs(brier - manual) < 1e-9, f"sklearn={brier:.6f}, manual={manual:.6f} "
          f"(sklearn {sklearn.__version__})")
except TypeError as e:
    check("V-MET-5 firma de brier_score_loss", False, f"{e} — ajustar y_prob/y_proba según versión")

print("=" * 70)
if FALLOS:
    print(f"RESULTADO: FAIL ({FALLOS})")
    sys.exit(1)
print("RESULTADO: PASS")
sys.exit(0)
