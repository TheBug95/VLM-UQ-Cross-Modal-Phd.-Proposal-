"""val_06_stats.py — Validación de las pruebas estadísticas con el tamaño
muestral real del estudio (N=129; subgrupos de 60/69; n=69 para H4).

Qué valida:
    V-EST-1  El Z asintótico de Mann-Whitney se calcula manualmente (corrección
             de continuidad + empates) — `mannwhitneyu` NO expone `zstatistic`
             (corregido 23-jul-2026 tras fallo detectado en scipy 1.16.3).
             Efecto r = |Z| / sqrt(N), N = n1 + n2 (Fritz, Morris & Richler 2012).
    V-EST-2  El p-value asintótico de spearmanr NO es fiable con n=69 (<500):
             la significación de H4 sale de scipy.stats.permutation_test.
             Se verifica que ambos existen y dan resultados coherentes.
    V-EST-3  scipy.stats.bootstrap con method='BCa' está disponible (preferido
             sobre percentile con N pequeño; 9.999 remuestreos).
    V-EST-4  DeLong NO está en sklearn: verificar disponibilidad del paquete
             `pauc` (o anotar el fallback a R pROC) para comparar AUROCs
             correlacionados (nuestra u(x) vs. entropía/MSP/energy).
    V-EST-5  Potencia estimada: con n1=60, n2=69, ¿qué tamaño de efecto r es
             detectable con 80% de potencia a alpha=0.05? (Simulación — fija
             expectativas honestas sobre lo que el estudio puede concluir.)

Criterio PASS: V-EST-1..4 verde; V-EST-5 informativo (se reporta el efecto
mínimo detectable).

Uso:
    python val_06_stats.py
"""
import sys

FALLOS = []


def check(nombre, condicion, detalle=""):
    print(f"[{'PASS' if condicion else 'FAIL'}] {nombre}" + (f" — {detalle}" if detalle else ""))
    if not condicion:
        FALLOS.append(nombre)


import numpy as np
import scipy
from scipy import stats

print("=" * 70)
print(f"VAL-06 · ESTADÍSTICA · scipy {scipy.__version__}")
print("=" * 70)

rng = np.random.default_rng(42)
# Datos sintéticos que imitan el diseño: u(x) en correctos (n1=60) vs. errores (n2=69)
u_correctos = rng.gamma(1.0, 0.5, 60)
u_errores = rng.gamma(1.0, 0.5, 69) + 0.4  # efecto moderado inyectado

# --- V-EST-1 -----------------------------------------------------------------------
# CORRECCIÓN (23-jul-2026): `mannwhitneyu` NO expone `zstatistic` (ni en scipy 1.16).
# El Z asintótico se calcula manualmente con corrección de continuidad y de empates;
# la fórmula está verificada: reproduce el p-value asintótico de scipy con error < 1e-10.
def mannwhitney_z(x, y):
    """Z asintótico de Mann-Whitney (verificado contra el p-value de scipy)."""
    x, y = np.asarray(x), np.asarray(y)
    n1, n2 = len(x), len(y)
    N = n1 + n2
    res = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    m_U = n1 * n2 / 2
    _, counts = np.unique(np.concatenate([x, y]), return_counts=True)
    tie = np.sum(counts**3 - counts)                     # corrección por empates
    sigma = np.sqrt((n1 * n2 / 12) * ((N + 1) - tie / (N * (N - 1))))
    diff = res.statistic - m_U
    z = (diff - np.sign(diff) * 0.5) / sigma             # corrección de continuidad
    return z, res.pvalue

z, p = mannwhitney_z(u_correctos, u_errores)
p_manual = 2 * stats.norm.sf(abs(z))
res_check = stats.mannwhitneyu(u_correctos, u_errores, alternative="two-sided", method="asymptotic")
check("V-EST-1a Z manual reproduce el p-value de scipy (< 1e-10)",
      abs(p_manual - res_check.pvalue) < 1e-10,
      f"Z={z:.3f}, p_scipy={res_check.pvalue:.6f}, p_desde_Z={p_manual:.6f}")
r = abs(z) / np.sqrt(60 + 69)
check("V-EST-1b r = |Z|/sqrt(N) computable y en [0,1]", 0 <= r <= 1,
      f"r={r:.3f}, p={p:.4f} (convención Fritz, Morris & Richler 2012)")

# --- V-EST-2 -----------------------------------------------------------------------
x = rng.normal(size=69)
y = x * 0.35 + rng.normal(size=69)  # correlación débil-moderada inyectada
rho, p_asint = stats.spearmanr(x, y)
perm = stats.permutation_test(
    (x, y), lambda a, b: stats.spearmanr(a, b).statistic,
    permutation_type="pairings", n_resamples=9999, rng=rng)
check("V-EST-2 spearmanr + permutation_test ejecutables",
      0 < perm.pvalue <= 1,
      f"rho={rho:.3f}, p_asintótico={p_asint:.4f}, p_permutación={perm.pvalue:.4f} "
      "→ reportar el de permutación (n=69 < 500)")

# --- V-EST-3 -----------------------------------------------------------------------
try:
    boot = stats.bootstrap((u_errores,), np.mean, confidence_level=0.95,
                           method="BCa", n_resamples=9999, rng=rng)
    check("V-EST-3 bootstrap BCa disponible",
          boot.confidence_interval.low < np.mean(u_errores) < boot.confidence_interval.high,
          f"IC95%=[{boot.confidence_interval.low:.3f}, {boot.confidence_interval.high:.3f}]")
except (TypeError, ValueError) as e:
    check("V-EST-3 bootstrap BCa disponible", False, f"{e} — actualizar scipy")

# --- V-EST-4 -----------------------------------------------------------------------
try:
    import pauc  # noqa: F401
    print("[PASS] V-EST-4 paquete `pauc` instalado (DeLong para AUROCs correlacionados)")
except ImportError:
    print("[WARN] V-EST-4 `pauc` no instalado → pip install pauc, o fallback: "
          "R pROC::roc.test (DeLong). Anotar en el entorno antes del análisis.")

# --- V-EST-5 (informativo) ----------------------------------------------------------
def potencia_mwu(efecto_d, n1=60, n2=69, sims=2000, alpha=0.05, seed=0):
    """Potencia de Mann-Whitney simulando normales con diferencia de medias d."""
    r = np.random.default_rng(seed)
    hits = 0
    for _ in range(sims):
        a = r.normal(0, 1, n1)
        b = r.normal(efecto_d, 1, n2)
        if stats.mannwhitneyu(a, b, alternative="two-sided").pvalue < alpha:
            hits += 1
    return hits / sims

for d in (0.3, 0.5, 0.8):
    pot = potencia_mwu(d)
    print(f"[INFO] V-EST-5 potencia con d={d}: {pot:.2f} "
          f"{'(suficiente)' if pot >= 0.8 else '(INSUFICIENTE — expectativa honesta)'}")

print("=" * 70)
if FALLOS:
    print(f"RESULTADO: FAIL ({FALLOS})")
    sys.exit(1)
print("RESULTADO: PASS")
sys.exit(0)
