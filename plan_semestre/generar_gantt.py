# -*- coding: utf-8 -*-
"""Genera el diagrama de Gantt del plan de trabajo del semestre 2026B."""
import sys
from pathlib import Path
from datetime import date, timedelta

try:
    sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
    from daimon_runtime import setup_plot
    setup_plot()
except Exception:
    pass  # fallback: estilo por defecto de matplotlib

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# (fase, subtarea, inicio, fin)
TASKS = [
    # ---- Sprint BIP 2026 (paper benchmark) ----
    ("Sprint BIP 2026", "S1 Diseño benchmark: 3 técnicas × 2 datasets × 2 VLMs", date(2026, 8, 5), date(2026, 8, 7)),
    ("Sprint BIP 2026", "S2 Implementación del pipeline de evaluación", date(2026, 8, 7), date(2026, 8, 14)),
    ("Sprint BIP 2026", "Datos Dayana: definición + buckets Dr Wu / México", date(2026, 8, 10), date(2026, 8, 14)),
    ("Sprint BIP 2026", "S3 Corridas y recolección de resultados", date(2026, 8, 12), date(2026, 8, 19)),
    ("Sprint BIP 2026", "S4 Análisis estadístico, tablas y figuras", date(2026, 8, 17), date(2026, 8, 21)),
    ("Sprint BIP 2026", "S5 Redacción del paper", date(2026, 8, 17), date(2026, 8, 26)),
    ("Sprint BIP 2026", "S6 Revisión interna y submission (límite 28-ago)", date(2026, 8, 26), date(2026, 8, 28)),
    # ---- UQ-VLM generalización ----
    ("UQ-VLM", "V1 Setup datasets: FairVLMed, RIM-ONE, REFUGE, ORIGA, ODIR-5K", date(2026, 8, 31), date(2026, 9, 11)),
    ("UQ-VLM", "V2 Adaptación multi-modelo: CogVLM, LLaVA-Med + 1 más", date(2026, 9, 7), date(2026, 9, 18)),
    ("UQ-VLM", "V3 Corridas multi-dataset × multi-modelo", date(2026, 9, 14), date(2026, 9, 25)),
    ("UQ-VLM", "V4 Generalización más allá de clasificación binaria", date(2026, 9, 21), date(2026, 10, 2)),
    ("UQ-VLM", "V5 Análisis y reporte (límite 2-oct)", date(2026, 9, 28), date(2026, 10, 2)),
    # ---- UQ-Segmentación ----
    ("UQ-Segmentación", "G1 Definición de la propuesta (single-pass, 1×, UQ por píxel)", date(2026, 9, 14), date(2026, 9, 25)),
    ("UQ-Segmentación", "G2 Prototipo con SAM sobre MM-ODIR-129", date(2026, 9, 21), date(2026, 10, 9)),
    ("UQ-Segmentación", "G3 Experimentos que respaldan la propuesta", date(2026, 10, 5), date(2026, 10, 30)),
    ("UQ-Segmentación", "G4 Extensión: más datasets y modelos de segmentación", date(2026, 11, 2), date(2026, 11, 20)),
    ("UQ-Segmentación", "G5 Comparativa con UQ del estado del arte", date(2026, 11, 9), date(2026, 11, 23)),
    ("UQ-Segmentación", "G6 Consolidación y documento (límite 23-nov)", date(2026, 11, 16), date(2026, 11, 23)),
    # ---- Cierre de semestre ----
    ("Cierre", "C1 Integración de resultados al documento de tesis", date(2026, 11, 24), date(2026, 12, 11)),
    ("Cierre", "C2 Preparación intensiva prueba de candidatura", date(2026, 11, 24), date(2026, 12, 18)),
    ("Cierre", "C3 Planificación del semestre 2027A", date(2026, 12, 14), date(2026, 12, 18)),
]

PHASE_COLORS = {
    "Sprint BIP 2026": "#d62728",
    "UQ-VLM": "#1f77b4",
    "UQ-Segmentación": "#2ca02c",
    "Cierre": "#9467bd",
}

fig, ax = plt.subplots(figsize=(16, 9))

ylabels = []
for i, (phase, name, start, end) in enumerate(TASKS):
    y = len(TASKS) - 1 - i
    ax.barh(y, (end - start).days, left=mdates.date2num(start),
            height=0.62, color=PHASE_COLORS[phase], edgecolor="white", linewidth=0.6)
    ylabels.append(name)

ax.set_yticks(range(len(TASKS)))
ax.set_yticklabels(list(reversed(ylabels)), fontsize=9)

# Regiones bloqueadas
ax.axvspan(mdates.date2num(date(2026, 11, 11)), mdates.date2num(date(2026, 11, 14)),
           color="gray", alpha=0.25, zorder=0)
ax.text(mdates.date2num(date(2026, 11, 12)), len(TASKS) - 0.3, "BIP\n11–13 nov",
        ha="center", va="bottom", fontsize=8, color="dimgray")
ax.axvline(mdates.date2num(date(2026, 8, 12)), color="gray", linestyle=":", linewidth=1.2)
ax.text(mdates.date2num(date(2026, 8, 12)), -0.9, "12-ago: conferencia TEC (sin trabajo)",
        rotation=90, va="bottom", ha="right", fontsize=8, color="dimgray")

# Hitos
for d, label in [(date(2026, 8, 28), "Submission BIP 2026"),
                 (date(2026, 10, 2), "Entrega UQ-VLM"),
                 (date(2026, 11, 23), "Entrega UQ-Segmentación"),
                 (date(2026, 12, 18), "Fin de semestre")]:
    ax.axvline(mdates.date2num(d), color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.text(mdates.date2num(d), len(TASKS) - 0.3, label, rotation=0,
            ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.set_xlim(mdates.date2num(date(2026, 8, 3)), mdates.date2num(date(2026, 12, 21)))
ax.set_ylim(-1, len(TASKS) + 1)
ax.grid(axis="x", linestyle=":", alpha=0.5)
ax.set_title("Plan de trabajo — Semestre 2026B (5-ago → 18-dic)", fontsize=14, fontweight="bold", pad=28)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

legend = [Patch(facecolor=c, label=p) for p, c in PHASE_COLORS.items()]
legend.append(Patch(facecolor="gray", alpha=0.25, label="Días bloqueados"))
ax.legend(handles=legend, loc="lower left", fontsize=9, framealpha=0.9)

fig.tight_layout()
out = Path(__file__).parent / "gantt_semestre_2026B.png"
fig.savefig(out, bbox_inches="tight", dpi=150)
print(f"Guardado: {out}")
