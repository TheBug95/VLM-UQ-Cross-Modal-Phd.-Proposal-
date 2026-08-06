# -*- coding: utf-8 -*-
"""Genera el calendario dia por dia del semestre 2026B (CSV + Markdown)."""
from pathlib import Path
from datetime import date, timedelta
import csv

OUT_DIR = Path(__file__).parent
START = date(2026, 8, 5)
END = date(2026, 12, 18)

CONFERENCIA = date(2026, 8, 12)          # miercoles, todo el dia
BIP_EVENTO = {date(2026, 11, 11), date(2026, 11, 12), date(2026, 11, 13)}
SPRINT_END = date(2026, 8, 28)
ETIQUETADO_END = date(2026, 9, 30)       # "de aqui a octubre"
DAYANA_WEEK = (date(2026, 8, 10), date(2026, 8, 14))

# Viernes alternos: A = pasantia (desde 14-ago), B = tutor (desde 21-ago)
def tipo_viernes(d):
    delta = (d - date(2026, 8, 14)).days
    if delta < 0 or delta % 7 != 0:
        return None
    return "A" if (delta // 7) % 2 == 0 else "B"

def fase_investigacion(d):
    """Devuelve (codigo, detalle) de la linea de investigacion activa."""
    if d <= date(2026, 8, 7):
        return "S1", "Sprint BIP — S1 Diseno benchmark (3 tec. x 2 datasets x 2 VLMs)"
    if d <= date(2026, 8, 14):
        return "S2", "Sprint BIP — S2 Implementacion del pipeline"
    if d <= date(2026, 8, 19):
        return "S3", "Sprint BIP — S3 Corridas y recoleccion de resultados"
    if d <= date(2026, 8, 21):
        return "S4/S5", "Sprint BIP — S4 Analisis estadistico + S5 Redaccion"
    if d <= date(2026, 8, 26):
        return "S5", "Sprint BIP — S5 Redaccion del paper"
    if d <= SPRINT_END:
        return "S6", "Sprint BIP — S6 Revision interna y submission (28-ago)"
    if d <= date(2026, 9, 11):
        return "V1", "UQ-VLM — V1 Setup datasets (FairVLMed, RIM-ONE, REFUGE, ORIGA, ODIR-5K)"
    if d <= date(2026, 9, 18):
        return "V2", "UQ-VLM — V2 Adaptacion multi-modelo (CogVLM, LLaVA-Med + 1)"
    if d <= date(2026, 9, 25):
        # solapamiento con UQ-Seg: mie/vie van a segmentacion
        if d.weekday() in (2, 4):
            return "G1", "UQ-Seg — G1 Definicion de la propuesta (single-pass, 1x, por pixel)"
        return "V3", "UQ-VLM — V3 Corridas multi-dataset x multi-modelo"
    if d <= date(2026, 10, 2):
        if d.weekday() in (2, 4):
            return "G1/G2", "UQ-Seg — G1 Diseno / G2 Prototipo SAM en MM-ODIR-129"
        return "V4/V5", "UQ-VLM — V4 Mas alla de clasif. binaria + V5 Reporte (2-oct)"
    if d <= date(2026, 10, 9):
        return "G2/G3", "UQ-Seg — G2 Prototipo SAM / G3 Experimentos de respaldo"
    if d <= date(2026, 10, 30):
        return "G3", "UQ-Seg — G3 Experimentos que respaldan la propuesta"
    if d <= date(2026, 11, 6):
        return "G4", "UQ-Seg — G4 Extension: mas datasets y modelos de segmentacion"
    if d <= date(2026, 11, 10):
        return "G4/G5", "UQ-Seg — G4 Extension / G5 Comparativa con UQ del SOTA"
    if d <= date(2026, 11, 20):
        return "G4/G5", "UQ-Seg — G4 Extension / G5 Comparativa SOTA (post-BIP)"
    if d <= date(2026, 11, 23):
        return "G5/G6", "UQ-Seg — G5 Comparativa SOTA / G6 Consolidacion (23-nov)"
    if d <= date(2026, 12, 11):
        return "C1/C2", "Cierre — C1 Integracion a tesis / C2 Prep. candidatura"
    return "C2/C3", "Cierre — C2 Candidatura / C3 Planificacion 2027A"

filas = []
d = START
while d <= END:
    if d.weekday() >= 5:  # sin fines de semana
        d += timedelta(days=1)
        continue

    dia = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes"][d.weekday()]

    if d == CONFERENCIA:
        filas.append([d.isoformat(), dia, "—", "—", "—", "—", "—", "—", "—",
                      "Conferencia en el TEC (todo el dia) — sin actividades", 0])
        d += timedelta(days=1)
        continue
    if d in BIP_EVENTO:
        filas.append([d.isoformat(), dia, "—", "—", "—", "—", "—", "—", "—",
                      "BIP 2026 — evento (comite organizador)", 0])
        d += timedelta(days=1)
        continue

    sprint = d <= SPRINT_END
    tesis = 1.0 if sprint else 2.0
    lit = 1.0 if sprint else 2.0
    kb = 0.5 if sprint else 1.0

    reuniones = []
    h_reunion = 0.0
    if d.weekday() == 1:
        reuniones.append("Reunion practicantes 14:30-15:30")
        h_reunion += 1.0
    if d.weekday() == 4:
        tv = tipo_viernes(d)
        if tv == "A":
            reuniones.append("Reunion pasantia (Angel y Micaela) 08:30-10:00")
            h_reunion += 1.5
        elif tv == "B" and d != date(2026, 11, 13):
            reuniones.append("Reunion con tutor 14:00-16:15")
            h_reunion += 2.25

    h_bip_org = 2.0 if d.weekday() == 2 else 0.0
    h_etiq = 2.0 if (d.weekday() in (0, 3) and d <= ETIQUETADO_END) else 0.0

    h_dayana = 0.0
    if DAYANA_WEEK[0] <= d <= DAYANA_WEEK[1] and d.weekday() in (0, 3):
        h_dayana = 2.0  # 2h lunes + 2h jueves = 4h semana del 10-ago

    cod, detalle = fase_investigacion(d)
    h_inv = round(8.0 - tesis - lit - kb - h_reunion - h_bip_org - h_etiq - h_dayana, 2)

    tareas_inv = f"{cod}: {detalle}"
    if h_etiq and d > SPRINT_END and d > ETIQUETADO_END:
        pass  # no aplica
    extras = []
    if h_etiq:
        extras.append(f"Herramienta etiquetado ({h_etiq:.1f}h)")
    if h_dayana:
        extras.append(f"Datos Dayana / buckets Dr Wu-Mexico ({h_dayana:.1f}h)")
    if h_bip_org:
        extras.append(f"Organizacion BIP ({h_bip_org:.1f}h)")

    filas.append([
        d.isoformat(), dia,
        f"{tesis:g}", f"{lit:g}", f"{kb:g}",
        f"{h_inv:g}", tareas_inv + ("; " + "; ".join(extras) if extras else ""),
        "; ".join(reuniones) if reuniones else "—",
        f"{(h_reunion + h_bip_org + h_etiq + h_dayana):g}",
        "Almuerzo 12:30-13:30 (1h)",
        f"{8:g}",
    ])
    d += timedelta(days=1)

HEADER = ["Fecha", "Dia", "Tesis (h)", "Literatura (h)", "Base conoc. Obsidian (h)",
          "Investigacion (h)", "Detalle de tareas del dia", "Reuniones",
          "Compromisos fijos semanales (h)", "Notas", "Total trabajo (h)"]

csv_path = OUT_DIR / "calendario_semestre_2026B.csv"
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(HEADER)
    w.writerows(filas)

md_path = OUT_DIR.parent / "Plan_Calendario_Diario_2026B.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Calendario dia por dia — Semestre 2026B\n\n")
    f.write("Jornada 8:30-17:30 con almuerzo 12:30-13:30 (8 h efectivas/dia, 40 h/semana).\n")
    f.write("Durante el **sprint BIP 2026 (5 -> 28 ago)** los bloques fijos se reducen a "
            "tesis 1 h, literatura 1 h y base de conocimiento 0.5 h para maximizar las horas del sprint.\n\n")
    f.write("| " + " | ".join(HEADER) + " |\n")
    f.write("|" + "|".join(["---"] * len(HEADER)) + "|\n")
    prev_week = None
    for r in filas:
        y, w, _ = date.fromisoformat(r[0]).isocalendar()
        if prev_week is not None and (y, w) != prev_week:
            f.write("|" + "|".join([" ** "] * 1) + "|" * (len(HEADER) - 1) + "\n")
        prev_week = (y, w)
        f.write("| " + " | ".join(str(c) for c in r) + " |\n")

# resumen de horas por semana
print(f"Dias laborables registrados: {len(filas)}")
print(f"CSV: {csv_path}")
print(f"MD:  {md_path}")
