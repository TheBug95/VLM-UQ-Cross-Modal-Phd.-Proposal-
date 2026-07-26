"""val_03_dataset.py — Validación del dataset MM-ODIR-129.

Qué valida:
    V-DATA-1  load_dataset("TheBug95/MM-ODIR-129") funciona nativamente
              (imagefolder + metadata.jsonl por split; el visor web roto es un
              fallo de conversión parquet del servidor, NO del dataset).
    V-DATA-2  Conteos: train=77, validation=26, test=26 (total 129).
    V-DATA-3  Campos esperados presentes (etiqueta de glaucoma y cdr_grade).
    V-DATA-4  Balance: 60 glaucoma / 69 no-glaucoma (análisis previo del
              annotations.json; si difiere, el diseño de H4 y los splits
              estratificados deben recalcularse).
    V-DATA-5  Imágenes decodificables con PIL, modo RGB.
    V-DATA-6  patient_id sin filtrado entre splits (sin data leakage por paciente).

NOTA DE PRIVACIDAD: split.json en el repo expone `doctor_name` (PII del
anotador). Este script NO lo imprime ni lo exporta. Nunca redistribuir ese
campo en resultados, tablas ni repositorios.

Uso:
    python val_03_dataset.py
"""
import sys

FALLOS = []


def check(nombre, condicion, detalle=""):
    print(f"[{'PASS' if condicion else 'FAIL'}] {nombre}" + (f" — {detalle}" if detalle else ""))
    if not condicion:
        FALLOS.append(nombre)


from datasets import load_dataset

print("=" * 70)
print("VAL-03 · DATASET MM-ODIR-129")
print("=" * 70)

ds = load_dataset("TheBug95/MM-ODIR-129")
print("Splits cargados:", {k: len(v) for k, v in ds.items()})

# --- V-DATA-2 ------------------------------------------------------------------
conteos = {k: len(v) for k, v in ds.items()}
check("V-DATA-2a train == 77", conteos.get("train") == 77, f"obtenido: {conteos.get('train')}")
check("V-DATA-2b validation == 26", conteos.get("validation") == 26, f"obtenido: {conteos.get('validation')}")
check("V-DATA-2c test == 26", conteos.get("test") == 26, f"obtenido: {conteos.get('test')}")
check("V-DATA-2d total == 129", sum(conteos.values()) == 129, f"obtenido: {sum(conteos.values())}")

# --- V-DATA-3 ------------------------------------------------------------------
cols = set(ds["train"].column_names)
print("Columnas:", sorted(cols))
check("V-DATA-3a existe columna de imagen", "image" in cols)
# El campo de etiqueta puede llamarse label/glaucoma/diagnostic...: reportar y
# fijar el nombre real aquí tras la primera ejecución.
candidatos_label = [c for c in cols if any(k in c.lower() for k in ("label", "glaucoma", "diag"))]
print("Candidatas a etiqueta:", candidatos_label)
check("V-DATA-3b existe alguna columna de etiqueta", len(candidatos_label) > 0)
# Verificado 22-jul-2026: el grado de severidad es `cup_to_disc_ratio`, ORDINAL 0–4
# (no continuo), no nulo en 70 muestras: los 69 Pathological + 1 Normal.
# Es el "cdr_grade" de la definición experimental → H4 corre sobre esos 69.
candidatos_cdr = [c for c in cols if "cdr" in c.lower() or "cup_to_disc" in c.lower()]
print("Candidatas a grado CDR:", candidatos_cdr)
check("V-DATA-3c existe cup_to_disc_ratio (grado ordinal para H4)", len(candidatos_cdr) > 0,
      "sin este campo, H4 no es ejecutable tal como está definida")
if candidatos_cdr:
    colc = candidatos_cdr[0]
    vals = [(ds[s][i][colc], ds[s][i]["label"]) for s in ds for i in range(len(ds[s]))]
    no_nulos = [(float(v), l) for v, l in vals if v is not None and str(v) != "nan"]
    pat = [v for v, l in no_nulos if l == "Pathological"]
    check("V-DATA-3d grado CDR no nulo en los 69 Pathological", len(pat) == 69,
          f"obtenido: {len(pat)} no nulos patológicos de {len(no_nulos)} totales")
    check("V-DATA-3e grado CDR es ordinal entero 0–4",
          all(v == int(v) and 0 <= v <= 4 for v, _ in no_nulos),
          f"rango observado: {min(v for v,_ in no_nulos)}–{max(v for v,_ in no_nulos)}")

# --- V-DATA-4 ------------------------------------------------------------------
if candidatos_label:
    col = candidatos_label[0]
    todos = [ds[s][col] for s in ds]
    planos = [v for split in todos for v in split]
    from collections import Counter
    dist = Counter(planos)
    print(f"Distribución de '{col}':", dict(dist))
    check("V-DATA-4 balance 60/69", sorted(dist.values()) == [60, 69],
          f"obtenido: {sorted(dist.values())} — si difiere, recalcular H4 y splits")

# --- V-DATA-5 ------------------------------------------------------------------
try:
    img = ds["train"][0]["image"]
    check("V-DATA-5a imagen decodifica (PIL)", hasattr(img, "size"), f"size={getattr(img, 'size', '?')}")
    check("V-DATA-5b modo RGB", img.mode == "RGB", f"modo={img.mode}")
except Exception as e:
    check("V-DATA-5 decodificación de imagen", False, f"{type(e).__name__}: {e}")

# --- V-DATA-6 ------------------------------------------------------------------
cand_pid = [c for c in cols if "patient" in c.lower()]
if cand_pid:
    colp = cand_pid[0]
    sets = {s: set(ds[s][colp]) for s in ds}
    inter_tv = sets["train"] & sets.get("validation", set())
    inter_tt = sets["train"] & sets.get("test", set())
    inter_vt = sets.get("validation", set()) & sets.get("test", set())
    check("V-DATA-6 sin solape de pacientes entre splits",
          not (inter_tv or inter_tt or inter_vt),
          f"solapes: train∩val={len(inter_tv)}, train∩test={len(inter_tt)}, val∩test={len(inter_vt)}")
else:
    print("[WARN] No se encontró columna de patient_id — verificar manualmente "
          "que los splits no comparten pacientes (riesgo de leakage).")

print("=" * 70)
if FALLOS:
    print(f"RESULTADO: FAIL ({FALLOS})")
    sys.exit(1)
print("RESULTADO: PASS")
sys.exit(0)
