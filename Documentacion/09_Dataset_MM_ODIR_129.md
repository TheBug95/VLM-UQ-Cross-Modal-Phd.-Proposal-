# 09 — Dataset MM-ODIR-129

> **Propósito:** documentar el dataset usado en el experimento: origen, estructura, anotaciones, riesgos y consideraciones éticas. Fuente primaria: `Analisis_Dataset_MM_ODIR_129.md` (verificado contra el repositorio el 21-jul-2026).

[⬅️ 08 — Discusión y Limitaciones](08_Discusion_y_Limitaciones.md) | [➡️ 10 — Guía de Reproducibilidad](10_Guia_Reproducibilidad.md)

---

## 9.1 Origen

MM-ODIR-129 (*Multi-Modal Optic Disc Image Dataset*) es un subconjunto de **129 imágenes de ODIR-5K** (Ocular Disease Intelligent Recognition, ~7.000 imágenes de fondo de ojo de screening en China) que fueron **re-anotadas por oftalmólogos de Costa Rica**. La motivación de la re-anotación: ODIR-5K etiqueta a nivel paciente por keywords (etiquetas ruidosas); MM-ODIR-129 tiene **etiqueta binaria por imagen, vista y descrita por un oftalmólogo** — ground truth de calidad experta. Licencia **MIT** (uso libre con atribución).

## 9.2 Estructura

| Propiedad | Valor |
|---|---|
| Imágenes | **129 fotos de fondo de ojo COMPLETAS** (NO recortes al disco), resolución variable (verificado visualmente: p. ej. 2592×1728 y 2304×2048 px) |
| Clases | `Normal` (60) / `Pathological` (69) — **todos los patológicos son glaucoma** |
| Splits oficiales | train: 77 (36N/41P) · validation: 26 (12N/14P) · test: 26 (12N/14P) |
| Tamaño en disco | ~834 MB (con máscaras) |
| Nomenclatura | `{patient_id}_{eye}.jpg` (p. ej. `1281_right.jpg`) |
| Estructura del repo | `annotations.json` (129 entradas) + `split.json` (anotaciones completas por split) + carpetas `train/`, `validation/`, `test/` con imágenes y máscaras `_cup`/`_disc` (PNG + NPY) |

**Descarga:** `huggingface_hub.snapshot_download(repo_id="TheBug95/MM-ODIR-129", repo_type="dataset")`. ⚠️ **No usar `load_dataset`:** el repo no tiene parquet y el visor de HF está roto ("Permission denied"). El pipeline carga con un loader propio que lee `annotations.json` + `split.json` + las carpetas por split (`src/data.py`).

## 9.3 Variables de anotación

Por imagen:

1. **Label binario** (`Normal` / `Pathological`).
2. **Transcripción clínica en texto libre** escrita por el oftalmólogo (inglés y español; con erratas humanas auténticas — no limpiar, no citar textualmente sin corregir entre corchetes).
3. **7 gradings ordinales de glaucoma** (en patológicos; campo `locs_data.glaucoma`):

| Signo | Escala | Rango clínico |
|---|---|---|
| `cup_to_disc_ratio` (→ `cdr_grade` en el pipeline) | 0–4 | ≤0.3 → 1.0 (total cupping) |
| `neuroretinal_rim` | 0–4 | ISNT preservada → pérdida total de rima |
| `disc_hemorrhage` | 0–1 | Hemorragia de Drance presente/ausente |
| `peripapillary_atrophy` | 0–2 | Ninguna → zona beta grande/progresiva |
| `rnfl_defect` | 0–3 | Ninguno → defecto en cuña + adelgazamiento difuso |
| `disc_pallor` | 0–2 | Color normal → palidez significativa |
| `vessel_changes` | 0–3 | Normal → bayoneting + nasalización |

4. **Máscaras de segmentación de copa y disco** (PNG + NPY) para las 69 imágenes patológicas — **no se usan en este experimento** salvo en el pooling oracle `roi`; son el puente al Pilar 4 de la tesis (segmentación, future work).

Nota verificada (22-jul-2026): `cdr_grade` del diseño corresponde al campo `cup_to_disc_ratio` del dataset — **ordinal entero 0–4** (no continuo), no nulo en 70 muestras (los 69 Pathological + 1 Normal). H4 corre sobre los 69 patológicos.

## 9.4 Artefactos de anotación (hallazgo de verificación visual)

Al inspeccionar muestras reales se encontró que `1281_right.jpg` (Pathological, train) contiene una **flecha negra dibujada sobre los píxeles** señalando el disco óptico. Riesgo: si este tipo de marcas aparece mayormente en imágenes patológicas, constituye un **atajo visual (cue espurio)** que el VLM podría explotar para responder, inflando su accuracy aparente y contaminando la interpretación clínica de la KL ("acertar por la flecha").

Acciones implementadas y pendientes:

1. ✅ **Auditoría automática de las 129 imágenes** (`src/data.py`, Paso 1.5): detector de regiones negras pequeñas de alto contraste → flag `has_annotation_artifact` en `master_table.csv` (sanity check #8: presente, con conteo por clase).
2. ⏳ **Análisis de robustez excluyendo las imágenes marcadas** (repetir métricas principales filtrando el CSV — sin re-cómputo del modelo): **pendiente obligatorio antes de cualquier submission** ([§8.4](08_Discusion_y_Limitaciones.md)).
3. Si el número de afectadas fuera grande: discutir en la sección de datos del paper y considerar enmascarar las marcas como preproceso declarado.

## 9.5 Análisis estadístico

- **Balance:** 60 Normal (46.5%) / 69 Pathological (53.5%) — dataset balanceado y curado, **no** distribución de screening (~6% prevalencia). AUROC y AUPRC se comportan bien bajo este balance; la prevalencia curada se declara como limitación para la traslación clínica.
- **Poder estadístico:** con N = 129 y 26 errores del modelo, el IC95% de un AUROC ~0.70 tiene anchura ±0.10–0.13. Evaluar solo en el test oficial (26 imgs) daría IC de ±0.20 — inaceptable. De ahí el protocolo: **evaluación principal sobre las 129** (legítimo: modelo frozen, nada se entrena), selección de variantes solo en train, y Monte Carlo CV para generalización ([§3.3](03_Hipotesis_y_Diseno_Experimental.md)).
- **Casos frontera incluidos deliberadamente:** hay `Normal` con CDR 0.6–0.7, atrofia peripapilar, fondo tessellado miópico y artefactos — **negativos difíciles** clínicamente reales (copete fisiológico, miopía que simula glaucoma). La frontera de la etiqueta es clínicamente correcta: lo que separa Normal de glaucoma no es el CDR solo sino la rima neuroretinal/RNFL. Ideal para el análisis de cuadrantes ([§6.7](06_Resultados_Experimentales.md)).
- **Sin acuerdo inter-anotador medido** (un solo grupo de oftalmólogos) y **sin metadatos demográficos** (limitación del propio dataset).
- **Solo glaucoma:** ninguna otra neuropatía óptica como control negativo.
- **Chequeo de clustering:** el `patient_id` embebido en el nombre de archivo permite verificar si ambos ojos de un paciente aparecen (potencial correlación intra-paciente).

## 9.6 Consideraciones éticas

- **PII:** `split.json` contiene `doctor_name` (identidad del oftalmólogo anotador). Regla dura del pipeline: **nunca** copiarlo a la tabla maestra ni a ningún artefacto (`src/data.py` lo excluye por diseño).
- **`patient_id`** en nombres de archivo y en `results_full.csv`: tratarlo con cuidado; no exponerlo públicamente.
- **⚠️ Doble ciego:** el dataset vive en la cuenta personal de Hugging Face del autor. Citarlo con URL en un PDF doble ciego equivale a firmar el paper. Decisión: en la submission se cita **anonimizado** (*"a publicly available multimodal glaucoma dataset derived from ODIR-5K [link withheld for double-blind review; will be provided in camera-ready]"*), salvo decisión contraria documentada. **No propagar la URL a artefactos de submission.** Opciones documentadas: espejo anónimo en anonymous.4open.science o declaración a los chairs.
- **Licencia MIT:** uso libre; mantener el dataset **fuera de control de versiones** (`.gitignore`).
- Datos de pacientes: las imágenes provienen de ODIR-5K (dataset público de investigación, ya anonimizado en origen); la re-anotación no añade datos de paciente identificables.

---

[⬅️ 08 — Discusión y Limitaciones](08_Discusion_y_Limitaciones.md) | [➡️ 10 — Guía de Reproducibilidad](10_Guia_Reproducibilidad.md)
