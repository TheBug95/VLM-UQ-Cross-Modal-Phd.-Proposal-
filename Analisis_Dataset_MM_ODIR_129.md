# Análisis Detallado del Dataset MM-ODIR-129 y sus Implicaciones para el Experimento BIP 2026

**Fecha:** 21 de julio de 2026 (actualizado el mismo día tras verificación visual de muestras reales)
**Fuente verificada:** https://huggingface.co/datasets/TheBug95/MM-ODIR-129 (dataset card + árbol de archivos + `annotations.json` + `split.json` + 2 imágenes de muestra descargadas e inspeccionadas visualmente el 21-jul-2026)
**Documentos que este análisis modifica:** `Definicion_Experimental_Minima_BIP2026.md` y `Codigo/AGENTS.md`

---

## 1. Ficha técnica verificada

| Propiedad | Valor |
|---|---|
| Nombre | MM-ODIR-129 — Multi-Modal Optic Disc Image Dataset |
| Origen | Subconjunto de imágenes de ODIR-5K, **re-anotadas por oftalmólogos de Costa Rica** |
| Formato de imagen | **Fotos de fondo de ojo COMPLETAS, resolución variable** (verificado visualmente: 2592×1728 y 2304×2048 px). NOTA: la ficha dice *"optic disc centered"*, pero la verificación con muestras reales muestra que **no son recortes al disco** — el disco aparece dentro del fundus completo; la frase describe el foco clínico de la anotación, no un encuadre geográfico |
| Tamaño | **129 imágenes** (834 MB en disco con máscaras) |
| Clases | `Normal` (60) / `Pathological` (69) — **todos los patológicos son glaucoma** |
| Splits oficiales | train: 77 (36N/41P) · validation: 26 (12N/14P) · test: 26 (12N/14P) |
| Anotación por imagen | label binario + **transcripción clínica en texto libre escrita por el oftalmólogo** + (en patológicos) **graduación ordinal de 7 signos glaucomatosos** |
| Máscaras | Copa y disco óptico (PNG + NPY) para las 69 imágenes patológicas |
| Licencia | MIT |
| Idiomas de anotación | Inglés y español |
| Estructura del repo | `annotations.json` (57.8 kB, 129 entradas con campos `id, image_filename, label, transcription, locs_data`) + `split.json` (149 kB, dict `{train, validation, test}` con anotaciones completas por split) + carpetas `train/`, `validation/`, `test/` con `<id>_<side>.jpg` y máscaras `_cup`/`_disc` |
| Nomenclatura de archivos | `{patient_id}_{eye}.jpg` — el ID de paciente está embebido en el nombre |
| ⚠️ Hallazgo de verificación | Al menos una imagen patológica (`1281_right.jpg`, train) contiene una **flecha negra dibujada sobre los píxeles** señalando el disco → ver §4.6 |

### Los 7 signos graduados (`locs_data.glaucoma`), cada uno ordinal

1. **cup_to_disc_ratio** (0–4): ≤0.3 → 1.0 (total cupping)
2. **neuroretinal_rim** (0–4): ISNT preservada → pérdida total de rima
3. **disc_hemorrhage** (0–1): hemorragia de Drance presente/ausente
4. **peripapillary_atrophy** (0–2): ninguna → zona beta grande/progresiva
5. **rnfl_defect** (0–3): ninguno → defecto en cuña + adelgazamiento difuso
6. **disc_pallor** (0–2): color normal → palidez significativa
7. **vessel_changes** (0–3): normal → bayoneting + nasalización

---

## 2. Qué cambia respecto a lo que asumía la definición experimental

La definición congelada asumía **ODIR-5K completo**: ~7,000 imágenes, ~295 pacientes glaucoma (~6%), etiquetas ruidosas a nivel paciente. MM-ODIR-129 es un objeto distinto, y el experimento hereda sus propiedades — para bien y para mal:

| Dimensión | ODIR-5K (lo asumido) | MM-ODIR-129 (la realidad) | Consecuencia |
|---|---|---|---|
| N | ~7,000 imágenes | **129 imágenes** | El poder estadístico cae drásticamente → rediseñar análisis (§4) |
| Positivos | ~295 pacientes (6%) | 69 imágenes (53%) | Dataset **balanceado y curado**, no distribución de screening → limitación a declarar |
| Calidad de etiqueta | Ruidosa (nivel paciente, keywords) | **Alta** (re-anotado por oftalmólogos, por imagen) | Mejor ground truth que el plan original — ventaja real |
| Tipo de imágenes | Fundus completo, multicámara | **Fundus completo, resolución variable** (verificado) | La tarea mantiene la dificultad real: el modelo debe localizar el disco → favorable para generar errores evaluables (mitiga §4.2) |
| Artefactos visuales | No documentados | **Flechas de anotación quemadas en píxeles** (≥1 confirmada) | Posible *cue* espurio correlacionado con la etiqueta → auditar (§4.6) |
| Split | Había que construirlo (patient-level) | **Ya viene dado** (split.json) | El Paso 1 se simplifica muchísimo |
| Descarga | Kaggle/Grand Challenge, fricción | `snapshot_download` de HF | Día 1 más simple |
| Extras | Solo etiquetas 8-clases | Transcripciones + 7 gradings + máscaras | Nuevas oportunidades analíticas (§5) |

---

## 3. Fortalezas del dataset para ESTE experimento

1. **Etiquetas de calidad experta, por imagen.** El problema #1 de ODIR-5K (label noise por etiquetado a nivel paciente) desaparece: cada imagen fue vista y descrita por un oftalmólogo. Para un paper donde la variable de referencia es "¿el modelo se equivocó?", tener un ground truth limpio es más importante que tener muchos datos.
2. **Negativos difíciles incluidos deliberadamente.** Revisando las transcripciones: hay imágenes etiquetadas `Normal` con CDR 0.6–0.7, atrofia peripapilar, fondo tessellado miópico y artefactos. Es decir, el dataset contiene **casos frontera** (copete fisiológico, miopía que simula glaucoma) — exactamente donde vive la incertidumbre clínica. La frontera de la etiqueta es clínicamente correcta: lo que separa Normal de glaucoma no es el CDR solo sino la rima neuroretinal/RNFL. Esto es oro para el análisis de cuadrantes y para el discurso de "ambigüedad clínica intrínseca" (Insight 2 del dossier).
3. **Las transcripciones de los oftalmólogos** habilitan (a) análisis cualitativo de errores con referencia experta (Fig 5 mucho más rica), y (b) future work directo de report generation y de prompts con biomarcadores.
4. **Los 7 gradings ordinales** habilitan un análisis extra casi gratis (ver §5.2): correlación entre la señal de incertidumbre y la severidad de la enfermedad.
5. **Las máscaras de copa/disco** conectan con el Pilar 4 de la tesis (segmentación) — no se usan en este paper, pero el dataset ya deja el puente construido para MICCAI OMIA 2027.
6. **Balance ~53/47.** Con clases balanceadas, AUROC y AUPRC se comportan bien y no hay que defender métricas bajo desbalance extremo.
7. **Tamaño manejable:** 129 imágenes × 2 prompts ≈ 258 inferencias ≈ **10–20 minutos de GPU**, no 4–8 horas. El cronograma se relaja y se puede iterar más.
8. **Fundus completo = escenario realista.** Al no ser recortes al disco, la tarea conserva la dificultad de screening real (el modelo debe localizar e interpretar el nervio óptico dentro del campo completo), lo que fortalece la validez externa del argumento clínico.

## 4. Debilidades y riesgos (con números honestos)

### 4.1 🔴 Poder estadístico: el cambio más importante

Con N=129, los intervalos de confianza se ensanchan mucho respecto al plan original (que asumía ~295 positivos):

- **Regla práctica:** con ~35–50 errores del modelo y ~80–95 aciertos (escenarios esperables), el IC95% de una AUROC de 0.70 tiene media anchura de **±0.10–0.13**. Si se reportara solo el split test oficial (26 imágenes), la anchura sería **±0.20 o peor** → inaceptable.
- **Consecuencia obligatoria:** la evaluación principal debe hacerse sobre las **129 imágenes completas** (legítimo porque nada se entrena — el argumento zero-shot de la definición se vuelve aún más fuerte), usando el train split (77) para selección de variantes/hiperparámetros y el val+test (52) como confirmación.
- **DeLong** queda estadísticamente débil con este N: se mantiene pero se declara exploratorio. Los effect sizes y los IC bootstrap pasan a ser la evidencia principal.
- **Reformulación del criterio de éxito:** AUROC puntual ≥ 0.65 sigue siendo la meta, pero hay que reportar con honestidad: si el IC95% excluye 0.5 → evidencia fuerte; si no, pero la estimación puntual es ≥ 0.65 con effect size mediano → evidencia *sugestiva*, aceptable para un proof-of-concept en BIP si se discute con franqueza.

### 4.2 🟠 Riesgo de "pocos errores" (el riesgo estadístico clave)

La UQ se evalúa sobre la variable "¿el modelo se equivocó?". Si MedGemma, ante imágenes **balanceadas y con etiqueta limpia** (aunque de fundus completo, lo que mantiene la dificultad real), resulta tener accuracy alta (digamos 90%), solo habría ~13 errores en 129 imágenes → **la evaluación de la alarma queda casi sin muestra positiva**.

Escenarios:

| Accuracy de MedGemma | Errores (de 129) | Evaluación de UQ |
|---|---|---|
| 60% (como sugiere VOLMO en glaucoma) | ~52 | Excelente poder |
| 75% | ~32 | Aceptable |
| 90% | ~13 | Insuficiente → contingencia |

**Mitigaciones:** (a) medir la accuracy base el día del piloto — es el primer número que hay que mirar; (b) si accuracy > 85%, la contingencia es reportar análisis de *calibración y correlación con severidad* (§5.2) como resultados principales y el error-detection como secundario, o ampliar con las imágenes "difíciles" de ODIR-5K original; (c) jamás inflar artificialmente los errores (p. ej. degradando imágenes) sin declararlo.

### 4.3 🟠 Advertencia de doble ciego (crítica para la submission)

El dataset vive en una cuenta personal de Hugging Face vinculable al autor, y el paper **debe citar el dataset**. Citar `TheBug95/MM-ODIR-129` en un PDF doble ciego equivale a firmar el paper. Opciones (decidir antes de escribir la sección de datos):
1. Citación anonimizada: *"a publicly available multimodal glaucoma dataset derived from ODIR-5K [link withheld for double-blind review; will be provided in camera-ready]"* — práctica aceptada.
2. Espejo anónimo en anonymous.4open.science con el dataset + código.
3. Declarar la situación a los chairs (aplica dado que el autor es también del comité — máxima transparencia recomendada).

### 4.4 🟡 Limitaciones a declarar en el paper (ya redactables)

- N=129: proof-of-concept; IC anchos reportados con honestidad.
- Prevalencia curada (~53%) ≠ prevalencia de screening (~6%): las métricas operativas no se trasladan directamente a población real.
- Resolución variable entre imágenes (fundus completo); el resize a 896×896 del pipeline puede degradar detalles finos de rima/RNFL en las imágenes de mayor resolución.
- Posibles artefactos de anotación quemados en píxeles (flechas) en algunas imágenes (ver §4.6).
- Solo glaucoma (ninguna otra neuropatía óptica como control negativo).
- Anotaciones de un solo grupo de oftalmólogos; sin acuerdo inter-anotador medido.
- Sin metadatos demográficos (limitación ya declarada por el propio dataset).

### 4.5 🟡 Detalles técnicos de carga

- El visor del dataset en HF está roto ("Permission denied") y **no hay parquet**: `load_dataset("TheBug95/MM-ODIR-129")` probablemente falle. La vía robusta es `huggingface_hub.snapshot_download(repo_id="TheBug95/MM-ODIR-129", repo_type="dataset")` + el loader propio que la propia ficha proporciona (lee `annotations.json` + `split.json` + archivos por carpeta).
- `split.json` contiene las anotaciones completas por split (verificado: dict con llaves `train`, `validation`, `test`), no solo nombres de archivo.
- Las transcripciones tienen erratas humanas auténticas ("raito", "appeatance") — esperable en anotación real; no limpiar, solo no citar textualmente en el paper sin corregir entre corchetes.

### 4.6 🟠 Artefactos de anotación quemados en las imágenes (hallazgo de verificación visual)

Al descargar muestras reales para verificar la ficha se encontró que `1281_right.jpg` (Pathological, train) contiene una **flecha negra dibujada sobre los píxeles** señalando el disco óptico. Riesgo: si este tipo de marcas aparece solo (o mayormente) en imágenes patológicas, constituye un **atajo visual** que el VLM podría explotar para responder, inflando su accuracy aparente y contaminando la interpretación clínica de la KL (el modelo podría "acertar por la flecha"). Acciones requeridas:

1. **Auditar las 129 imágenes** (revisión visual asistida o detector simple de regiones negras pequeñas de alto contraste fuera del borde del fundus) y registrar un flag `has_annotation_artifact` en la tabla maestra.
2. Reportar cuántas imágenes marcadas hay por clase (Normal vs. Pathological).
3. **Análisis de robustez:** repetir las métricas principales excluyendo las imágenes marcadas (no cuesta re-cómputo del modelo; es filtrar el CSV).
4. Mencionar el hallazgo en limitaciones; si el número de imágenes afectadas es grande, discutirlo en la sección de datos del paper y considerar enmascarar las marcas como preproceso declarado.

---

## 5. Nuevas oportunidades analíticas que este dataset habilita

### 5.1 Análisis cualitativo de errores con referencia experta (mejora directa de Fig 5)

Cada cuadrante de la matriz 2×2 (correcto/incorrecto × KL alta/baja) puede ilustrarse con la imagen + la respuesta del modelo + **la descripción del oftalmólogo** como referencia. Esto eleva mucho la calidad del análisis cualitativo respecto al plan con ODIR-5K (que no tenía descripciones).

### 5.2 Correlación incertidumbre–severidad (análisis extra barato, recomendado)

Sobre los 69 casos patológicos: correlación de Spearman entre u(x) y el grading de CDR (0–4) — y opcionalmente un score compuesto de los 7 signos. Si la KL crece con la severidad/ambigüedad estructural, es evidencia independiente de que la señal captura algo clínicamente real (y funciona incluso si la accuracy del modelo es alta, mitigando §4.2). Cuesta unas líneas sobre el CSV. Con n=69, ρ≈0.33 es detectable con 80% de poder. **Se propone añadirla como análisis exploratorio H4.**

### 5.3 Future work inmediato que el dataset deja servido

- Prompts enriquecidos con biomarcadores ("CDR=0.8, rima adelgazada...") — la KL entre prompt biomarcador e imagen (Pilar 2/4).
- Evaluación de *report generation*: similitud entre la descripción generada por MedGemma y la transcripción experta.
- Segmentación copa/disco con las máscaras (Pilar 4, MICCAI OMIA 2027).

---

## 6. Cambios requeridos en `Definicion_Experimental_Minima_BIP2026.md`

| # | Sección afectada | Cambio |
|---|---|---|
| 1 | §3.2 (dataset) | Reescribir: MM-ODIR-129 (129 imágenes de fundus completo, origen ODIR-5K, re-anotación experta, splits oficiales, licencia MIT). Mantener la justificación de "por qué este dataset" con las fortalezas de §3 de este análisis |
| 2 | §3.4 (evaluación y splits) | Simplificar: no hay que construir patient-level split ni derivar etiquetas por keywords. Evaluación principal sobre las 129; selección de variantes en train (77); confirmación en val+test (52). Eliminar el análisis agregado a nivel paciente (o reducirlo a nota: el `patient_id` del nombre de archivo permite chequeo de clustering si ambos ojos aparecen) |
| 3 | §2 (hipótesis) y §8.1 (éxito) | Añadir H4 exploratoria (correlación KL–severidad). Reformular criterio de éxito con lenguaje de IC honesto (§4.1) y el riesgo de pocos errores (§4.2) |
| 4 | §4 Paso 1 (tabla maestra) | Nueva especificación: leer `annotations.json` + `split.json`; columnas `{image_filename, patient_id, eye, label, split, transcription, cdr_grade, ...7 gradings, has_masks, has_annotation_artifact}`. Descarga vía `snapshot_download` |
| 5 | §4 Paso 3 (corrida) | 129 × 2 prompts ≈ 10–20 min GPU. El piloto de 20 imágenes pasa a ser ~15% del dataset — mantenerlo igual. Añadir Paso 1.5: auditoría de artefactos de anotación (§4.6) |
| 6 | §6 (métricas) | Añadir nota de poder: IC bootstrap como evidencia principal; DeLong exploratorio; Spearman para H4 |
| 7 | §10 sanity check #7 | Cambiar "≈295 pacientes (~6%)" → "60 Normal / 69 Pathological = 129 total; train 77 / val 26 / test 26" |
| 8 | Sección de limitaciones del paper (§7.3, V. Discussion) | Incorporar las 7 limitaciones de §4.4 |
| 9 | Nueva consideración transversal | Manejo de doble ciego con dataset propio (§4.3) — decisión requerida antes de redactar |
| 10 | Nueva consideración transversal | Análisis de robustez excluyendo imágenes con artefactos de anotación (§4.6) |

## 7. Cambios requeridos en `Codigo/AGENTS.md`

1. Stack: dataset = `TheBug95/MM-ODIR-129` (HF, MIT), descarga con `snapshot_download` + loader propio (no `load_dataset` — visor roto).
2. Estructura de datos esperada: `data/mm_odir_129/{annotations.json, split.json, train/, validation/, test/}`.
3. Decisiones congeladas: sustituir "ODIR-5K, etiqueta binaria por keywords" por "MM-ODIR-129, label Normal/Pathological por imagen; glaucoma = única condición patológica; imágenes de fundus completo de resolución variable".
4. Sanity check #7 actualizado (conteos 60/69, splits 77/26/26) + nuevo check: flag `has_annotation_artifact` presente en la tabla maestra.
5. Tiempo estimado de corrida: minutos, no horas (afecta la nota de reanudabilidad — sigue siendo buena práctica pero ya no es crítica).
6. Nota de seguridad/ética: dataset público MIT; tratar el vínculo autor–dataset por el doble ciego (no incluir URL identificable en artefactos que vayan al PDF).
7. Referencias rápidas: añadir la URL del dataset.

---

## 8. Veredicto del análisis

**El cambio de dataset es una mejora neta para un primer paper en BIP, a cambio de poder estadístico.** Se gana: etiquetas de calidad experta (el argumento más importante para un paper de UQ), negativos difíciles clínicamente reales, transcripciones para análisis cualitativo, gradings para el análisis de severidad, imágenes de fundus completo (escenario realista), splits listos y un cronograma más holgado. Se pierde: tamaño muestral (129 vs ~7,000), lo que obliga a (a) evaluar sobre el dataset completo, (b) reportar ICs honestos, (c) reformular el criterio de éxito como proof-of-concept, y (d) vigilar el riesgo de "muy pocos errores" si MedGemma acierta mucho. Además, introduce dos obligaciones nuevas: **resolver la citación del dataset bajo doble ciego** antes de redactar, y **auditar los artefactos de anotación** (flechas quemadas en píxeles) con su análisis de robustez correspondiente.
