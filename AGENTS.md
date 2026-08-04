# AGENTS.md — BIP 2026 / Glaucoma Uncertainty Experiment

> **Estado actual del directorio:** la especificación experimental (v2) sigue siendo la fuente de verdad. Ya existe infraestructura mínima de proyecto (`pyproject.toml`, `requirements.txt`, `config.yaml`, entorno virtual `.venv`, directorios `src/`, `data/`, `results/`, `figures/`) y los scripts de validación en `validacion/`. El código fuente del pipeline experimental (`src/data.py`, `src/inference.py`, etc.) aún no está implementado. Todo lo técnico sigue en `Definicion_Experimental_Minima_BIP2026.md`.

---

## 1. Visión general del proyecto

**Nombre informal:** Experimento mínimo BIP 2026 — *Cross-Modal Representation Disagreement as a Lightweight Uncertainty Signal for Glaucoma Detection in Medical Vision-Language Models*.

**Objetivo:** demostrar que, cuando MedGemma-4B se equivoca al detectar glaucoma en una foto de fondo de ojo, el desacuerdo interno entre su representación visual y su representación textual (medido por divergencia KL) es significativamente mayor que cuando acierta. Esa señal se usa para triage: derivar al oftalmólogo los casos más inciertos.

**Idioma oficial del proyecto:** español (documentación, comentarios, nombres de variables libres, redacción científica). Los nombres técnicos, clases, funciones, prompts en inglés y citas se mantienen en inglés.

**Archivos fuente de verdad:**
- `Definicion_Experimental_Minima_BIP2026.md` — definición congelada del experimento (v2, dataset MM-ODIR-129): hipótesis H1–H4, protocolo, ablaciones, figuras, tablas, sanity checks.
- `Analisis_Dataset_MM_ODIR_129.md` — análisis del dataset: fortalezas, riesgos estadísticos (N=129), artefactos de anotación, advertencia de doble ciego.
- `Revision_Propuesta_BIP2026.md` — revisión crítica de la propuesta (bugs identificados y corregidos en el diseño).
- `BIP2026_Dossier_Maestro.md` y `research/` — dossier de investigación de fondo (10 dimensiones).

---

## 2. Stack tecnológico (especificado, aún no implementado)

| Capa | Tecnología esperada |
|------|---------------------|
| Lenguaje | Python 3.10+ |
| Modelo VLM | `google/medgemma-4b-it` (MedGemma 4B instruct) vía HuggingFace — **requiere aceptar licencia HAI-DEF y `HF_TOKEN`** |
| Framework de ML | PyTorch + **`transformers>=4.51.3`** (obligatorio: API de hidden states válida desde esa versión) |
| Dataset | **`TheBug95/MM-ODIR-129`** (Hugging Face, licencia MIT) — ver §3 |
| Estadística / métricas | `scipy>=1.11`, `scikit-learn>=1.7`, bootstrap BCa, DeLong (exploratorio) |
| Visualización | Matplotlib / Seaborn |
| Gestión de dependencias | `requirements.txt` + `pyproject.toml` (instalación editable con `pip install -e .`) |

**Requisitos de hardware documentados:**
- ~16 GB VRAM en bfloat16 para MedGemma 4B (~6–8 GB en 4-bit con `bitsandbytes`, opcional).
- GPU recomendada; la corrida completa es corta (ver §5).

---

## 3. Dataset: MM-ODIR-129 (decisiones y advertencias)

- **Qué es:** 129 fotos de fondo de ojo **completas** (resolución variable; NO son recortes al disco — verificado visualmente), originarias de ODIR-5K y **re-anotadas por oftalmólogos de Costa Rica**. 60 `Normal` / 69 `Pathological` (todos los patológicos = glaucoma).
- **Contenido por imagen:** label binario, transcripción clínica del oftalmólogo, y (en patológicos) 7 gradings ordinales de glaucoma (`cup_to_disc_ratio` 0–4, `neuroretinal_rim` 0–4, `disc_hemorrhage` 0–1, `peripapillary_atrophy` 0–2, `rnfl_defect` 0–3, `disc_pallor` 0–2, `vessel_changes` 0–3). Las 69 patológicas tienen máscaras de copa/disco (PNG + NPY) — **no se usan en este experimento** (future work).
- **Splits oficiales (usarlos tal cual):** train 77 (36N/41P) · validation 26 (12N/14P) · test 26 (12N/14P).
- **Descarga:** `huggingface_hub.snapshot_download(repo_id="TheBug95/MM-ODIR-129", repo_type="dataset", local_dir="data/mm_odir_129")`. ⚠️ **NO usar `load_dataset`** (el repo no tiene parquet y el visor está roto); cargar con un loader propio que lea `annotations.json` + `split.json` y las carpetas por split (la ficha del dataset incluye código de ejemplo).
- **Estructura esperada tras descarga:** `data/mm_odir_129/{annotations.json, split.json, train/, validation/, test/}`; archivos `{patient_id}_{eye}.jpg` (+ `_cup`/`_disc` masks).
- **Protocolo de evaluación (congelado):** como nada se entrena (modelo frozen, señal training-free), la **evaluación principal es sobre las 129 imágenes**; el split train (77) se usa para seleccionar la variante KL y ajustar T de temperature scaling; val+test (52) es confirmación.
- **⚠️ Artefactos de anotación:** al menos una imagen patológica (`1281_right.jpg`, train) tiene una **flecha negra dibujada sobre los píxeles**. Obligatorio: auditoría visual/automática de las 129 imágenes y flag `has_annotation_artifact` en la tabla maestra (Paso 1.5 de la definición) + análisis de robustez excluyéndolas.
- **⚠️ Doble ciego:** el dataset está en la cuenta personal de HF del autor. En el PDF del paper se cita de forma anonimizada ("link withheld for double-blind review") salvo decisión contraria documentada. No incluir la URL en artefactos que vayan a la submission.

---

## 4. Estructura actual del directorio

```text
G:/.../BIP 2026/Codigo/
├── AGENTS.md                                  # este archivo
├── Definicion_Experimental_Minima_BIP2026.md  # especificación completa del experimento (v2)
├── Guia_Conceptual_y_Algoritmo_BIP2026.md     # explicación pedagógica paso a paso (documento hermano)
├── Plan_de_Validacion_BIP2026.md              # matriz de verificación pre-implementación
├── Analisis_Dataset_MM_ODIR_129.md            # análisis del dataset (consolidado desde la raíz, 24-jul)
├── Revision_Propuesta_BIP2026.md              # revisión de la propuesta, ronda 2 (22-jul)
├── revision_critica_propuesta_bip2026.md      # revisión de la propuesta, ronda 1 (17-jul)
├── Propuesta BIP 2026.md                      # semilla original de la propuesta (23-jun)
├── bip2026_pilares_analysis.md                # articulación con los 4 pilares de la tesis
├── BIP2026_Dossier_Maestro.md                 # dossier de investigación (resumen de research/)
├── research/                                  # investigación de fondo: 10 dimensiones + cross-verificación + insights
├── test_inference.py                          # test con mocks de src/inference.py (scaffolding TDD)
├── config.yaml                                # hiperparámetros de diseño + fallback de IDs
├── pyproject.toml                             # metadatos del proyecto e instalación editable
├── requirements.txt                           # dependencias versionadas
├── .venv/                                     # entorno virtual (creado, dependencias parciales)
├── src/                                       # código fuente del pipeline
│   ├── __init__.py
│   ├── config.py                              # carga config.yaml; resuelve IDs desde HF si puede
│   ├── data.py                                # descarga MM-ODIR-129 y genera master_table.csv
│   ├── uncertainty.py                         # pooling ponderado por ROI (máscaras de disco)
│   ├── inference.py                           # pipeline MedGemma: logits yes/no + 54+27 variantes KL/JSD
│   └── evaluation.py                          # AUROC/AUPRC y comparación de variantes (mean vs roi)
├── data/                                      # datasets y tablas maestras (no versionar)
│   ├── mm_odir_129/                           # dataset descargado (annotations.json, split.json, imágenes)
│   └── master_table.csv                       # 129 filas × 15 columnas (generado)
├── results/                                   # CSVs y artefactos (no versionar)
├── figures/                                   # figuras generadas (no versionar)
└── validacion/                                # scripts de validación pre-implementación
    ├── val_01_environment.py
    ├── val_02_tokenizer.py
    ├── val_03_dataset.py
    ├── val_04_generate_api.py
    ├── val_05_metrics.py
    ├── val_06_stats.py
    ├── val_07_pilot.py
    ├── val_08_resultados.py
    └── val_09_calibracion.py                    # sanity sintético de Platt/bins/ECE/TPR@FPR
```

Datos de verificación del dataset en `../data/mm_odir_129_preview/` (annotations.json, split.json y 2 imágenes de muestra descargadas el 21-jul).

---

## 5. Organización esperada del código

La infraestructura de configuración y la tabla maestra ya existen (`src/config.py` lee `config.yaml`; `src/data.py` descarga MM-ODIR-129 y genera `data/master_table.csv`). El resto del pipeline aún se implementa. El documento de diseño define estas piezas conceptuales:

1. **`data/master_table.csv`** — ya generado por `src/data.py`. Una fila por imagen, construida desde `annotations.json` + `split.json` (descargados con `snapshot_download`):
   `image_filename, patient_id, eye, label (0=Normal, 1=Pathological), split, transcription, cdr_grade, neuroretinal_rim, disc_hemorrhage, peripapillary_atrophy, rnfl_defect, disc_pallor, vessel_changes, has_masks, has_annotation_artifact`.
   **Mapeo verificado (22-jul-2026):** el campo `cdr_grade` del diseño es `cup_to_disc_ratio` en el dataset — ordinal entero 0–4 (no continuo), no nulo en 70 muestras (los 69 Pathological + 1 Normal). H4 corre sobre esos 69. OJO: `split.json` contiene `doctor_name` (PII del anotador) — nunca copiarlo al master table ni a ningún artefacto.
2. **`src/data.py`** — descarga (`snapshot_download`), loader propio, construcción de `master_table.csv`, auditoría de artefactos (Paso 1.5).
3. **`src/inference.py`** — pipeline de inferencia sobre MedGemma (detalles de API en §6.2):
   - prefill con `output_hidden_states=True`;
   - extracción de hidden states en capas 17, 26 y 34;
   - máscara por ID de token de imagen (¡nunca slicing fijo `[:, :256, :]`!);
   - 1 paso de decode greedy para logits de "yes"/"no" y hidden state del token de respuesta.
4. **`src/uncertainty.py`** — pooling ponderado por ROI (máscaras de disco, *oracle*), pooling ponderado por **atención cruzada** (deployable, sin máscaras externas) y heatmaps de atención. Responde a la crítica de *dilución espacial* (disco óptico = 5–10% de la imagen). **Regla numérica dura (detectada en val_07, 23-jul-2026):** la KL/JSD sobre hidden states se computa SIEMPRE con `F.log_softmax` en **float64** tras **z-score normalization** — las *massive activations* de Gemma colapsan `softmax` a distribuciones degeneradas incluso en float64. JSD = `scipy.spatial.distance.jensenshannon` **al cuadrado** (devuelve distancia, no divergencia). Computa además los baselines de igual costo (entropy, MSP, energy, TS) y el baseline 2× verbalized confidence (P5: parsing directo del número 0–100; u(x)=1−conf/100; solo sobre P1).
5. **`src/evaluation.py`** — análisis estadístico sobre el CSV en **formato largo** (implementado, 29-jul-2026). Selección de la variante ganadora SOLO en train (excluye poolings oracle), bootstrap CI BCa (9.999 remuestreos) para AUROC/AUPRC, Mann-Whitney + effect size, Spearman con permutación (H4), sensitivity@80%spec, TPR a FPR fijos (5%/10%/20%), curvas accuracy-coverage, baselines de igual costo (entropy, 1-MSP, energy) y señal combinada `rank(KL)+rank(1-MSP)` (sin parámetros). **Calibración estilo FUSE §5.2 (añadido 03-ago-2026):** Platt scaling ajustado SOLO en train por señal, bins equiprobables (10 ≈ 13 obs/bin), ECE (+sensibilidad con 5 bins), correlaciones de calibración Pearson/Spearman y Brier, con IC bootstrap percentil 95%; flag `in_sample` cuando el split de análisis es train. `--all-signals` para tabla rápida de AUROC por variante. Guarda `results/evaluation_summary.csv`. ⚠️ Las correlaciones de calibración son evidencia SECUNDARIA (Platt es monótona por construcción); la evidencia principal es ECE + reliability diagram.
6. **`src/figures.py`** — Figuras 2–10 y Tablas 1–5 sobre formato largo (reusa helpers de `src.evaluation`; la ganadora se elige en train o se pasa con `--signal`). Fig 2 boxplot, Fig 3 ROC/PR, Fig 4 accuracy-coverage, Fig 5 cuadrantes, Fig 10 reliability diagram de calibración (Platt en train, bins equiprobables); T1 resultados, T2 ablaciones (groupby directo en formato largo), T3 comparativa de la literatura, T5 discriminación + calibración lado a lado (espejo de FUSE Table 1).
7. **`results/results_full.csv`** — CSV central (una fila por imagen × prompt; esquema en §6.3). **Escritura incremental (append) para reanudabilidad.**

**Volumen de cómputo:** 129 imágenes × 2 prompts = 258 inferencias ≈ 10–20 min de GPU. La reanudabilidad es buena práctica pero ya no es crítica.

---

## 6. Especificaciones operativas (leer ANTES de implementar)

### 6.1 Prompts congelados (texto literal)

- **P1 (principal):** `"Does this fundus image show glaucoma? Answer yes or no."`
- **P4 (contraste):** system prompt `"You are an expert ophthalmologist."` + la misma pregunta del usuario.
- **P5 (verbalized confidence — baseline 2×, añadido 26-jul-2026):** segundo turno tras la respuesta de P1: `"How confident are you in your answer? Reply with a number from 0 to 100."` Se extrae el número por parsing directo (no logits); u(x) = 1 − conf/100. Solo sobre P1 (129 inferencias extra, ~5 min de GPU).
- Nota: el chat template de Gemma 3 no tiene rol de sistema propio — pliega el system prompt dentro del primer turno de usuario. Es el comportamiento esperado; no "arreglarlo".

### 6.2 Detalles de la API de HuggingFace (aquí viven los bugs)

- **Logits yes/no:** salen de `outputs.scores[0]` (logits del primer token generado; en greedy puro equivalen a los logits crudos de la última posición del prefill — verificable contra forward manual), NO de `hidden_states`. **IDs verificados (tokenizer Gemma-3/MedGemma):** primarios `yes`=4443, `no`=1904 (el chat template termina en `<start_of_turn>model\n`, así que el primer token de respuesta NO lleva espacio inicial); alternativos a inspeccionar: `Yes`=10784, `No`=3771, `▁yes`=11262, `▁no`=951, `▁Yes`=8438, `▁No`=2301. La lista se inspecciona UNA vez con el tokenizador (val_02) y se congela como constante en `src/inference.py`. Sanity check #4: P(yes)+P(no) ≈ 1.
- **Hidden states (CORREGIDO tras verificación con transformers ≥4.51):** `outputs.hidden_states` es una tupla por paso de generación. Con `max_new_tokens=1` **solo existe `hidden_states[0]`** (el prefill); `hidden_states[1]` sería IndexError. El hidden state que **condiciona** el primer token de respuesta es `hidden_states[0][capa+1][:, -1, :]` (última posición del prefill) → este es nuestro p_text primario (mantiene el claim single-pass). Los de **imagen/prompt** también salen de `hidden_states[0]` (máscara por posición). Variante/ablación: con `max_new_tokens=2`, el estado *tras* generar el token es `hidden_states[1][capa+1][:, -1, :]`.
- **Indexación de capas:** la tupla por paso tiene `num_layers + 1 = 35` entradas (índice 0 = embeddings escalados por √2560 — no mezclar con capas en la KL). "Capas 17, 26, 34" = índices `[17]`, `[26]`, `[34]` de la tupla ≡ capas 16, 25, 33 del decoder si se cuenta desde 0; convención congelada: índice de tupla, donde 34 = última capa.
- **Máscara de tokens de imagen:** construir con `input_ids == model.config.image_token_index` (= 262144, `<image_soft_token>`; IDs vecinos verificados: `<start_of_image>`=255999, `<end_of_image>`=256000) — NUNCA hardcodear sin validar (val_02 lo verifica). Deben ser exactamente 256 posiciones contiguas tras `<start_of_image>` (sanity check #3).
- **Vocabulario y proyección visual (rescatado del Frente A, 22-jul):** citar por separado 262.144 (entradas del tokenizer) y 262.208 (`vocab_size` de la embedding matrix en config HF). Los soft tokens de imagen son 1152-dim en SigLIP y 2560-dim solo tras el proyector — la KL se computa SIEMPRE sobre hidden states del decoder (2560-dim), nunca sobre salidas del vision encoder.
- **Bug conocido:** en algunas versiones de `transformers`, `output_hidden_states` no cascadea al vision tower. No nos afecta: solo usamos hidden states del DECODER. Si aparece, workaround: `model.vision_tower()` directo o actualizar `transformers`.
- **Generación:** `model.generate(..., max_new_tokens=1, do_sample=False, output_hidden_states=True, output_scores=True, return_dict_in_generate=True)` bajo `torch.inference_mode()`, dtype `bfloat16`, `device_map="auto"`.
- **Imagen:** pasar por el `AutoProcessor` tal cual (él hace resize 896×896 y normalización) — sin preproceso manual; CLAHE quedó fuera del alcance mínimo.

### 6.3 Esquema de `results_full.csv` (FORMATO LARGO, una fila por imagen × prompt × variante)

```
image_filename, patient_id, prompt_id (P1|P4), split,
logit_yes, logit_no, p_yes, pred (0|1), entropy_answer, msp_answer, energy_answer,
label (0|1), correct (0|1), inference_ms,
signal_type (kl_v_t|kl_t_v|jsd|cosine|kl_prompt_L34),
layer (solo 34; las capas 17/26 colapsan numéricamente),
tau (1.0|2.0|4.0),
pooling (mean|max|roi|attn|topk|normw|rollout|headspec),
value
```
~23.600 filas (129 imágenes × 2 prompts × variantes). Las columnas de observación (logits, baselines, label) se repiten en cada fila de la misma observación; las señales viven en `value`. Todas las variantes se guardan en la MISMA pasada: las ablaciones son análisis sobre tabla, no re-cómputo. La variante final "nuestro método" se elige SOLO con el split train y se reporta congelada (ganadora en la corrida completa: `kl_t_v` capa 34, τ=1, pooling `max`). Nota: las filas `kl_prompt_L34` quedaron con layer/tau desalineados en el CSV; `src.evaluation.load_results()` las normaliza al cargar (fix pendiente en `src.inference`).

### 6.4 Self-consistency (baseline multi-pass)

50 imágenes (subconjunto estratificado por clase) × 10 muestras a temperatura **1.5** (no 0.7: T<1 *afila* la distribución y con p_yes mediana ≈ 0.9999 el voto queda unánime — detectado 29-jul-2026 junto con un bug que comparaba logits en vez del token muestreado, lo que producía AUROC 0.5; corregido en `infer_self_consistency`) → fracción de "sí" → entropía como u(x). Se implementa como flag separado (`--self-consistency`, `--sc-temp` para la temperatura) para no mezclar muestreo con la pasada determinista principal.

---

## 7. Comandos de build, ejecución y test

### 7.1 Crear y activar el entorno virtual

En Windows (Git Bash / PowerShell):

```bash
python -m venv .venv
source .venv/Scripts/activate            # Git Bash
# .venv\Scripts\Activate.ps1            # PowerShell
```

En Linux/Mac:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 7.2 Instalar dependencias

Opción A — desde `requirements.txt` (recomendada; en GPU instalar torch con el índice CUDA correspondiente):

```bash
pip install -r requirements.txt
# Para GPU NVIDIA, previo a lo anterior:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Opción B — instalación editable del paquete:

```bash
pip install -e .
```

### 7.3 Validaciones pre-implementación (orden recomendado)

```bash
python validacion/val_01_environment.py
python validacion/val_02_tokenizer.py
python validacion/val_03_dataset.py
python validacion/val_05_metrics.py
python validacion/val_06_stats.py
python validacion/val_04_generate_api.py   # requiere GPU + licencia
python validacion/val_07_pilot.py          # requiere GPU + licencia
python validacion/val_09_calibracion.py    # sanity sintético de calibración (sin GPU)
```

### 7.4 Comandos del pipeline (por implementar)

```bash
# Descarga de datos + tabla maestra + auditoría de artefactos
python -m src.data

# Piloto de 20 imágenes con sanity checks (OBLIGATORIO antes de la corrida)
python -m src.inference --pilot --n 20

# Corrida completa (258 inferencias, ~10–20 min de GPU)
python -m src.inference --run-full

# Corrida completa con atenciones (más lento, eager attention)
python -m src.inference --run-full --attentions

# Repeticiones con múltiples semillas (resultados con columna seed)
python -m src.inference --run-full --seeds 42 123 456
python -m src.inference --pilot --n 20 --seeds 42 123
python -m src.inference --self-consistency --seeds 42 123

# Self-consistency (50 imágenes × 10 muestras)
python -m src.inference --self-consistency

# Verbalized confidence (P5, 129 inferencias extra sobre P1, ~5 min)
python -m src.inference --verbalized

# Estadísticas y figuras
python -m src.evaluation
python -m src.figures
```

**Nota:** para ejecutar cualquier comando con MedGemma se requiere `HF_TOKEN` y la licencia HAI-DEF aceptada en HuggingFace. El dataset se descarga automáticamente vía `datasets`.

---

## 8. Convenciones de desarrollo

### 8.1 Estilo de código
- No hay linter ni formateador configurado aún. Se recomienda **PEP 8** y `snake_case`.
- Los nombres científicos fijados deben respetarse: `u(x)`, `p_vis`, `p_text`, `KL(p_vis || p_text)`, `KL(p_text || p_vis)`, `JSD`.

### 8.2 Reproducibilidad
- Fijar semillas: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`; `torch.backends.cudnn.deterministic = True` si aplica.
- Registrar versiones de `torch`, `transformers`, Python y GPU en cada corrida.
- Greedy decoding (determinista) para la pasada principal.
- Inicializar Git desde el día 1 con `.gitignore` mínimo: `data/`, `results/`, `*.pt`, `__pycache__/`. (El repo anónimo para el double-blind se construirá sobre esto.)

### 8.3 Decisiones de diseño congeladas (no negociables sin justificación documentada)
- **Modelo:** `google/medgemma-4b-it` (frozen).
- **Dataset:** `TheBug95/MM-ODIR-129`; label binario `Normal`/`Pathological` por imagen; glaucoma = única condición patológica; fundus completo, resolución variable.
- **Prompts:** P1 y P4 con el texto literal de §6.1.
- **Entrada:** vía `AutoProcessor` (resize 896×896 interno); sin preproceso manual.
- **Capas de interés:** índices `[17]`, `[26]`, `[34]` de la tupla de hidden states.
- **Tokens de imagen:** exactamente 256, identificados por máscara sobre `image_token_id`.
- **Respuesta:** scoring por logits del primer token generado (`scores[0]`); nunca parseo de texto libre.
- **Evaluación:** principal sobre las 129 imágenes; selección de hiperparámetros solo en train (77); val+test (52) como confirmación.
- **Auditoría:** flag `has_annotation_artifact` obligatorio antes de reportar resultados.

---

## 9. Estrategia de testing

No hay framework de testing configurado. El diseño impone **8 sanity checks** que deben pasar en el piloto de 20 imágenes antes de cualquier corrida completa:

| # | Sanity check | Resultado esperado |
|---|---|---|
| 1 | `KL(p || p)` con la misma distribución | Exactamente 0 |
| 2 | Misma imagen dos veces, mismo prompt | `u(x)` idéntico (greedy, seed fija) |
| 3 | Máscara de image tokens sobre 5 ejemplos | Marca exactamente 256 posiciones contiguas tras `<start_of_image>` |
| 4 | `P(yes) + P(no)` | ≈ 1.0 (salvo ε numérico) |
| 5 | Imagen totalmente negra con P1 | KL visiblemente alta vs. imágenes normales |
| 6 | Distribución de `P(yes)` sobre el dataset | No colapsada en 0 o 1 |
| 7 | Conteo en `master_table.csv` | 60 Normal / 69 Pathological = 129; train 77 / val 26 / test 26 |
| 8 | Flag `has_annotation_artifact` | Presente en la tabla maestra, conteo por clase reportado |

**Además, en el piloto se mide primero la accuracy base de MedGemma:** si es >85%, activar la vigilancia del riesgo "pocos errores" (la evaluación de UQ tendría poca muestra positiva; la definición §8.1/§8.3 contempla la contingencia: H4/calibración pasan a primer plano).

**Recomendación futura:** convertir los checks 1–4 en tests unitarios (`pytest`) antes de escalar el código.

---

## 10. Consideraciones de seguridad

- No hay secretos, credenciales ni `.env` en el directorio actual. El token de HuggingFace se maneja por variable de entorno `HF_TOKEN` — nunca commitearlo.
- El acceso a MedGemma requiere aceptar la licencia HAI-DEF. No compartir tokens de acceso personales.
- MM-ODIR-129 es público bajo licencia MIT; mantenerlo fuera de control de versiones (`.gitignore`).
- **Doble ciego:** la URL del dataset identifica al autor; en el paper se cita anonimizada. No propagar la URL a artefactos de submission.
- `results_full.csv` contiene `patient_id`; tratarlo con cuidado y no exponerlo públicamente.

---

## 11. Notas críticas para el agente de código

- **Este proyecto ya tiene infraestructura mínima (`requirements.txt`, `pyproject.toml`, `config.yaml`, `.venv`, `src/config.py`).** El siguiente paso lógico es completar la instalación de dependencias, descargar el dataset y comenzar por la tabla maestra + piloto.
- **Bug crítico ya anticipado:** no usar slicing fijo `[:, :256, :]` para los tokens de imagen — máscara por `image_token_id` (§6.2).
- **La señal principal es `u(x) = KL(p_vis || p_text)`**, donde `p_vis` = mean pooling de los 256 tokens visuales y `p_text` = hidden state de la última posición del prefill (el estado que condiciona el primer token de respuesta: `hidden_states[0][capa+1][:, -1, :]`, §6.2).
- **Se computan ambas direcciones de KL y JSD** en la misma pasada (54 columnas de variantes); la variante reportada se elige solo con train.
- **El objetivo de éxito:** AUROC ≥ 0.65 en detección de errores con IC bootstrap 95% reportado honestamente (con N=129 el IC es ancho: ±0.10–0.13 — lenguaje de "evidencia fuerte" vs. "sugestiva" según si excluye 0.5). H4: correlación Spearman positiva entre u(x) y `cdr_grade` en los 69 patológicos.
- **Punto Go/No-Go:** día 4, al terminar la estadística principal. Si H0 no se rechaza, seguir el plan de contingencia de la Sección 8.3 de la definición.
- **Robustez numérica verificada (29-jul-2026, `validacion/val_08_resultados.py` + forward manual de las 129 imágenes en GPU distinta):** los logits son **bitwise reproducibles** entre GPUs (0/129 predicciones cambian). La KL está **winsorizada** en ln(1/eps)=23.03 (`epsilon: 1.0e-10` en `config.yaml`; 53/129 imágenes en el techo — esto es por diseño, no un bug). ⚠️ Un `epsilon` distinto desplaza TODOS los valores de KL en una constante (ln del ratio de eps): mantener `epsilon` fijo entre corridas y no comparar valores absolutos entre corridas con eps distinto. Ruido real entre GPUs/backends ≈ 0.4 nats; ranking estable (Spearman 0.964, ΔAUROC 0.016). Reglas: derivación clínica por percentil de la cohorte (nunca umbral absoluto de nats); reportar solo métricas de ranking (AUROC/AUPRC).

### 11.1 Métodos de explicación considerados y descartados como señal UQ

| Método | Por qué se descartó como señal UQ | Dónde se documenta |
|---|---|---|
| **Grad-CAM sobre tokens de visión** | Requiere backward pass (no es single-pass); en transformers decoder-only los hidden states no son "píxeles"; bfloat16/4-bit hace gradientes inestables. | Future work del paper |
| **Integrated Gradients (IG)** | Requiere 50–300 pasadas (rompe el framing 1× costo); equivalente a métodos multi-pass ya descartados. | Future work del paper |
| **Atención cruzada** | Ya implementada como ablación deployable. Resultado: AUROC 0.559 < mean 0.655 → la atención del modelo frozen no está alineada con la tarea. | Tabla de ablaciones del paper |
| **ROI con máscara de disco** | Implementada como *oracle* (no deployable). Resultado: AUROC 0.889 → cota superior que cuantifica la dilución espacial. | Tabla de ablaciones del paper |

**Conclusión:** la señal principal deployable es `mean` pooling. Grad-CAM/IG quedan como future work por costo computacional y por pertenecer a la familia XAI (explicación), no UQ (detección de errores).

---

## 12. Referencias rápidas

- Documento de diseño: `Definicion_Experimental_Minima_BIP2026.md`
- Análisis del dataset: `Analisis_Dataset_MM_ODIR_129.md`
- Revisión crítica: `Revision_Propuesta_BIP2026.md`
- Advertencias de código (dims 02 y 07): `research/bip2026_dim02.md`, `research/bip2026_dim07.md`
- Modelo: https://huggingface.co/google/medgemma-4b-it
- Dataset: https://huggingface.co/datasets/TheBug95/MM-ODIR-129 (⚠️ no citar en el PDF — doble ciego, §3)
