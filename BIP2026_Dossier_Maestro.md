# BIP 2026 — Investigación Consolidada: Propuesta de Paper

**Título propuesto:** *Cross-Modal Representation Disagreement as a Lightweight Uncertainty Signal for Glaucoma Detection in Medical Vision-Language Models*

**Investigador:** Miguel Guillermo Abreu Cárdenas  
**Tutor:** Saúl Calderón Ramírez  
**Fecha de compilación:** 2026-06-23  
**Venue objetivo:** 8th IEEE International Conference on BioInspired Processing (BIP 2026), Costa Rica, 11–13 Noviembre 2026  
**Deadline interno:** 31 de julio de 2026

---

## Estructura de este Dossier

Este documento maestro consolida **10 dimensiones de investigación profunda**, **cross-verificación de hallazgos**, y **insights cross-dimensionales** para apoyar la redacción del paper BIP 2026 y posicionar el trabajo dentro del roadmap de la tesis doctoral.

| Archivo | Contenido | Líneas | Estado |
|---------|-----------|--------|--------|
| `research/bip2026_dim01.md` | Paper inspirador: "Between the Layers Lies the Truth" (intra-layer KL, LightGBM, signature maps) | ~377 | ✅ Completo |
| `research/bip2026_dim02.md` | Arquitectura MedGemma 4B: hidden states, MedSigLIP, GPU reqs, loading code | ~380 | ✅ Completo |
| `research/bip2026_dim03.md` | Cross-modal uncertainty: benchmarks (VLM-UQBench, UMPIRE, VIG-TUQ), biological analogy | ~469 | ✅ Completo |
| `research/bip2026_dim04.md` | Dataset ODIR-5K / MM-ODIR: estructura, descarga, preprocessing, splits | ~308 | ✅ Completo |
| `research/bip2026_dim05.md` | Glaucoma detection & métricas: SOTA, AUPRC vs AUROC, Brier, umbrales clínicos | ~440 | ✅ Completo |
| `research/bip2026_dim06.md` | Baselines training-free: MSP, entropy, TS, energy, Mahalanobis, MC-Dropout, TTA, SE | ~742 | ✅ Completo |
| `research/bip2026_dim07.md` | Código ejecutable: loading MedGemma, hidden-state extraction, KL, ODIR-5K Dataset, visualización | ~438 | ✅ Completo |
| `research/bip2026_dim08.md` | BIP 2026 conference: formato 6-8 páginas, CMT, double-blind, IEEE Xplore, fees | ~398 | ✅ Completo |
| `research/bip2026_dim09.md` | Tesis doctoral — 4 pilares: UQ, XAI, few-shot, segmentación; roadmap y brechas | ~301 | ✅ Completo |
| `research/bip2026_dim10.md` | Experimental design: baselines, ablaciones, métricas, análisis estadístico, reproducibilidad | ~459 | ✅ Completo |
| `research/bip2026_cross_verification.md` | Cross-verificación de hallazgos (Alta/Media/Baja/Conflicto) | ~200 | ✅ Completo |
| `research/bip2026_insight.md` | 9 insights cross-dimensionales con implicaciones para paper y tesis | ~300 | ✅ Completo |

---

## Resumen Ejecutivo

### La Contribución (en 3 oraciones)

Este paper propone que la **divergencia KL entre las representaciones visuales y textuales** de un Medical Vision-Language Model (MedGemma 4B) operando en **un solo forward pass** y sin entrenamiento adicional, constituye una señal de incertidumbre *lightweight*, interpretable y clínicamente accionable para la detección de glaucoma en imágenes de fondo de ojo. A diferencia de métodos como MC-Dropout (multi-pass, costoso) o entropía predictiva (solo textual, no detecta alucinaciones visuales), nuestra señal captura el **desacuerdo cross-modal** entre lo que el modelo "ve" y lo que el modelo "dice". Evaluamos esta hipótesis en el dataset ODIR-5K (5,000 pacientes, ~6% glaucoma), comparando contra 8+ baselines de UQ y reportando AUROC, AUPRC, Brier score y ECE.

### Por qué esto importa para la tesis

Los cuatro pilares de la tesis doctoral se articulan naturalmente alrededor de este método:

1. **UQ (Pilar 1):** KL cross-modal es una nueva familia de métodos *single-pass + feature-based + cross-modal* que llena un vacío en el espacio metodológico (Insight 1).
2. **XAI (Pilar 2):** La dirección de la KL (vision→text vs text→vision) es inherentemente explicativa: diagnóstica *qué tipo* de incertidumbre existe (Insight 2, Insight 3).
3. **Few-Shot (Pilar 3):** La KL puede filtrar pseudo-labels y seleccionar support sets de alta calidad en adaptación few-shot (Insight 7).
4. **Segmentación (Pilar 4):** La KL global puede descomponerse en mapas de riesgo espacial para segmentación de disco/copa óptica (Insight 1, Insight 3).

Este paper no es una contribución aislada; es el **nodo central** del roadmap doctoral: desde la revisión sistemática y el trabajo de MC-Dropout en GECCO 2026, hasta la segmentación text-guided y la validación clínica prospectiva (Insight 9).

---

## Hallazgos Clave por Dimensión (Resumen de 1 Página Cada Uno)

### Dim 01 — "Between the Layers Lies the Truth" (Paper Inspirador)

- **Método:** Para cada token relevante, convierte las post-MLP activations de cada capa ℓ en una distribución de probabilidad vía softmax temperado. Computa la matriz L×L de divergencias KL pairwise entre capas (signature map). Aplana la matriz y entrena un clasificador LightGBM para predecir correctitud.
- **Resultados:** AUPRC ~0.82 in-distribution, superior a probing en transferencia cross-dataset (+2.86 AUPRC), robusto bajo cuantización 4-bit (+1.94 AUPRC).
- **Adaptación para BIP:** En lugar de intra-layer KL (capa vs capa), computamos **cross-modal KL** (vision encoder vs text decoder). Esto preserva la propiedad "single-pass + feature-based" pero añade la dimensión cross-modal que es arquitectónicamente inherente a los VLMs.
- **Referencia:** `research/bip2026_dim01.md` | Sección 2.1–2.5, Equations 1–5.

### Dim 02 — MedGemma 4B Architecture & Feasibility

- **Modelo:** `google/medgemma-4b-it` (~4B parámetros, multimodal IT). Gemma 3 architecture.
- **Vision encoder:** MedSigLIP (SigLIP-400M fine-tuned médico), 27 capas, 1152 hidden, 16 heads, input 896×896, patch 14, 4096 patches → 256 soft tokens projected a 2560-dim.
- **Decoder:** 34 capas transformer, hidden size 2560, vocab 262k tokens, bidirectional attention on image tokens, interleaved local/global attention on text.
- **GPU:** ~16GB VRAM para inference bfloat16 single-pass; ~6-8GB con 4-bit quantization.
- **Bug conocido:** `output_hidden_states` no cascadea correctamente al vision encoder en algunas versiones de transformers. Workaround: extraer projected vision tokens desde los primeros 256 positions del decoder hidden states, o llamar `model.vision_tower()` directamente.
- **Carga:** `AutoModelForImageTextToText` / `Gemma3ForConditionalGeneration` desde HuggingFace con `transformers>=4.50`.
- **Referencia:** `research/bip2026_dim02.md` | Secciones 1–4, Tabla de specs.

### Dim 03 — Cross-Modal Uncertainty: Estado del Arte

- **VLM-UQBench (feb 2026):** Benchmark de 600 samples con uncertainty visual, textual y cross-modal. Evalúa 9 métodos: white-box (entropy, perplexity, MSP, PMI, semantic entropy, p(True)) y black-box (lexical similarity, DegMat, LUQ).
- **UMPIRE (feb 2026):** Training-free framework usando incoherence-adjusted semantic volume con DPPs. Outperforms baselines en AUROC, ECE, AURAC.
- **VIG-TUQ (may 2026):** Visual-Grounded Token Uncertainty. Usa Jensen-Shannon entre predicciones con/sin imagen + attention weights. Single-pass.
- **Expert-CFG (ICCV 2025):** Usa predictive entropy para detectar respuestas no confiables en MedVLMs, luego aplica classifier-free guidance con anotaciones expertas. AUC > 0.8 en VQA-RAD/SLAKE.
- **Insight biológico:** El cerebro humano detecta conflictos cross-modal (efecto McGurk, ventriloquismo) como señal de alerta. La divergencia KL entre modalidades es una formalización computacional de este principio.
- **Trampa:** KL es asimétrica y numéricamente inestable cuando denominador→0. Alternativas: JSD, Cauchy-Schwarz divergence.
- **Referencia:** `research/bip2026_dim03.md` | Secciones 2–4, 6.

### Dim 04 — Dataset: ODIR-5K / MM-ODIR

- **No existe un dataset separado "MM-ODIR".** El término se refiere a la estructura multimodal nativa de ODIR-5K: imágenes fundus + edad/sexo del paciente + keywords diagnósticos por ojo.
- **Estructura:** 5,000 pacientes, ~7,000 imágenes (left/right), 8 labels (N, D, G, C, A, H, M, O). Glaucoma: 207 training, 30 off-site test, 58 on-site test (~6% total).
- **Descarga:** Grand Challenge official, Kaggle mirrors (andrewmvd, sunlight081788, jeftaadriel), Academic Torrents.
- **Preprocessing:** Resize 896×896, normalize [-1, 1], RGB. CLAHE opcional para ablación.
- **Split obligatorio:** Patient-level (no image-level) para evitar data leakage. Ambos ojos del mismo paciente deben estar en el mismo fold.
- **Código:** `pandas` melt a one-row-per-image, `train_test_split` estratificado por paciente y label G.
- **Referencia:** `research/bip2026_dim04.md` | Secciones 1–5, Snippets de código.

### Dim 05 — Métricas y Estándares Clínicos

- **SOTA en ODIR-5K:** CNN/Transformer methods logran AUC promedio 0.79–0.98, pero glaucoma-specific AUC raramente se reporta separado. Multi-label average oculta minority-class performance.
- **Debate AUROC vs AUPRC:** NeurIPS 2024 (McDermott et al.) argumenta que AUROC es más robusto a imbalance y menos sesgado por subgrupos que AUPRC. Recomendación: reportar **ambas**, pero interpretar AUROC como métrica principal de ranking y AUPRC como secundaria de precision-operational.
- **Brier Score:** Mide calibración probabilística. SOTA glaucoma con CC-LS loss: Brier 0.098 vs BCE 0.195. Nuestro método training-free puede aplicar temperature scaling post-hoc para mejorar Brier.
- **Umbrales clínicos:** Screening glaucoma requiere sensitivity >80% (ideal >90%). SD-OCT: sens 79%, esp 92%. Tonometry: sens 48%, esp 94%. El objetivo es sensibilidad comparable a SD-OCT con specificity aceptable.
- **Patient-level evaluation:** ODIR-5K labels son por paciente, no por imagen. Evaluar por imagen introduce label noise en casos unilaterales.
- **Referencia:** `research/bip2026_dim05.md` | Secciones 1–5, Tabla de thresholds clínicos.

### Dim 06 — Baselines de Uncertainty Estimation (Training-Free)

- **Output-based (single-pass):** MSP, Entropy, Temperature Scaling + Entropy, Energy Score, Mahalanobis Distance, Cosine Distance to nearest neighbor.
- **Probing-based (lightweight training):** SAPLMA (MLP on hidden states, 60–80% accuracy predicting truthfulness). Requiere labeled correct/incorrect para entrenar probe, pero VLM permanece frozen.
- **Multi-pass (más costosos):** MC-Dropout (25–100 passes, limitado en modelos modernos sin dropout), Test-Time Augmentation (múltiples augmentaciones visuales), Semantic Entropy (múltiples samples + clustering NLI).
- **Limitaciones críticas:** Entropy/MSP fallan en **high-confidence errors** (modelo confiadamente equivocado). Semantic Entropy es el más robusto pero 5–10× más caro. MC-Dropout requiere acceso a dropout masks (no siempre disponible en modelos frozen).
- **Nuestro diferenciador:** Somos **single-pass + feature-based + cross-modal**. Ningún baseline existente ocupa las 3 propiedades simultáneamente (Insight 1).
- **Librerías:** Torch-Uncertainty (Python, implementa TS, MC-Dropout, ensembles, ECE), Uncertainty Toolbox (regression), laplace-torch (Bayesian approximations).
- **Referencia:** `research/bip2026_dim06.md` | Tabla comparativa de 12 baselines, Secciones 2–11.

### Dim 07 — Código Ejecutable: Pipeline End-to-End

- **Paso 1:** Cargar MedGemma 4B con `AutoModelForImageTextToText` + `torch.bfloat16` + `device_map="auto"`.
- **Paso 2:** Preprocess ODIR-5K: `Resize(896,896)`, `ToTensor()`, `Normalize(mean=0.5, std=0.5)` para rango [-1,1].
- **Paso 3:** Forward pass con `output_hidden_states=True`.
- **Paso 4:** Extraer projected vision tokens: `prefill[-1][:, :256, :]` (256 tokens, 2560-dim). Extraer text tokens: `prefill[-1][:, 256:, :]`.
- **Paso 5:** Mean-pooling sobre tokens, aplicar softmax temperado: `F.softmax(mean / tau, dim=-1)`.
- **Paso 6:** KL divergence: `F.kl_div(text_dist.log(), vision_dist, reduction='batchmean')`.
- **Paso 7:** Guardar CSV: `image_id, kl_divergence, predicted_answer, ground_truth`.
- **Paso 8:** Estadística: `scipy.stats.mannwhitneyu` (KL correct vs incorrect), `sklearn.metrics.average_precision_score` (AUPRC), `sklearn.metrics.brier_score_loss`.
- **Paso 9:** Visualización: `seaborn.boxplot` + `stripplot` overlay (colores: gris + #7c3aed), `matplotlib` PR curve.
- **Repositorios clave:** VL-Uncertainty (perturbation-based), UMPIRE (pseudocode), Expert-CFG (token-level entropy), Torch-Uncertainty (framework completo).
- **Referencia:** `research/bip2026_dim07.md` | 9 pasos de pipeline, 8 snippets de código.

### Dim 08 — BIP 2026: Conference Requirements & Strategy

- **Conference:** 8th IEEE International Conference on BioInspired Processing (BIP 2026). Costa Rica, 11–13 Nov 2026. IEEE Conference #71710.
- **Format:** 6–8 páginas, IEEE conference template (`\documentclass[conference]{IEEEtran}`). $50/página extra >8.
- **Submission:** Microsoft CMT (no EasyChair). Double-blind review. Sin metadata de autores, sin logos institucionales en figuras.
- **Preprints:** IEEE permite arXiv preprints, pero double-blind implica que reviewers no deberían buscar. **Safest:** no publicar título/abstract en arXiv antes de notificación. Post-aceptación: actualizar con copyright IEEE + DOI.
- **Track:** *Health, Biodiversity, Agriculture and Physical Sciences* → *"Bioinspired image and sound processing for health and life sciences"* o *"Bioinspired artificial intelligence"*.
- **Bio-inspired framing:** No es suficiente aplicar CNN a fondo de ojo. Debe argumentar justificación biológica: procesamiento multisensorial humano, detección de conflictos cross-modal, códigos de población neuronal (Insight 5).
- **Ethics statement:** Dataset público y de-identificado. No requiere IRB adicional, pero incluir declaración: *"We use publicly available de-identified datasets (ODIR-5K) previously collected under IRB-approved protocols. No new patient data was acquired."*
- **Reproducibility:** Código en GitHub anónimo (anonymous.4open.science), `requirements.txt`, random seeds, hardware specs.
- **Referencia:** `research/bip2026_dim08.md` | Secciones 1–12, Tabla de fees estimados.

### Dim 09 — Articulación con los 4 Pilares de la Tesis Doctoral

- **Pilar 1 (UQ):** BIP 2026 generaliza "Between the Layers" (intra-layer) a cross-modal. Extensiones futuras: calibración KL→ECE, adaptive triage (activar MC-Dropout solo cuando KL es alta), distribuciones no Gaussianas (Wasserstein, Fréchet).
- **Pilar 2 (XAI):** KL es explicación nativa. Direcciones: KL alta vis→text = language prior bias; KL alta text→vis = visual ambiguity; ambas altas = intrinsic clinical ambiguity. Future: attribution maps sobre KL, chain-of-thought explicativo.
- **Pilar 3 (Few-Shot):** KL como curador de pseudo-labels y selector de support sets. Filtrar ejemplos con alta divergencia para evitar confirmation bias. Future: prompt learning con regularización KL, meta-learning de dificultad vía KL.
- **Pilar 4 (Segmentación):** Segmentación clásica (OpenCV/scikit-image) genera biomarcadores (CDR, VCDR) que enriquecen prompts textuales. KL entre prompt enriquecido e imagen = señal de coherencia biomarcador-visual. Future: text-guided segmentation, uncertainty maps espaciales.
- **Roadmap de tesis:**
  - Fase 1 (Hecho): Revisión sistemática ~50 papers + IEEE CBMS 2026 + ophthalmo_capture.
  - Fase 2 (Hecho/En curso): Between the Layers + GECCO 2026 (MC-Dropout + CMA-ES).
  - Fase 3 (Ahora): **BIP 2026** — Cross-modal KL for glaucoma UQ.
  - Fase 4 (Futura inmediata): CIARP 2026 + MICCAI 2027 OMIA (segmentación text-guided + few-shot adaptativo).
  - Fase 5 (Tesis): Síntesis en *"A Multimodal Uncertainty Framework for Trustworthy Glaucoma AI"* + despliegue ophthalmo_capture v2.0 + validación clínica prospectiva.
- **Referencia:** `research/bip2026_dim09.md` | Secciones 2–8, Roadmap completo.

### Dim 10 — Diseño Experimental y Baselines

> **⚠️ Nota de actualización (26-jul-2026):** esta sección resume la investigación de junio (9 baselines). El set congelado vigente está en `Definicion_Experimental_Minima_BIP2026.md` PARTE 5 (v2.3): MSP, Entropía (equivalente exacta a MSP en binario — sanity check teórico), TS+Entropía, Energy (1×) + **Verbalized Confidence** (2×, P5) + Self-Consistency (10×). Mahalanobis, MC-Dropout, SAPLMA, Semantic Entropy (degenera en binario = Self-Consistency), Deep Ensembles y reader study quedaron eliminados con justificación documentada.

- **Baselines a comparar:** Entropy, TS+Entropy, MSP, Energy Score, Mahalanobis, MC-Dropout (upper bound teórico), Semantic Entropy (upper bound semántico), SAPLMA-like probe (baseline learned), Human expert (reader study).
- **Ablaciones:**
  - Modalidades: solo vision vs solo text vs cross-modal.
  - Capas del decoder: 25%, 50%, 75%, 100%, mean-pooling layer-wise.
  - Pooling: mean, max, CLS, last token.
  - Temperatura: T ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}.
  - Dirección KL: KL(vis||text), KL(text||vis), symmetric, JSD.
- **Métricas:** AUROC (primary), AUPRC (secondary), Brier Score, ECE, Sensitivity@80%Specificity, Cohen's Kappa, Accuracy vs Coverage Curve (selective classification).
- **Análisis estadístico:** DeLong's test para comparar AUROC, bootstrap 10k stratificado para AUPRC CI, Mann-Whitney U / t-test para KL distributions, Cohen's d / r effect sizes, FDR correction (Benjamini-Hochberg).
- **Power analysis:** Con N=207 glaucoma cases, ΔAUROC de ~0.03–0.05 es detectable con 80% power. Mejoras >0.05 son robustamente significativas.
- **Datasets:** ODIR-5K (primary), REFUGE (secondary, 400/400/400 split), RIM-ONE (cross-dataset generalization).
- **Prompts:** P1: "Does this image show glaucoma?" (directo); P2: "Classify this fundus image..." (tarea); P3: "Describe this image and mention if glaucoma is present." (descriptivo); P4: "You are an expert ophthalmologist..." (system prompt); P5: "Analyze this retinal fundus photograph... optic disc cupping..." (biomarcadores); P6: "Question: Is glaucoma present? Answer yes or no and explain..." (estructurado).
- **Análisis de errores:** Matriz 2×2 (correct/incorrect × low/high KL). Cuadrante crítico: Incorrect + Low KL (overconfident errors). Visualizar 5-10 ejemplos por cuadrante.
- **Reproducibilidad:** Fijar seeds (42), `cudnn.deterministic=True`, reportar GPU/VRAM, `requirements.txt`, tiempo de inferencia por imagen, ML Reproducibility Checklist (15 items), STARD-AI reporting guidelines.
- **Referencia:** `research/bip2026_dim10.md` | Secciones 1–9, Tablas de baselines y prompts.

---

## Insights Cross-Dimensionales (Resumen Ejecutivo)

### Insight 1: El "Punto Dulce" Metodológico — Single-Pass + Feature-Based + Cross-Modal

Ningún método existente combina estas 3 propiedades. La Figura 1 del paper debe visualizar este espacio 3D y posicionar nuestra contribución en el octante vacío. Esto comunica instantáneamente la novedad metodológica.

### Insight 2: La Asimetría de KL como Diagnóstico de "Modo de Fallo"

- KL(vis||text) alta → visual under-utilization (el texto ignora la imagen).
- KL(text||vis) alta → textual over-reach (el texto alucina lo que no está en la imagen).
- Ambas altas → intrinsic clinical ambiguity (caso genuinamente difícil).
- Ambas bajas → confident agreement (caso típico, bien representado).

Esto transforma la KL de un número de incertidumbre a un **diagnóstico estructural**.

### Insight 3: La Incertidumbre como Explicación Nativa — De Post-Hoc a Nativa

Grad-CAM/SHAP son post-hoc: añaden un modelo explicativo externo. La KL cross-modal es nativa: la divergencia *es* la razón de la incertidumbre. No se explica *qué* predice el modelo, sino *por qué* no está seguro.

### Insight 4: ODIR-5K como "Laboratorio de Incertidumbre"

ODIR-5K no es solo un dataset; es un laboratorio natural: desbalance extremo (6%), labels a nivel paciente (introduce label noise), múltiples patologías concurrentes, y variabilidad de calidad. Cada propiedad que lo hace desafiante para clasificación lo hace ideal para validar UQ.

### Insight 5: Justificación Bio-Inspirada Genuina

La detección de conflictos cross-modal en el cerebro humano (efecto McGurk, ventriloquismo) es un principio neurocientífico bien establecido. La divergencia KL entre modalidades es una formalización computacional de este mecanismo. Esto no es una metáfora forzada; es una justificación científica que encaja perfectamente en el scope de BIP (bio-inspired processing).

### Insight 6: Escalabilidad como Argumento de Impacto

Comparación de costos estimados (single imagen, GPU 16GB):

| Método | Passes | VRAM | Tiempo | Training |
|--------|--------|------|--------|----------|
| MC-Dropout (25) | 25 | 16GB | ~60s | No (pero requiere dropout) |
| Semantic Entropy (10) | 10+NLI | 16GB+ | ~30s | No |
| Deep Ensemble (5 modelos) | 5 | 80GB | ~12s | Sí (5×) |
| SAPLMA Probe | 1 | 16GB | ~2s | Sí (probe) |
| **KL Cross-Modal** | **1** | **16GB** | **~2s** | **No** |

Este costo computacional mínimo hace el método viable para deployment en el Global South (hospitales rurales, Latinoamérica), donde los recursos son limitados.

### Insight 7: La Paradoja del Few-Shot — Calidad sobre Cantidad

En few-shot, 10 ejemplos con alta KL son peores que 5 ejemplos con baja KL. La KL puede funcionar como **curador de support sets**: seleccionar ejemplos que maximicen la coherencia cross-modal. Esto planta la semilla para el siguiente paper post-BIP.

### Insight 8: El Camino Crítico para Julio 31 (5 semanas)

- **Semana 1 (ya):** Implementar pipeline end-to-end. Cargar MedGemma, extraer hidden states, computar KL, guardar CSV.
- **Semana 2:** Ejecutar experimentos en ODIR-5K. Computar baselines. Ablaciones (dirección, capa, temperatura).
- **Semana 3:** Análisis estadístico (DeLong, bootstrap, t-test). Visualizaciones (boxplots, PR curves, cuadrantes de error).
- **Semana 4:** Redactar paper 6-8 páginas IEEE. Double-blind compliance. Submitir a CMT.
- **Riesgos críticos:** Bug de hidden states (mitigación: workaround de decoder), licencia MedGemma (aceptar ahora), download ODIR-5K (descargar hoy), power estadístico limitado (reportar effect sizes + CI).

### Insight 9: La Posición Estratégica de BIP 2026 en el Roadmap Doctoral

BIP 2026 es el **nodo central** de la tesis: desde la revisión sistemática y GECCO 2026 (MC-Dropout), hasta la segmentación text-guided y validación clínica. La secuencia de venues recomendada:
1. BIP 2026 (Jul 2026): Proof of concept.
2. CIARP 2026 (Sep 2026): Extensión latinoamericana / few-shot.
3. MICCAI 2027 OMIA (Mar 2027): Segmentación + validación clínica.
4. IEEE TMI / JBHI (2027–2028): Journal version consolidada.

---

## Recomendaciones Inmediatas para el Usuario

1. **Aceptar la licencia Health AI Developer Foundations** en Hugging Face para descargar MedGemma 4B hoy mismo.
2. **Descargar ODIR-5K** desde Grand Challenge o Kaggle y verificar la estructura del CSV.
3. **Implementar el pipeline mínimo** (dim07, pasos 1-7) en una muestra de 20 imágenes para validar que los hidden states se extraen correctamente.
4. **Verificar el bug de vision hidden states** en tu versión de transformers. Si `output_hidden_states` no retorna vision hidden states, usar el workaround del decoder (primeros 256 tokens = projected vision).
5. **Definir el prompt definitivo** para el paper. Recomendación inicial: P4 (system prompt como experto) + P1 (directo binario). Evaluar 2-3 variantes en ablación.
6. **Crear cuenta en Microsoft CMT** para BIP 2026 y verificar el formato de submission (double-blind PDF).
7. **Preparar el repositorio anónimo** en anonymous.4open.science con el código de inference y evaluación.
8. **No publicar el título/abstract en arXiv** antes de la notificación de BIP (para preservar double-blind).

---

## Referencias Maestras (Citas Recurrentes en Múltiples Dimensiones)

- [^1] Badash, Z. N., & Belinkov, Y. (2026). *Between the Layers Lies the Truth: Uncertainty Estimation in LLMs Using Intra-Layer Local Information Scores*. arXiv:2603.22299.
- [^2] Sellergren, A., et al. (2025). *MedGemma: Technical Report*. arXiv:2507.05201.
- [^3] Liang, X., et al. (2025). *Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models*. ICCV 2025.
- [^4] Kuhn, L., et al. (2024). *Detecting hallucinations in large language models using semantic entropy*. Nature.
- [^5] Guo, C., et al. (2017). *On calibration of modern neural networks*. ICML.
- [^6] Liu, W., et al. (2020). *Energy-based out-of-distribution detection*. NeurIPS.
- [^7] Hendrycks, D., & Gimpel, K. (2017). *A baseline for detecting misclassified and out-of-distribution examples in neural networks*. ICLR.
- [^8] Azaria, A., & Mitchell, T. (2023). *The Internal State of an LLM Knows When its Lying*. arXiv:2304.13734.
- [^9] Orlando, J. I., et al. (2020). *REFUGE Challenge: A unified framework for evaluating automated methods for glaucoma assessment*. Medical Image Analysis.
- [^10] Sounderajah, V., et al. (2025). *The STARD-AI reporting guideline for diagnostic accuracy studies using artificial intelligence*. Nature Medicine.
- [^11] Shi, D., et al. (2025). *EyeCLIP: A multimodal visual-language foundation model for computational ophthalmology*. Nature Medicine / arXiv:2409.06644.
- [^12] McDermott, J., et al. (2024). *A Closer Look at AUROC and AUPRC under Class Imbalance*. NeurIPS 2024.
- [^13] Saito, T., & Rehmsmeier, M. (2015). *The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets*. PLOS ONE.
- [^14] Pineau, J., et al. (2021). *Improving reproducibility in machine learning research*. NeurIPS Reproducibility Checklist.
- [^15] Lambert, B., et al. (2024). *Trustworthy clinical AI solutions: A unified review of uncertainty quantification in Deep Learning models for medical image analysis*. Artificial Intelligence in Medicine.
- [^16] Huang, X., et al. (2023). *Artificial intelligence in glaucoma: opportunities, challenges, and future directions*. BioMedical Engineering OnLine.

---

*Este documento maestro fue compilado a partir de 12 archivos de investigación detallados. Para profundizar en cualquier dimensión, consultar el archivo correspondiente en `research/`.*
