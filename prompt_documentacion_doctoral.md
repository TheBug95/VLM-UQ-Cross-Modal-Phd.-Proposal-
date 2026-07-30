# Prompt para Generar la Documentación Doctoral del Proyecto BIP 2026

> **Instrucción:** Copia y pega este prompt completo en una nueva conversación de Antigravity (o el agente de tu elección) para que genere toda la documentación estructurada en la carpeta `Documentacion/`.

---

## Prompt

Eres un investigador doctoral en Inteligencia Artificial Médica. Tu tarea es generar la **documentación técnica completa y de calidad doctoral** de un proyecto de investigación que ya tiene resultados experimentales. Esta documentación constituye el avance formal de una propuesta de técnica de **Uncertainty Quantification (UQ) para Vision-Language Models (VLMs)** en el contexto de una tesis doctoral.

### Contexto del Proyecto

**Título:** *Cross-Modal Representation Disagreement as a Lightweight Uncertainty Signal for Glaucoma Detection in Medical Vision-Language Models*

**Objetivo:** Demostrar que cuando MedGemma-4B (un VLM médico basado en Gemma 3) se equivoca al clasificar una imagen de fondo de ojo como glaucoma/normal, el **desacuerdo entre su representación visual interna y su representación textual** (medido por divergencia KL entre los hidden states de los tokens de imagen y el token de respuesta) es significativamente mayor que cuando acierta. Esta señal de incertidumbre es **training-free** (no requiere entrenamiento), **single-pass** (se extrae en una sola pasada de inferencia) y **cross-modal** (explota la naturaleza bimodal del VLM).

**Contribuciones originales (verificadas como sin precedente en la literatura):**
1. **Señal KL cross-modal:** Primera extracción y uso de la divergencia KL entre los hidden states de los tokens visuales y el token de respuesta del decoder como señal de incertidumbre. Ningún trabajo previo extrae KL entre representaciones internas de modalidades distintas dentro del mismo forward pass de un VLM.
2. **Fusión parameter-free por ranks `rank(KL) + rank(1-MSP)`:** Combinación original de la señal cross-modal (espacio de representaciones internas) con la señal de output (Maximum Softmax Probability) mediante agregación de rangos sin hiperparámetros. Verificado: esta combinación específica NO ha sido propuesta en ningún paper previo. La técnica general de rank fusion existe en information retrieval (Reciprocal Rank Fusion, Cormack et al., 2009), pero su aplicación para fusionar señales de UQ heterogéneas (cross-modal + output-space) en VLMs es nueva.

**Estado:** El experimento está **completo con todos los baselines**. Tenemos:
- Pipeline de inferencia implementado y verificado (`src/inference.py`)
- 129 imágenes × 2 prompts = 258 inferencias deterministas ejecutadas
- 98 variantes de señales de incertidumbre evaluadas (KL, JSD, coseno × capas × temperaturas × 8 estrategias de pooling)
- **Verbalized Confidence (P5):** 129 inferencias extra (2×), baseline de confianza declarada
- **Self-Consistency (SC):** 50 imágenes × 10 muestras × 2 prompts a T=1.5 (10×), baseline multi-pass
- Figuras del paper generadas (Fig 2-9: boxplots, ROC/PR, accuracy-coverage, cuadrantes, costo-vs-AUROC, verbalized degeneracy, SC boxplots)
- Tablas de resultados (T1 resultados principales, T2 ablaciones 98 variantes, T3 comparativa con literatura, **T4 costo-beneficio**)
- Evaluación estadística completa (AUROC con BCa bootstrap, Mann-Whitney U, Spearman H4)

### Resultados Principales Obtenidos

Estos son los **datos reales verificados** del experimento. Úsalos textualmente en las tablas:

**Accuracy base de MedGemma:** 79.8% (103/129 correctas con P1)

**Señal ganadora:** `kl_t_v_L34_tau1.0_max` (KL del token de respuesta hacia los tokens de imagen, capa 34, temperatura 1.0, max pooling)
- AUROC = 0.661 [0.522, 0.772] (BCa 95%, 9999 remuestreos)
- AUPRC = 0.329 [0.183, 0.478]
- Mann-Whitney U: p = 0.0057, effect size r = 0.223
- Sensitivity @ 80% Specificity = 0.423

**Señal combinada** `rank(KL) + rank(1-MSP)` **(CONTRIBUCIÓN ORIGINAL — sin precedente en la literatura)**:
- AUROC = 0.698 [0.596, 0.787]
- AUPRC = 0.375 [0.228, 0.545]
- Mann-Whitney U: p = 0.00096
- Mejora de +5.5% AUROC sobre KL sola y +11.9% sobre 1-MSP sola
- **Originalidad:** La fusión por ranks de una señal de espacio de representaciones internas (KL cross-modal) con una señal de espacio de salida (1-MSP) es nueva. No existe paper previo que combine estas dos familias de señales UQ de esta manera. La técnica de rank aggregation per se viene de IR (Cormack 2009), pero su instanciación aquí es original.

**Baselines de igual costo (single-pass, 1×):**
| Señal | AUROC | AUPRC | Mann-Whitney p |
|-------|-------|-------|----------------|
| KL cross-modal (nuestra) | 0.661 | 0.329 | 0.0057 |
| Entropy | 0.624 | 0.277 | 0.0258 |
| 1 - MSP | 0.624 | 0.277 | 0.0258 |
| Energy | 0.560 | 0.242 | 0.1745 |
| rank(KL) + rank(1-MSP) | 0.698 | 0.375 | 0.00096 |

**Baselines de mayor costo (multi-pass):**
| Señal | Costo | AUROC | Evaluación |
|-------|-------|-------|------------|
| Verbalized Confidence (P5) | 2× | 0.519 | 129 imgs |
| SC frac_other | 10× | 0.655 | 50 imgs (SC) |
| SC entropía binaria | 10× | 0.573 | 50 imgs (SC) |
| SC entropía 3-vías | 10× | 0.552 | 50 imgs (SC) |

**Comparación justa en el subconjunto SC (50 imágenes):**
| Señal | Costo | AUROC |
|-------|-------|-------|
| KL cross-modal (nuestra) | 1× | 0.781 |
| rank(KL)+rank(1-MSP) | 1× | 0.731 |
| SC frac_other | 10× | 0.655 |
| 1-MSP | 1× | 0.573 |
| SC entropía binaria | 10× | 0.573 |
| SC entropía 3-vías | 10× | 0.552 |

**Hallazgos clave de los baselines multi-pass:**
- **Verbalized Confidence:** MedGemma solo declara 2 valores de confianza: 90% (n=11) y 95% (n=118). La señal es casi constante → AUROC 0.519 (≈azar). Sobreconfianza extrema: declara 95% pero accuracy real = 80.5%.
- **Self-Consistency a T=1.5:** La mayoría de muestras generan tokens fuera de formato ("based", "i", "the", "it") en vez de "yes"/"no". `frac_other` (fracción de respuestas fuera de formato) es la mejor señal SC (AUROC 0.655) pero inferior a KL (1×). La entropía binaria sí/no es casi nula (el modelo casi nunca dice "no" al muestrear).
- **Dominancia Pareto:** KL cross-modal (1×) domina a TODOS los baselines multi-pass (10×) en AUROC con 10× menos cómputo.

**Accuracy-Coverage (derivando el X% más incierto):**
| Cobertura | Accuracy KL | n retenidos |
|-----------|-------------|-------------|
| 95% | 80.3% | 122 |
| 90% | 81.0% | 116 |
| 80% | 83.5% | 103 |
| 70% | 84.4% | 90 |
| 50% | 89.1% | 64 |

**Datos del modelo:**
- Modelo: `google/medgemma-4b-it` (MedGemma 4B, basado en Gemma 3)
- Parámetros: ~4B, frozen (sin fine-tuning)
- Vision encoder: MedSigLIP → 256 tokens de imagen × 2560 dim
- Decoder: 34 capas de transformer decoder-only
- Inferencia: greedy decoding, max_new_tokens=1, bfloat16
- Token IDs verificados: yes=4443, no=1904, image_soft_token=262144

**Datos del dataset:**
- MM-ODIR-129: 129 fotos de fondo de ojo completas (no recortes)
- 60 Normal / 69 Pathological (todos glaucoma)
- Re-anotadas por oftalmólogos de Costa Rica
- Splits: train 77 (36N/41P) / validation 26 (12N/14P) / test 26 (12N/14P)
- 7 gradings ordinales de glaucoma por imagen patológica (CDR 0-4, neuroretinal rim, etc.)
- 69 imágenes con máscaras de segmentación de copa/disco

### Estructura del Código Fuente (ya implementado)

```
src/
├── config.py      — Carga centralizada de config.yaml, resolución de Token IDs desde HF
├── data.py        — Descarga MM-ODIR-129, construye master_table.csv, auditoría de artefactos
├── inference.py   — Pipeline MedGemma: logits yes/no, hidden states, 98 variantes KL/JSD/coseno + verbalized (P5) + self-consistency (SC)
├── uncertainty.py — 8 estrategias de pooling: mean, max, roi, attn, topk, normw, rollout, headspec
├── evaluation.py  — AUROC/AUPRC BCa, Mann-Whitney, Spearman H4, accuracy-coverage, baselines
└── figures.py     — Generación de figuras 2-9 y tablas T1-T4 del paper
```

### Figuras Disponibles (ya generadas)

Las siguientes figuras están en la carpeta `figures/` del proyecto. **DEBES copiarlas a `Documentacion/assets/` y embederlas** con `![descripción](assets/nombre.png)` en cada documento donde correspondan:

1. `fig2_boxplot.png` — Boxplot de KL cross-modal: correctos vs. incorrectos (p=0.0057)
2. `fig2_boxplot_entropy.png`, `fig2_boxplot_1msp.png`, `fig2_boxplot_energy.png`, `fig2_boxplot_rankcombo.png` — Boxplots de cada baseline
3. `fig3_roc_pr.png` — Curvas ROC y Precision-Recall de todas las señales
4. `fig4_accuracy_coverage.png` — Accuracy vs. Coverage (derivando el X% más incierto)
5. `fig5_quadrants.png` — Ejemplos de cuadrantes (correcto+baja u, correcto+alta u, error+alta u, error+baja u) con transcripciones clínicas
6. `fig6_correlacion_senales.png` — Correlación entre señales
7. `fig7_costo_vs_auroc.png` — **NUEVA:** Scatter de costo computacional (1×, 2×, 10×) vs. AUROC. Panel izquierdo: cohorte 129, panel derecho: subconjunto SC 50. Demuestra dominancia Pareto de KL (1×) sobre baselines multi-pass (10×)
8. `fig8_verbalized.png` — **NUEVA:** Panel izquierdo: distribución de confianza declarada (solo 2 valores: 90 y 95, n=11 y n=118). Panel derecho: calibración declarada vs. real (sobreconfianza: dice 95% pero acierta 80.5%)
9. `fig9_sc_boxplots.png` — **NUEVA:** Boxplots correcto/incorrecto de SC frac_other (AUROC=0.655, p=0.054) y entropía 3-vías (AUROC=0.552, p=0.294)
10. `heatmap_2472_right.png`, `heatmap_2759_left.png`, `heatmap_3086_right.png` — Heatmaps de atención sobre fundus reales

### Instrucciones de Generación

Genera la documentación **completa** en archivos Markdown separados dentro de la carpeta:
```
G:\My Drive\Dropbox\Migue\Doctorado\Años Doctorado\2do Año\Primer semestre\BIP 2026\Codigo\Documentacion\
```

Crea primero la subcarpeta `Documentacion/assets/` y copia allí todas las figuras de `figures/` que uses.

#### Archivos a generar (en este orden):

---

### 📄 01_Indice_General.md
**Índice maestro** con links a todos los documentos. Incluye:
- Tabla de contenidos con hipervínculos a cada documento
- Resumen ejecutivo del proyecto (1 párrafo)
- Estado actual del avance (checklist)
- Glosario de términos técnicos usados en toda la documentación

---

### 📄 02_Marco_Teorico.md
**Marco teórico profundo** (~3000-4000 palabras). Incluye:
- §2.1 Vision-Language Models (VLMs): arquitectura, pre-entrenamiento, estado del arte (CogVLM, LLaVA-Med, MedGemma)
- §2.2 Uncertainty Quantification en Deep Learning: taxonomía (aleática vs. epistémica), métodos clásicos (MC-Dropout, Deep Ensembles, Temperature Scaling)
- §2.3 UQ para LLMs y VLMs: Semantic Entropy, Verbalized Confidence, UMPIRE, VIG-TUQ, SAPLMA
- §2.4 El problema de la dilución espacial en imágenes médicas: por qué mean pooling falla cuando la región de interés (disco óptico) es 5-10% de la imagen
- §2.5 Cross-Modal Disagreement como señal de incertidumbre: intuición, formulación matemática
- §2.6 Glaucoma y diagnóstico por imagen de fondo de ojo: cup-to-disc ratio, signos clínicos, importancia del triage
- Usa diagramas Mermaid para ilustrar la arquitectura de un VLM genérico y la taxonomía de métodos UQ

---

### 📄 03_Hipotesis_y_Diseno_Experimental.md
**Hipótesis y diseño experimental completo** (~2500 palabras). Incluye:
- §3.1 Hipótesis formales:
  - **H1:** La señal KL cross-modal tiene AUROC > 0.5 (mejor que azar) para detectar errores del modelo
  - **H2:** La señal KL supera los baselines de igual costo (entropy, 1-MSP, energy)
  - **H3:** La señal KL es complementaria a los baselines (la combinación rank(KL)+rank(1-MSP) supera a ambas)
  - **H4:** En imágenes patológicas, la incertidumbre se correlaciona con la severidad del glaucoma (CDR grade)
- §3.2 Variables:
  - **Independiente:** Imagen de fondo de ojo + prompt textual
  - **Dependiente:** Valor de la señal de incertidumbre u(x)
  - **Controladas:** Modelo frozen, greedy decoding, seed 42, resolución 896×896
- §3.3 Protocolo experimental:
  - Selección de variante ganadora SOLO en train (77 imgs)
  - Confirmación en val+test (52 imgs)
  - Evaluación principal sobre las 129 imágenes (justificación: modelo frozen = no hay sobreajuste)
- §3.4 Prompts congelados (P1 y P4, texto literal)
- §3.5 Métricas de evaluación: AUROC, AUPRC, Mann-Whitney U (r), Sensitivity@80%Spec, AURC, Excess-AURC
- Diagrama Mermaid del flujo experimental end-to-end

---

### 📄 04_Arquitectura_Tecnica.md
**Arquitectura técnica detallada del pipeline** (~3000 palabras). Incluye:
- §4.1 Arquitectura de MedGemma-4B:
  - Vision encoder (MedSigLIP): 896×896 → patches 14×14 → 256 tokens × 1152-dim → proyector → 2560-dim
  - Decoder (Gemma 3): 34 capas, 2560-dim, causal self-attention
  - Diagrama Mermaid detallado del flujo de datos desde la imagen hasta el token de respuesta
- §4.2 Extracción de representaciones internas:
  - p_vis: hidden states de los 256 tokens de imagen en capa 34
  - p_text: hidden state de la última posición del prefill (= token que condiciona la respuesta)
  - Explicar por qué capa 34 (las capas 17, 26 colapsan numéricamente)
- §4.3 Conversión a distribuciones de probabilidad:
  - `F.log_softmax(vec / τ, dim=0, dtype=float64)`
  - Por qué se necesita float64 (massive activations de Gemma colapsan softmax en float32)
  - Por qué NO se usa z-score normalization (aplana la varianza de KL)
- §4.4 Cálculo de la divergencia KL:
  - KL(p_vis || p_text), KL(p_text || p_vis), JSD, distancia coseno
  - Fórmulas matemáticas en LaTeX
- §4.5 Estrategias de pooling (8 variantes):
  - mean, max, roi (oracle, requiere máscaras), attn (cross-attention), topk (Top-K por norma L2), normw (norm-weighted), rollout (Attention Rollout, Abnar & Zuidema 2020), headspec (selección de cabezas visuales)
  - Diagrama Mermaid comparando las 8 estrategias
- §4.6 Baselines de igual costo:
  - Entropy: H(p_yes, p_no)
  - MSP: max(p_yes, p_no)
  - Energy: -logsumexp(logit_yes, logit_no)
- §4.7 Señal combinada `rank(KL) + rank(1-MSP)` **(contribución original)**:
  - Motivación: KL captura desacuerdo cross-modal (espacio de representaciones), MSP captura confianza de output (espacio de logits). Son señales complementarias por naturaleza.
  - Formulación: `u_combo(x) = rank(KL(x)) + rank(1 - MSP(x))` — sin parámetros, sin normalización
  - Por qué rank fusion y no suma directa: las escalas de KL (~21-23 en este experimento) y 1-MSP (0-1) son incompatibles; los ranks las hacen comparables sin introducir hiperparámetros
  - Originalidad: verificado por búsqueda exhaustiva en la literatura — NO existe paper previo que proponga esta combinación específica para UQ en VLMs
  - Fórmula en LaTeX
- §4.8 Sanity checks (los 8 checks implementados, con resultados reales del piloto)

---

### 📄 05_Implementacion_Software.md
**Documentación técnica del código** (~3000 palabras). Incluye:
- §5.1 Stack tecnológico: Python 3.10+, PyTorch, transformers ≥4.51.3, scipy, scikit-learn
- §5.2 Estructura del proyecto (árbol de directorios completo con descripciones)
- §5.3 Módulo por módulo:
  - `src/config.py`: carga de config.yaml, resolución de Token IDs, determinismo
  - `src/data.py`: descarga del dataset, tabla maestra, auditoría de artefactos
  - `src/inference.py`: pipeline de inferencia, formato largo, self-consistency
  - `src/uncertainty.py`: 8 estrategias de pooling
  - `src/evaluation.py`: análisis estadístico, selección de variante ganadora
  - `src/figures.py`: generación de figuras
- §5.4 Diagrama de flujo del pipeline (Mermaid):
  - Desde `python -m src.data` → `python -m src.inference --pilot` → `python -m src.inference --run-full` → `python -m src.evaluation` → `python -m src.figures`
- §5.5 Formato de datos:
  - `master_table.csv`: esquema de 15 columnas
  - `results_full.csv`: esquema de formato largo (image × prompt × variante)
  - `evaluation_summary.csv`: esquema de resumen
- §5.6 Reproducibilidad: semillas, determinismo CUDA, versiones de dependencias

---

### 📄 06_Resultados_Experimentales.md
**Resultados completos con análisis** (~4000 palabras). Este es el documento más importante. Incluye:
- §6.1 Accuracy base de MedGemma (79.8%): contexto, distribución de P(yes)
- §6.2 Selección de variante ganadora (en train):
  - Top-10 variantes por AUROC en train (tabla con datos reales de T2)
  - Justificación de `kl_t_v_L34_tau1.0_max`
- §6.3 Resultados principales (H1):
  - **EMBEDER** `fig2_boxplot.png`: la señal KL es mayor en errores que en aciertos
  - AUROC = 0.661 [0.522, 0.772], p = 0.0057
  - Tabla T1 completa con datos reales
- §6.4 Comparación con baselines (H2):
  - **EMBEDER** `fig3_roc_pr.png`: curvas ROC y PR de todas las señales
  - Tabla comparativa con AUROC, AUPRC, Mann-Whitney p
- §6.5 Señal combinada (H3) — **CONTRIBUCIÓN ORIGINAL**:
  - rank(KL) + rank(1-MSP): AUROC = 0.698, mejora de +5.5% sobre KL sola y +11.9% sobre 1-MSP
  - Enfatizar que esta combinación es propuesta original del autor: la fusión parameter-free por ranks de una señal cross-modal (interna) con una señal de output (MSP) no ha sido publicada previamente
  - Discutir la complementariedad empírica: KL detecta errores que MSP no detecta y viceversa
  - Explicar por qué rank fusion es superior a suma directa (incompatibilidad de escalas KL~22 vs MSP~0-1)
- §6.6 Accuracy-Coverage (aplicación clínica):
  - **EMBEDER** `fig4_accuracy_coverage.png`
  - Tabla de accuracy por nivel de cobertura (datos reales)
  - Interpretación clínica: derivando el 30% más incierto, accuracy sube de 79.8% a 84.4%
- §6.7 Análisis cualitativo:
  - **EMBEDER** `fig5_quadrants.png`: ejemplos de cuadrantes con transcripciones
  - Discusión de los 4 cuadrantes
- §6.8 Heatmaps de atención:
  - **EMBEDER** los 3 heatmaps: `heatmap_2472_right.png`, `heatmap_2759_left.png`, `heatmap_3086_right.png`
  - Análisis de dónde mira el modelo
- §6.9 Confirmación en val+test:
  - AUROC = 0.569 [0.200, 0.938] — alta varianza por n=52 (solo 6 errores)
  - Discusión honesta de la limitación estadística
- §6.10 Baselines multi-pass:
  - §6.10.1 Verbalized Confidence (P5, 2×):
    - AUROC = 0.519 (≈ azar) — la señal es casi inútil
    - El modelo solo declara 2 valores de confianza: 90% (n=11) y 95% (n=118)
    - Sobreconfianza: declara 95% → accuracy real 80.5%; declara 90% → accuracy real 72.7%
    - **EMBEDER** `fig8_verbalized.png`: distribución degenerada + calibración sobreconfiada
    - Interpretación: los VLMs instrucción-tuned están tan alineados que no usan la escala completa de confianza
  - §6.10.2 Self-Consistency (SC, 10×, T=1.5):
    - 50 imágenes × 10 muestras × 2 prompts, temperatura 1.5 (elegida alta para evitar votos unánimes)
    - Hallazgo clave: la mayoría de muestras generan tokens fuera de formato ("based", "i", "the") en vez de "yes"/"no"
    - Mejor señal SC: `frac_other` (fracción fuera de formato) → AUROC = 0.655, pero MWU p = 0.054 (no significativa)
    - Entropía 3-vías: AUROC = 0.552, p = 0.294 (no significativa)
    - **EMBEDER** `fig9_sc_boxplots.png`: boxplots frac_other y entropía 3-vías
    - Interpretación: MedGemma no es robusto al muestreo estocástico — la inestabilidad generativa ("deriva") puede ser en sí misma una señal proxy de incertidumbre
  - §6.10.3 Comparación costo-beneficio (Tabla T4 + Fig 7):
    - **EMBEDER** `fig7_costo_vs_auroc.png`: scatter costo (1×/2×/10×) vs. AUROC
    - **Resultado central:** KL cross-modal (1×) domina a TODOS los baselines multi-pass en el gráfico Pareto
    - En comparación justa (50 imgs del subconjunto SC): KL (1×) AUROC = 0.781 vs. SC frac_other (10×) = 0.655
    - Tabla T4 con datos reales de los 12 métodos evaluados
- §6.11 Tabla T2 de ablaciones:
  - Top-20 variantes ordenadas por AUROC
  - Análisis de tendencias: max > mean, τ=1 > τ>1, kl_t_v > kl_v_t
- §6.12 Tabla T3 comparativa con la literatura:
  - Nuestro método vs. MC-Dropout, Semantic Entropy, UMPIRE, VIG-TUQ, SAPLMA
  - Explicar por qué la comparación es sobre propiedades (single-pass, training-free, cross-modal) y no sobre AUROC directo (diferentes datasets/modelos)

---

### 📄 07_Ablaciones_y_Analisis_Profundo.md
**Ablaciones detalladas** (~3000 palabras). Incluye:
- §7.1 Efecto del tipo de divergencia: KL vs. JSD vs. Coseno
  - ¿Por qué KL(p_text || p_vis) supera a KL(p_vis || p_text)?
  - Interpretación: la asimetría de KL captura "sorpresa" del modelo
- §7.2 Efecto de la estrategia de pooling (8 variantes):
  - Tabla completa con AUROC por pooling (datos reales de T2)
  - ¿Por qué max > mean? (captura la activación pico, no diluye)
  - ¿Por qué roi (oracle) no ayuda? (análisis y discusión)
  - ¿Por qué topk y normw no superan a max? (posible explicación)
- §7.3 Efecto de la temperatura τ:
  - τ=1 > τ=2 > τ=4: la distribución aplanada pierde la señal
- §7.4 Efecto de la capa:
  - Solo capa 34 funciona; capas 17 y 26 colapsan a KL ≈ 0 para todos
  - Interpretación: las capas tardías codifican la decisión semántica
- §7.5 Efecto del prompt:
  - P1 (AUROC 0.661) > P4 con system prompt de experto (AUROC 0.614)
  - Interpretación: ¿el system prompt "alinea" las representaciones y reduce el desacuerdo?
- §7.6 Señal KL de prompt (imagen vs. texto del prompt):
  - AUROC = 0.483 (peor que azar)
  - Interpretación: la señal útil está en la respuesta, no en el prompt
- §7.7 Análisis de baselines multi-pass vs. single-pass:
  - ¿Por qué Verbalized Confidence falla? Degeneración de la escala (solo 90/95), sobreconfianza instruction-tuned
  - ¿Por qué Self-Consistency falla? Inestabilidad generativa a T=1.5 (drift fuera de formato), frac_other como proxy accidental
  - La señal cross-modal (interna) captura información que la señal de output (logits/texto generado) no puede: el desacuerdo entre modalidades ocurre ANTES de la generación

---

### 📄 08_Discusion_y_Limitaciones.md
**Discusión crítica y limitaciones** (~2500 palabras). Incluye:
- §8.1 Interpretación teórica: ¿qué mide el desacuerdo cross-modal?
  - Cuando el modelo "ve" algo ambiguo pero "dice" algo seguro → KL alta
  - Conexión con calibración: modelos sobreconfiados tienen baja entropy pero alta KL cross-modal
- §8.2 Contribuciones originales a la tesis doctoral (dos contribuciones verificadas):
  - **Contribución 1 — Señal KL cross-modal:** Primer método de UQ que es simultáneamente single-pass, training-free Y cross-modal. Nadie antes extrajo KL entre hidden states de tokens visuales y textuales del decoder de un VLM.
  - **Contribución 2 — Fusión parameter-free `rank(KL) + rank(1-MSP)`:** Primera combinación de una señal de espacio de representaciones internas (cross-modal KL) con una señal de espacio de output (MSP) mediante agregación de rangos. Verificado por búsqueda exhaustiva: no existe paper previo que proponga esta combinación. La complementariedad empírica (AUROC 0.698 > 0.661 KL sola > 0.624 MSP sola) demuestra que ambos espacios aportan información ortogonal.
  - Apertura para generalizar a otros VLMs (LLaVA-Med, CogVLM) y otras tareas médicas
- §8.3 Limitaciones honestas:
  - N=129 es pequeño (potencia estadística limitada)
  - AUROC 0.661 es moderado (no suficiente para deployment clínico directo)
  - Solo un modelo (MedGemma-4B) y un dataset (MM-ODIR-129)
  - Confirmación en val+test débil (AUROC 0.569, CI muy ancho)
  - Self-Consistency evaluado sobre subconjunto de 50 imágenes (no 129), lo cual limita la comparabilidad directa
  - Verbalized Confidence degenerada (solo 2 valores), lo que sugiere una limitación fundamental de los VLMs instruction-tuned para auto-evaluarse
  - Heatmaps no validados cuantitativamente contra segmentaciones ground truth
- §8.4 Riesgos identificados:
  - Massive activations: riesgo de colapso numérico si se cambia de float64
  - Dependencia de la versión de transformers (≥4.51.3 obligatorio)
- §8.5 Trabajo futuro (roadmap para la tesis):
  - Escalar a datasets más grandes (RIM-ONE, REFUGE, ORIGA)
  - Probar con otros VLMs (LLaVA-Med, GPT-4V, Gemini Pro Vision)
  - Gradient-weighted pooling como alternativa a max pooling
  - Integración con triage automatizado en clínica oftalmológica

---

### 📄 09_Dataset_MM_ODIR_129.md
**Documentación completa del dataset** (~1500 palabras). Incluye:
- §9.1 Origen: ODIR-5K → re-anotación por oftalmólogos de Costa Rica
- §9.2 Estructura: 129 imágenes, splits, formato de archivos
- §9.3 Variables de anotación: 7 gradings ordinales, transcripciones clínicas, máscaras
- §9.4 Artefactos de anotación: `1281_right.jpg` con flecha negra quemada
- §9.5 Análisis estadístico: distribución de clases, balance por split, CDR grades
- §9.6 Consideraciones éticas: PII (doctor_name), doble ciego, licencia MIT

---

### 📄 10_Guia_Reproducibilidad.md
**Guía de reproducibilidad completa** (~1500 palabras). Incluye:
- §10.1 Requisitos de hardware y software
- §10.2 Instalación paso a paso (desde cero)
- §10.3 Configuración de HuggingFace (HF_TOKEN, licencia HAI-DEF)
- §10.4 Comandos del pipeline en orden
- §10.5 Validaciones pre-implementación (7 scripts de validación)
- §10.6 Verificación de resultados: checksums, conteos esperados

---

### 📄 11_Conclusiones_y_Proximos_Pasos.md
**Conclusiones formales y roadmap doctoral** (~1500 palabras). Incluye:
- §11.1 Contribuciones de esta fase (dos contribuciones originales verificadas):
  1. **Contribución original #1:** Primera demostración empírica de cross-modal disagreement (KL entre hidden states visuales y textuales del decoder) como señal de UQ en VLMs médicos — sin precedente en la literatura
  2. **Contribución original #2:** Fusión parameter-free `rank(KL) + rank(1-MSP)` — primera combinación por ranks de una señal de espacio de representaciones internas con una señal de espacio de output para UQ en VLMs — verificado como sin precedente en la literatura
  3. Framework de evaluación con 98 variantes y baselines de igual costo
  4. Pipeline training-free, single-pass, costo computacional 1×
- §11.2 Verificación de hipótesis (tabla con H1-H4: verificada/parcial/pendiente)
- §11.3 Roadmap doctoral:
  - Fase 2: Escalar a múltiples datasets y VLMs
  - Fase 3: Generalizar a otras tareas médicas (dermatología, radiología)
  - Fase 4: Integración con sistemas de triage clínico
  - Defensa de la técnica de UQ como contribución doctoral

---

### Reglas de Calidad Obligatorias

1. **Idioma:** Todo en español. Los términos técnicos en inglés se mantienen en inglés (KL divergence, hidden states, pooling, AUROC, etc.).
2. **Datos reales:** NUNCA inventes números. Usa exclusivamente los datos proporcionados arriba o los que extraigas de los archivos CSV del proyecto.
3. **Figuras:** Copia las figuras de `figures/` a `Documentacion/assets/` y embédelas con sintaxis Markdown estándar `![descripción](assets/nombre.png)`.
4. **Diagramas:** Usa bloques Mermaid ```mermaid``` para diagramas de arquitectura, flujos de datos, taxonomías y pipelines.
5. **Tablas:** Usa tablas Markdown con alineación y datos reales formateados a 3 decimales.
6. **Referencias:** Cita la literatura relevante en formato inline (Autor, Año) — no necesitas BibTeX.
7. **Fórmulas:** Usa LaTeX inline `$...$` y bloques `$$...$$` para todas las fórmulas matemáticas.
8. **Estructura:** Cada documento debe tener navegación al anterior/siguiente.
9. **Honestidad:** Reporta las limitaciones con total transparencia. Este es un avance doctoral, no un paper final.
10. **Cross-references:** Usa links relativos entre documentos `[ver §6.3](06_Resultados_Experimentales.md#63-resultados-principales-h1)`.
11. **Originalidad:** Siempre que menciones `rank(KL) + rank(1-MSP)`, enfatiza explícitamente que es una **contribución original del autor** (no viene de ningún paper previo). La técnica genérica de rank fusion existe en information retrieval (Cormack et al., 2009), pero su instanciación para combinar señales de UQ heterogéneas (cross-modal + output-space) en VLMs es nueva. Lo mismo aplica para la señal KL cross-modal entre hidden states del decoder.

### Archivos del proyecto a leer para extraer datos

Lee los siguientes archivos antes de generar la documentación:

- `config.yaml` — hiperparámetros del experimento
- `src/config.py`, `src/data.py`, `src/inference.py`, `src/uncertainty.py`, `src/evaluation.py`, `src/figures.py` — código fuente completo
- `figures/tabla_t1_resultados.csv` — Tabla T1 (resultados principales, incluye AURC y Excess-AURC)
- `figures/tabla_t2_ablaciones.csv` — Tabla T2 (98 variantes con AUROC)
- `figures/tabla_t3_comparativa.csv` — Tabla T3 (comparativa con literatura)
- `figures/tabla_t4_costo_beneficio.csv` — **NUEVA:** Tabla T4 (costo computacional vs. AUROC para todos los métodos)
- `results/evaluation_summary.csv` — Resumen de evaluación con BCa CI
- `results/acc_cov_P1_winner.csv` — Accuracy-coverage de la señal ganadora
- `results/results_verbalized.csv` — **NUEVO:** 129 filas de confianza verbalizada (P5)
- `results/results_self_consistency.csv` — **NUEVO:** 101 filas (50 imgs × 2 prompts) de self-consistency
- `results/analisis_verbalized.py` — Script de análisis del baseline verbalized
- `results/analisis_sc_v2.py` — Script de análisis del baseline self-consistency (voto 3-vías)
- `Definicion_Experimental_Minima_BIP2026.md` — Especificación experimental congelada
- `Guia_Conceptual_y_Algoritmo_BIP2026.md` — Guía pedagógica paso a paso
- `Analisis_Dataset_MM_ODIR_129.md` — Análisis del dataset
- `AGENTS.md` — Especificaciones técnicas detalladas

### Ruta de la documentación

```
G:\My Drive\Dropbox\Migue\Doctorado\Años Doctorado\2do Año\Primer semestre\BIP 2026\Codigo\Documentacion\
├── assets/                          ← Copiar aquí las figuras de figures/
│   ├── fig2_boxplot.png
│   ├── fig2_boxplot_entropy.png
│   ├── fig2_boxplot_1msp.png
│   ├── fig2_boxplot_energy.png
│   ├── fig2_boxplot_rankcombo.png
│   ├── fig3_roc_pr.png
│   ├── fig4_accuracy_coverage.png
│   ├── fig5_quadrants.png
│   ├── fig6_correlacion_senales.png
│   ├── fig7_costo_vs_auroc.png      ← NUEVA
│   ├── fig8_verbalized.png          ← NUEVA
│   ├── fig9_sc_boxplots.png         ← NUEVA
│   ├── heatmap_2472_right.png
│   ├── heatmap_2759_left.png
│   └── heatmap_3086_right.png
├── 01_Indice_General.md
├── 02_Marco_Teorico.md
├── 03_Hipotesis_y_Diseno_Experimental.md
├── 04_Arquitectura_Tecnica.md
├── 05_Implementacion_Software.md
├── 06_Resultados_Experimentales.md
├── 07_Ablaciones_y_Analisis_Profundo.md
├── 08_Discusion_y_Limitaciones.md
├── 09_Dataset_MM_ODIR_129.md
├── 10_Guia_Reproducibilidad.md
└── 11_Conclusiones_y_Proximos_Pasos.md
```

Comienza generando la carpeta `assets/` copiando las figuras, luego genera los 11 documentos en orden. Para cada documento, sigue la estructura de secciones indicada arriba y asegúrate de que tenga profundidad doctoral.
