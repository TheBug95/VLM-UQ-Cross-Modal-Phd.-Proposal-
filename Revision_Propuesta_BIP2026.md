# Revisión Detallada de la Propuesta BIP 2026

**Fecha de revisión:** 17 de julio de 2026
**Documentos revisados:** `Propuesta BIP 2026.md`, `BIP2026_Dossier_Maestro.md`, `bip2026_experimental_plan.md`, `bip2026_pilares_analysis.md`, `research/bip2026_dim01–dim10.md`, `research/bip2026_cross_verification.md`, `research/bip2026_insight.md`
**Deadline asumido:** 31 de julio de 2026 (14 días naturales desde hoy)

---

## 0. Veredicto Ejecutivo

| Pregunta | Respuesta corta |
|----------|-----------------|
| ¿Es publicable la idea? | **Sí.** El posicionamiento (single-pass + feature-based + cross-modal, aplicación oftalmológica, framing bio-inspirado) es coherente y defendible para el nivel de BIP. |
| ¿Es posible en 2 semanas? | **Sí, pero solo con el alcance mínimo** de la `Propuesta` (1 dataset, 3–5 baselines, 2 prompts). El plan experimental completo (9 baselines × 3 datasets × 6 prompts + reader study) **no es ejecutable** en 2 semanas. |
| ¿El deadline 31-jul está confirmado? | **No.** Al 17-jul el CFP de bipconference.org sigue con la sección "Important Dates" vacía. Histórico: 15-ago (BIP 2022), 20-ago/13-sep (BIP 2021). El 31-jul es un deadline autoimpuesto. **Confirmar hoy en CMT o con los chairs.** Si el deadline real es mediados de agosto, el plan respira 2 semanas más. |
| ¿Hay errores técnicos? | **Sí, 3 críticos:** (1) bug de indexación de los tokens de imagen en el código propuesto; (2) el lado "texto" de la KL es el *prompt*, no la *respuesta* del modelo — la interpretación "lo que ve vs. lo que dice" está sobre-vendida; (3) baselines conceptualmente rotos para este modelo (MC-Dropout) o degenerados (Semantic Entropy binaria). |
| ¿Qué falta? | Una **definición operacional exacta de qué son p_vis y p_text**. Es el hueco central y probablemente la razón por la que la propuesta "aún no te queda clara". |

**Recomendación:** proceder, pero ejecutar primero las 5 decisiones de la Sección 7 (hoy), corregir los 3 errores críticos, y congelar el alcance mínimo antes de escribir código.

---

## 1. SECCIÓN DEDICADA: La Propuesta Explicada en Detalle

Esta sección existe porque los 4 documentos describen la idea a distintos niveles de abstracción y con detalles inconsistentes entre sí. Aquí está la propuesta completa, de principio a fin, en un solo lugar.

### 1.1 La idea en una frase

> Si un modelo de visión-lenguaje médico (MedGemma) "ve" algo distinto de lo que "dice", la distancia entre ambas representaciones internas debería ser grande — y esa distancia, medida con divergencia KL en **una sola pasada hacia adelante** y **sin entrenar nada**, puede servir como señal de incertidumbre para saber cuándo el modelo probablemente se está equivocando al detectar glaucoma.

### 1.2 El problema que resuelve

Los métodos de estimación de incertidumbre (UQ) usados en imagen médica tienen costos que los hacen poco prácticos en entornos de recursos limitados (el argumento del *Global South* es uno de los ganchos del paper):

- **MC-Dropout / Deep Ensembles:** 5–100 pasadas por imagen (o 5 modelos en memoria).
- **Semantic Entropy:** 10+ generaciones muestreadas + un modelo NLI externo para agrupar respuestas.
- **Entropía / MSP sobre el texto generado:** baratos, pero solo miran la salida textual — no detectan cuando el modelo *alucina* ignorando la imagen (el modo de fallo más peligroso en MedVLMs, documentado en la literatura como *language prior bias*).

El espacio metodológico que ocupa la propuesta (Insight 1 del dossier): **single-pass + basada en features internas + cross-modal**. Ningún método publicado combina las tres. Esa es la novedad.

### 1.3 El método, paso a paso (versión operacional)

```
ENTRADA: imagen de fondo de ojo (ODIR-5K) + prompt "Does this fundus image show glaucoma?"
│
├─ 1. MedSigLIP (vision encoder de MedGemma) codifica la imagen 896×896
│     → 4096 patches → pooling → 256 "soft tokens" visuales proyectados a 2560-dim
│
├─ 2. Los 256 tokens visuales se insertan en la secuencia del decoder (Gemma 3, 34 capas)
│     junto con los tokens del prompt → forward pass (prefill)
│
├─ 3. Se extraen los hidden states de la última capa:
│     • posiciones de los tokens de imagen  → representación "visual"  (256 × 2560)
│     • posiciones de los tokens de texto   → representación "textual" (T × 2560)
│
├─ 4. Mean-pooling sobre cada conjunto → dos vectores de 2560 dimensiones
│
├─ 5. Softmax temperado sobre la dimensión de features:
│     p_vis = softmax(v / τ)      p_text = softmax(t / τ)
│
├─ 6. Divergencia:  u(x) = KL(p_vis ‖ p_text)
│
├─ 7. En paralelo, del primer token generado: P(yes) vs. P(no)
│     → predicción binaria → etiqueta "correcto / incorrecto" vs. ground truth
│
SALIDA por imagen: { id, u(x), P(yes), predicción, ground_truth, correcto? }
```

**La hipótesis experimental:** `u(x)` es significativamente mayor en las imágenes donde el modelo se equivoca que en las que acierta.

**Cómo se evalúa la hipótesis:**
1. **Mann-Whitney U** (o t-test si hay normalidad): ¿las dos distribuciones de KL (correctos vs. incorrectos) difieren? Con effect size (r o Cohen's d), no solo p-value.
2. **AUROC / AUPRC de u(x) como detector de errores:** se usa el valor de KL como score continuo para predecir la variable binaria "el modelo se equivocó". AUROC > 0.5 (idealmente ≥ 0.65–0.70) = la señal tiene valor.
3. **Accuracy vs. Coverage (selective classification):** si se abstiene en el k% de casos con mayor KL, ¿sube la accuracy en el resto? Esta es la métrica con significado clínico directo: "enviando el 10% más incierto a revisión humana, la accuracy del sistema sube de X a Y".
4. **Matriz de cuadrantes 2×2** (correcto/incorrecto × KL alta/baja): el cuadrante peligroso es *incorrecto + KL baja* (errores con sobre-confianza). El objetivo del método es que ese cuadrante quede lo más vacío posible.

### 1.4 Qué NO es la propuesta

- **No entrena nada.** MedGemma queda congelado. No hay fine-tuning, no hay probes, no hay LoRA.
- **No compite en accuracy de clasificación.** El paper no propone un mejor clasificador de glaucoma; propone una mejor forma de *saber cuándo no confiar* en el clasificador. Esto es importante porque el clasificador base es mediocre (MedGemma-4B en glaucoma: F1 ≈ 63.7% según VOLMO) — lejos de ser un problema, eso genera abundantes errores con los que evaluar la UQ.
- **No requiere labels de entrenamiento.** La señal KL no usa ground truth; el ground truth solo se usa para *evaluar* la señal.

### 1.5 De dónde viene la idea (linaje intelectual)

1. **"Between the Layers Lies the Truth" (Badash, Belinkov & Freiman, 2026, arXiv:2603.22299):** convierte las activaciones post-MLP de cada capa de un LLM en distribuciones (softmax temperado) y mide el *desacuerdo entre capas* con KL pairwise → una "signature map" L×L que predice correctitud. Ojo con los números al citarlo: su AUPRC ~0.82 es el promedio **bajo cuantización 4-bit**, no in-distribution (in-dist hay paridad o −1.4 a −1.8 pp vs. probing), el +2.86 pp es solo en Llama-3.1-8B, y su predictor es **supervisado** (LightGBM) — nuestra versión training-free es justo la diferenciación. **La propuesta BIP generaliza este principio de intra-capa (capa vs. capa, misma modalidad) a cross-modal (visión vs. texto, mismo forward).**
2. **Literatura de alucinaciones en VLMs:** los MedVLMs exhiben *modality bias* — el prior lingüístico domina sobre la evidencia visual. El desacuerdo cross-modal es, en teoría, el detector natural de ese fallo.
3. **Neurociencia (el gancho bio-inspirado):** el cerebro humano detecta conflictos entre modalidades sensoriales (efecto McGurk, ilusión del ventrílocuo) y los usa como señal de alerta. La KL entre modalidades es la formalización computacional de ese principio. **Ojo:** hoy esto está *afirmado* en los documentos pero no *argumentado con citas* — la cadena defendible (verificada 22-jul-2026) es: McGurk & MacDonald (1976, Nature 264:746–748, el fenómeno) → Ernst & Bülthoff (2004, TiCS 8(4):162–169, integración Bayesiana de modalidades) → Botvinick et al. (2001, Psych Review 108(3):624–652, *conflict monitoring*) → Yeung et al. (2004, Psych Review 111(4):931–959, conflicto → detección de errores). Sin esa cadena el framing bio-inspirado queda como metáfora decorativa ante un reviewer de BIP.

### 1.6 Cómo encaja en la tesis

Es el nodo central del roadmap doctoral: generaliza el trabajo de GECCO (MC-Dropout, caro) y "Between the Layers" (intra-modal) hacia UQ cross-modal, y siembra los otros tres pilares: la **dirección** de la KL como explicación nativa (XAI), la KL como curador de pseudo-labels/support sets (few-shot), y la KL descompuesta espacialmente como mapa de riesgo para segmentación (OD/OC). Los próximos venues (CIARP 2026 → MICCAI OMIA 2027 → journal) cuelgan de que este proof-of-concept salga.

### 1.7 Escenarios de resultado posibles

| Escenario | Lectura | Consecuencia |
|-----------|---------|--------------|
| AUROC(error detection) ≥ 0.70, significativo | Éxito pleno | Paper fuerte: señal útil, barata y novedosa. |
| AUROC 0.60–0.70 | Éxito parcial | Publicable en BIP: comparable a baselines baratos a costo mínimo; discutir cuándo funciona y cuándo no (análisis por cuadrantes y subgrupos). |
| AUROC ≈ 0.5 | Resultado negativo | Plan B (Sección 6): pivotear a análisis de *por qué* no funciona + variante JSD/por-capas, o reformular como estudio de limitaciones de UQ basada en features en MedVLMs. Publicable pero más débil; mejor saberlo en la semana 1. |

---

## 2. ¿Es Posible? (Análisis de Factibilidad)

### 2.1 Tiempo

La `Propuesta` estima: 2 h (dataset) + 1 día (forward passes) + 1 día (estadística) + 3–4 días (escritura) ≈ **6–7 días efectivos**. Con 14 días naturales disponibles hay colchón razonable **si y solo si**:
- el código funciona en los primeros 2 días (hay un bug conocido de `output_hidden_states` y un bug de indexación nuevo, ver §4.1–4.2);
- el dataset y la licencia del modelo están resueltos el día 1 (hoy ninguno de los dos está verificado, ver §2.3);
- el alcance queda congelado en el mínimo.

**No es posible** ejecutar el `bip2026_experimental_plan.md` completo: 9 baselines × 6 dimensiones de ablación × 6 prompts × 3 datasets + reader study con 3 oftalmólogos + análisis de subgrupos + power analysis Monte Carlo. Eso es un paper de journal de 3–4 meses. Los dos documentos describen, de facto, **dos papers distintos**, y hay que declarar ganador al de la `Propuesta` (versión mínima) hoy.

### 2.2 Cómputo

- MedGemma-4B: pesos ≈ 8.6 GB → ~10–12 GB VRAM en bfloat16 con activaciones; ~4–5 GB en 4-bit NF4 (T4 de Colab → ir en 4-bit; bf16 cómodo en L4/A100).
- ~7,000 imágenes × ~1–2 s/imagen (prefill de ~280 tokens + 1 token generado) ≈ **2–4 horas por corrida completa**. Perfectamente viable.
- Las ablaciones multiplican corridas: cada prompt extra = otra corrida completa. Con 2 prompts y extracción de *todas las capas y ambas direcciones de KL en la misma pasada* (guardar todo de una vez), se evita re-correr: **diseñar el CSV de salida para guardar capas {25%, 50%, 75%, 100%} × direcciones {KL(v‖t), KL(t‖v), JSD} × temperaturas {1, 2, 4} en una sola corrida**. Esto convierte las ablaciones en análisis sobre el CSV, no en re-cómputo. Es la decisión de ingeniería más importante del proyecto.
- ⚠️ Pendiente: confirmar que tienes GPU ≥16 GB disponible estas 2 semanas.

### 2.3 Datos y modelo (estado verificado hoy)

| Recurso | Estado al 17-jul | Acción |
|---------|------------------|--------|
| ODIR-5K imágenes + `data.xlsx` | **No verificado localmente.** En el workspace BIP no hay datos; en la carpeta GECCO solo hay scripts y JSONs de resultados de optimización (los scripts asumen el dataset pero las imágenes no están ahí). La `Propuesta` dice "dataset local creado por mí (2 h)" — hay que localizarlo o descargarlo. | **Hoy:** localizar el dataset donde sea que viva (otro disco, Colab, Kaggle) y copiarlo al workspace. |
| Licencia MedGemma (HAI-DEF) en HuggingFace | Sin verificar | **Hoy:** aceptar la licencia y probar descarga del checkpoint. |
| GPU ≥16 GB | Sin verificar | **Hoy:** confirmar hardware (local o cloud). |

### 2.4 Estadística

- Glaucoma en ODIR-5K: 207 (train) + 30 (off-site test) + 58 (on-site test) ≈ **295 pacientes positivos** (~6%).
- Punto clave que los documentos no explotan: **como el modelo es frozen y la señal es training-free, no hay riesgo de contaminación — se puede evaluar sobre el dataset completo (~7,000 imágenes, ~295 pacientes G)** en lugar de solo el test oficial (88 positivos). Los splits train/val solo se necesitan para los componentes que sí ajustan algo: temperature scaling, Mahalanobis, probe SAPLMA (si se incluyen). Esto mejora sustancialmente el poder estadístico y hay que argumentarlo explícitamente en el paper.
- Con N≈295 positivos: ΔAUROC de ~0.05 detectable con 80% de poder. Suficiente.
- El power analysis del plan usa N=207 asumiendo train split; hay que rehacerlo con el conjunto de evaluación real que se decida.

### 2.5 El deadline

El CFP público **sigue sin fechas** al 17-jul (verificado hoy en bipconference.org). El 31-jul es un deadline autoimpuesto; el histórico sugiere mediados de agosto o incluso septiembre. Como eres del comité, **confirma la fecha real hoy**: si es más tarde, se habilitan REFUGE como validación externa y ablaciones más ricas; si es el 31-jul, el plan mínimo es obligatorio.

---

## 3. Qué Falta (Gaps, en orden de prioridad)

1. **Definición operacional exacta de p_vis y p_text.** Ningún documento fija: (a) ¿qué tokens entran en "texto" — solo el prompt, o también la respuesta generada?; (b) ¿qué capa(s)?; (c) ¿qué pooling?; (d) ¿qué τ y qué ε para estabilidad numérica?; (e) ¿dirección canónica de la KL? Sin esto no hay método que implementar ni que escribir en la Sección de Métodos. **Esta es la decisión #1 (ver §7).**
2. **Protocolo de respuesta y de "correcto".** Hoy el plan parsea texto libre ("yes"/"no") del output generado — frágil. Falta: scoring por logits de "yes"/"no" en el primer token generado (genuinamente single-pass, robusto, y habilita P(yes) como probabilidad de clase).
3. **Guardar los logits del token de respuesta en el CSV.** Sin ellos no existen los baselines MSP / Entropy / Energy / TS — todos se calculan sobre la distribución del token de respuesta. El CSV planeado (id, KL, respuesta, gt) es insuficiente.
4. **Decisión de conjunto de evaluación** (todo el dataset por ser zero-shot vs. solo test oficial) y **manejo patient-level vs. image-level**: ODIR etiqueta por paciente pero incluye *diagnostic keywords por ojo* — usarlos para asignar label por imagen reduce el label noise en casos unilaterales. Falta especificarlo.
5. **Diseño del CSV "todo en una pasada"** (capas × direcciones × temperaturas) para que las ablaciones sean análisis, no re-cómputo.
6. **Presupuesto de cómputo y hardware confirmado** (quién corre qué, dónde, cuánto tarda).
7. **Plan de contingencia si AUROC ≈ 0.5** (ver §6). Los documentos asumen éxito; ninguno dice qué se hace si la señal no separa.
8. **Posicionamiento contra los vecinos más cercanos:** VIG-TUQ (arXiv:2605.27136 — Jensen-Shannon entre predicciones con/sin imagen + attention; **solo su score de atención es single-pass: su score JSD requiere un 2º forward sin imagen** — *muy* cercano conceptualmente), UMPIRE (arXiv:2602.24195, **multi-sample**, no single-pass), VLM-UQBench (arXiv:2602.09214), Expert-CFG (ICCV 2025, expert-in-the-loop), y los baselines de probing barato **Semantic Entropy Probes** (Kossen et al. 2024, arXiv:2406.15927) e **INSIDE** (Chen et al., ICLR 2024). Related work sin estos es un riesgo de rechazo directo; el claim "nadie ha publicado cross-modal KL como UQ en MedVLMs para oftalmología" debe suavizarse a "to the best of our knowledge, first in ophthalmic MedVLMs" + tabla comparativa de propiedades.
9. **Citas neurocientíficas reales** para el framing bio-inspirado (cadena verificada: McGurk & MacDonald 1976 → Ernst & Bülthoff 2004 → Botvinick et al. 2001 → Yeung et al. 2004; ver §3 item 3). Hoy el argumento bio-inspirado es una afirmación, no un argumento.
10. **Estructura página por página del paper** (la `Propuesta` la promete y no existe en ningún documento). Propuesta mínima: Abstract (0.25) + Intro (1) + Related Work (0.75) + Methods (1.5) + Experiments/Results (2) + Discussion+Limitations (0.75) + Conclusion (0.25) + References (1) ≈ 7.5 páginas.
11. **Ethics statement, repo anónimo (anonymous.4open.science), cuenta/registro CMT, checklist de anonimato** (sin logos institucionales en figuras — relevante porque eres del comité y del TEC).
12. **Figuras planeadas concretas:** Fig 1 (pipeline + octante de novedad 3D del Insight 1), Fig 2 (boxplot KL correcto/incorrecto), Fig 3 (ROC + PR), Fig 4 (accuracy-coverage), Fig 5 (ejemplos de los 4 cuadrantes). Falta asignar quién las hace y cuándo.

---

## 4. Qué Hay Mal (Errores Concretos)

### 4.1 🔴 Bug de indexación de los tokens de imagen (crítico)

El código de `dim07` extrae:
```python
vision_tokens = last_layer[:, :256, :]   # "primeros 256 tokens = imagen"
text_tokens   = last_layer[:, 256:, :]
```
**Esto es incorrecto.** Con `apply_chat_template`, la secuencia real es:
```
<bos><start_of_turn>user\n {system prompt} {prompt texto} <start_of_image> [256 tokens de imagen] <end_of_image>\n<end_of_turn>\n<start_of_turn>model\n
```
Los tokens de imagen están **en medio de la secuencia**, desplazados por todos los tokens de sistema y de prompt. `[:, :256, :]` capturaría `<bos>` + tokens de texto del sistema/prompt — es decir, la KL compararía "texto vs. texto" y el resultado sería un artefacto sin sentido. **Corrección obligatoria:** localizar las posiciones con una máscara sobre `input_ids == config.image_token_id` (o el índice de `<start_of_image>`), no con slicing fijo.

### 4.2 🔴 El lado "texto" es la pregunta, no la respuesta (brecha semántica crítica)

En el pipeline, `text_tokens` = tokens del system prompt + prompt del usuario. La respuesta del modelo ("yes"/"no") **no entra en la KL**. Consecuencias:
- Lo que se mide es desacuerdo entre la imagen y *una pregunta fija idéntica para todas las imágenes*, no entre "lo que el modelo ve" y "lo que el modelo diagnostica".
- La varianza de p_text entre muestras existe solo porque los estados ocultos de las posiciones de texto atienden a los tokens de imagen (atención causal del decoder) — es decir, p_text ya está contaminada/condicionada por la imagen. La interpretación "dos fuentes independientes que discrepan" se debilita.
- **Corrección recomendada:** incluir los hidden states de la **respuesta generada** (posiciones de decode) en el lado texto, o mejor aún, definir el lado texto como los estados del *token de respuesta*. Eso sí es "lo que el modelo dice". Mantener como ablación: KL(imagen‖prompt) vs. KL(imagen‖respuesta).

### 4.3 🔴 Softmax sobre la dimensión de features no es una distribución interpretable

`softmax(v/τ)` sobre las 2560 dimensiones del hidden state produce una "distribución sobre features", que no tiene semántica probabilística clara (no es distribución sobre clases, ni sobre tokens, ni sobre conceptos). El paper inspirador hace lo mismo pero comparando *capas del mismo token* (mismo espacio, misma semántica) — la comparación cross-modal hereda la heurística sin la misma justificación. No invalida el enfoque (la pregunta empírica "¿predice errores?" sigue siendo válida), pero:
- hay que presentarlo honestamente como heurística inspirada en [Badash & Belinkov], no como "distribución de probabilidad de las representaciones";
- la ablación con **JSD, KL simétrica y una medida sin softmax (cosine / Cauchy-Schwarz)** deja de ser opcional: es la defensa contra la crítica obvia del reviewer.

### 4.4 🟠 "Single forward pass" es impreciso tal como está el código

El snippet usa `generate(max_new_tokens=5)` = 1 prefill + hasta 5 decodes, y luego parsea texto libre. Dos problemas: (a) el claim de marketing del paper debe decir "single image encoding + greedy answer", o mejor, (b) reemplazar por **1 prefill + 1 token con scoring de logits yes/no**: genuinamente single-pass, sin parsing frágil, y produce P(yes) para todos los baselines.

### 4.5 🟠 MC-Dropout es un baseline conceptualmente roto para MedGemma

Gemma 3 no tiene capas de dropout activas en inferencia (configs típicos con dropout 0.0). "Habilitar dropout en test-time" requeriría modificar el modelo — contradice el framing "frozen, training-free". El plan ya lo intuye ("upper bound teórico") pero lo mantiene en la tabla. **Decisión:** eliminarlo y reemplazarlo por **Test-Time Augmentation** (5 augmentaciones visuales; multi-pass, agnóstico de arquitectura, fácil de implementar) como referencia multi-pass.

### 4.6 🟠 Semantic Entropy degenera en una tarea binaria

Con respuesta sí/no hay solo 2 clases semánticas; SE se reduce a la entropía de la frecuencia de yes/no en N muestras — es decir, **self-consistency**, que es un baseline legítimo pero hay que llamarlo por su nombre y computarlo así (10 muestras a T=0.7 en un subconjunto de test, no en las 7,000 imágenes).

### 4.7 🟠 Inconsistencias entre documentos

| Tema | `Propuesta` | `experimental_plan` / Dossier |
|------|-------------|-------------------------------|
| Alcance | Mínimo, 1 dataset | 9 baselines, 3 datasets, 6 prompts, reader study |
| Timeline | 2 semanas | 5 semanas (Insight 8, escrito el 23-jun) |
| Métrica headline | "AUPRC ~0.72" | AUROC primaria, AUPRC secundaria: McDermott et al. (NeurIPS 2024, arXiv:2401.06091) muestran que la supuesta ventaja de AUPRC puede ser un artefacto in-distribution — lo correcto es **reportar ambas con IC** y no predicar la superioridad de ninguna |
| Dataset | "MM-ODIR" | "No existe MM-ODIR" (dim04) → usar ODIR-5K |
| N en power analysis | — | 207 (train) pero eval split sin especificar |

Además: el "AUPRC ~0.72" del diagrama de la `Propuesta` es una expectativa inventada — no debe llegar al paper (ni al abstract) como afirmación previa al experimento.

### 4.8 🟡 Menores

- **Human baseline (reader study con 3 oftalmólogos):** inviable en 2 semanas → mover a future work.
- **REFUGE / RIM-ONE:** fuera del alcance mínimo; REFUGE (400 test, ~40 positivos) solo si el deadline real se confirma posterior al 31-jul.
- **Claim de novedad** redactado en absoluto ("nadie ha publicado...") → suavizar + tabla comparativa.
- **Brier score sobre KL min-max-escalada** (dim07) no es calibración real; si se reporta Brier/ECE, que sea sobre P(yes) con temperature scaling ajustada en val.
- El `F.kl_div(text_log, vision_dist)` del snippet computa KL(vision‖text) — la convención de PyTorch es traicionera; fijar la dirección canónica y verificarla con un test unitario trivial (KL(p‖p)=0).

---

## 5. Riesgos y Mitigaciones (tabla consolidada)

| # | Riesgo | Prob. | Impacto | Mitigación |
|---|--------|-------|---------|------------|
| 1 | Deadline real ≠ 31-jul (sin confirmar) | Media | Alto (planificación errada) | Confirmar hoy en CMT/chairs; re-planificar con la fecha real |
| 2 | Bug de indexación de image tokens (§4.1) | Alta si no se corrige | Fatal (resultados sin sentido) | Máscara por `image_token_id`; sanity check visual con 5 imágenes el día 2 |
| 3 | AUROC ≈ 0.5 (la señal no separa) | Media | Alto | Detectarlo el día 3–4 con la primera corrida; activar Plan B (§6); ablación imagen‖respuesta y por-capas como rescate |
| 4 | Dataset no localizado / licencia sin aceptar | Media | Alto (bloquea día 1) | Resolver hoy; si ODIR tarda, empezar con 20 imágenes de cualquier mirror de Kaggle |
| 5 | Bug conocido de `output_hidden_states` en transformers (dim02) | Media | Medio | Workaround documentado: llamar `model.vision_tower` directamente o tomar posiciones del decoder; fijar versión de transformers |
| 6 | Scope creep (el plan completo tienta) | Alta | Alto | Alcance mínimo congelado por escrito hoy; todo lo demás → sección Future Work |
| 7 | Clasificador base muy débil en glaucoma (F1≈64%) | Conocido | Medio | No es bug, es feature: más errores = mejor evaluación de UQ; reportar accuracy del base con honestidad y estratificar UQ por clase (detectar FN de glaucoma es lo clínicamente relevante) |
| 8 | Reviewer cuestiona el softmax-sobre-features (§4.3) | Alta | Medio | Framing honesto + ablación JSD/cosine + citar paper inspirador |
| 9 | Reviewer cuestiona bio-inspired como metáfora | Media | Medio | Citas neuro reales + 1 párrafo de mecanismo (conflict monitoring), no solo la metáfora |
| 10 | Re-identificación por double-blind (eres del comité/TEC) | Baja | Medio | Repo anónimo, sin logos, no arXiv previo, PDF sin metadata |

---

## 6. Plan de Contingencia (si la señal no funciona)

1. **Rescate técnico (día 4–5):** probar la variante imagen‖respuesta (§4.2), JSD, capas medias (50–75%), y τ alto. La literatura de probing sugiere que las capas medias-tardías codifican mejor la "verdad" — si la última capa no separa, las medias pueden hacerlo.
2. **Rescate de framing (día 6+):** reformular como *estudio de diagnóstico*: "cross-modal hidden-state disagreement does NOT predict errors in frozen MedVLMs for glaucoma — an analysis of why" + matriz de cuadrantes + comparación con lo que sí funciona (self-consistency). Resultado negativo bien analizado es publicable en BIP, aunque más débil.
3. **No hacer:** meter LightGBM sobre signature maps aplanadas (el método completo del paper inspirador) — rompe el framing training-free y no cabe en 2 semanas. Eso es el paper de CIARP/journal.

---

## 7. Decisiones a Tomar HOY (17-jul)

1. **Confirmar deadline real** (CMT / chairs del BIP — tienes acceso por el comité).
2. **Aceptar licencia MedGemma + probar descarga** del checkpoint `google/medgemma-4b-it`.
3. **Localizar ODIR-5K** (imágenes + data.xlsx) y copiarlo al workspace; verificar conteo de positivos G.
4. **Fijar la definición del método** (una línea por ítem):
   - Lado visión: estados de las posiciones de imagen (máscara por token id), capa(s) {50%, 75%, 100%}, mean-pooling.
   - Lado texto: **estados del token de respuesta generado** (primario) + prompt (ablación).
   - Divergencias: KL(v‖t), KL(t‖v), JSD (guardar las 3).
   - τ ∈ {1, 2, 4} (guardar las 3). ε=1e-10.
5. **Congelar alcance mínimo por escrito:** ODIR-5K completo (evaluación zero-shot, patient-level), prompts P1+P4, baselines {MSP, Entropy, TS+Entropy, Energy, Self-Consistency(10 muestras, subset)}, métricas {AUROC, AUPRC+bootstrap CI, Sens@80%Esp, Accuracy-Coverage}, stats {Mann-Whitney + effect size, DeLong}, 5 figuras. Todo lo demás = Future Work.

---

## 8. Plan Día a Día (17 → 31 jul)

| Día | Fecha | Trabajo |
|-----|-------|---------|
| 1 | vie 17 | Las 5 decisiones de §7. Setup de entorno. Descarga modelo + datos. |
| 2 | sáb 18 | Pipeline mínimo en 20 imágenes. Verificar máscara de image tokens (§4.1) y workaround de hidden states. Sanity checks: KL(p‖p)=0, dirección correcta. |
| 3 | dom 19 | Corrida completa ODIR (~7,000 imágenes, 2–4 h) con CSV "todo en una pasada". |
| 4 | lun 20 | Baselines desde logits guardados + estadística principal (Mann-Whitney, AUROC/AUPRC + CI). **Go/No-Go según §6.** |
| 5 | mar 21 | Figuras 2–4 (boxplot, ROC/PR, accuracy-coverage). Análisis de cuadrantes. |
| 6 | mié 22 | Ablaciones sobre el CSV (capas, direcciones, τ, prompt P1 vs P4). Self-consistency en subset. |
| 7 | jue 23 | Buffer para re-corridas. Ejemplos cualitativos de cuadrantes (Fig 5). Fig 1 (pipeline). |
| 8–11 | vie 24 – lun 27 | Escritura del paper (estructura de §3.10; el dossier ya tiene el 70% del contenido). |
| 12 | mar 28 | Revisión interna (idealmente con Saúl), pulido de related work (§3.8) y framing bio-inspirado (§3.9). |
| 13 | mié 29 | Double-blind check, repo anónimo, formato IEEE, ethics statement, PDF final. |
| 14 | jue 30 – vie 31 | Buffer + submission en CMT. |

Si el deadline real resulta ser posterior (p. ej. mediados de agosto): insertar REFUGE como validación externa y el análisis de subgrupos (calidad de imagen, patologías concurrentes) entre los días 7 y 12.

---

## 9. Resumen de una Línea

**La idea es buena y publicable, el plan mínimo es ejecutable en 2 semanas, pero hoy la propuesta tiene un hueco conceptual central (qué es exactamente p_text), un bug de indexación que invalidaría todos los resultados, dos baselines rotos, y un deadline sin confirmar — las 5 decisiones de la Sección 7 resuelven todo esto antes de escribir la primera línea de código.**
