# Definición Experimental Mínima — Paper BIP 2026

**Título del paper:** *Cross-Modal Representation Disagreement as a Lightweight Uncertainty Signal for Glaucoma Detection in Medical Vision-Language Models*
**Fecha:** 20 de julio de 2026 — **Versión 2 (21-jul-2026): actualizada al dataset MM-ODIR-129** (ver `Analisis_Dataset_MM_ODIR_129.md`) — **Versión 2.1 (22-jul-2026): correcciones propagadas desde el Plan de Validación** (`Plan_de_Validacion_BIP2026.md`): `load_dataset` sí funciona, `transformers>=4.51.3`, checkpoint MedGemma v1.0.1, VRAM realista (~10–12 GB bf16 / ~4–5 GB NF4), `cup_to_disc_ratio` ordinal 0–4 como grado de severidad para H4, bootstrap BCa, permutación para Spearman (n=69), reformulación McDermott (AUROC/AUPRC), números corregidos de "Between the Layers" (AUPRC ~0.82 bajo 4-bit) y VIG-TUQ (no single-pass en su score JSD), licencias ODIR-5K/HAI-DEF, advertencia PII `doctor_name`. — **Versión 2.2 (23-jul-2026): batería val_01–val_07 completamente en verde** (cero supuestos sin verificar): Z de Mann-Whitney manual (sin `zstatistic`), KL/JSD con `log_softmax` en float64 (massive activations), VRAM medido 5.6–7.5 GB en 4-bit, 4.3 s/muestra (~20 min para 258 corridas), reproducibilidad bit-exacta. Documento hermano nuevo: `Guia_Conceptual_y_Algoritmo_BIP2026.md` (explicación pedagógica paso a paso). — **Versión 2.3 (26-jul-2026): baselines actualizados tras análisis crítico** — nota de equivalencia exacta MSP≡entropía en tarea binaria (sanity check teórico del pipeline), nuevo baseline **Verbalized Confidence** (2×, P5) que cierra la escalera de costo 1×/2×/10×, y lista ampliada de baselines eliminados con justificación (Deep Ensembles, Semantic Entropy como forma degenerada de Self-Consistency, Conformal Prediction como wrapper ortogonal). — **Versión 2.4 (03-ago-2026): protocolo de calibración estilo FUSE §5.2** — nueva §6.7: TPR a FPR fijos (5%/10%/20%), Platt scaling ajustado solo en train, ECE con bins equiprobables, correlaciones de calibración (Pearson/Spearman) y Brier con IC bootstrap; Fig 10 (reliability diagram) y Tabla T5 nuevas; sanity checks sintéticos en `validacion/val_09_calibracion.py`.
**Propósito de este documento:** definir, con el máximo detalle posible, el experimento mínimo necesario para demostrar nuestro punto en BIP 2026. Está escrito asumiendo que el lector no sabe nada del tema: cada concepto se explica desde cero y cada decisión se justifica ("por qué así y no de otra forma").
**Filosofía:** BIP no es una conferencia de altísima exigencia. Este experimento es un **primer contacto** para validar que la idea funciona. Todo lo que sea profundización queda explícitamente declarado como *future work* (Sección 9).

---

# PARTE 0 — ¿Qué queremos demostrar?

En una frase, para un humano que no sabe de IA:

> Los modelos de IA que "ven" una imagen médica y "responden" preguntas sobre ella a veces se equivocan con total confianza — lo cual es peligroso en medicina. Nosotros proponemos una forma barata y rápida de detectar cuándo el modelo probablemente está por equivocarse: **medir qué tan en desacuerdo están, dentro del modelo, la parte que procesa la imagen y la parte que produce el lenguaje**. Si ambas partes "no se ponen de acuerdo", el caso es sospechoso y debería revisarlo un humano.

El experimento mínimo debe demostrar **una sola afirmación central**:

> **Afirmación central (H1):** cuando MedGemma se equivoca al detectar glaucoma en una foto de fondo de ojo, el desacuerdo interno entre su representación visual y su representación textual (medido con divergencia KL) es, en promedio, significativamente mayor que cuando acierta — lo suficiente como para usar ese desacuerdo como alarma automática de "este caso necesita revisión humana".

Todo lo demás (baselines, ablaciones, métricas extra) existe únicamente para que esa afirmación sea creíble y esté bien medida. Si un elemento del diseño no contribuye a probar H1 de forma creíble, se corta.

---

# PARTE 1 — Conceptos base (explicados desde cero)

## 1.1 ¿Qué es un VLM y cómo "piensa" MedGemma?

Un **VLM (Vision-Language Model)** es una red neuronal que recibe como entrada **una imagen + un texto** (por ejemplo, una foto de retina y la pregunta "¿esto muestra glaucoma?") y produce como salida **texto** (por ejemplo, "sí" o "no").

MedGemma 4B tiene dos grandes componentes:

1. **El codificador visual (MedSigLIP).** Toma la imagen (que redimensionamos a 896×896 píxeles), la corta en parches de 14×14 píxeles (como un rompecabezas de 64×64 = 4,096 parches), y transforma cada parche en un vector de números. Luego, por una capa de *pooling*, esos 4,096 vectores se resumen en **256 "tokens visuales"** (vectores de 2,560 números cada uno). Estos 256 vectores son "lo que el modelo vio", en formato matemático.

2. **El decodificador de lenguaje (un transformer tipo Gemma 3, de 34 capas).** Recibe la secuencia completa: los tokens de texto de la pregunta + los 256 tokens visuales, todos en el mismo "idioma matemático" (vectores de 2,560 números). Los procesa capa por capa (cada capa mezcla información entre tokens mediante *atención*) y al final produce la respuesta, token por token.

**¿Qué es un token?** La unidad mínima que procesa el modelo. En texto, un token es aproximadamente un pedazo de palabra (la frase "Does this image show glaucoma?" son ~7 tokens). En imagen, cada token es uno de los 256 resúmenes de regiones de la foto.

**¿Qué es un hidden state (estado oculto)?** Después de pasar por cada capa del decodificador, cada token tiene un vector actualizado de 2,560 números: eso es su *hidden state* en esa capa. Es "lo que el modelo está pensando" sobre ese token en ese punto de procesamiento. Los hidden states son el sismógrafo interno del modelo: no son la respuesta final, pero contienen información sobre cómo se está formando la respuesta.

**Punto clave para nuestra idea:** como los tokens de imagen y los tokens de texto terminan viviendo en el mismo espacio de 2,560 dimensiones dentro del decodificador, **podemos compararlos matemáticamente entre sí**. Eso es lo que hace posible medir el "desacuerdo" entre modalidades sin entrenar nada.

## 1.2 ¿Qué es "incertidumbre" y por qué queremos medirla?

En screening de glaucoma (detectar la enfermedad en población general antes de que cause ceguera), el modelo **va a equivocarse a veces**. Eso es inevitable. Lo que sí podemos evitar es que se equivoque *sin avisar*:

- **Error con aviso (aceptable):** el modelo dice "no estoy seguro" → el caso se envía a un oftalmólogo.
- **Error sin aviso (peligroso):** el modelo dice "sano" con total confianza, pero el paciente tiene glaucoma → no se deriva, el paciente pierde visión.

La **cuantificación de incertidumbre (UQ)** es el campo que construye señales numéricas `u(x)` tales que: **u(x) alto ⟺ es probable que el modelo se equivoque en x**. Una buena señal de UQ permite *selective classification*: el sistema responde automáticamente los casos seguros y abstiene (deriva al humano) los casos inciertos.

**¿Por qué hace falta una señal NUEVA?** Porque las existentes tienen problemas prácticos:
- **MC-Dropout / Ensembles:** hay que correr el modelo 5–100 veces por imagen (o entrenar 5 modelos). Caro, lento, inviable en hospitales con pocos recursos.
- **Entropía del texto de salida:** barata, pero solo mira *las palabras* de la respuesta. Falla exactamente en el peor caso: cuando el modelo "alucina" — responde con fluidez y confianza ignorando lo que muestra la imagen (fenómeno documentado como *language prior bias*: el modelo dice lo que "suena médicamente plausible" según su entrenamiento de texto, no lo que ve).

Nuestra apuesta: el desacuerdo entre "lo que el modelo vio" y "lo que el modelo dice" debería encenderse precisamente en esos casos de alucinación/conflicto, donde la entropía textual está ciega.

## 1.3 Softmax: cómo convertir números arbitrarios en una "distribución"

Un hidden state es una lista de 2,560 números cualesquiera (pueden ser negativos, enormes, lo que sea). Para comparar dos vectores como si fueran "opiniones probabilísticas", primero los convertimos en **distribuciones de probabilidad**: listas de números no negativos que suman exactamente 1.

La función **softmax** hace eso:

```
p_i = exp(z_i / τ) / Σ_j exp(z_j / τ)
```

**Ejemplo numérico (τ=1):** vector `[2, 1, 0.1]` →
`exp(2)=7.39, exp(1)=2.72, exp(0.1)=1.11` → suma=11.21 →
softmax = `[0.66, 0.24, 0.10]`. Ya suma 1.

**¿Qué hace la temperatura τ?**
- **τ pequeña (<1):** exagera las diferencias → la distribución se vuelve "tajante" (un ganador casi con todo el peso).
- **τ grande (>1):** aplana las diferencias → la distribución se acerca a uniforme.

Ejemplo con `[2, 1, 0.1]`: con τ=0.5 → `[0.85, 0.13, 0.02]` (más tajante); con τ=3 → `[0.47, 0.33, 0.20]` (más plana).

**¿Por qué necesitamos temperatura aquí?** Porque los valores crudos de los hidden states tienen una escala arbitraria que depende del modelo. Si aplicamos softmax con τ=1 y la distribución sale extremadamente tajante (un solo número ~1 y el resto ~0), todas las distribuciones se parecen entre sí y la KL deja de ser informativa. La temperatura nos da una perilla para encontrar el régimen donde las diferencias entre muestras se ven mejor. No sabemos a priori qué τ es mejor → la tratamos como ablación pequeña (τ ∈ {1, 2, 4}), elegida en el split de entrenamiento, no en test.

**Nota de implementación numérica (confirmada empíricamente 23-jul-2026, val_07):** Gemma presenta *massive activations* — en la capa 34 algunos componentes del hidden state llegan a magnitudes de 10²–10³. Con τ=1, `softmax` en float32 colapsa 2559 de 2560 componentes a **exactamente 0** y `log(0) = -inf` → la KL sale `inf`. Por eso la KL **nunca** se computa como `softmax().log()`, sino con `F.log_softmax` en **float64** (resta el máximo antes de exponenciar → log-probabilidades finitas). Consecuencia científica ya anticipada arriba: a τ=1 las distribuciones son casi delta (KL del orden de cientos de nats); esperamos que la señal informativa viva en τ>1 o en capas medias — una razón más para que τ y capa se elijan *solo en train*.

**Advertencia de honestidad científica (importante para la redacción):** aplicar softmax sobre la *dimensión de features* no produce una distribución "sobre clases" ni "sobre conceptos" — es una **heurística** heredada del paper que nos inspira ("Between the Layers Lies the Truth", que hace exactamente esto pero entre capas de un LLM). Nosotros la adoptamos, la declaramos como heurística, y la defendemos empíricamente: la pregunta del experimento no es "¿es esto una probabilidad teológicamente pura?" sino "¿este número separa los errores de los aciertos?". Además incluimos alternativas (JSD, coseno) como ablación.

## 1.4 Divergencia KL: medir qué tan distintas son dos distribuciones

Dadas dos distribuciones `p` (referencia) y `q` (aproximación), la **divergencia de Kullback-Leibler** es:

```
KL(p ‖ q) = Σ_i p_i · log(p_i / q_i)
```

**Propiedades (cada una importa para el diseño):**

1. **Siempre ≥ 0.** Cero solo si p y q son idénticas. Interpretación: "cuánta información se pierde si uso q para representar p".
2. **Asimétrica:** KL(p‖q) ≠ KL(q‖p). En general.
3. **Sin cota superior:** puede ser enorme si q asigna casi 0 a algo a lo que p le da peso (por eso sumamos un ε=10⁻¹⁰ a q para evitar log(∞) — estabilidad numérica).

**Ejemplo numérico con 4 categorías:**
- `p = [0.7, 0.2, 0.08, 0.02]`, `q = [0.65, 0.25, 0.07, 0.03]` (muy parecidas) → KL ≈ 0.006 (casi cero).
- `p = [0.7, 0.2, 0.08, 0.02]`, `q = [0.25, 0.25, 0.25, 0.25]` (uniforme) → KL ≈ 0.46 (mucho mayor).

**¿Por qué KL y no otra distancia?**

| Alternativa | Por qué no como medida principal |
|---|---|
| **Distancia euclidiana** entre los vectores crudos | Sensible a la escala absoluta de los activaciones (que varía por capa y por modelo); no compara *forma* de distribución. |
| **Similitud coseno** | Mide solo el ángulo entre dos vectores: ignora cómo está repartida la "masa" entre dimensiones. Dos vectores pueden tener ángulo similar pero estructuras internas muy distintas. |
| **KL** | Compara la *forma completa* de dos distribuciones; es la medida estándar de "desacuerdo entre distribuciones" en teoría de información; es la que usa el paper que nos inspira (comparabilidad directa); y su **asimetría es explotable** (ver abajo). |
| **JSD (Jensen-Shannon)** | Variante simétrica y acotada de KL. Muy buena opción, la incluimos como ablación, pero perdemos la interpretación direccional. **Nota de implementación (verificada):** `scipy.spatial.distance.jensenshannon` devuelve la *distancia* (√JSD), no la divergencia — hay que elevarla al cuadrado (cota ln 2 ≈ 0.693); `val_05` lo comprueba numéricamente. |

**La asimetría como ventaja diagnóstica (idea diferenciadora del paper):**
- `KL(p_vis ‖ p_text)` alto: la imagen "cree" cosas que el texto no respalda → el lenguaje está ignorando la evidencia visual (*visual under-utilization*).
- `KL(p_text ‖ p_vis)` alto: el texto "afirma" cosas que la imagen no muestra → posible alucinación (*textual over-reach*).

Por eso computamos **ambas direcciones** (más JSD) en la misma corrida: no cuesta nada extra y le da al paper una dimensión explicativa que los baselines no tienen.

## 1.5 ¿Qué significa "training-free, single-pass" y por qué es el gancho del paper?

- **Training-free:** no entrenamos, no ajustamos, no fine-tuneamos nada. Usamos MedGemma exactamente como se descarga. Por qué importa: (a) es lo que hace el método usable en un hospital sin equipo de ML; (b) es lo que lo hace reproducible; (c) es lo que lo hace rápido de implementar en 2 semanas.
- **Single-pass:** una sola pasada hacia adelante por imagen (más la generación de 1 token de respuesta). Por qué importa: los métodos competidores (MC-Dropout, Semantic Entropy) necesitan 10–100 pasadas. En el paper mostramos una tabla de costo: nuestro método ~1×, los otros 10–100×. Ese contraste es el argumento de impacto para el *Global South*.

## 1.6 ¿Qué es "correctitud" y cómo se mide el error del modelo?

Para evaluar si nuestra alarma funciona, necesitamos saber en qué casos el modelo se equivocó:

1. Le preguntamos al modelo: "¿esta imagen muestra glaucoma? sí/no".
2. Obtenemos su respuesta como **probabilidad**: P(sí) y P(no) leyendo directamente los *logits* del primer token generado (ver §3.6).
3. La predicción es "sí" si P(sí) > 0.5.
4. Comparamos contra la etiqueta real del dataset MM-ODIR-129 (`Pathological` = glaucoma / `Normal`), asignada por oftalmólogos a cada imagen → cada imagen queda marcada como **correcta** o **incorrecta**.

La variable que queremos predecir con nuestra señal KL **no es** "¿tiene glaucoma?" sino **"¿el modelo se equivocó?"**. Son preguntas distintas y confundirlas es un error común. Nuestra unidad de análisis es: `u(x) = KL del caso x` como predictor de `error(x) ∈ {0,1}`.

---

# PARTE 2 — Hipótesis formales

**H1 (principal):** la distribución de u(x) = KL(p_vis ‖ p_text) en casos *incorrectos* es estocásticamente mayor que en casos *correctos* — medido por (a) Mann-Whitney U con p < 0.05 y effect size reportado, y (b) AUROC de u(x) como detector de errores, con intervalo de confianza bootstrap del 95% reportado explícitamente y valor puntual objetivo ≥ 0.65. (Ver §6.6 sobre el poder estadístico con N=129 y el lenguaje honesto de "evidencia fuerte" vs. "evidencia sugestiva" según si el IC excluye 0.5.)

**H2 (secundaria, comparativa):** u(x) obtiene AUROC de detección de errores **al menos comparable** (diferencia no significativa por DeLong, exploratorio con este N) a los baselines baratos (entropía, MSP, energy) a costo computacional 1×, y aporta señal complementaria (correlación imperfecta con ellos).

**H3 (secundaria, operativa):** al abstener el 10% de casos con mayor u(x), la accuracy del sistema en el 90% restante mejora de forma visible (curva accuracy-coverage creciente y por encima de la de los baselines baratos).

**H4 (exploratoria, añadida en v2 — correlación incertidumbre–severidad):** en los 69 casos patológicos, u(x) correlaciona positivamente (Spearman) con la severidad estructural de la enfermedad medida por el grading ordinal de CDR (0–4) asignado por el oftalmólogo. Justificación: si la señal crece con la severidad/ambigüedad estructural, es evidencia independiente de que captura algo clínicamente real — y funciona como resultado aunque la accuracy del modelo sea alta (mitiga el riesgo de "pocos errores", §8.3).

**H0 (nula):** u(x) no separa errores de aciertos mejor que el azar (AUROC ≈ 0.5). Si H0 no se rechaza, activamos el plan de contingencia (Sección 8.3).

**¿Por qué el objetivo es AUROC ≥ 0.65 y no 0.95?** Porque detectar errores de un modelo con una señal barata es un problema genuinamente difícil: en la literatura de UQ, señales con AUROC 0.65–0.75 ya se consideran útiles para triage clínico, y el paper que nos inspira reporta AUPRC ~0.82 **bajo cuantización 4-bit** — in-distribution obtiene paridad o −1.4 a −1.8 pp vs. probing — con un método *con entrenamiento* (LightGBM supervisado; Badash, Belinkov & Freiman, arXiv:2603.22299). Nuestro método es training-free: aspirar a 0.65–0.75 es realista y suficiente para el argumento clínico (derivar el 10% más incierto ya mejora el sistema).

---

# PARTE 3 — Decisiones de diseño (cada una con su "por qué")

## 3.1 ¿Por qué MedGemma 4B y no otro modelo?

| Criterio | MedGemma 4B | Alternativas descartadas |
|---|---|---|
| **Es un VLM médico real** | Pre-entrenado específicamente con imágenes médicas (incluye retina) sobre Gemma 3 | CLIP/LLaVA genéricos: no representan el escenario clínico real que motiva el paper |
| **Pesos abiertos y descargables** | HuggingFace (`google/medgemma-4b-it`), solo requiere aceptar licencia HAI-DEF | Modelos cerrados (GPT-4V, Gemini): no dan acceso a hidden states → nuestra técnica sería imposible |
| **Tamaño manejable** | ~4B parámetros: pesos ≈ 8.6 GB → **medido en val_07 (23-jul-2026): 5.6–7.5 GB VRAM pico en 4-bit NF4**; estimado ~10–12 GB en bfloat16 | MedGemma 27B: no mejora en glaucoma según VOLMO y exige más GPU; además el argumento "Global South" pide modelos pequeños. En Colab T4 (16 GB) usar 4-bit; bf16 cómodo en L4/A100 |
| **Acceso a hidden states** | Arquitectura Gemma 3 en `transformers` permite extraerlos | Modelos tipo CLIP puro (EyeCLIP, RetiZero): no tienen decodificador generativo — la KL imagen↔respuesta no existiría |
| **Experiencia previa del equipo** | Ya trabajado en GECCO 2026 | Empezar con un modelo nuevo gastaría días en setup |

**Dato que juega a nuestro favor:** MedGemma es *medianamente malo* en glaucoma (F1 ≈ 63.7% según VOLMO). Para un paper de clasificación sería un problema; para un paper de UQ es ideal: genera muchos errores con los que evaluar la alarma. Hay que reportar su accuracy base con total honestidad y presentarla como motivación: "el modelo se equivoca con frecuencia → hace falta una alarma".

## 3.2 ¿Por qué MM-ODIR-129 y no ODIR-5K completo? (reescrito en v2)

**El dataset es `TheBug95/MM-ODIR-129` (Hugging Face):** 129 fotos de fondo de ojo **completas** (resolución variable; verificado visualmente — no son recortes al disco), provenientes de ODIR-5K pero **re-anotadas imagen por imagen por oftalmólogos de Costa Rica**. 60 Normal / 69 Pathological (todos los patológicos son glaucoma). Cada imagen incluye: label binario, **transcripción clínica escrita por el oftalmólogo**, y en patológicos, **graduación ordinal de 7 signos glaucomatosos** (CDR, rima neuroretinal, hemorragia de Drance, atrofia peripapilar, defecto RNFL, palidez, cambios vasculares). Las 69 patológicas incluyen máscaras de copa y disco (para future work). Splits oficiales incluidos: train 77 / validation 26 / test 26. **Nota de licencias (verificada 22-jul-2026):** el repo declara licencia MIT, pero el dataset base ODIR-5K **no publica licencia formal** → en el paper citar Li et al. 2021 (arXiv:2102.07978, DOI 10.1007/978-3-030-71058-3_11) + URL del challenge, sin afirmar licencia permisiva. **Nota de privacidad:** `split.json` expone `doctor_name` (PII del anotador) — nunca redistribuirlo en artefactos, tablas ni repositorios.

**Razones de la elección:**

1. **Etiquetas de calidad experta, por imagen.** El problema #1 de ODIR-5K (label noise por etiquetado a nivel paciente) desaparece. Para un paper cuya variable de referencia es "¿el modelo se equivocó?", un ground truth limpio vale más que miles de imágenes ruidosas.
2. **Negativos difíciles deliberados:** hay `Normal` con CDR 0.6–0.7, atrofia peripapilar y fondo miópico — casos frontera reales donde vive la incertidumbre clínica. Ideal para el análisis de cuadrantes y el discurso de ambigüedad intrínseca.
3. **Transcripciones y gradings** habilitan análisis cualitativo con referencia experta (Fig 5) y la hipótesis H4 (correlación con severidad) — imposibles con ODIR-5K puro.
4. **Fundus completo = escenario realista:** el modelo debe localizar el nervio óptico dentro del campo completo, como en screening real.
5. **Cero fricción operativa:** descarga directa desde Hugging Face, splits listos, licencia MIT.
6. **Costo aceptado (y declarado):** N=129 limita el poder estadístico → protocolo estadístico adaptado (§6.6) y framing de proof-of-concept.

**Descartados para el mínimo (future work):** ODIR-5K completo (~7,000 imágenes, para escalar la validación), REFUGE (validación externa, solo ~40 positivos en test), RIM-ONE (159 imágenes).

**⚠️ Doble ciego:** el dataset está en la cuenta personal de Hugging Face del autor. Citarlo directamente en un PDF double-blind rompe el anonimato. Decisión pendiente antes de redactar: citación anonimizada ("link withheld for double-blind review"), espejo en anonymous.4open.science, o declaración a los chairs.

**⚠️ Artefactos de anotación:** al menos una imagen patológica contiene una flecha negra dibujada sobre los píxeles (hallazgo de verificación visual). Si las marcas se concentran en patológicas, son un atajo visual que el modelo podría explotar → auditoría obligatoria (Paso 1.5) y análisis de robustez.

## 3.3 ¿Por qué clasificación binaria de glaucoma y no multi-label?

1. **Es el foco de la tesis doctoral** (los 4 pilares giran alrededor de glaucoma).
2. **Simplicidad del experimento mínimo:** una sola pregunta, una sola respuesta sí/no, una sola noción de error. Multi-label multiplicaría las decisiones sin añadir nada a la prueba de H1.
3. **Glaucoma es la clase más dura:** minoritaria en screening real, sutil visualmente (excavación del disco óptico), y donde MedGemma rinde peor. Si la alarma funciona aquí, el caso es convincente.
4. En MM-ODIR-129 la tarea ya es binaria por diseño (`Normal` vs. `Pathological`=glaucoma), con la frontera clínica correcta: lo que separa las clases no es el CDR solo sino la rima neuroretinal/RNFL.

## 3.4 ¿Cómo se evalúa? (reescrito en v2: splits oficiales + evaluación completa)

**Principio rector:** como el modelo es frozen y la señal es training-free, **nada se entrena con el dataset** — no hay fuga de información posible: las etiquetas solo se usan para *evaluar*, nunca para *construir* la señal. Por tanto:

- **Evaluación principal: las 129 imágenes completas.** Con N=129 no podemos permitirnos reportar solo 26 (el IC95% de una AUROC sobre 26 imágenes sería ±0.20 — inaceptable). Usar todo el dataset es legítimo y necesario.
- **Split train (77 imágenes):** se usa para toda decisión de diseño ajustable: selección de la variante KL final (capa × dirección × τ × pooling), ajuste de T para Temperature Scaling, y elección de umbrales.
- **Splits validation + test (52 imágenes):** confirmación — se reportan las métricas principales también ahí para mostrar que la elección hecha en train se mantiene.
- **No hace falta construir splits a nivel paciente** (las etiquetas son por imagen, asignadas por el oftalmólogo). El `patient_id` embebido en el nombre de archivo (`{patient_id}_{eye}.jpg`) se registra en la tabla maestra y permite un chequeo de robustez por clustering si ambos ojos de un paciente aparecen (nota en el paper, sin análisis adicional en el alcance mínimo).

Esto simplifica enormemente el plan original: no hay derivación de etiquetas por keywords ni split 70/15/15 manual.

## 3.5 ¿Por qué KL y no otra métrica? (justificación completa de la señal)

En §1.4 definimos la KL como concepto; aquí va la **decisión de diseño completa**, porque es la pregunta más probable de un revisor: *"¿por qué divergencia KL y no otra cosa?"*

**Punto de partida: qué tipo de objeto matemático necesitamos.** Nuestra señal construye dos distribuciones sobre **el mismo soporte** (el vocabulario del modelo): `p_vis` (pooling de los 256 tokens visuales → cabeza LM → softmax con τ) y `p_text` (última posición del prefill → misma cabeza → softmax). "¿Cuánto discrepan las dos vías?" es entonces, *por construcción*, una pregunta de **divergencia entre distribuciones** — no de distancia entre vectores. La KL es la herramienta canónica para ese problema exacto.

**Por qué fallan las candidatas obvias:**

| Candidata | Por qué no |
|---|---|
| **Entropía de cada lado** | Mide incertidumbre *dentro* de una distribución, no **acuerdo entre dos**. Las dos vías pueden tener entropía bajísima (ambas seguras) y discrepar totalmente — exactamente el fallo que buscamos. La entropía es ciega al conflicto. |
| **Euclidiana / coseno sobre logits crudos** | Los logits no son probabilidades: su escala absoluta no significa nada. Y Gemma tiene *massive activations* (componentes gigantes) que secuestran cualquier norma. El coseno ignora magnitudes; la euclidiana queda dominada por ellas. |
| **L1 / L2 sobre las probabilidades** | Tratan toda la masa igual; L2 queda dominada por el top-1 y no distingue "casi acierto" de "disparate". Sin interpretación probabilística. |

**Por qué pierden las rivales serias:**

| Rival | Qué aporta | Por qué KL gana |
|---|---|---|
| **Total Variación (TV)** | Simétrica, acotada en [0,1], intuitiva | **Tosca**: si ambas distribuciones tienen su pico en tokens distintos, TV = 1 siempre — no distingue si el pico equivocado cayó en un token plausible o en uno absurdo. Insensible a *cuánto* se equivoca. |
| **Jensen–Shannon (JS)** | Simétrica, acotada (≤ ln 2 ≈ 0.693), finita siempre | La rival más fuerte (por eso entra como ablación, §1.4). Pero: (a) al estar acotada **satura** — no diferencia "muy mal" de "catastróficamente mal", y en UQ clínica esa cola es justo lo que importa; (b) pierde la **direccionalidad** (§1.4: la asimetría es nuestra ventaja diagnóstica); (c) la KL se descompone como cross-entropy − entropía, conectando con la loss de entrenamiento. |
| **Wasserstein (EMD)** | "Costo de transportar" una distribución a otra | Necesita una *métrica de fondo* entre elementos: ¿cuánto cuesta mover masa del token "yes" al token "artery"? El vocabulario es **categórico, no métrico** — no hay distancia natural entre tokens. Wasserstein es para espacios continuos. Además, cara de computar. |
| **χ²** | Σ (p−q)²/q | Hipersensible a probabilidades diminutas (denominador al cuadrado): con las colas exponenciales del softmax y las massive activations es un desastre numérico. Lo que en KL fue un bug visible y arreglable (float32 → `log_softmax` en float64, val_07), en χ² sería inmanejable. |
| **Hellinger / Bhattacharyya** | Afinidad acotada | Mismo argumento de saturación que JS, sin la conexión con cross-entropy ni la tradición en UQ. |

**Las razones positivas (lo que KL nos da y ninguna otra junta):**

1. **Cero con significado** (desigualdad de Gibbs): KL = 0 ⇔ acuerdo perfecto → hipótesis nula limpia.
2. **Unidades = nats**, las mismas de la cross-entropy con que el modelo fue entrenado: medimos desacuerdo en el mismo idioma en que el modelo aprendió.
3. **Sensibilidad a las colas**: si `p_text ≈ 0` donde `p_vis > 0`, la KL se dispara → es la firma matemática del **error confiado** (nuestro efecto McGurk artificial). Esa propiedad no es un bug: es el detector.
4. **Asimetría explotable**: dirección causal del modelo (la visión informa, el lenguaje decide) → las dos direcciones tienen lecturas clínicas distintas (§1.4).
5. **Encaje en el marco UQ**: predictive entropy, información mutua (BALD), entropía semántica — toda la literatura vive en la familia de Shannon (la información mutua *es* una KL esperada). Nos permite escribir: "señal de la misma familia teórica, pero en **una sola pasada**" (1× vs 10–100×).
6. **Costo extra = cero**: forma cerrada, sin muestreo, diferenciable — la base entera del claim *single-pass*.
7. **Score ordenante** para Mann-Whitney / AUROC / Spearman: solo necesitamos un escalar que ordene los 129 casos, y la KL lo da con orden bien definido.

**Ejemplo numérico (vocabulario de 3 tokens):**
- Acuerdo: `p_vis=[0.9, 0.05, 0.05]`, `p_text=[0.85, 0.1, 0.05]` → KL ≈ 0.004 nats.
- Conflicto confiado: misma `p_vis`, `p_text=[0.001, 0.9, 0.099]` → KL ≈ 6.2 nats.

La entropía de `p_text` es baja en ambos casos (no ve el problema); TV vale ~0.9 y no gradúa severidad; JS se satura en 0.693 aunque el descarte sea de 10⁻³⁰. La KL crece sin techo con la severidad del descarte (de 0.001 a 10⁻⁶ la lleva de ~6.2 a ~12.4): la severidad queda **cuantificada, no recortada**.

**Matiz de honestidad (anticipando al revisor):** la KL no es una distancia en sentido estricto (no es simétrica ni cumple la desigualdad triangular). No nos afecta: no necesitamos un espacio métrico, necesitamos un score que ordene casos — y para eso sobra.

**Coartada empírica:** no le pedimos al revisor que crea la teoría. La batería val_07 computa **18 variantes** (capa × τ × pooling × dirección) y JS entra como ablación (§1.4) — la elección "KL, esta dirección, esta τ" es **testeable y reportable**, no dogma. Y su único riesgo práctico (ceros exactos por massive activations → KL = inf) ya fue cazado y documentado en val_07.

---

# PARTE 4 — Protocolo experimental, paso a paso

**Paso 0 — Preparación (día 1).** Aceptar licencia MedGemma en HuggingFace (`huggingface-cli login` con `HF_TOKEN`); descargar `google/medgemma-4b-it` (**checkpoint v1.0.1**, que corrige el manejo del end-of-image token); cargar el dataset con `load_dataset("TheBug95/MM-ODIR-129")` — **verificado 22-jul-2026: funciona nativamente** (imagefolder + `metadata.jsonl` por split; el visor web roto es solo un fallo de conversión parquet del servidor). `snapshot_download` solo hace falta para máscaras y campos extra (`split.json`); fijar seeds (42); registrar versiones (`transformers>=4.51.3` — la estructura de `outputs.hidden_states` documentada en AGENTS.md §6.2 solo es válida desde esa versión — `torch`, Python, GPU); exportar `CUBLAS_WORKSPACE_CONFIG=:4096:8` y fijar TF32 explícito. **Estado 23-jul-2026: batería `Codigo/validacion/val_01`–`val_07` ejecutada y en verde — este paso ya está de facto validado; al re-correr en otra máquina, repetir la batería primero.** *Salida: entorno funcional.*

**Paso 1 — Tabla maestra (día 1).** De `annotations.json` + `split.json` construir un DataFrame con una fila por imagen:
`{image_filename, patient_id, eye, label (0=Normal, 1=Pathological), split, transcription, cdr_grade, neuroretinal_rim, disc_hemorrhage, peripapillary_atrophy, rnfl_defect, disc_pallor, vessel_changes, has_masks, has_annotation_artifact}`.

**Mapeo verificado (22-jul-2026):** `cdr_grade` = campo `cup_to_disc_ratio` del dataset — **ordinal entero 0–4** (no continuo), no nulo en 70 muestras (los 69 Pathological + 1 Normal). H4 corre Spearman sobre esos 69.
*Salida: `master_table.csv`.*

**Paso 1.5 — Auditoría de artefactos de anotación (día 1–2, NUEVO en v2).** Revisar las 129 imágenes (visualmente o con un detector simple de regiones negras pequeñas de alto contraste fuera del borde del fundus) para marcar flechas/marcas dibujadas sobre los píxeles. Registrar `has_annotation_artifact` y el conteo por clase. *Salida: flag en `master_table.csv` + conteo para el paper.*

**Paso 2 — Piloto de 20 imágenes (día 2).** Correr el pipeline completo en 20 imágenes (~10 de cada clase) verificando los **sanity checks** (Sección 10). **Primer número a mirar: la accuracy base de MedGemma** — si en el piloto es >85%, activar la vigilancia del riesgo "pocos errores" (§8.3). *Salida: CSV piloto + verificación de que la máscara de tokens de imagen marca las 256 posiciones correctas. Sin piloto aprobado no se pasa al Paso 3.*

**Paso 3 — Corrida completa (día 2–3).** Para cada imagen × cada prompt (P1, P4) — 258 inferencias, ~10–20 min de GPU:
```
imagen 896×896 (via AutoProcessor, sin preproceso manual) → prefill (output_hidden_states=True) →
  máscara image tokens (por image_token_id, nunca slicing fijo) → hidden states capas {17, 26, 34}
  → 1 paso de decode (greedy) → logits yes/no (de outputs.scores[0]) + hidden state del token de respuesta
→ computar y guardar en UNA fila del CSV:
  image_filename, patient_id, prompt_id, split,
  logit_yes, logit_no, P(yes), pred, label, correct,
  entropy_answer, msp_answer, energy_answer,        # baselines "gratis"
  KL_v‖t, KL_t‖v, JSD  × capas {17,26,34} × τ {1,2,4} × pooling {mean,max},
  KL usando prompt (ablación), tiempo_ms
```
*Salida: `results_full.csv` (la pieza central del experimento). Guardado incremental (append por fila) para reanudabilidad.*

**Paso 4 — Baselines derivados (día 3):** (a) **TS+Entropy — sin GPU:** ajustar T en train minimizando ECE, aplicar a todos. (b) **Verbalized Confidence (P5 — GPU ~5 min):** para cada imagen, un segundo turno tras la respuesta P1 preguntando la confianza 0–100 → columna `verbalized_conf` (parsing directo del número, no logits; solo sobre P1). (c) **Self-Consistency (GPU):** para 50 imágenes (subconjunto estratificado, por costo), 10 muestras a temperatura 0.7 → fracción de "sí" → entropía de la frecuencia como u(x). *Salida: columnas extra en el CSV (`T` de calibración, `verbalized_conf`, `sc_yes_fraction`).*

**Paso 5 — Estadística principal (día 3–4):** selección de la variante KL en train → congelar → evaluar en las 129 (principal) y en val+test (confirmación): Mann-Whitney U + effect size r = \|Z\|/√N — **Z calculado manualmente con corrección de continuidad y empates** (`mannwhitneyu` no expone `zstatistic`; fórmula verificada en val_06: reproduce el p-value de scipy con error < 1e-10); AUROC con bootstrap CI **BCa** (9.999 remuestreos; preferido sobre percentile con N pequeño — `scipy.stats.bootstrap(method='BCa')`); AUPRC con bootstrap CI; DeLong vs. mejor baseline (exploratorio — no está en sklearn: paquete `pauc` o R `pROC`); Spearman entre señales; **H4: Spearman entre u(x) y `cup_to_disc_ratio` en los 69 patológicos — con n=69 el p-value asintótico de `spearmanr` no es fiable (<500): la significación sale de `scipy.stats.permutation_test`, con Kendall tau-b como sensibilidad**; análisis de robustez excluyendo imágenes con `has_annotation_artifact`. *Salida: tabla de resultados principal.*

**Paso 6 — Análisis operativo y cualitativo (día 4–5):** curvas accuracy-coverage de todas las señales; matriz de cuadrantes 2×2 (correcto/incorrecto × KL alta/baja, corte en la mediana de KL); selección de 2–3 ejemplos por cuadrante **con su transcripción del oftalmólogo como referencia**. *Salida: Figuras 4 y 5.*

**Paso 7 — Figuras finales y tablas (día 5–6).** Sección 7 de este documento.

**Paso 8 — Redacción (días 6–12).** Estructura y reparto de páginas ya definidos (§7.3). Incluye la decisión de citación anonimizada del dataset (§3.2).

---

# PARTE 5 — Los 6 baselines: qué son y por qué están

Un baseline es un método alternativo con el que comparamos nuestra señal para responder: "¿lo nuestro aporta algo respecto a lo obvio?". Criterio de selección: cubrir la **escalera de costo** (1× → 2× → 10×) con métodos estándar en la literatura (el reviewer los reconoce), declarando el costo de cada uno — la comparación honesta es contra los de **igual costo**; los de mayor costo son la barra práctica de eficiencia. Con N=129 no hay poder estadístico para separar una docena de métodos (IC95% de una AUROC ≈ ±0.10–0.13): el set parsimonioso es una decisión estadística, no pereza.

| # | Baseline | Qué es, en simple | Familia | Costo | Por qué está |
|---|---|---|---|---|---|
| 1 | **MSP** (Maximum Softmax Probability) | La confianza del modelo = P(clase predicha). Incertidumbre = 1 − P. | Confianza de salida | 1× | El baseline más clásico (Hendrycks & Gimpel 2017). Si no le ganamos al método más tonto posible, no hay paper. |
| 2 | **Predictive Entropy** | Entropía de la distribución {P(sí), P(no)}: −Σ p·log p. Máxima cuando P(sí)=0.5. | Confianza de salida | 1× | Estándar en UQ. **Equivalencia exacta en binario (sanity check teórico):** en una distribución Bernoulli, H(p) es función monótona de max(p, 1−p) → MSP y entropía producen rankings IDÉNTICOS y por tanto la *misma* AUROC. No son "casi" equivalentes: son el mismo baseline con dos nombres. Se reportan ambos a propósito: si sus AUROCs difieren, hay un bug en el pipeline (validación interna gratis). |
| 3 | **Temperature Scaling + Entropy** | Igual que #2 pero los logits se dividen por T ajustada en train para calibrar | Calibración post-hoc | 1× | Responde al reviewer que dice "tu señal solo mide mala calibración". Es la defensa mínima de calibración (Guo et al. 2017). |
| 4 | **Energy Score** | E(x) = −log(exp(logit_yes)+exp(logit_no)). Menos sensible a sobre-confianza que MSP | OOD detection | 1× | Estándar en detección de anomalías (Liu et al. 2020); gratis con los logits guardados. No es rank-equivalente a MSP (depende de max(z) y del gap, no solo del gap). |
| 5 | **Verbalized Confidence** | Segundo turno tras la respuesta (prompt P5): el modelo reporta su propia confianza 0–100; u(x) = 1 − conf/100 | Confianza verbalizada (LLM-native) | 2× | El baseline estándar de la literatura LLM (Kadavath et al. 2022; Lin et al. 2022): "¿para qué medir nada si basta con preguntarle al modelo qué tan seguro está?". Es la omisión más atacable en review 2026 si falta; la literatura reporta que suele perder contra MSP por sobre-confianza — resultado cómodo para nosotros. Cierra la escalera de costo por el medio. |
| 6 | **Self-Consistency** | 10 respuestas muestreadas a T=0.7; incertidumbre = desacuerdo entre ellas (fracción de "sí" cerca de 0.5) | Muestreo multi-pass | 10× | Es la versión honesta de "Semantic Entropy" para una tarea binaria (SE con 2 clases semánticas degenera en esto). Representa a la familia multi-pass (10× costo) — si le ganamos o empatamos con 1×, el argumento de eficiencia queda cerrado. |

**Baselines eliminados explícitamente (y la razón, que irá en el paper o en respuesta a reviewers):**
- **MC-Dropout:** Gemma 3 no tiene dropout activo en inferencia; exigiría modificar el modelo → contradice nuestro propio framing *frozen/training-free*. Se menciona en Related Work con esta justificación.
- **Deep Ensembles / métodos evidential / Bayes-by-Backprop:** requieren múltiples copias del modelo (4B cada una) o re-entrenarlo → contradice el framing completo del paper (UQ barata para el Global South). Se posicionan en Related Work con la tabla de costos.
- **Semantic Entropy (Farquhar et al. 2024):** necesita respuestas de forma libre con diversidad semántica; en tarea binaria degenera matemáticamente en la entropía de la fracción de "sí" muestreado = nuestro baseline #6. Se declara así en el paper (convierte la omisión aparente en rigor).
- **Mahalanobis / SAPLMA (probes):** requieren *ajustar* algo con datos etiquetados (Gaussianas clase-condicional o un MLP sobre hidden states) → no son training-free; compararlos contra nosotros es injusto a su favor. Future work.
- **Conformal Prediction:** no es un competidor sino un *wrapper* ortogonal que da garantía de cobertura a cualquier score (incluido el nuestro); con N=129 el split de calibración queda raquítico. Future work (calibración/ECE).
- **Reader study con oftalmólogos:** inviable logísticamente en 2 semanas. Future work.

**Criterio de lectura de la tabla final:** no necesitamos ganarle a todo. El resultado ideal es: nuestra KL ≥ baselines de igual costo {1–4} en AUROC (la barra honesta), y comparable a Verbalized {5} (2×) y Self-Consistency {6} (10×) — la barra de eficiencia. Si KL queda por debajo pero su correlación con los baselines es baja, el discurso pasa a ser "señal complementaria" (se combinan y mejoran — una línea de análisis extra con la media de los z-scores de ambas señales, reportable sin entrenar nada).

---

# PARTE 6 — Métricas: qué es cada una y por qué se eligió

Recordemos qué se mide: tenemos una señal continua u(x) (la KL) y una verdad binaria por imagen (¿el modelo se equivocó?). Las métricas evalúan **qué tan bien u(x) ordena a los errores por encima de los aciertos**.

## 6.1 AUROC (métrica principal)

**Qué es, en simple:** toma un caso incorrecto al azar y un caso correcto al azar. AUROC es la probabilidad de que el incorrecto tenga u(x) mayor que el correcto.
- AUROC = 0.5 → la señal no sabe nada (moneda al aire).
- AUROC = 1.0 → todos los errores tienen KL mayor que todos los aciertos (alarma perfecta).
- AUROC ≥ 0.65 → alarma útil para triage (nuestro objetivo).

**Por qué es la principal:** (a) no depende de elegir un umbral de KL — evalúa el ordenamiento completo; (b) es el estándar de facto en UQ/OOD, el reviewer lo espera; (c) McDermott et al. (NeurIPS 2024, arXiv:2401.06091) mostraron que la supuesta ventaja de AUPRC sobre AUROC puede ser un artefacto in-distribution — por eso AUROC encabeza y **ambas se reportan con IC**, sin predicar superioridad de ninguna.

## 6.2 AUPRC (métrica secundaria obligatoria)

**Qué es:** el área bajo la curva Precision-Recall. Responde: "si activo la alarma en los casos de mayor KL, ¿qué fracción son errores de verdad (precision) y qué fracción de todos los errores logro atrapar (recall)?".

**Por qué además de AUROC:** con clases desbalanceadas, una AUROC puede verse inflada por aciertos fáciles en la mayoría; AUPRC es más estricta (Saito & Rehmsmeier 2015). En MM-ODIR-129 la clase *diagnóstica* está balanceada (53/47), pero la variable a predecir es "error del modelo", cuya prevalencia depende de la accuracy base — si el modelo acierta mucho, los errores serán minoría y AUPRC vuelve a ser la métrica estricta. Se reporta con su intervalo de confianza.

## 6.3 Sensitivity @ 80% Specificity (métrica clínica)

**Qué es:** fijamos el umbral de alarma de modo que el 80% de los *aciertos* queden por debajo (no se molesta al sistema en el 80% de los casos buenos), y medimos qué fracción de los *errores* queda por encima (atrapados).

**Por qué:** es el lenguaje en que piensa un clínico (sensibilidad/especificidad), ancla la métrica a un punto operativo concreto, y la literatura de screening de glaucoma exige sensibilidad alta aun a costa de especificidad. Una sola fila en la tabla, gran valor de credibilidad clínica.

## 6.4 Accuracy vs. Coverage (la métrica del argumento final)

**Qué es:** ordenamos los casos de mayor a menor u(x). Vamos "tapando" (absteniendo/derivando al humano) el 5%, 10%, 15%... más incierto y graficamos la accuracy del modelo en lo que queda.

**Por qué es la figura que vende el paper:** traduce la señal a una decisión de política clínica: *"derivando solo el 10% de los casos a un oftalmólogo, la accuracy automática sube de X% a Y%"*. Es la demostración directa de H3 y del valor práctico del método. Se dibuja para nuestra KL y para cada baseline en el mismo gráfico: la curva que sube más rápido gana.

## 6.5 Mann-Whitney U + effect size (el test de H1)

**Qué es:** un test estadístico que compara dos distribuciones (KL de correctos vs. KL de incorrectos) **sin asumir que son normales** — compara rangos, no medias.

**Por qué este y no el t-test:** las divergencias KL tienen distribución asimétrica con cola larga a la derecha (la mayoría de valores pequeños, algunos enormes) — el t-test asume normalidad y puede engañar. Protocolo: se corre Shapiro-Wilk; si ambas distribuciones fueran normales, t-test; si no (lo esperado), Mann-Whitney. Se reporta **siempre el effect size** (r = Z/√N, o Cohen's d si aplica): el p-value solo dice "hay diferencia"; el effect size dice si la diferencia *importa* (r: 0.1 pequeño, 0.3 mediano, 0.5 grande). Con N=129 el effect size es aún más importante que el p-value.

## 6.6 Bootstrap 95% CI, DeLong y la nota de poder con N=129 (actualizado en v2)

- **Bootstrap CI:** remuestrear los datos con reemplazo 9.999 veces, recalcular AUROC/AUPRC cada vez, y reportar el intervalo **BCa** (sesgo y asimetría corregidos; con N pequeño es preferible al de percentiles — `scipy.stats.bootstrap(method='BCa')`). Responde "¿qué tan estable es tu número?". **Con N=129 es LA evidencia principal del paper**, no un accesorio: regla práctica — con ~35–50 errores y ~80–95 aciertos, el IC95% de una AUROC ≈ 0.70 tendrá media anchura de ±0.10–0.13.
- **DeLong's test:** test estándar para saber si la diferencia entre dos AUROC sobre los *mismos datos* es significativa. Con N=129 tiene poder limitado → se reporta como **exploratorio** para la única comparación primaria: nuestra KL vs. el mejor baseline barato. α=0.05.
- **Spearman para H4:** correlación de rangos entre u(x) y el grading de CDR en los 69 patológicos (ρ≈0.33 detectable con 80% de poder). No asume linealidad ni normalidad — apropiado para gradings ordinales.
- **Lenguaje honesto de resultados:** si el IC95% de la AUROC excluye 0.5 → "evidencia fuerte"; si no, pero la estimación puntual es ≥0.65 con effect size mediano → "evidencia sugestiva" (aceptable para proof-of-concept en BIP, discutiéndolo con franqueza).
- **¿Por qué no correcciones por comparaciones múltiples?** Porque congelamos UNA comparación primaria (nuestra señal elegida en train vs. mejor baseline). El resto se declara exploratorio.

## 6.7 Calibración de la señal de incertidumbre (añadido en v2.4, 03-ago-2026; protocolo FUSE §5.2)

**Motivación:** una buena señal u(x) no solo debe *rankear* errores (discriminación) sino *cuantificar* el riesgo (calibración: P(correct | u⋆) ≈ 1 − u⋆). Se adopta el protocolo de evaluación de FUSE §5.2 (a su vez basado en Guo et al. 2017):

- **TPR a FPR fijos (discriminación operativa):** TPR de detección de errores interpolado a FPR ∈ {5%, 10%, 20%} (TPR@FPR=20% ≡ sensitivity@80%spec, ya congelada). Invariante monotónica → se computa sobre la señal cruda.
- **Normalización a [0,1] — Platt scaling:** la KL está en nats; para el test de calibración se ajusta una sigmoide 1-feature u → P(error) **SOLO con el split train** (respeta "selección/ajuste solo en train") y se aplica a las 129. Cada señal (ganadora, baselines, rank-combo) tiene su propio Platt. Cuando el análisis es sobre train se etiqueta como in-sample (optimista).
- **Binning (Guo et al. 2017):** 10 bins **equiprobables** (~13 obs/bin con N=129); por bin, u media calibrada vs. tasa de error empírica.
- **ECE (versión incertidumbre):** media ponderada |u_media − error_empírico| por bin. Depende del nº de bins → convención fija n_bins=10 y se reporta sensibilidad con n_bins=5.
- **Correlaciones de calibración:** Pearson y Spearman entre u media y error empírico por bin. **Secundarias:** Platt es monótona por construcción, así que correlaciones altas no prueban calibración por sí solas; la evidencia principal es ECE + reliability diagram (Fig 10).
- **Brier score:** u_calibrada vs. error binario (convención verificada en val_05).
- **IC bootstrap percentil 95%** para ECE y correlaciones (Platt fijo, remuestreo de observaciones; 1.999 remuestreos). Con ~13 obs/bin estas métricas son ruidosas → mismo lenguaje honesto de §6.6.
- **Artefactos nuevos:** Fig 10 (reliability diagram multi-señal) y Tabla T5 (discriminación + calibración lado a lado, espejo de la Table 1 de FUSE). Sanity checks sintéticos en `validacion/val_09_calibracion.py`.

---

# PARTE 7 — Figuras, tablas y estructura del paper

## 7.1 Las 5 figuras (qué muestra cada una y para qué existe)

| Fig | Contenido | Función en el argumento |
|---|---|---|
| **Fig 1** | Diagrama del pipeline: imagen+prompt → MedGemma → extracción de hidden states (máscara) → pooling → softmax → KL → alarma. Incluir el recuadro del "octante vacío" (single-pass × feature-based × cross-modal). | El reviewer entiende el método en 20 segundos y ve la novedad posicionada. |
| **Fig 2** | Boxplot + stripplot de u(x) en correctos vs. incorrectos (con p-value y effect size). | La evidencia visual directa de H1: las dos distribuciones se separan. |
| **Fig 3** | Curvas ROC y PR de nuestra señal y los 6 baselines (panel doble). | La comparación cuantitativa de un vistazo. |
| **Fig 4** | Curvas accuracy-coverage de todas las señales. | El argumento clínico: abstención selectiva mejora el sistema. |
| **Fig 5** | 4–8 imágenes ejemplo de los cuadrantes (acierto confiado / acierto dudoso / **error confiado** / error detectado) con la respuesta del modelo, el valor de KL y **la transcripción del oftalmólogo** como referencia. | Humaniza el método, muestra los modos de fallo con respaldo experto, y exhibe el cuadrante peligroso (error + KL baja). |

## 7.2 Las 3 tablas

| Tabla | Contenido |
|---|---|
| **T1** | Resultados principales: para cada señal (KL + 6 baselines): AUROC [CI], AUPRC [CI], Sens@80%Esp, costo relativo (1×/2×/10×). Incluye fila de accuracy base del modelo. |
| **T2** | Ablaciones: capa {17,26,34} × dirección {v‖t, t‖v, JSD} × τ {1,2,4} (mejor pooling) → AUROC. Más la fila imagen↔prompt vs. imagen↔respuesta. Más la fila de robustez sin artefactos de anotación. |
| **T3** | Comparativa de propiedades vs. métodos de la literatura (MC-Dropout, Sem. Entropy, UMPIRE — multi-sample, VIG-TUQ — single-pass solo en su score de atención; su JSD necesita 2º forward, SAPLMA — supervisado, **nuestro**): columnas = single-pass?, training-free?, cross-modal?, costo. — la tabla que cristaliza la novedad. |

## 7.3 Estructura y presupuesto de páginas (formato IEEE, 6–8 págs.)

| Sección | Páginas | Contenido clave |
|---|---|---|
| Abstract | 0.25 | Problema (errores silenciosos), idea (KL cross-modal), resultado (AUROC=X [IC], abstención 10% → +Y accuracy), eficiencia (1× costo). |
| I. Introduction | 1.0 | Glaucoma y ceguera evitable → VLMs médicos → errores con sobre-confianza → UQ cara → nuestra solución barata + framing bio-inspirado (conflicto multisensorial, cadena McGurk 1976 → Ernst & Bülthoff 2004 → Botvinick 2001 → Yeung 2004) + 3 contribuciones en bullets. |
| II. Related Work | 0.75 | UQ en imagen médica; UQ en VLMs (VIG-TUQ arXiv:2605.27136 — ojo: solo su score de atención es single-pass, su JSD requiere 2º forward sin imagen — UMPIRE arXiv:2602.24195 multi-sample, VLM-UQBench arXiv:2602.09214 — obligatorio citar y diferenciarse); baselines de probing barato Semantic Entropy Probes (arXiv:2406.15927) e INSIDE (ICLR 2024); "Between the Layers" (arXiv:2603.22299) como antecesor directo; Tabla T3. |
| III. Methods | 1.5 | Notación; arquitectura MedGemma (1 párrafo); extracción con máscara; definición formal de u(x) con fórmula; direcciones de KL como diagnóstico; bio-inspiración (½ columna con citas neuro). |
| IV. Experiments | 2.0 | Setup (dataset MM-ODIR-129 con citación anonimizada, protocolo de evaluación justificado, prompts, baselines); T1 + Fig 2–3; T2; Fig 4; Fig 5 + mini-análisis de cuadrantes. |
| V. Discussion & Limitations | 0.75 | Qué significa; **limitaciones honestas: N=129 (proof-of-concept, IC anchos), prevalencia curada ~53% ≠ screening real, resolución variable con resize a 896, posibles artefactos de anotación en píxeles, solo glaucoma, un solo modelo, anotadores de un solo grupo sin inter-anotador, accuracy base modesta, validación clínica pendiente**. |
| VI. Conclusion & Future Work | 0.25 | Recap + roadmap (todo lo de la Sección 9 de este documento). |
| References | 1.0 | ~25–30 refs. |

---

# PARTE 8 — Criterios de éxito, lectura de resultados y contingencia

## 8.1 Definición de éxito (decidida ANTES de correr, para no mover el arco)

- **Éxito pleno (evidencia fuerte):** AUROC(KL) ≥ 0.65 con IC95% bootstrap que excluye 0.5, KL ≥ baselines baratos (o DeLong no significativo a favor de ellos), curva accuracy-coverage claramente creciente, y H4 con ρ positivo significativo. → Paper con discurso "señal útil y barata".
- **Éxito parcial (publicable):** AUROC puntual ≥ 0.65 pero el IC incluye 0.5 (evidencia *sugestiva*), o KL < baselines pero correlación baja con ellos. → Discurso "proof-of-concept + señal complementaria": mostrar que combinar KL + entropía (media de z-scores, sin entrenar) supera a cada una sola. Sigue siendo una contribución honesta para BIP.
- **Vigilancia de pocos errores:** si la accuracy base de MedGemma > 85% (pocos errores para evaluar UQ), los resultados principales pasan a ser **calibración + H4 (correlación con severidad)** y el error-detection se reporta como secundario con sus ICs anchos declarados.
- **No-Go (H0 no rechazada):** AUROC ≈ 0.5 en la mejor variante de train. → Activar §8.3.

## 8.2 Punto Go/No-Go: **día 4**, al terminar la estadística principal. No se escribe una línea del paper antes de esto.

## 8.3 Plan de contingencia (si H0 no se rechaza)

1. **Rescate técnico (día 4–5, sin re-cómputo de GPU):** revisar sobre el mismo CSV las variantes en *train* (capas medias, JSD, τ alta, imagen↔prompt). La literatura de probing dice que las capas medias codifican mejor la verdad que la última — es plausible que la señal esté ahí.
2. **Rescate de framing (día 5+):** reformular como estudio de diagnóstico negativo: *"cross-modal hidden-state disagreement does not predict errors in frozen MedVLMs: an analysis"*, con cuadrantes, H4, y comparación con lo que sí funciona (self-consistency). Es publicable en BIP como contribución negativa bien documentada, aunque más débil.
3. **Prohibido bajo presión:** meter LightGBM sobre signature maps (el método completo del paper inspirador) — rompe el framing training-free y no cabe en el calendario. Eso es el siguiente paper.

---

# PARTE 9 — Qué queda EXPLÍCITAMENTE para Future Work (lista para la sección final del paper)

1. Escalar la validación a ODIR-5K completo (~7,000 imágenes) y validación externa en REFUGE y RIM-ONE.
2. Comparación contra probes supervisados (SAPLMA) y métodos multi-pass completos (Semantic Entropy con NLI, TTA).
3. Más prompts (descriptivo, biomarcadores CDR/VCDR desde segmentación clásica) y análisis de sensibilidad al prompt.
4. Extensión multi-label y a otros VLMs (LLaVA-Med, MedGemma-27B).
5. Reader study con oftalmólogos: correlación entre incertidumbre humana y KL; medición de acuerdo inter-anotador en el dataset.
6. Descomposición espacial de la KL (mapas de riesgo sobre la imagen) y segmentación copa/disco con las máscaras de MM-ODIR-129 (Pilar 4, MICCAI OMIA 2027).
7. Calibración de la KL como probabilidad de error (ECE) y umbrales clínicos formales.
8. Combinación adaptativa: KL como triage para activar métodos caros solo en casos inciertos.
9. *Report generation*: similitud entre descripciones generadas por MedGemma y las transcripciones expertas del dataset.
10. **Extensión de la técnica a modelos de segmentación (desacuerdo cross-depth, "layer-BALD"):** la misma señal de desacuerdo interno aplicada a segmentación copa/disco (conecta con la entrada 6, que provee los datos y las máscaras). La versión literal "imagen vs. máscara" no es viable (no comparten espacio para una KL); la formulación viable es comparar, por píxel, las distribuciones sobre clases leídas con **la misma cabeza de segmentación** a distintas profundidades del decoder — training-free en arquitecturas con *deep supervision* (nnU-Net DSV, UNet++). Con K > 2 salidas, el desacuerdo se formula como **Jensen–Shannon del conjunto** = información mutua entre la clase del píxel y la profundidad, análogo estructural a BALD/MC-Dropout pero "muestreando profundidad" en vez de parámetros: el decoder como *ensemble gratuito* en una sola pasada. Ventajas sobre el caso VLM: la señal es un **mapa espacial de desacuerdo** (UQ + XAI simultáneos; el escalar de triage = media o percentil 95 del mapa) y el error es **continuo** (1−Dice → Spearman directo, más poder estadístico que el split binario). Obstáculos documentados: compatibilidad dimensional de la cabeza (sin adaptadores entrenados, o se rompe el claim training-free), upsampling determinista, autocorrelación espacial (inferencia a nivel imagen), y revisión bibliográfica enfocada en UQ single-pass para segmentación antes de reclamar novedad. Es el mismo problema metodológico que la selección de capas óptimas en el VLM → narrativa de tesis coherente: "del token al píxel".

  **Nota técnica (24-jul-2026) — viabilidad en transformers (SAM y afines):** en transformers el obstáculo dimensional desaparece (residual stream de ancho constante → cualquier bloque se lee con la cabeza final, misma lógica del *logit lens* y de nuestra lectura de capas 17/26/34 en MedGemma). **SAM:** capturar bloques intermedios del image encoder (ViT) en una sola pasada → neck compartido (convoluciones ya incluidas, deterministas) → K corridas del mask decoder (~2% del cómputo cada una → costo total ≈ 1.05×); máscaras binarias → JS por píxel sobre distribuciones Bernoulli. Caveats de diseño: SAM es condicionado por prompt (la señal es u(x, prompt) → protocolo de prompts fijo) y *class-agnostic* (error = 1−Dice contra máscara de referencia; usar variantes médicas tipo MedSAM/SAM-Med2D o aceptar desempeño base — para evaluar UQ basta varianza en calidad). El **IoU predicho de SAM es el baseline natural a vencer** (cabeza de confianza entrenada): igualarlo o superarlo con una señal training-free es un resultado fuerte. **Testbed ideal: MaskFormer/Mask2Former** — entrenados con deep supervision nativa (cada capa del decoder ya produce predicción de máscaras) → el desacuerdo entre profundidades es literalmente gratis.

**Por qué conviene declarar todo esto:** (a) protege la novedad ("lo sabemos y está en camino"); (b) convierte las debilidades del alcance mínimo en roadmap; (c) es exactamente el pipeline de la tesis doctoral — el paper queda articulado con CIARP 2026 y MICCAI OMIA 2027.

---

# PARTE 10 — Sanity checks obligatorios (antes de confiar en cualquier número)

| # | Check | Resultado esperado | Si falla |
|---|---|---|---|
| 1 | `KL(p ‖ p)` con la misma distribución | Exactamente 0 | Bug en la fórmula o en la convención de `F.kl_div` (recordar: input=log q, target=p → computa KL(p‖q)) |
| 2 | Misma imagen dos veces, mismo prompt | u(x) idéntico (greedy, seed fija) | No-determinismo: revisar dtype/seed |
| 3 | Máscara de image tokens sobre 5 ejemplos | Marca exactamente 256 posiciones contiguas tras `<start_of_image>` | Bug de indexación (el bug crítico de la revisión) |
| 4 | P(sí) + P(no) | = 1.0 (salvo ε) | Lista de variantes de tokens yes/no incompleta |
| 5 | Imagen totalmente negra (corrupta) con P1 | KL visiblemente alta vs. imágenes normales | Si una corrupción extrema no mueve la señal, algo está mal en la extracción |
| 6 | Distribución de P(yes) sobre el dataset | No colapsada en 0 o 1 para todas las imágenes | Si el modelo dice siempre lo mismo, la tarea/prompt está mal planteada |
| 7 | Conteo en `master_table.csv` | **60 Normal / 69 Pathological = 129 total; train 77 / val 26 / test 26** | Error en la lectura de `annotations.json` / `split.json` |
| 8 | Flag `has_annotation_artifact` | Presente en la tabla maestra, con conteo por clase reportado | Auditoría de artefactos (Paso 1.5) pendiente — riesgo de cue espurio |

---

# Anexo — Glosario relámpago

- **Token:** unidad mínima de entrada del modelo (pedazo de palabra o parche de imagen).
- **Hidden state:** vector de 2,560 números que representa "lo que el modelo piensa" de un token tras una capa.
- **Prefill / decode:** prefill = procesar toda la entrada de una vez; decode = generar la respuesta token a token.
- **Logit:** puntuación cruda (sin normalizar) que el modelo asigna a cada palabra posible.
- **Softmax / temperatura:** convierte logits (o cualquier vector) en distribución; τ controla qué tan tajante es.
- **KL / JSD:** distancias entre distribuciones; KL es asimétrica, JSD simétrica y acotada.
- **UQ (uncertainty quantification):** técnicas para que el modelo "diga cuándo no sabe".
- **AUROC / AUPRC:** métricas de ordenamiento; probabilidad de que un error tenga más señal que un acierto / área precision-recall.
- **Selective classification / accuracy-coverage:** abstenerse en los casos inciertos y medir cómo mejora la accuracy en el resto.
- **Mann-Whitney U / DeLong / bootstrap / Spearman:** tests e intervalos para afirmar diferencias y correlaciones sin hacer trampa estadística.
- **Training-free / single-pass:** sin entrenar nada / una sola pasada por imagen — las dos propiedades de eficiencia que vendemos.
- **Ground truth:** la etiqueta real (diagnóstico del oftalmólogo en MM-ODIR-129) usada solo para evaluar.
- **Cue espurio:** pista visual accidental (p. ej. una flecha de anotación) que el modelo podría usar en lugar de la patología real.
- **MM-ODIR-129:** nuestro dataset — 129 fundus completos de ODIR-5K re-anotados por oftalmólogos con label, transcripción y 7 gradings de glaucoma.

---

*Este documento es la definición congelada del experimento mínimo (v2.2, dataset MM-ODIR-129, batería de validación en verde). Cualquier cambio posterior (nuevo baseline, nueva ablación, nuevo dataset) se anota como desviación justificada o se difiere a future work. Documentos hermanos: `Guia_Conceptual_y_Algoritmo_BIP2026.md` (explicación pedagógica), `AGENTS.md` (reglas operativas para el agente de código), `Plan_de_Validacion_BIP2026.md` (evidencia de verificación).*
