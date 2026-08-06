"""Glosario de conceptos del experimento BIP 2026 (español).

Estos textos se reutilizan en dos sitios:
1. Los paneles "¿Qué estoy viendo?" del dashboard.
2. El system prompt del asistente IA (app/assistant.py).

Mantener el lenguaje claro y divulgativo: el dashboard también es una pieza
de comunicación científica.
"""

RESUMEN_EXPERIMENTO = """\
**¿De qué va este experimento?**

Cuando un modelo de lenguaje con visión (MedGemma-4B) analiza una foto de fondo
de ojo para detectar glaucoma, queremos saber **cuándo no podemos confiar en su
respuesta** — sin reentrenarlo y sin pasadas extra de cómputo.

La idea central: el modelo "ve" la imagen (representación visual) y "lee" la
pregunta (representación textual). Si esas dos representaciones internas están
**en desacuerdo** (medido con divergencia KL entre sus hidden states), es más
probable que el modelo se esté equivocando. A esa señal la llamamos **u(x)**.

Con u(x) podemos hacer **triage clínico**: derivar al oftalmólogo los casos más
inciertos y aceptar automáticamente los más seguros, subiendo la accuracy del
sistema completo.

**Cifras del estudio:** 129 imágenes (60 normales / 69 con glaucoma),
258 inferencias greedy, modelo congelado (training-free), costo 1× (una sola
pasada por imagen). Señal ganadora: **KL texto→visión, capa 34, τ=1, pooling max**.
"""

CONCEPTOS: dict[str, str] = {
    "u(x)": """\
**u(x) — la señal de incertidumbre**

Es un número por imagen: cuanto más alto, menos confiable es la respuesta del
modelo para esa imagen. En nuestra variante ganadora,
u(x) = KL(p_texto ‖ p_visión): comparamos la distribución interna del modelo en
el token de respuesta contra la de los 256 tokens visuales (capa 34 del decoder,
softmax con temperatura τ=1, pooling por máximo). Se computa en float64 por las
*massive activations* de Gemma (en baja precisión la softmax colapsa).""",

    "kl": """\
**Divergencia KL (Kullback-Leibler)**

Mide qué tan distintas son dos distribuciones de probabilidad. KL(p‖q) = 0 si
son idénticas; crece cuando difieren. No es simétrica: KL(p‖q) ≠ KL(q‖p).
Aquí la aplicamos sobre los hidden states del decoder (2.560 dimensiones)
convertidos a distribuciones con softmax/τ. Computamos ambas direcciones
(texto→visión y visión→texto) más la JSD, y la ganadora se eligió solo con el
split de entrenamiento.""",

    "jsd": """\
**Divergencia Jensen-Shannon (JSD)**

Versión simétrica y acotada de la KL: JSD(p,q) = ½KL(p‖m) + ½KL(q‖m) con
m = (p+q)/2. Ojo técnico: `scipy.spatial.distance.jensenshannon` devuelve la
*distancia* (raíz cuadrada); nosotros usamos el cuadrado.""",

    "auroc": """\
**AUROC (área bajo la curva ROC)**

Probabilidad de que una imagen donde el modelo se equivoca tenga un u(x) mayor
que una donde acierta, escogidas al azar. 0.5 = azar (la señal no dice nada);
1.0 = separación perfecta. Es la métrica principal del estudio. Con N=129 el
intervalo de confianza bootstrap 95% es ancho (±0.10–0.13), así que lo
reportamos siempre junto al valor.""",

    "auprc": """\
**AUPRC (área bajo la curva precisión-recall)**

Como el AUROC pero sensible al desbalance de clases: aquí los "positivos" son
los errores del modelo (~20% de los casos), así que la AUPRC es más informativa
que el AUROC cuando importa capturar errores. El baseline al azar es la
prevalencia de errores (~0.20).""",

    "triage": """\
**Triage / derivación selectiva (accuracy-coverage)**

Regla operativa: ordenar las imágenes por u(x), derivar al oftalmólogo el X% más
incierto y aceptar la respuesta del modelo en el resto. La curva
accuracy-coverage muestra cómo sube la accuracy al bajar la cobertura.
**Regla dura del proyecto:** la derivación se define por *percentil de la
cohorte*, nunca por un umbral absoluto de nats (los valores absolutos de KL
dependen del epsilon numérico y del hardware).""",

    "calibracion": """\
**Calibración (ECE, diagrama de confiabilidad, Platt)**

Un modelo está calibrado si cuando dice "90% seguro" acierta ~90% de las veces.
El **ECE** (Expected Calibration Error) resume el desvío en bins equiprobables.
Aplicamos **Platt scaling** (regresión logística sobre la señal) ajustado SOLO
en train para no contaminar la evaluación. Nota honesta: las correlaciones
post-Platt son evidencia secundaria (Platt es monótona por construcción); la
evidencia principal es el ECE y el reliability diagram.""",

    "pooling": """\
**Pooling de tokens visuales**

Los 256 tokens de imagen se colapsan en un solo vector antes de la KL.
Variantes: `mean` (promedio), `max` (máximo por dimensión — la ganadora),
`topk`, `normw` (ponderado por norma), `attn` (ponderado por atención cruzada),
`rollout` (attention rollout), `headspec` (cabezas especializadas) y
`roi` (ponderado por la máscara del disco óptico — **oracle**: usa información
externa, no es deployable, y solo existen filas para las 69 patológicas, así que
su AUROC no es comparable con el resto). **Hallazgo contraintuitivo:** el roi
oracle obtiene AUROC 0.349 (n=69) — *peor* que el azar. El 0.889 del piloto de
20 imágenes era un artefacto de muestra pequeña y no se reprodujo. Hipótesis
documentada: la tensión informativa no vive solo en el disco; el contexto global
del fundus (que `max` sí ve) aporta a la señal.""",

    "tau": """\
**Temperatura τ**

Suaviza las distribuciones antes de la KL: softmax(logits/τ). τ=1 es la softmax
cruda (ganadora); τ=2 y τ=4 son ablaciones. Regla numérica dura: siempre en
float64 y sin normalización previa (z-score o norma L2 aplastan la señal).""",

    "baselines": """\
**Baselines de igual costo (1×)**

Señales de incertidumbre que también cuestan una sola pasada:
- **Entropía** de la distribución yes/no de la respuesta.
- **1 − MSP** (Maximum Softmax Probability): 1 menos la probabilidad del token
  elegido.
- **Energía**: −logsumexp de los logits yes/no.
- **rank(KL)+rank(1−MSP)**: combinación sin parámetros de nuestra señal con
  1−MSP por rangos. Es la mejor señal global (AUROC 0.698) sin costo extra.

Baselines más caros (fuera del framing 1×): self-consistency (10 muestras, 10×)
y verbalized confidence (preguntar al modelo qué tan seguro está, 2×).""",

    "p1_p4": """\
**Prompts P1 y P4**

- **P1 (principal):** "Does this fundus image show glaucoma? Answer yes or no."
- **P4 (contraste):** el mismo prompt con system prompt de experto
  ("You are an expert ophthalmologist."). El chat template de Gemma 3 pliega el
  system prompt dentro del primer turno de usuario (comportamiento esperado).
Sirven para comprobar si la señal es robusta al encuadre de la pregunta.""",

    "h4": """\
**Hipótesis H4: severidad del glaucoma — RECHAZADA**

H4 proponía que u(x) sería mayor en glaucomas más severos (más ambiguos
visualmente), medido con la correlación de Spearman entre u(x) y el grado
ordinal `cup_to_disc_ratio` (0–4, anotado por oftalmólogos) en los 69 casos
patológicos. **Resultado: ρ = +0.001, p = 0.99 → H4 rechazada.** La
interpretación es interesante y la reportamos con honestidad: la señal detecta
**errores del modelo**, no la gravedad de la enfermedad.""",

    "limites": """\
**Límites del estudio (lectura honesta)**

- **N=129** es pequeño: los IC bootstrap 95% del AUROC son anchos (±0.10–0.13).
  Hablamos de "evidencia sugestiva", no concluyente, salvo que el IC excluya 0.5.
- Dataset de una sola fuente (re-anotado de ODIR-5K); riesgo de artefactos de
  anotación (al menos una imagen tiene una flecha dibujada; existe el flag
  `has_annotation_artifact` y análisis de robustez excluyéndolas).
- La KL está winsorizada en ln(1/ε)=23.03 nats por diseño (ε=1e-10).
- Los valores absolutos de KL NO son comparables entre hardware/corridas: solo
  importan métricas de ranking (AUROC/AUPRC) y percentiles de cohorte.
- Herramienta de investigación, no un dispositivo médico.""",
}

# Descripciones cortas de cada mapa de pooling (tab «Mapas de pooling»)
MAPAS_POOLING: dict[str, str] = {
    "mean": "Media uniforme: los 256 tokens pesan igual (1/256). Es la referencia — no mira a ningún sitio.",
    "max": "Ganadora. Frecuencia con la que cada token es el máximo por dimensión: los puntos calientes son regiones donde el modelo concentra sus activaciones extremas.",
    "topk": "Sobreviven solo los 25 tokens de mayor norma L2 (blanco) y el resto se descarta. Muestra qué regiones tienen las activaciones más fuertes.",
    "normw": "Cada token pesa según su norma L2: versión suave de topk — las regiones muy activas dominan sin descartar el resto.",
    "attn": "Atención cruzada de la capa 34: cuánto 'mira' el token de respuesta a cada región de la imagen al decidir (media de todas las cabezas).",
    "rollout": "Attention rollout: propaga la atención por las 34 capas, capturando caminos indirectos respuesta → texto → imagen.",
    "headspec": "Solo las 4 cabezas de atención más 'visuales' de la capa 34: filtra las cabezas posicionales que añaden ruido uniforme.",
    "roi": "Oracle: la máscara real del disco óptico (anotada por oftalmólogos) promediada por celda. Solo existe para las 69 patológicas. AUROC 0.349 — restringir al disco *empeora* la señal.",
}

# Textos cortos para los paneles "¿Qué estoy viendo?" de cada gráfica
PANELES: dict[str, str] = {
    "roc_pr": """\
**¿Qué estoy viendo?** Dos formas de medir qué tan bien u(x) separa los errores
del modelo de los aciertos. **Izquierda (ROC):** cuanto más se arquea la curva
hacia la esquina superior izquierda, mejor; la diagonal punteada es el azar
(AUROC = 0.5). **Derecha (PR):** más útil con clases desbalanceadas (los errores
son ~20% de los casos); la línea punteada es el baseline al azar = prevalencia
de errores. Pasa el cursor sobre las curvas para ver los umbrales de u(x).""",

    "boxplot": """\
**¿Qué estoy viendo?** Distribución de u(x) en dos grupos: imágenes donde el
modelo **acertó** (verde) y donde **falló** (rojo). Si la señal funciona, la caja
roja queda más arriba (los errores tienen mayor incertidumbre). Cada punto es
una imagen; pasa el cursor para ver cuál. La prueba de Mann-Whitney evalúa si la
diferencia es significativa.""",

    "histograma": """\
**¿Qué estoy viendo?** Histograma de u(x) separado por la etiqueta real
(Normal vs. Glaucoma). No es la hipótesis principal (H1 es sobre errores, no
sobre clases), pero muestra si la incertidumbre se relaciona con la condición
clínica — relacionado con H4 (más severidad → más incertidumbre).""",

    "triage": """\
**¿Qué estoy viendo?** El simulador de derivación clínica. Mueve el slider:
el sistema deriva al oftalmólogo el % de casos con mayor u(x) y acepta la
respuesta del modelo en el resto. La curva muestra la **accuracy de los casos
retenidos** para cada nivel de cobertura; la línea punteada es la accuracy sin
triage (derivar al azar no mejora nada). Los contadores muestran cuántos errores
del modelo caen dentro de los casos derivados vs. cuántos capturaría una
derivación aleatoria del mismo tamaño.""",

    "cuadrantes": """\
**¿Qué estoy viendo?** Cada punto es una imagen: eje X = confianza del modelo
(p_yes), eje Y = nuestra señal u(x). Verde = acierto, rojo = error. Lo ideal:
los puntos rojos concentrados arriba (u(x) alto → derivable). La línea vertical
marca el umbral de derivación del slider. Los casos rojos arriba a la derecha
son los más peligrosos: el modelo dice "glaucoma" con confianza y se equivoca —
nuestra señal los detecta.""",

    "mapas": """\
**¿Qué estoy viendo?** El fundus está dividido en un grid de **16×16 = 256 tokens
visuales** (lo que el modelo "ve"). Cada panel muestra **cuánto pesa cada token**
en una de las 8 técnicas de pooling que agregan esos 256 vectores en uno solo
antes de calcular u(x). Cada mapa está normalizado por separado (escala propia):
compara *dónde* concentra el peso cada técnica, no los valores absolutos entre
paneles. La técnica ganadora es `max`; `roi` es el oracle con la máscara real
del disco (solo patológicas). Estos mapas se extrajeron con una pasada de GPU
adicional (`src/extract_pooling_maps.py`).""",

    "galeria": """\
**¿Qué estoy viendo?** Las 129 fotos de fondo de ojo del estudio. El borde indica
si MedGemma acertó (verde) o falló (rojo) la clasificación con el prompt P1.
Usa los filtros y haz click en cualquier imagen para ver su ficha: predicción,
u(x) y en qué percentil de la cohorte cae su incertidumbre. Las imágenes con
⚠️ tienen artefactos de anotación detectados en la auditoría.""",
}
