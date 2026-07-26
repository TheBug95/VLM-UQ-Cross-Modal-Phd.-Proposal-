# Guía Conceptual y Algoritmo del Experimento — BIP 2026

**Qué es este documento:** la explicación completa de lo que vamos a hacer, escrita
como si nunca hubieras visto nada de esto. Cada pieza pequeña (tokens, hidden states,
softmax, KL, AUROC, bootstrap…) viene con su intuición, un ejemplo numérico calculado
a mano, y —lo más importante— **por qué se eligió así y no de otra forma**.
Si algo del diseño cambia, este documento se actualiza.

**Estado al escribirlo (23-jul-2026):** batería de validación val_01–val_07 **completamente
en verde** — todo lo que aquí se afirma sobre el modelo, el dataset y las librerías está
verificado empíricamente (ver `Plan_de_Validacion_BIP2026.md`).

---

## 1. La idea en una frase

> **Cuando un modelo de visión-lenguaje "ve" una cosa en la imagen pero "dice" otra en
> su respuesta, ese desacuerdo interno se puede medir con un número — y ese número nos
> dice cuándo el modelo está a punto de equivocarse, sin necesidad de saber la respuesta
> correcta.**

La analogía que usaremos en el paper es la del cerebro humano: si tus ojos ven a una
persona decir "ga" pero tus oídos oyen "ba", tu cerebro detecta el **conflicto entre
sentidos** y se pone en alerta (efecto McGurk). Ese "algo no cuadra" es gratis — no
necesitas saber cuál es la verdad para sentir la contradicción. Nosotros construimos
el equivalente computacional: medimos el desacuerdo entre la **representación interna
de la imagen** y la **representación interna de la respuesta** del modelo, con una
fórmula matemática llamada **divergencia de Kullback-Leibler (KL)**.

**Para qué sirve:** en un hospital sin especialistas, el modelo responde "¿hay glaucoma?
sí/no". Si además escupiera un número que diga "en este caso estoy internamente
contradictorio, mejor que lo vea un humano", el sistema derivaría al especialista justo
los casos peligrosos. Eso es *cuantificación de incertidumbre* (UQ), y lo nuestro es
que la sacamos **de una sola pasada del modelo** (los métodos existentes necesitan
10–100 pasadas → 10–100× más caros → inviables donde hace falta).

---

## 2. Qué queremos demostrar exactamente

No queremos demostrar que el modelo diagnostica bien. Queremos demostrar que
**nuestro número (la KL) sabe distinguir cuándo se equivoca**.

- **H1 (hipótesis principal):** cuando el modelo se equivoca, su desacuerdo interno
  u(x) = KL(visión ‖ texto) es **mayor** que cuando acierta. Si eso es cierto con
  evidencia estadística razonable, la señal sirve para triage.
- **H4 (hipótesis secundaria):** en los casos con glaucoma, a **mayor severidad** de la
  enfermedad (grado del daño del nervio óptico, anotado por oftalmólogos), el modelo se
  pone más inseguro (o más seguro — medimos la correlación y la contamos sea cual sea
  el signo). Esto conecta la señal con algo clínicamente interpretable.

El experimento es deliberadamente **pequeño y honesto**: 129 imágenes, un modelo,
una pregunta. Es un primer contacto para validar que la señal *existe*; el trabajo
futuro escala y profundiza.

---

## 3. Los bloques de construcción (cada pieza micro, explicada)

### 3.1 ¿Cómo "lee" el modelo una imagen? — los 256 tokens visuales

Los modelos de lenguaje solo entienden **tokens**: pedazos discretos numerados. El texto
"Does this fundus image show glaucoma?" se trocea en ~10 tokens de palabras/subpalabras.
¿Y la imagen?

1. La imagen se redimensiona a **896×896 píxeles** (lo hace el procesador del modelo,
   no nosotros — verificado en val_02).
2. Un "ojito" convolucional la corta en parches de 14×14 píxeles:
   896/14 = **64 parches por lado → 64×64 = 4.096 parches**.
3. Cada parche pasa por la torre de visión (SigLIP) y sale como un vector de 1.152
   números: su "descripción numérica".
4. Luego un *pooling* 4×4 agrupa los parches de 16 en 16 (promedia cada bloque de
   4×4 parches vecinos): **4.096 / 16 = 256 vectores**.
5. Esos 256 vectores se proyectan a 2.560 números (el tamaño interno del modelo de
   lenguaje) y se insertan en la secuencia **como si fueran 256 palabras más**.

**Conclusión micro:** la imagen entra al modelo como **256 tokens visuales**, cada uno
un vector de 2.560 números que describe una región de la retina. En el texto del prompt
aparecen como 256 repeticiones del token especial `<image_soft_token>` (ID 262144 —
verificado). Eso es lo que nos permite **encontrarlos**: buscamos en la secuencia las
posiciones con ese ID y sabemos exactamente qué 256 vectores son "la imagen".

**¿Por qué nos importa contarlos?** Porque nuestra señal necesita *solo* la parte visual
de la "opinión interna" del modelo. Si tomáramos la secuencia completa (imagen +
pregunta) estaríamos mezclando lo que el modelo *ve* con lo que *lee* — y la medida de
desacuerdo quedaría contaminada. La regla dura: **máscara por ID de token, nunca
"las primeras 256 posiciones"** (con la plantilla de chat, el bloque de imagen queda en
medio de la secuencia; un slicing fijo era el bug #1 que anticipó la revisión).

### 3.2 ¿Qué es un hidden state? — la "opinión interna" de cada capa

El modelo de lenguaje es una pila de **34 capas** idénticas. La secuencia (imagen +
pregunta) entra, y cada capa va "re-escribiendo" el vector de cada posición mezclando
información de todas las demás (atención). Tras cada capa, cada posición tiene un vector
nuevo de 2.560 números: eso es un **hidden state**.

Intuición: es lo que el modelo "piensa" de esa posición *a esa altura de su razonamiento*.
Las capas bajas captan cosas superficiales (bordes, sintaxis); las medias, conceptos
("esto parece una copa óptica agrandada"); las altas, la decisión que se avecina.

**Dato clave verificado:** al pedirle a `generate()` los hidden states, devuelve una
tupla con **35 entradas**: la posición 0 son los *embeddings* de entrada (la materia
prima, aún sin razonar — no la usamos en la KL) y las posiciones 1–34 son las salidas
de las 34 capas.

### 3.3 El mapa: de dónde sale cada ingrediente

Aquí está el corazón del diseño. Necesitamos **tres cosas** de una sola ejecución:

| Ingrediente | Qué es | De dónde sale exactamente |
|---|---|---|
| **p_vis** | Resumen de "lo que el modelo vio" | Los 256 hidden states de los tokens de imagen (máscara por ID 262144), en una capa dada, resumidos en un solo vector (pooling) |
| **p_text** | "Lo que el modelo está a punto de decir" | El hidden state de la **última posición de la secuencia**, en la misma capa |
| **p_yes / p_no** | La respuesta concreta | Los logits (puntos) que el modelo asigna a los tokens "yes" y "no" en su primera palabra generada |

**¿Por qué la última posición es "lo que está a punto de decir"?** Los modelos de
lenguaje funcionan así: para producir la siguiente palabra, la última posición de la
secuencia se convierte en el "resumen de todo lo anterior" y de ese vector se calculan
los puntos de cada palabra candidata del vocabulario. Es decir, el hidden state de la
última posición **es literalmente el estado del que nace la respuesta**. Si el modelo
va a decir "yes", ese vector ya lo "sabe".

Y el detalle técnico que costó una corrección: con `max_new_tokens=1`, la librería solo
devuelve los hidden states del **prefill** (la pasada que procesa el prompt completo);
el estado que condiciona la respuesta es `hidden_states[0][capa][:, -1, :]` — última
posición del prefill. No existe un `hidden_states[1]` (eso sería con 2+ tokens
generados, y queda como ablación). Verificado en val_04/val_07.

**¿Por qué no usar el hidden state del prompt ("la pregunta")?** Porque la pregunta es
siempre la misma ("Does this fundus image show glaucoma?") — no varía entre aciertos y
errores. Lo que cambia entre un acierto y un error es lo que el modelo *va a decir* de
esta imagen concreta. Por eso la comparación honesta es **imagen vs. respuesta**, no
imagen vs. pregunta (esa comparación también la guardamos, como ablación).

### 3.4 Softmax y la temperatura τ — cómo se convierte un vector en "distribución"

La KL solo compara **distribuciones de probabilidad** (listas de números positivos que
suman 1). Nuestros vectores son 2.560 números arbitrarios (pueden ser negativos,
enormes, lo que sea). La máquina que los convierte en distribución es el **softmax**:

```
p_i = exp(z_i / τ) / Σ_j exp(z_j / τ)
```

En palabras: a cada número se le aplica la exponencial (lo vuelve positivo y exagera
las diferencias) y luego se divide entre la suma total (para que todo sume 1).

**Ejemplo a mano:** vector `[2, 1, 0.1]` con τ=1:
`exp(2)=7.39, exp(1)=2.72, exp(0.1)=1.11` → suma 11.21 → distribución `[0.66, 0.24, 0.10]`.
El más grande se lleva la mayor parte, pero no todo.

**La temperatura τ es una perilla de "nitidez":**

| τ | Cálculo sobre `[2, 1, 0.1]` | Resultado | Efecto |
|---|---|---|---|
| 0.5 | `[e⁴, e², e⁰·²] = [54.6, 7.4, 1.2]` → /63.2 | `[0.86, 0.12, 0.02]` | más **tajante** (el ganador arrasa) |
| 1 | (arriba) | `[0.66, 0.24, 0.10]` | normal |
| 2 | `[e¹, e⁰·⁵, e⁰·⁰⁵] = [2.72, 1.65, 1.05]` → /5.42 | `[0.50, 0.30, 0.19]` | más **plana** |
| 4 | | `[0.41, 0.32, 0.27]` | casi uniforme |

**¿Por qué la necesitamos aquí?** Porque los hidden states de Gemma tienen componentes
gigantescos (del orden de cientos, las llamadas *massive activations*). Con τ=1 el
softmax puede salir prácticamente delta: `[0.9999…, ~0, ~0, …]` — y si todas las
distribuciones son "un pico en alguna dimensión", la KL solo mide si los picos coinciden,
perdiendo toda la riqueza. τ>1 aplana la distribución y deja ver la estructura.
**No sabemos qué τ es la buena** → probamos τ ∈ {1, 2, 4}, y la elegimos *solo con el
split de entrenamiento* (nunca con test — eso sería hacer trampa estadística).

**Advertencia de honestidad (la repetimos siempre):** aplicar softmax sobre la dimensión
de *features* no produce una "probabilidad sobre clases" en ningún sentido teórico puro —
es una **heurística** (la misma del paper que nos inspira). La defendemos empíricamente:
la pregunta no es "¿es una probabilidad teológicamente correcta?" sino "¿este número
separa errores de aciertos?". Eso lo decide el experimento.

### 3.5 La divergencia KL — la fórmula estrella, explicada desde cero

**Intuición:** tienes dos distribuciones sobre las mismas categorías. La KL(p ‖ q) mide
**cuánto te sorprende q si el mundo de verdad funciona como p**. Si son idénticas,
sorpresa cero. Si q pone casi toda su masa donde p no pone nada, sorpresa enorme.

**Fórmula:**

```
KL(p ‖ q) = Σ_i p_i · ln(p_i / q_i)
```

Se lee: recorre cada categoría i; toma su probabilidad según p (eso pesa cuánto "importa"
esa categoría); multiplícala por el logaritmo del cociente entre lo que p dice y lo que
q dice. Si p y q coinciden en la categoría, el cociente es 1 y el logaritmo es 0 → no
aporta. Si p dice "mucha" y q dice "poca", el cociente es grande → aporta mucho.

**Ejemplo calculado a mano** (este exacto está programado en val_05 como autoverificación):

p = `[0.65, 0.35]` (lo que "ve"), q = `[0.50, 0.50]` (lo que "dice")

- Categoría 1: 0.65 · ln(0.65/0.50) = 0.65 · ln(1.3) = 0.65 · 0.2624 = **+0.1705**
- Categoría 2: 0.35 · ln(0.35/0.50) = 0.35 · ln(0.7) = 0.35 · (−0.3567) = **−0.1248**
- **KL(p‖q) = 0.1705 − 0.1248 = 0.0457 nats**

(Fíjate que un término puede salir negativo — lo que nunca sale negativa es la *suma
total*; la demostración está en §6.2.)

**Tres propiedades que usamos todos los días del experimento:**

1. **KL ≥ 0 siempre**, y = 0 solo si p = q exactamente. Por eso sirve como "medidor de
   desacuerdo": 0 = acuerdo total.
2. **Es asimétrica:** KL(p‖q) ≠ KL(q‖p). En nuestro ejemplo al revés: KL(q‖p) =
   0.5·ln(0.5/0.65) + 0.5·ln(0.5/0.35) = 0.5·(−0.2624) + 0.5·(0.3567) = **0.0472** —
   cercano pero distinto. Esta asimetría no es un defecto: es una *feature* (ver §3.7).
3. **No está acotada arriba:** puede valer desde 0 hasta infinito. Por eso reportamos
   también la JSD, que sí está acotada (§3.7).

### 3.6 El detalle numérico que casi nos tumba: log_softmax en float64

En la teoría, KL(p‖q) se programa como `p * (log p − log q)`. En la práctica hubo un
bug real (lo cazó val_07 el 23-jul-2026): los hidden states de las capas tardías de
Gemma tienen componentes de magnitud 100–1.000. Con τ=1, el softmax en float32 colapsa
**2.559 de 2.560 componentes a exactamente 0** (la exponencial de un número muy negativo
se redondea a cero en la computadora). Y `log(0) = −∞` → **KL = ∞**.

La solución es una identidad matemática que las librerías implementan de forma segura:
`log_softmax(z)_i = z_i − max(z) − log Σ exp(z_j − max(z))` — restar el máximo antes de
exponenciar no cambia el resultado matemático (se cancela arriba y abajo en la fracción
del softmax) pero evita el desbordamiento. Todo el cómputo de KL/JSD se hace con
`log_softmax` en **float64**. Verificado: KL finita y KL(p‖p) = 0 exacto.

### 3.7 Las dos direcciones y la JSD — por qué computamos "de más"

Como la KL es asimétrica, calculamos las dos direcciones **en la misma pasada** (no
cuesta cómputo extra: ambas usan los mismos vectores ya extraídos):

- **KL(p_vis ‖ p_text):** "¿cuánto le sorprende al modelo lo que *va a decir*, dado lo
  que *vio*?" — nuestra dirección principal. Si la respuesta no se desprende de la
  imagen, este número crece.
- **KL(p_text ‖ p_vis):** la dirección espejo — "¿la imagen sorprende dado lo que va
  a decir?" (útil si el modelo se "enamora" de su respuesta e ignora la imagen).
- **JSD (Jensen-Shannon):** el promedio de cada una contra la distribución media:
  `JSD = ½ KL(p‖m) + ½ KL(q‖m)` con `m = (p+q)/2`. Es **simétrica y acotada** (máximo
  ln 2 ≈ 0.693 nats), a costa de perder la interpretación direccional. Va como ablación.

En total, por imagen guardamos **54 variantes**: 3 divergencias × 3 capas {17, 26, 34}
× 3 temperaturas {1, 2, 4} × 2 pooling {mean, max}. Guardar todo de una vez convierte
las ablaciones en *análisis sobre el CSV*, no en re-cómputos de GPU — la decisión de
ingeniería más importante del proyecto.

### 3.8 La respuesta yes/no: softmax renormalizada a dos candidatos

El modelo puntúa su próxima palabra sobre **todo el vocabulario (262.208 tokens)**. A
nosotros solo nos interesan dos: `yes` (ID 4443) y `no` (ID 1904) — verificados contra
el tokenizer. Entonces:

```
P(yes) = exp(logit_yes) / (exp(logit_yes) + exp(logit_no))     (y P(no) = 1 − P(yes))
```

**¿Por qué renormalizar en vez de tomar la probabilidad "cruda" del vocabulario?**
Porque la cruda diluye la masa entre sinónimos ("Yes", " yes", "sí", "definitely"…) y
entre miles de tokens irrelevantes; lo que nosotros preguntamos es un problema **binario
por diseño**, y forzar la elección entre exactamente dos opciones hace las probabilidades
comparables entre imágenes. Sanity check permanente: P(yes) + P(no) = 1 exactamente.

**¿Por qué `yes`/`no` minúsculas y sin espacio?** La plantilla de chat termina en
`<start_of_turn>model\n` — el modelo responde "a renglón seguido", así que su primera
palabra no lleva espacio inicial. Verificado en val_02 (las variantes con espacio o
mayúscula se inspeccionan una vez y se congelan, por robustez).

---

## 4. El algoritmo, paso a paso

Esta es la receta completa. Cada paso indica su *salida* y dónde vive en el código.

### Fase 0 — Preparación (una sola vez)

1. Aceptar la licencia HAI-DEF en HuggingFace; `huggingface-cli login`.
2. Descargar `google/medgemma-4b-it` (checkpoint **v1.0.1**) y cargarlo en 4-bit NF4
   (5.6–7.5 GB VRAM medidos) o bf16 si hay GPU grande.
3. Cargar el dataset: `load_dataset("TheBug95/MM-ODIR-129")` → 77 train / 26 val /
   26 test (verificado en val_03).
4. Fijar seeds (42), `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 explícito, batch = 1.
5. **Salida:** entorno congelado (versiones registradas para la sección de
   reproducibilidad del paper).

### Fase 1 — Tabla maestra (sin GPU, ~5 min)

6. Construir `data/master_table.csv`: una fila por imagen con
   `{image_filename, patient_id (de split.json), eye, label (0=Normal, 1=Pathological),
   split, transcription, cdr_grade (= cup_to_disc_ratio, ordinal 0–4), 7 gradings de
   signos, has_masks, has_annotation_artifact}`.
7. **Regla de privacidad:** el campo `doctor_name` de `split.json` **nunca** entra a la
   tabla ni a ningún artefacto (PII del anotador).
8. Auditoría: marcar imágenes con artefactos de anotación (para el análisis de robustez).
9. **Salida:** `master_table.csv` (129 filas).

### Fase 2 — Inferencia (GPU, ~20 min medidos: 4.3 s × 258 corridas)

Para **cada imagen** × **cada prompt** (P1 directo; P4 con "You are an expert
ophthalmologist."):

10. Armar el chat template con la imagen; verificar máscara de imagen = 256 posiciones.
11. `generate(max_new_tokens=1, do_sample=False, output_scores=True,
    output_hidden_states=True, return_dict_in_generate=True)` bajo `inference_mode`.
12. Extraer de `scores[0]`: logit_yes, logit_no → P(yes) renormalizada, predicción,
    entropía/MSP/energy de la respuesta (baselines baratos).
13. Extraer de `hidden_states[0]` (el prefill), para las capas 17, 26 y 34:
    - los 256 vectores de imagen (máscara por ID 262144) → pooling mean y max → p_vis;
    - el vector de la última posición → p_text.
14. Calcular las **54 variantes** (3 divergencias × 3 capas × 3 τ × 2 pooling) con
    `log_softmax` en float64.
15. **Escribir la fila al CSV en modo append** (reanudable si se cae).
16. **Salida:** `results/results_full.csv` — 258 filas × ~100 columnas.

### Fase 3 — Baselines multi-pass (GPU, solo un subconjunto)

**16b. Verbalized Confidence (baseline 2×, añadido 26-jul-2026):** para cada imagen,
    un segundo turno tras la respuesta de P1 (prompt P5: "How confident are you in your
    answer? Reply with a number from 0 to 100.") → parsing directo del número →
    `verbalized_conf`; u(x) = 1 − conf/100. Es el baseline verbal estándar de la
    literatura LLM — cierra la escalera de costo por el medio (1×/2×/10×).
17. **Self-Consistency:** en 50 imágenes (estratificadas), 10 muestras a T=0.7 →
    incertidumbre = entropía de la fracción de "yes". Representa a la familia multi-pass
    (10× costo) — si le ganamos con 1×, el argumento de eficiencia queda cerrado.
18. **Temperature Scaling:** ajustar T minimizando ECE *solo en train*, aplicar a todos.

### Fase 4 — Selección de la variante (sin GPU, solo train)

19. Con las 77 filas de train, evaluar las 54 variantes como detector de errores
    (AUROC contra la etiqueta `correct`).
20. **Elegir UNA** (capa × dirección × τ × pooling) **y congelarla**. Todo lo que se
    reporte después usa esa, sin volver a mirar alternativas en test. Esto es lo que
    separa ciencia de *p-hacking*.

### Fase 5 — Estadística y figuras (sin GPU)

21. **Test principal:** Mann-Whitney U (errores vs. aciertos) con p-value y tamaño de
    efecto r = |Z|/√N (Z manual verificado en val_06).
22. **AUROC** de la variante congelada con IC95% bootstrap **BCa** (9.999 remuestreos);
    AUPRC con su IC (ambas, sin predicar superioridad de ninguna — lección McDermott).
23. **DeLong** (`pauc`) contra el mejor baseline barato (exploratorio).
24. **H4:** Spearman(u(x), cup_to_disc_ratio) en los 69 patológicos, significación por
    **permutación** (n=69: el p asintótico no es fiable), Kendall tau-b de sensibilidad.
25. Curva accuracy-coverage ("si derivo el x% más incierto, ¿qué accuracy queda?").
26. Robustez: repetir todo excluyendo imágenes con `has_annotation_artifact`.
27. **Salida:** Tablas T1–T3 y Figuras 1–5.

### Punto Go/No-Go (día 4)

- AUROC ≥ 0.65 con IC que **excluye 0.5** → evidencia fuerte → escribir paper con
  discurso "señal útil y barata".
- IC que **incluye 0.5** pero punto estimado > 0.6 → evidencia sugestiva → paper de
  primer contacto con ese lenguaje honesto.
- AUROC ≈ 0.5 → plan de contingencia (capas medias, JSD, τ alta sobre el mismo CSV;
  si nada: estudio de limitaciones — también publicable en BIP).

---

## 5. Micro-decisiones: qué, alternativa, y por qué

| Decisión | Alternativa descartada | Por qué así |
|---|---|---|
| p_text = última posición del **prefill** | hidden state tras generar el token (max_new_tokens=2) | Es el estado que *produce* la respuesta; mantiene el claim **single-pass**. La otra queda como ablación |
| Máscara por ID 262144 | slicing fijo `[:, :256]` | Con chat template el bloque de imagen queda en medio; slicing fijo = bug garantizado |
| Capas {17, 26, 34} | todas las capas / solo la última | 50%, 75%, 100% de profundidad: la literatura de probing dice que las medias-tardías codifican mejor "la verdad"; 3 puntos bastan para el mínimo |
| τ ∈ {1, 2, 4} | τ aprendida / muchas τ | Ablación pequeña y controlada; se elige en train |
| Pooling mean (primario) + max (ablación) | atención aprendida | Training-free: nada que aprender. Mean es robusto a dimensiones gigantes; max capta "la señal más fuerte" |
| yes/no renormalizado | probabilidad cruda del vocabulario | Problema binario por diseño → comparabilidad entre muestras |
| Greedy, 1 token | sampling, respuestas largas | Determinista (reproducibilidad bit-exacta verificada) y la señal vive en la *primera* palabra |
| batch = 1 | batching | Secuencias de largo variable, 258 corridas ≈ 20 min: la simplicidad gana |
| 4-bit NF4 en T4 | bf16 | VRAM medido 5.6–7.5 GB vs ~10–12 GB bf16; en Colab T4 es la opción segura |
| Selección de variante solo en train | elegir la mejor en todo el dataset | Elegir mirando test contamina todos los p-values e ICs posteriores |
| Mann-Whitney + r | t-test | Las KL no son normales (colas largas); el test de rangos no asume distribución |
| Bootstrap BCa | percentile | Con N=129 el BCa corrige sesgo y asimetría del remuestreo |
| Spearman + permutación (H4) | p-value asintótico | Con n=69 el asintótico no es fiable; la permutación no asume nada |
| AUROC **y** AUPRC | solo AUPRC | McDermott et al. 2024: la ventaja de AUPRC puede ser artefacto → se reportan ambas con IC |

---

## 6. Justificaciones matemáticas en detalle

### 6.1 Por qué τ aplana o afila el softmax

Divide entre τ *dentro* de la exponencial: `p_i ∝ exp(z_i / τ)`.

- Si τ es grande, los cocientes `z_i/τ` se hacen todos pequeños y parecidos → las
  exponenciales salen parecidas → al normalizar, la distribución se acerca a uniforme.
- Si τ es pequeño, los cocientes se exageran → el mayor se dispara → distribución delta.

Caso límite útil de recordar: τ → ∞ da exactamente la uniforme (todas las categorías
con 1/2560), y ahí KL(p‖q) → 0 *para todo par* — por eso τ tampoco puede ser gigante:
aplanar demasiado borra la señal. El régimen informativo está en algún punto medio, y
por eso τ es ablación, no constante.

### 6.2 Por qué KL nunca es negativa (desigualdad de Gibbs, explicada)

Usamos el hecho `ln(x) ≤ x − 1` para todo x > 0 (la curva del logaritmo siempre queda
por debajo de su tangente en x=1). Aplicado a x = q_i/p_i:

```
ln(q_i/p_i) ≤ q_i/p_i − 1
−KL(p‖q) = Σ p_i · ln(q_i/p_i) ≤ Σ p_i · (q_i/p_i − 1) = Σ q_i − Σ p_i = 1 − 1 = 0
```

Como −KL ≤ 0, entonces **KL ≥ 0**. Y la igualdad solo ocurre cuando q_i = p_i en todas
las categorías. Traducción: la KL es un "termómetro de desacuerdo" calibrado — nunca
marca bajo cero, y marca exactamente 0 solo con acuerdo perfecto. Por eso es seguro
usarla como score: los números grandes *siempre* significan más desacuerdo.

**Y por qué es asimétrica:** la KL pesa cada categoría con p_i. Las categorías que p
considera importantes mandan; si q les asigna poco, castigo grande. Pero las categorías
que *q* considera importantes y p no, apenas cuentan en KL(p‖q) (peso p_i ≈ 0) — y sí
en KL(q‖p). Cada dirección pregunta algo distinto; por eso guardamos ambas.

### 6.3 AUROC: qué es realmente ese número

Definición sin fórmulas: **toma un caso al azar del grupo "errores" y uno del grupo
"aciertos". AUROC es la probabilidad de que el error tenga u(x) mayor que el acierto.**

- AUROC = 0.5 → como tirar una moneda: la señal no ordena nada.
- AUROC = 1.0 → todos los errores tienen más señal que todos los aciertos: orden perfecto.
- AUROC = 0.65 → 65 de cada 100 parejas (error, acierto) quedan bien ordenadas.

¿Por qué 0.65 ya es útil? Porque para triage no necesitas orden perfecto: si derivas al
especialista el 10–20% con más u(x), y ese grupo está enriquecido en errores, el sistema
completo mejora. Los detectores de humo no son perfectos: son *útilmente* sensibles.

Se calcula sin elegir ningún umbral (integra todos los umbrales posibles), por eso es la
métrica principal: ningún reviewer puede acusarnos de haber "afinado el corte".

### 6.4 Mann-Whitney U y el tamaño de efecto r = |Z|/√N

**Mann-Whitney U** contesta: "¿los errores tienen *sistemáticamente* más u(x) que los
aciertos?" sin asumir que los datos sean normales (las KL tienen colas larguísimas → el
t-test clásico sería inapropiado). Funciona con **rangos**: junta los 129 valores,
los ordena de menor a mayor, y suma en qué posiciones cayeron los errores vs. los
aciertos. Si los errores ocuparan posiciones al azar, la suma da un valor esperado; si
se concentran arriba, la suma es anómalamente grande → p-value pequeño.

**El p-value solo dice "existe diferencia", no "qué tan grande".** Para la magnitud se
reporta r = |Z|/√N (convención de Fritz, Morris & Richler 2012), donde Z es el
estadístico estandarizado de la U y N = 129. Regla de lectura: 0.1 pequeño, 0.3 mediano,
0.5 grande. Ojo implementado: scipy **no** devuelve Z directamente (bug detectado en
val_06) → lo calculamos a mano con corrección de continuidad y empates, y el script
autoverifica que el p-value derivado de nuestro Z coincide con el de scipy (< 1e-10).

### 6.5 Bootstrap BCa: intervalos de confianza sin fórmulas

Pregunta: "tu AUROC es 0.70, ¿qué tan estable es ese número?" Respuesta bootstrap:

1. De tus 129 filas, sortea **129 con reemplazo** (algunas se repiten, otras faltan) →
   un "mundo paralelo" estadísticamente equivalente.
2. Recalcula el AUROC en ese mundo. Repite 9.999 veces → 9.999 AUROCs.
3. El histograma de esos 9.999 valores *es* la incertidumbre de tu estimación. El IC95%
   recorta el 2.5% de cada cola.

**BCa** (*bias-corrected and accelerated*) es la versión que además corrige dos
imperfecciones: si la distribución remuestreada está sesgada respecto al valor original,
y si es asimétrica. Con N=129 (pequeño) esas correcciones importan → BCa en vez del
percentil simple. Regla del paper: **el IC es la evidencia principal**, no un adorno —
con ~40 errores y ~90 aciertos, un AUROC 0.70 tendrá IC de ±0.10–0.13, y el discurso
("fuerte" vs. "sugestiva") depende de si excluye 0.5.

### 6.6 Spearman + permutación (H4)

**Spearman** es una correlación sobre *rangos*: ordena los 69 pacológicos por u(x),
ordénalos por grado CDR (0–4), y mide si los órdenes coinciden (ρ entre −1 y 1). No
asume linealidad ni normalidad — perfecto para un grado ordinal.

**El problema:** el p-value "de fábrica" usa una aproximación que solo es fiable con
n > 500. Con n=69 puede mentir. Solución: **test de permutación** —

1. Calcula ρ con los datos reales.
2. Baraja los grados CDR al azar (rompiendo cualquier relación verdadera) y recalcula ρ.
3. Repite 9.999 veces: obtienes la distribución de "ρs que saldrían por puro azar".
4. El p-value = fracción de azares que superan tu ρ real.

No asume nada; es exacto salvo por el número de barajadas. Como sensibilidad reportamos
Kendall tau-b (otra correlación de rangos, más conservadora con empates — y el CDR
ordinal tiene muchos empates).

---

## 7. Qué está verificado y dónde

| Afirmación usada en esta guía | Verificada en |
|---|---|
| 256 tokens de imagen, contiguos tras `<start_of_image>`, máscara por ID 262144 | val_02, val_07 |
| IDs yes=4443 / no=1904; plantilla termina en `<start_of_turn>model\n` | val_02 |
| hidden_states[0][capa][:, -1, :] = estado que condiciona la respuesta; 35 entradas | val_04, val_07 |
| scores[0] = logits del primer token; equivalencia con forward manual | val_04 |
| KL con log_softmax float64 finita; KL(p‖p)=0; reproducibilidad bit-exacta | val_07 |
| load_dataset funciona; 77/26/26; 60 Normal/69 Pathological; CDR ordinal 0–4 en 69 | val_03 |
| F.kl_div dirección correcta; JSD = jensenshannon²; AUROC/AUPRC con scores continuos | val_05 |
| Z manual de Mann-Whitney; BCa; permutación; potencia con 60/69 | val_06 |
| Tiempo 4.3 s/muestra (~20 min total); VRAM 5.6–7.5 GB | val_07 |

---

## 8. Glosario mínimo

- **Token:** pedazo discreto (palabra, subpalabra o parche de imagen) numerado que el
  modelo procesa.
- **Hidden state:** vector de 2.560 números = lo que el modelo "piensa" de una posición
  tras una capa dada.
- **Logit:** punto crudo que el modelo da a cada palabra candidata antes de softmax.
- **Softmax / τ:** convierte un vector en distribución; τ controla qué tan tajante.
- **KL(p‖q):** sorpresa de q si el mundo es p; ≥ 0, asimétrica.
- **JSD:** versión simétrica y acotada (≤ ln 2) de la KL.
- **AUROC:** probabilidad de que un error tenga más señal que un acierto.
- **Mann-Whitney / r:** test de "¿un grupo tiene más que el otro?" sin asumir normalidad
  + tamaño del efecto.
- **Bootstrap BCa:** intervalo de confianza por remuestreo, corregido por sesgo/asimetría.
- **Spearman + permutación:** correlación de rangos con p-value exacto por barajadas.
- **Single-pass:** todo se extrae de UNA ejecución del modelo por imagen (frente a
  10–100 de los métodos multi-pass).
- **Training-free:** nada que entrenar; la señal sale de lo que el modelo ya computó.

*Documento hermano de `Definicion_Experimental_Minima_BIP2026.md` (la especificación
formal), `AGENTS.md` (las reglas operativas para el agente de código) y
`Plan_de_Validacion_BIP2026.md` (la evidencia de que todo lo aquí afirmado está verificado).*
