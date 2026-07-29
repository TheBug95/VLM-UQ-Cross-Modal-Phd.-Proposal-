# Reporte del Experimento BIP 2026 — Estado Actual (29-jul-2026)

**Qué es este documento:** explicación completa y ordenada de qué estamos haciendo, cómo se calcula cada número, cuáles son las variantes de la señal, qué valores tiene cada una hoy, y el reporte de la combinación KL + MSP. Todo con datos reales de `results/results_full.csv` (129 imágenes × 2 prompts, corrida completa).

---

## 1. El experimento en una página

**Pregunta de investigación:** cuando MedGemma-4B se equivoca al detectar glaucoma en una foto de fondo de ojo, ¿el desacuerdo entre su representación visual y su representación textual es mayor que cuando acierta?

**Si la respuesta es sí**, ese desacuerdo sirve como señal de alarma barata (una sola pasada del modelo, sin entrenar nada) para decidir qué pacientes derivar al oftalmólogo.

**El flujo, paso a paso:**

```
Foto de fondo de ojo          Pregunta: "Does this fundus image
        │                     show glaucoma? Answer yes or no."
        │                              │
        └──────────► MedGemma-4B ◄─────┘
                     (1 sola pasada)
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
   256 tokens de      último token      logits de la
   imagen (lo que     (lo que está      respuesta:
   el modelo VIO)     por responder)    P(yes), P(no)
            │               │               │
            ▼               ▼               ▼
         p_vis  ◄──KL──►  p_text      baselines:
         (vector)  u(x)   (vector)    entropy, MSP, energy
```

- **u(x) = divergencia KL entre p_vis y p_text.** Hipótesis: u(x) alto ⟺ el modelo está por equivocarse.
- **Datos:** MM-ODIR-129 — 129 fotos (60 normales, 69 glaucoma), ya procesadas las 129 × 2 prompts.
- **MedGemma base:** acierta el **79.8%** (P1) — se equivoca en 26 de 129 casos. Nuestro trabajo es detectar esos 26 errores.

---

## 2. Cómo se calcula cada valor (con ejemplo real)

Tomemos la imagen `1281_right.jpg` (prompt P1). Cada número del CSV se produce así:

| Paso | Qué se hace | Ejemplo real |
|---|---|---|
| 1 | La imagen entra por el processor de MedGemma → se vuelve **256 tokens de imagen**; el texto del prompt se tokeniza aparte | 256 tokens de imagen + ~30 de texto |
| 2 | Una pasada del modelo. Guardamos los **hidden states de la capa 34** (la última) | matriz de ~286 × 2560 números |
| 3 | **p_text** = el vector de la **última posición** (el estado desde el que el modelo genera su respuesta) | vector de 2560 números |
| 4 | **p_vis** = los 256 vectores de imagen **resumidos en uno** por un *pooling* (mean, max, etc.) | vector de 2560 números |
| 5 | Ambos vectores se convierten a **distribuciones de probabilidad** sobre el vocabulario (262.208 palabras) con un softmax con temperatura τ | dos distribuciones |
| 6 | Se calcula la **divergencia KL** entre ellas (y JSD, y coseno) | `kl_t_v = 22.19` |
| 7 | En paralelo, de los logits de la respuesta salen los **baselines**: P(yes)=0.99999, entropy, MSP, energy | `entropy = 0.00017` |

**Todo esto sale de UNA sola pasada del modelo** (~4.5 segundos por imagen). Las 97 variantes son re-cortes sobre los mismos números ya capturados — no cuestan inferencia extra.

---

## 3. Las 97 variantes: qué se varía en cada una

Una variante = **(tipo de divergencia) × (pooling) × (temperatura)**.

### 3.1 Tipos de divergencia (qué se mide)

| Tipo        | Definición           | Interpretación                                                            |
| ----------- | -------------------- | ------------------------------------------------------------------------- |
| `kl_t_v`    | KL(texto ‖ imagen)   | ¿El texto dice cosas que la imagen no respalda? (dirección "alucinación") |
| `kl_v_t`    | KL(imagen ‖ texto)   | ¿La imagen contiene cosas que el texto no captura?                        |
| `jsd`       | Jensen-Shannon       | Versión simétrica de KL                                                   |
| `cosine`    | 1 − similitud coseno | Distancia angular entre vectores **crudos** (sin softmax)                 |


### 3.2 Poolings (cómo se resumen los 256 tokens de imagen en un vector)

| Pooling    | Qué hace                                                                 | ¿Usable en la realidad?                          |
| ---------- | ------------------------------------------------------------------------ | ------------------------------------------------ |
| `mean`     | Promedio simple de los 256 tokens                                        | Sí                                               |
| `max`      | Se queda con el valor máximo de cada dimensión                           | Sí                                               |
| `attn`     | Pondera por la atención que el último token pone en cada token de imagen | Sí                                               |
| `topk`     | Promedia solo los 26 tokens (~10%) con mayor norma                       | Sí                                               |
| `normw`    | Pondera cada token por su norma (sin parámetros)                         | Sí                                               |
| `rollout`  | Pondera por la atención propagada a través de todas las capas            | Sí                                               |
| `headspec` | Usa solo las 4 cabezas de atención más "visuales"                        | Sí                                               |

### 3.3 Temperatura τ (cómo de plana es la distribución)

| τ | Efecto sobre los valores KL (ejemplo `kl_t_v_mean`) |
|---|---|
| 1 | mediana 21.7 ← **la que mejor funciona** |
| 2 | mediana 10.0 (aplana ~a la mitad) |
| 4 | mediana 0.8 (aplana demasiado, pierde discriminación) |

**Total: 4 tipos × 8 poolings × 3 τ = 96 variantes.** Además: 3 baselines (entropy, 1−MSP, energy) y la combinación por ranks (sección 7).

---

## 4. Capas: usamos UNA sola, la 34

El modelo tiene 34 capas (+ embeddings). Originalmente se probaron las capas **17, 26 y 34**, pero en el piloto se descubrió que en las capas 17 y 26 la KL **colapsa**: los valores quedan degenerados (casi idénticos para todas las imágenes), porque a esa profundidad las representaciones de imagen y texto aún no están diferenciadas para la tarea. **Todo el análisis quedó congelado en la capa 34** (la última), donde la representación ya está orientada a responder la pregunta.

---

## 5. Valores actuales de cada variante (tabla)

Estadísticos reales sobre las 129 imágenes, prompt P1, **τ=1** (la temperatura ganadora). Para τ=2 los valores KL bajan aprox. a la mitad; para τ=4 a ~una décima parte.

### 5.1 Familia `kl_t_v` (la ganadora está aquí)

| Pooling                          | min   | mediana   | max   | n   |
| -------------------------------- | ----- | --------- | ----- | --- |
| **max** ← **variante congelada** | 21.41 | **22.77** | 23.02 | 129 |
| mean                             | 4.88  | 21.71     | 23.02 | 129 |
| rollout                          | 5.63  | 21.84     | 23.02 | 129 |
| normw                            | 5.42  | 21.67     | 23.02 | 129 |
| topk                             | 6.40  | 21.92     | 23.02 | 129 |
| attn                             | 5.20  | 16.98     | 23.02 | 129 |
| headspec                         | 4.72  | 17.04     | 23.02 | 129 |


### 5.2 Familia `kl_v_t` (dirección contraria)

| Pooling      | min   | mediana | max   | n   |
| ------------ | ----- | ------- | ----- | --- |
| max          | 9.00  | 22.88   | 23.03 | 129 |
| mean         | 8.26  | 21.02   | 23.03 | 129 |
| rollout      | 8.58  | 21.20   | 23.03 | 129 |
| normw        | 8.20  | 21.02   | 23.03 | 129 |
| topk         | 5.76  | 18.17   | 23.03 | 129 |
| attn         | 5.77  | 18.64   | 23.03 | 129 |
| headspec     | 5.91  | 18.42   | 23.03 | 129 |


### 5.3 Familias `jsd`, `cosine`, `kl_prompt`

| Variante    | min   | mediana | max   | Nota                                           |
| ----------- | ----- | ------- | ----- | ---------------------------------------------- |
| jsd_max     | 0.692 | 0.693   | 0.693 | **Saturada** en el techo (ln 2): no discrimina |
| jsd_mean    | 0.670 | 0.693   | 0.693 | Casi saturada                                  |
| cosine_mean | 0.709 | 0.802   | 0.912 | Escala [0, 2]; no usa τ                        |
| cosine_max  | 0.931 | 0.960   | 1.008 | Muy concentrada                                |


### 5.4 Baselines (salen de los logits de la respuesta, costo 1×)

| Baseline | min    | mediana | max    | Lectura                                          |
| -------- | ------ | ------- | ------ | ------------------------------------------------ |
| entropy  | 0.000  | 0.001   | 0.693  | Casi siempre ~0: el modelo casi nunca duda       |
| msp      | 0.500  | 0.9999  | 1.000  | Saturado arriba: confianza altísima casi siempre |
| energy   | −25.25 | −22.25  | −17.69 | Escala negativa                                  |

---

## 6. Resultados de detección de errores (lo importante)

**La tarea:** ordenar las 129 imágenes por u(x) y ver si los 26 errores quedan arriba. Métricas: **AUROC** (0.5 = azar, 1.0 = perfecto; ranking global) y **AURC / Excess-AURC** (área bajo la curva riesgo-cobertura de selective prediction — el error del modelo integrado sobre todos los niveles de derivación; Excess normalizado: 0 = oracle, 1 = azar, menor = mejor). IC = intervalo de confianza bootstrap del 95% (9.999 remuestreos BCa).

| Señal (P1)                                   | AUROC     | IC 95%             | AURC     | Excess-AURC | p (Mann-Whitney) |
| -------------------------------------------- | --------- | ------------------ | -------- | ----------- | ---------------- |
| **KL congelada (`kl_t_v` max τ=1, capa 34)** | **0.661** | [0.522, 0.772]     | 0.1425   | 0.670       | 0.006            |
| 1−MSP (baseline gratis)                      | 0.624     | [0.491, 0.739]     | 0.1536   | 0.732       | 0.026            |
| entropy (baseline)                           | 0.624     | [0.491, 0.739]     | 0.1536   | 0.732       | 0.026            |
| energy (baseline)                            | 0.560     | [0.432, 0.680]     | 0.1828   | 0.895       | 0.175            |
| **Combinación rank(KL)+rank(1−MSP)**         | **0.698** | **[0.596, 0.787]** | **0.0955** | **0.407** | **0.001**        |

Lectura honesta:
- La KL sola **supera al azar con significancia** (su IC excluye 0.5) y cumple el objetivo del proyecto (≥ 0.65) justo. También es la #1 entre las 97 variantes por Excess-AURC — las dos métricas coinciden.
- La ventaja sobre el baseline gratis (1−MSP) es **modesta** (+0.037): la KL no reemplaza a la confianza del modelo, la complementa (sección 7).
- **El AURC amplifica el veredicto sobre la combinación:** en Excess-AURC la combinación marca 0.407 vs 0.670 de la KL sola — 39% más cerca del oracle. El AURC premia acertar en la *cabeza* de la lista de derivación, que es donde la combinación es fuerte (57% de precisión en los 7 casos más inciertos vs 29% de la KL sola).
- **Concordancia entre métricas:** Spearman(AUROC, Excess-AURC) = −0.51 sobre las variantes — se correlacionan moderadamente pero no son redundantes: AUROC evalúa el ranking global, AURC los puntos operativos de triage.
- **Correlación entre señales** (Spearman, `figures/fig6_correlacion_senales.png`): KL⊥MSP = 0.27 (complementarias), KL⊥energy = −0.01 (totalmente independientes), entropy≡MSP = 1.00 (la misma señal), energy–MSP = 0.84 (redundantes — por eso energy no aporta).
- En P4 (prompt con rol de oftalmólogo) todo es más débil: KL 0.614, MSP 0.629, combinación 0.654.

**Triage (accuracy-coverage, P1):** si se deriva al oftalmólogo el 50% de casos con mayor u(x), la accuracy en el 50% que el modelo responde solo sube de **79.8% → 89.1%**. Y en el otro extremo: la combinación tiene una **zona verde** — los **39 casos menos inciertos (30.2%) son todos correctos** (el primer error aparece en la posición 40 del ranking; verificado recomputando desde el CSV en `results/verificacion_zona_verde.py`, donde están las posiciones de los 26 errores). Advertencias: la frontera de la zona es suave (Δu ≈ 0.004 entre las posiciones 39 y 40) y la medida es in-sample (IC 95% de la tasa de error en la zona verde hasta ~7.7% por regla de tres). Lectura clínica: auto-responder la cola (~30% sin errores en esta cohorte), derivar la cabeza, y la zona gris intermedia al especialista.

---

## 7. Reporte de la combinación KL + MSP

### 7.1 Qué es

```
u(x) = rank(KL) + rank(1 − MSP)
```

- **KL**: nuestra señal cross-modal interna (la ganadora congelada).
- **1−MSP**: la duda del modelo en su respuesta (1 − confianza).
- **rank()**: en vez de sumar los valores crudos (incomparables: KL≈22.8, MSP≈0.0003), cada imagen toma su **puesto en la fila** de cada señal. La suma de puestos es la nueva u(x). Más alto = más sospechoso.

No tiene **ningún parámetro que ajustar** — no puede sobreajustarse como la regresión logística (que probamos y aprendió a ignorar el MSP).

### 7.2 Por qué es válida (en una frase)

Las dos señales son **testigos que vieron cosas distintas**: su correlación es de apenas 0.27 (P1) y 0.02 (P4). La KL atrapa errores donde el modelo respondió confiado pero hubo tensión interna imagen↔texto; el MSP atrapa errores donde el modelo vaciló aunque no hubo tensión interna. Cada una ve lo que la otra no — y la suma de puestos premia a quien es sospechoso para *al menos una* de las dos.

### 7.3 Resultados

| Evaluación                                                        | AUROC                             |
| ----------------------------------------------------------------- | --------------------------------- |
| Combinación en las 129 imágenes (P1)                              | **0.698** [0.596, 0.787], p=0.001 |
| — KL sola (comparación)                                           | 0.661 [0.522, 0.772]              |
| — 1−MSP solo (comparación)                                        | 0.624 [0.491, 0.739]              |
| Monte Carlo CV, 200 splits (KL congelada)                         | **0.698 ± 0.062**                 |
| Monte Carlo CV, 200 splits (re-seleccionando la KL en cada split) | **0.648 ± 0.087**                 |
| Combinación en P4                                                 | 0.654 [0.539, 0.754]              |

**Interpretación:** la combinación es la mejor señal que tenemos, es consistente en ambos prompts, y su desempeño se mantiene al re-muestrear poblaciones de pacientes (Monte Carlo). Advertencia honesta: con N=129, su IC se solapa con el de la KL sola, así que se reporta como **análisis exploratorio** — la señal primaria del paper sigue siendo la KL congelada, y la combinación demuestra que la señal cross-modal **aporta información que la confianza de salida no tiene**.

---

## 8. Lo que NO funcionó (también es resultado)

| Hipótesis / variante | Resultado | Conclusión |
|---|---|---|
| H4: u(x) correlaciona con la severidad (cdr_grade) | Spearman rho = +0.001, p = 0.99 | **Rechazada.** La señal detecta errores, no severidad |
| **Verbalized confidence (P5, baseline 2×)** | **AUROC 0.519 (azar).** El modelo declara 95% en 118/129 imágenes y 90% en 11 — solo 2 valores distintos. Mal calibrada (dice 95%, acierta 80.5%) | "Simplemente preguntarle al modelo" no funciona: la confianza verbal es degenerada. Nuestro 1× (0.661/0.698) supera claramente al baseline 2×. Tampoco aporta a las combinaciones (verb+KL+MSP = 0.694 ≤ 0.698) |
| Fusión de las dos direcciones KL (ablación exhaustiva: 8 poolings × 3 τ × suma/max/min/asimetría/ranks/JSD) | Ninguna fusión supera a kl_t_v sola (mejor: rk_sum 0.649 < 0.661); añadir v→t a la combinación estrella siempre la baja (0.698 → 0.681) | La señal del error vive solo en la dirección t→v (alucinación); v→t es mayormente ruido. "La fusión ayuda" solo en configs donde t_v ya es azar (~0.50) |
| ROI oracle (máscaras del disco) | AUROC 0.35 en las 69 con máscara | El 0.889 del piloto fue artefacto de muestra pequeña; no se reproduce |
| Distancia coseno | AUROC 0.42–0.57 | No aporta; la KL va mejor |
| JSD | Saturada en su techo (0.693) con τ=1 | No discrimina en este régimen |
| Capas 17 y 26 | KL colapsa (valores degenerados) | Solo la capa 34 es útil |
| τ = 2 y τ = 4 | Aplanan demasiado la distribución | τ = 1 gana siempre |

---

## 9. Verificación y robustez (29-jul-2026)

**Verificación independiente** (`validacion/val_08_resultados.py`, 19/19 PASS): invariantes del CSV (p_yes=softmax de logits, entropy binaria, MSP, energy, consistencia del formato largo), AUROC y AUPRC recomputados a mano sin sklearn (coincidencia exacta al 6º decimal: 0.660941 y 0.329312), combinación por ranks recomputada (0.697535), identidad AUROC≡U/(n_e·n_c), `kl_div` contra Σ p·ln(p/q) a mano, IC bootstrap estable a cambio de semilla.

**Robustez entre GPUs y re-computación manual** (forward de las 129 imágenes en Colab, GPU distinta, backend eager): los logits son **bitwise idénticos** (0 de 129 predicciones cambian) → el input y el forward son perfectamente reproducibles entre hardware. La diferencia observada en la KL (+4.54 nats, casi constante) se explicó por completo por el `eps` de clamp: el pipeline usa `eps=1e-10` (`config.yaml`, techo ln(1/eps)=23.03) y el snippet de verificación usó 1e-12 (techo 27.63); 116/129 imágenes difieren en exactamente ln(1e-10/1e-12)=4.605. El ruido numérico real entre GPUs/backends es pequeño (std ≈ 0.4 nats) y no afecta al ranking: Spearman = 0.964, AUROC 0.645 vs 0.661 (Δ=0.016, dentro del IC bootstrap).

**Hallazgo adicional — la KL está winsorizada:** el clamp en `eps` pone un techo en ln(1/eps)=23.03, y 53 de 129 imágenes están en el techo. Esto recorta la cola extrema (donde la softmax picada sobre hidden states crudos es numéricamente ruidosa); de hecho el AUROC con techo 1e-10 (0.661) es ligeramente mejor que con 1e-12 (0.645).

**Implicaciones:** (a) mantener `epsilon` fijo entre corridas — un eps distinto desplaza TODOS los valores en una constante (ln del ratio); (b) la derivación clínica se define por percentil de la cohorte, no por umbral absoluto de nats; (c) la evaluación por AUROC es robusta a hardware, backend de atención y convención de eps.

---

## 10. Resumen ejecutivo (los números que importan)

| Pregunta                            | Respuesta hoy                                                 |
| ----------------------------------- | ------------------------------------------------------------- |
| ¿Funciona la señal KL cross-modal?  | **Sí, modestamente:** AUROC 0.661 [0.522, 0.772], p=0.006     |
| ¿Supera al baseline gratis (MSP)?   | Por poco sola (+0.037); **claramente en combinación** (0.698) |
| ¿Cuál es la configuración ganadora? | `kl_t_v`, capa 34, τ=1, pooling `max`                         |
| ¿Sirve para triage?                 | Sí: derivando el 50% más incierto, accuracy 79.8% → 89.1%     |
| ¿Correlaciona con severidad?        | **No** (H4 rechazada)                                         |
| ¿Generaliza a pacientes nuevos?     | Estimación honesta (Monte Carlo): combinación 0.648–0.698     |
| Prompt con rol (P4) vs simple (P1)  | P1 mejor en todo                                              |

**Archivos de soporte:** datos en `results/results_full.csv` (formato largo), resumen estadístico en `results/evaluation_summary.csv`, análisis de la combinación en `results/analisis_combinacion.py` y `results/analisis_ranks_mc.py`, figuras en `figures/fig2`–`fig5`.
