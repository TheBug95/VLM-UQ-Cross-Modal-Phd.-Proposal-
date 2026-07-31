# 12 — Verificación y Validación

> **Documento distintivo de esta tesis:** el pipeline no solo se ejecutó — se verificó con **código independiente**, recomputación a mano y re-ejecución en hardware distinto. Aquí se documenta esa verificación, la robustez numérica de la señal y las lecciones aprendidas de los bugs que la verificación encontró.

[⬅️ 11 — Conclusiones y Próximos Pasos](11_Conclusiones_y_Proximos_Pasos.md) | [⬅️ Volver al Índice](01_Indice_General.md)

---

## 12.1 Filosofía de verificación

Principio rector: **nunca verificar con el mismo código que pudo tener el bug**. Aplicado así:

- **Código independiente:** `val_08_resultados.py` reimplementa las métricas desde cero (sin importar `src/`), leyendo solo el CSV de resultados.
- **Recomputación a mano:** AUROC por fórmula de ranks sin sklearn; KL contra la suma $\sum p \ln(p/q)$ manual; AUPRC por su definición.
- **Controles positivo y negativo:** identidades matemáticas que deben cumplirse exactamente (AUROC ≡ U/(n_e·n_c); JSD ≤ ln 2; coseno ∈ [0, 2]); el empate documentado de `1307_right.jpg`.
- **Re-ejecución en otro hardware:** forward manual de las 129 imágenes en una GPU distinta (Colab, backend eager) contra los valores del CSV.

## 12.2 Validaciones pre-implementación (`val_01`–`val_07`)

Antes de escribir el pipeline se validaron por separado (detalle en [§10.5](10_Guia_Reproducibilidad.md)): entorno (`val_01`), tokenizer (`val_02` — origen de los IDs congelados `yes` = 4443, `no` = 1904, `image_soft_token` = 262.144, `<start_of_image>` = 255.999, `<end_of_image>` = 256.000, y de la regla "máscara por ID, nunca slicing fijo"), dataset (`val_03`), API de `generate` (`val_04` — origen de la corrección "solo existe `hidden_states[0]` con `max_new_tokens=1`"), métricas (`val_05`), estadística (`val_06`) y piloto (`val_07` — origen de la regla numérica dura: softmax cruda en **float64**; las massive activations de Gemma colapsan la softmax en baja precisión).

## 12.3 `validacion/val_08_resultados.py` — 19 checks independientes (todos PASS)

Grupos de checks:

1. **Invariantes del CSV:** $p_{yes}$ = softmax de los logits; entropy binaria consistente; MSP y energy consistentes; consistencia del formato largo (misma observación, mismos valores de columnas de observación); rangos teóricos (JSD ≤ ln 2, coseno ∈ [0, 2]); conteos (129 imágenes, 60N/69P).
2. **Métricas recomputadas a mano (sin sklearn):** AUROC por fórmula de ranks = **0.660941** (coincidencia exacta al 6º decimal con el pipeline); AUPRC manual = **0.329312**; identidad **AUROC ≡ U/(n_e·n_c)** verificada (el Mann-Whitney U y el AUROC son el mismo estadístico — control de coherencia entre reportes).
3. **Combinación por ranks recomputada:** AUROC = **0.697535** exacto.
4. **`kl_div` contra $\sum_i p_i \ln(p_i / q_i)$ manual:** verifica la **DIRECCIÓN** de la KL (que `kl_t_v` sea efectivamente KL(texto‖imagen) y no al revés — un bug de dirección habría sido invisible en los AUROC agregados).
5. **Bootstrap con semilla distinta:** IC compatible (estabilidad del BCa).
6. **Empate documentado:** `1307_right.jpg` tiene $\ell_{yes} = \ell_{no} = 17.0$ exactos → $p_{yes} = 0.5$; el desempate determinista está documentado y no es un artefacto.

**Resultado: 19/19 PASS** (29-jul-2026).

## 12.4 Verificación desde el modelo (re-ejecución cross-GPU)

Forward manual de las 129 imágenes en una **GPU distinta** (Colab, backend de atención eager), recomputando la KL independientemente del pipeline (`results/verificacion_manual_kl.csv`):

- **Logits bitwise idénticos:** 0 de 129 predicciones cambian entre GPUs → el input y el forward son perfectamente reproducibles entre hardware.
- **Ranking KL estable:** Spearman(KL manual, KL csv) = **0.964**; AUROC 0.645 vs. 0.661 (Δ = 0.016, dentro del IC bootstrap).
- **La diferencia de valores (+4.54 nats, casi constante) se explicó por completo:** el snippet de verificación usó `eps = 1e-12` (techo 27.63) y el pipeline `eps = 1e-10` (techo 23.03); 116/129 imágenes difieren en exactamente $\ln(10^{-10}/10^{-12}) = 4.605$ nats. No era un bug: era la constante del clamp.
- **Ruido numérico real entre GPUs/backends:** std ≈ **0.4 nats** (eager vs. sdpa) — sin efecto en el ranking ni en las métricas.

## 12.5 Robustez numérica (reglas duras)

1. **KL winsorizada:** el clamp `eps = 1e-10` pone un techo en $\ln(1/\varepsilon) = 23.03$ nats; **53 de 129 imágenes están en el techo**. Es por diseño: recorta la cola numéricamente ruidosa (donde la softmax picada sobre hidden states crudos es inestable). Dato a favor: el AUROC con techo 1e-10 (0.661) es ligeramente **mejor** que con 1e-12 (0.645).
2. **`epsilon` fijo entre corridas:** un eps distinto desplaza TODOS los valores en una constante ($\ln$ del ratio de eps) → no comparar nats absolutos entre corridas con eps distinto.
3. **Derivación clínica por percentil de la cohorte, nunca por umbral absoluto de nats** (los valores absolutos no son portables entre eps/hardware).
4. **Reportar solo métricas de ranking** (AUROC/AUPRC/AURC), que son invariantes a esas constantes.
5. **float64 obligatorio** en la conversión a distribuciones; **sin normalización previa** (z-score/L2 aplana la KL a ~0). Ver [§4.3](04_Arquitectura_Tecnica.md).

## 12.6 Verificación de la zona verde (`results/verificacion_zona_verde.py`)

Recomputación independiente desde `results_full.csv` del hallazgo clínico central ([§6.6](06_Resultados_Experimentales.md)):

- Ordenando las 129 imágenes por $u_{combo}(x)$, las **posiciones de los 26 errores** son exactamente: [40, 46, 50, 51, 60, 62, 63, 69, 70, 78, 79, 80, 81, 85, 86, 89, 97, 98, 103, 104, 105, 119, 123, 125, 128, 129].
- **Primer error en la posición 40** → los 39 casos menos inciertos (30.2%) son todos correctos. **Cruce exacto** con la curva accuracy-coverage de la Figura 4 (accuracy 100% a cobertura 30%).
- Advertencias verificadas al mismo tiempo: frontera suave (Δu = 0.004 entre posiciones 39 y 40) y cota in-sample (IC 95% de la tasa de error en la zona hasta ~7.7%, regla de tres sobre 0/39).

## 12.7 Lecciones aprendidas (bugs encontrados por verificación)

| Bug | Cómo se detectó | Corrección |
|---|---|---|
| **Self-consistency: voto por logits en vez de token muestreado** | AUROC exactamente 0.500 (sospechoso: una señal real no da azar perfecto) | Comparar el token **generado** en cada muestra, no los logits |
| **SC: chequeo por ID exacto de "yes"** | Las variantes del token (`Yes`, `▁yes`…) no se contaban; conteo de votos sesgado | Clasificar la muestra por su string/tokens contra la lista de variantes verificadas en val_02 |
| **SC: deriva fuera de formato a T=1.5** | Los `sc_samples` mostraban "based", "i", "the" | No es bug sino hallazgo: nació la señal `frac_other` ([§6.10.2](06_Resultados_Experimentales.md)) |
| **Oracle de AURC invertido** | El oracle daba PEOR que el azar (imposible por definición) | Signo del oracle corregido; añadido control "oracle ≤ señal ≤ azar" |
| **Index misalignment en pandas** | NaNs silenciosos al combinar Series con índices distintos (la combinación daba valores absurdos) | Alinear por `image_filename` explícitamente antes de sumar ranks |
| **Capas 17/26 con KL colapsada** | Valores degenerados en el piloto (todas las imágenes iguales) | Congelar el análisis en capa 34 ([§7.4](07_Ablaciones_y_Analisis_Profundo.md)) |
| **`hidden_states[1]` con `max_new_tokens=1`** | IndexError en val_04 | Usar `hidden_states[0]` (prefill) — documentado en [§4.2](04_Arquitectura_Tecnica.md) |
| **Filas `kl_prompt_L34` con layer/τ desalineados** | Inconsistencia detectada al cargar el CSV | `src.evaluation.load_results()` las normaliza al cargar (fix pendiente en `src.inference`) |

**Lección meta:** cada uno de estos bugs habría sobrevivido a una lectura casual de los resultados (los números "parecían razonables"). Solo los controles estructurales (identidades matemáticas, azar exacto, oracle > azar, recomputación independiente) los hicieron visibles. Esa es la justificación práctica de la filosofía de [§12.1](#121-filosofía-de-verificación) — y una metodología que la tesis adopta de forma permanente.

---

[⬅️ 11 — Conclusiones y Próximos Pasos](11_Conclusiones_y_Proximos_Pasos.md) | [⬅️ Volver al Índice](01_Indice_General.md)
