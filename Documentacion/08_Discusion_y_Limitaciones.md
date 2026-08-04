# 08 — Discusión y Limitaciones

> **Propósito:** interpretar qué significan los resultados, posicionarlos frente al estado del arte, declarar las contribuciones doctorales y — con la misma prominencia — las limitaciones, riesgos y pendientes. Este es un avance doctoral honesto, no un paper final.

[⬅️ 07 — Ablaciones y Análisis Profundo](07_Ablaciones_y_Analisis_Profundo.md) | [➡️ 09 — Dataset MM-ODIR-129](09_Dataset_MM_ODIR_129.md)

---

## 8.1 Interpretación teórica: ¿qué mide el desacuerdo cross-modal?

La lectura que proponemos es: $u(x) = D_{KL}(p_{text}\Vert p_{vis})$ mide **cuánto de lo que el modelo está a punto de afirmar no está respaldado por su propia representación visual del caso**. Cuando el modelo "ve" algo ambiguo pero "dice" algo seguro, la KL sube — y empíricamente eso coincide con el error (p = 0.006).

Conexión con calibración: MedGemma es un modelo **sobreconfiado** (entropy de respuesta mediana ≈ 0.001; confianza verbalizada degenerada en 95%). En un modelo así, los baselines de output están ciegos: el modelo nunca "duda" en voz alta. La KL cross-modal capta la tensión **que el output esconde**: baja entropy y alta KL simultáneas son la firma del error sobreconfiado. Esto explica también por qué la combinación por ranks funciona: MSP y KL son dos ventanas distintas (output e interna) sobre el mismo riesgo, y cada una ve errores que la otra no ve (Spearman = 0.27).

Lo que la señal **no** mide: la severidad de la enfermedad (H4 rechazada, [§11.2](11_Conclusiones_y_Proximos_Pasos.md)). La KL responde a la dificultad *del modelo con el caso*, no a la gravedad clínica: un glaucoma muy avanzado y "fácil de leer" produce KL baja; un copete fisiológico benigno pero ambiguo, KL alta. Para triage de errores del modelo es exactamente lo que se quiere; para estratificación clínica de severidad, no sirve — y lo reportamos como resultado negativo.

## 8.1b Posicionamiento frente al estado del arte

Contra la tabla de trabajos cercanos ([§2.4](02_Marco_Teorico.md)):

- **vs. NoLan / VCD / ConVis (decodificación contrastiva):** ellos contrastan la distribución de salida bajo **dos condiciones de entrada distintas** (multimodal vs. solo texto; original vs. distorsionada/reconstruida) para medir sesgo o alucinación → 2+ pasadas y una pregunta distinta. Nosotros medimos desacuerdo **intra-modelo en 1 pasada** para UQ. Son **complementarios, no competidores** — y nuestro costo es la mitad o menos.
- **vs. FUSE (procesos gaussianos + diversidad semántica):** misma filosofía training-free, pero ellos muestrean N respuestas y ajustan GPs sobre encoders congelados; nosotros hacemos un **cálculo cerrado en una inferencia**.
- **vs. Dropout Decoding (ensamble por máscaras de tokens visuales):** ellos **destruyen** información (enmascaran subconjuntos de tokens) para medir consenso entre corridas; nosotros **leemos la información completa una sola vez**.
- **vs. FairCLIP (Sinkhorn, equidad en glaucoma):** su crítica a la asimetría de KL se responde reportando **ambas direcciones siempre** ([§6.9b](06_Resultados_Experimentales.md)) + JSD como variante simétrica ([§7.1](07_Ablaciones_y_Analisis_Profundo.md)). Su objetivo (equidad) es ortogonal al nuestro (detección de errores); su dataset, **Harvard-FairVLMed** (10.000 imágenes SLO + notas clínicas), es el candidato natural para escalar la validación (Fase 2).
- **vs. Grad-CAM / Integrated Gradients:** descartados como señal UQ: requieren backward pass (rompen el framing single-pass 1×), los hidden states de un decoder-only no son "píxeles", y bfloat16/4-bit hace los gradientes inestables. Pertenecen a XAI (explicación), no a UQ (detección de errores) → future work.

## 8.2 Contribuciones originales a la tesis doctoral

Dos contribuciones verificadas como **sin precedente en la literatura** (análisis de novedad por especialista externo):

1. **Señal KL cross-modal intra-modelo:** primer método de UQ que es simultáneamente **single-pass, training-free y cross-modal**. Nadie antes extrajo la divergencia KL entre los hidden states de los tokens visuales y el estado textual del decoder de un VLM como señal de incertidumbre. Validación empírica: AUROC 0.661 [0.522, 0.772], p = 0.006, dominancia Pareto sobre baselines 2×/10×.
2. **Fusión parameter-free `rank(KL) + rank(1−MSP)`:** primera combinación por agregación de rangos de una señal del **espacio de representaciones internas** (KL cross-modal) con una del **espacio de salida** (MSP) para UQ en VLMs. La técnica genérica de rank fusion existe en IR (Cormack et al., 2009), pero esta instanciación es nueva. La complementariedad empírica (0.698 > 0.661 KL sola > 0.624 MSP solo; Excess-AURC 0.407 vs. 0.670/0.732) demuestra que ambos espacios aportan información ortogonal.

Ambas son **generalizables** por construcción: nada en la formulación ata la señal a MedGemma ni a glaucoma — se abre la puerta a LLaVA-Med, CogVLM y otras tareas médicas (roadmap en [§11.3](11_Conclusiones_y_Proximos_Pasos.md)).

## 8.3 Limitaciones honestas

| # | Limitación | Impacto |
|---|---|---|
| 1 | **N = 129 es pequeño** | IC anchos (±0.10–0.13); poder limitado para comparaciones pareadas |
| 2 | **AUROC 0.661 es moderado** | No suficiente para deployment clínico directo; útil como componente de triage, no como decisor |
| 3 | **Un solo modelo y un solo dataset** | MedGemma-4B + MM-ODIR-129; generalización a otros VLMs/tareas es conjetura |
| 4 | **H4 rechazada** | La señal no se correlaciona con severidad (rho = +0.001, p = 0.99): detecta errores, no gravedad |
| 5 | **Confirmación val+test débil** | AUROC 0.569 [0.200, 0.938] con solo 6 errores — estadísticamente vacía |
| 6 | **Maldición del ganador** | Monte Carlo CV anidado: 0.581 ± 0.116 (KL sola) — la generalización real está en ~0.58–0.66 |
| 7 | **SC evaluado en subconjunto de 50** | No comparable directamente con la cohorte; la comparación justa va etiquetada ([§6.10.3](06_Resultados_Experimentales.md)) |
| 8 | **Verbalized degenerada** | Solo 2 valores (90/95): posible limitación **fundamental** de VLMs instruction-tuned para auto-evaluarse |
| 9 | **Heatmaps sin validación cuantitativa** | No se midieron contra las segmentaciones ground truth de copa/disco |
| 10 | **KL winsorizada** | Techo $\ln(1/\varepsilon)=23.03$ recorta la cola (24/129 por encima de 23.0 nats en la variante ganadora, sin tocar el techo exacto; el techo exacto solo se satura en la dirección espejo, 61/129); los valores absolutos no son portables entre $\varepsilon$/hardware → derivación clínica **por percentil de cohorte, nunca por umbral absoluto** |
| 11 | **Softmax restringido** | $p_{yes}$ y MSP son scores binarios sobre 2 logits, no probabilidades calibradas de vocabulario completo (el SC reveló masa fuera de yes/no) |
| 12 | **Zona verde in-sample** | 30.2% sin errores en ESTA cohorte; en pacientes nuevos el error esperado en la zona es ≲ 1/39 (regla de tres, IC hasta ~7.7%) |
| 13 | **Prevalencia curada (~53% patológicos)** | No es prevalencia de screening (~6%): las métricas operativas no se trasladan directamente a población real |

## 8.4 Riesgos identificados

- **Massive activations:** riesgo de colapso numérico si se baja de float64 (la softmax degenera) — la regla float64 es dura ([§4.3](04_Arquitectura_Tecnica.md)).
- **Dependencia de versión:** `transformers >= 4.51.3` obligatorio (API de hidden states).
- **Sensibilidad a `epsilon`:** un eps distinto desplaza TODOS los valores de KL en una constante → mantener `epsilon` fijo entre corridas y no comparar nats absolutos entre corridas con eps distinto.
- **⏳ PENDIENTE OBLIGATORIO (diseño congelado):** el **análisis de robustez excluyendo imágenes con `has_annotation_artifact`** (p. ej., la flecha quemada de `1281_right.jpg`). Es barato (filtrar el CSV, no re-cómputo) y **debe reportarse antes de cualquier submission** — si la flecha funciona como cue espurio correlacionado con la etiqueta, podría inflar la accuracy base y contaminar la interpretación clínica de la KL ([§9.4](09_Dataset_MM_ODIR_129.md)).
- **Doble ciego:** la URL del dataset identifica al autor; en el PDF del paper se cita anonimizada ([§9.6](09_Dataset_MM_ODIR_129.md)).

- **Calibración con N pequeño:** ECE y correlaciones con ~13 obs/bin son ruidosas y Platt es monótona por construcción → la calibración ([§6.13](06_Resultados_Experimentales.md)) es evidencia **secundaria** (ECE + reliability diagram), nunca el claim principal.

## 8.5 Trabajo futuro (roadmap para la tesis)

1. **Escalar a datasets más grandes:** Harvard-FairVLMed (10.000 imgs), RIM-ONE, REFUGE, ORIGA — poder estadístico real y prevalencias diversas.
2. **Otros VLMs:** LLaVA-Med, CogVLM, GPT-4V, Gemini Pro Vision — ¿la señal es propiedad del mecanismo o del modelo?
3. **Gradient-weighted pooling** como alternativa a max pooling (con costo declarado: deja de ser 1× puro).
4. **Barrido fino de capas** 27–34 y early-exit (fases de signature maps del pipeline doctoral).
5. **Integración con triage clínico automatizado:** el esquema de 3 zonas ([§6.6](06_Resultados_Experimentales.md)) como protocolo operativo en clínica oftalmológica; estudio prospectivo.
6. **XAI como complemento (no como UQ):** Grad-CAM/IG sobre los casos de alta KL para explicación al oftalmólogo.

---

[⬅️ 07 — Ablaciones y Análisis Profundo](07_Ablaciones_y_Analisis_Profundo.md) | [➡️ 09 — Dataset MM-ODIR-129](09_Dataset_MM_ODIR_129.md)
