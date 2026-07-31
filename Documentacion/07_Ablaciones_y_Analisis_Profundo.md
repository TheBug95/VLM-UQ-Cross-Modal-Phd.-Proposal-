# 07 — Ablaciones y Análisis Profundo

> **Propósito:** responder, una por una, las preguntas de diseño que el experimento puso a prueba: tipo de divergencia, estrategia de pooling, temperatura, capa, prompt, y por qué los baselines multi-pass fallan. Todos los números provienen de la corrida completa (129 imágenes, prompt P1 salvo indicación; fuente: `figures/tabla_t2_ablaciones.csv` y `Reporte_Experimento_BIP2026.md` §8).

[⬅️ 06 — Resultados Experimentales](06_Resultados_Experimentales.md) | [➡️ 08 — Discusión y Limitaciones](08_Discusion_y_Limitaciones.md)

---

## 7.1 Efecto del tipo de divergencia: KL vs. JSD vs. Coseno

| Tipo | Mejor AUROC (129) | Lectura |
|---|---|---|
| **kl_t_v** (texto‖imagen) | **0.661** (max, τ=1) | Ganadora: dirección de la alucinación |
| kl_v_t (imagen‖texto) | 0.566 (max, τ=1) | Espejo: riqueza trivial imagen>texto — ruido |
| jsd | 0.572 (rollout, τ=1) | Simétrica pero **saturada** en su techo |
| cosine | 0.514–0.569 según pooling | Sobre vectores crudos; no aporta |
| kl_prompt | 0.483 | Peor que azar — la señal vive en la respuesta, no en el prompt |

**¿Por qué KL(t‖v) supera a KL(v‖t)?** Por la asimetría de la KL: $D_{KL}(p_{text}\Vert p_{vis})$ pondera las discrepancias por donde **el texto** pone probabilidad — si la respuesta afirma algo que la representación visual no respalda, el término $\ln(p_{text}/p_{vis})$ explota exactamente ahí (dirección de la alucinación). La dirección contraria pondera por donde la **imagen** pone masa, y una retinografía siempre contiene más contenido que un sí/no (vasos, mácula, periferia, ruido): esa dirección mide una constante estructural, no el error. Datos completos y la figura del espejo en [§6.9b](06_Resultados_Experimentales.md).

**Ablación exhaustiva de fusión de direcciones** (`results/analisis_fusion_todas.py`): 8 poolings × 3 τ × 6 esquemas de fusión (suma, max, min, asimetría, rank-suma, JSD). Resultados: **ninguna fusión supera a kl_t_v sola** (la mejor, rank-suma, llega a 0.649 < 0.661); añadir v→t a la combinación estrella siempre la **baja** (0.698 → 0.681); y "la fusión ayuda" solo en configuraciones donde kl_t_v ya era azar (~0.50) — es decir, rescata señales muertas, no mejora la viva.

**¿Por qué JSD no discrimina?** Porque las distribuciones $p_{vis}$ y $p_{text}$ son casi disjuntas (brecha modal): la JSD se satura en su techo de $\ln 2 \approx 0.693$ (mediana 0.693 con τ=1, ver Reporte §5.3). Una señal pegada al techo para casi todas las imágenes no puede rankear.

**Coseno** opera sobre los vectores crudos sin softmax: con massive activations, el ángulo entre vectores lo dominan unas pocas dimensiones extremas comunes a todos los casos → AUROC 0.42–0.57, sin aporte.

**Conclusión de la sección:** la señal del error vive específicamente en la **sorpresa dirigida** $D_{KL}(p_{text}\Vert p_{vis})$ — la asimetría de KL no es un defecto aquí, es el mecanismo.

---

## 7.2 Efecto de la estrategia de pooling (8 variantes)

AUROC (129 imgs, kl_t_v, τ=1; roi solo en las 69 con máscaras):

| Pooling | AUROC | Lectura |
|---|---|---|
| **max** ★ | **0.661** | Congelado en train (0.728) |
| rollout | 0.533 | Atención propagada: no alineada con la tarea |
| normw | 0.525 | Ponderar por norma no basta |
| mean | 0.516 | Dilución espacial confirmada |
| topk | 0.484 | Top-26 por norma: peor que azar en esta familia |
| attn | 0.479 | Cross-attention del último token: peor pooling deployable |
| headspec | 0.469 | Cabezas "visuales": la peor |
| roi (oracle) | **0.349** (n=69) | **No se reproduce el 0.889 del piloto** — era artefacto de muestra pequeña |

**¿Por qué max > mean?** El máximo elemento a elemento conserva las **activaciones pico**: si en algún token de imagen (el disco, una hemorragia) hay una activación extrema que tensiona con el texto, `max` la preserva; `mean` la diluye 256 veces. Es la respuesta empírica directa al riesgo de dilución espacial ([§2.5](02_Marco_Teorico.md)).

**¿Por qué roi (oracle) no ayuda?** Contraintuitivo e importante: darle al pooling la máscara exacta del disco **empeora** la señal en la corrida completa (0.349, n = 69). El 0.889 del piloto fue artefacto de muestra pequeña (no se reproduce). Hipótesis: la tensión informativa no vive solo en el disco — el contexto global del fundus (que `max` sí ve) aporta a la señal; y restringir el pooling a una región pequeña lo hace más ruidoso numéricamente. Queda como hallazgo a investigar en Fase 2.

**¿Por qué la atención (attn, rollout, headspec) rinde mal?** Porque la atención de un modelo **frozen, zero-shot** no está alineada con la tarea de glaucoma: los heatmaps ([§6.8](06_Resultados_Experimentales.md)) muestran atención difusa por el fondo. Ponderar por esa atención amplifica el ruido en vez de la región diagnóstica.

**¿Por qué topk y normw no superan a max?** La norma L2 del hidden state es un proxy imperfecto de "importancia": los tokens de norma alta no son necesariamente los diagnósticos (de hecho las *massive activations* de Gemma producen tokens con norma gigante sistemáticamente). Promediar los 26 de mayor norma (topk) o ponderar por norma (normw) mezcla señal y artefacto; el máximo por dimensión evita promediar.

---

## 7.3 Efecto de la temperatura τ

Ejemplo con la familia `kl_t_v` + max:

| τ | AUROC (129) | Mediana KL (nats) | Efecto |
|---|---|---|---|
| **1** | **0.661** | ~22.8 | La que mejor funciona |
| 2 | 0.578 | ~10.0 | Aplana ~a la mitad: pierde contraste |
| 4 | 0.525 | ~0.8 | Aplana demasiado: distribución casi uniforme, sin discriminación |

τ = 1 gana siempre, en todas las familias y poolings. Interpretación: la temperatura > 1 homogeniza las distribuciones, y al homogenizar desaparece precisamente la estructura que distingue "texto anclado" de "texto alucinado". El diseño probó τ > 1 como remedio a la brecha modal; los datos dijeron que la brecha **es** la señal, no el problema.

---

## 7.4 Efecto de la capa

Solo la **capa 34** (la última) produce señal útil. En las capas 17 y 26 la KL **colapsa**: valores degenerados, casi idénticos para todas las imágenes (detectado en el piloto; el esquema `results_full.csv` quedó congelado con `layer = 34` únicamente).

**Interpretación:** en las capas intermedias las representaciones de imagen y texto aún están en modo "procesamiento genérico" — no han formado la decisión. En la última capa, el estado textual ya está orientado a responder la pregunta clínica y el contraste con la evidencia visual se hace visible. Este hallazgo conecta con la literatura de probing (SAPLMA y sigs.: las representaciones de "verdad" viven en capas medias-tardías), y deja una línea de future work: barrido fino de capas 27–34 y early-exit ([§8.5](08_Discusion_y_Limitaciones.md)).

---

## 7.5 Efecto del prompt

| Señal | P1 (simple) | P4 (system: oftalmólogo experto) |
|---|---|---|
| KL cross-modal | **0.661** | 0.614 |
| 1−MSP | 0.624 | **0.629** |
| Combinación | **0.698** | 0.654 |

**P1 gana en todo** (con la excepción marginal de MSP). Interpretación de trabajo: el system prompt de experto **"alinea" las representaciones** — condiciona al decoder a comportarse como oftalmólogo, homogenizando el estado textual y **reduciendo el desacuerdo cross-modal** que es nuestra señal. Paradoja útil para el paper: hacer el prompt más "profesional" hace al modelo algo mejor calibrado en output (MSP sube) pero **menos transparente internamente** (KL baja). Para UQ cross-modal, el prompt ingenuo es preferible.

---

## 7.6 Señal KL de prompt (imagen vs. texto del prompt)

`kl_prompt_L34`: KL entre los tokens de imagen y los tokens **del prompt textual** (no de la respuesta). AUROC = **0.483 — peor que azar**.

**Interpretación (control negativo valioso):** la señal útil no es "cualquier" divergencia imagen-texto. El prompt es idéntico para todas las imágenes ("Does this fundus image show glaucoma?..."), así que su divergencia con la imagen mide sobre todo variación visual entre casos — no el estado de decisión del modelo. La señal vive en la **respuesta que el modelo está a punto de emitir** (`p_text`), no en la pregunta. Este control blinda la interpretación de H1: no es que "KL alta = imagen rara", sino "KL alta = respuesta desanclada".

---

## 7.7 Análisis de baselines multi-pass vs. single-pass

**¿Por qué falla Verbalized Confidence?** Dos razones verificadas: (a) **degeneración de la escala** — solo declara 90 (n=11) y 95 (n=118); una señal con 2 valores no puede rankear 129 casos (AUROC 0.519); (b) **sobreconfianza instruction-tuned** — el alignment enseña al modelo a sonar seguro: dice 95% donde acierta 80.5%. La confianza verbal no está anclada en el estado interno real del modelo. Detalles y figura en [§6.10.1](06_Resultados_Experimentales.md).

**¿Por qué falla Self-Consistency?** A T=1.5 MedGemma **deriva fuera del formato** ("based", "i", "the") en la mayoría de las muestras — la entropía sí/no muere (casi nunca dice "no") y la señal informativa termina siendo accidental: `frac_other` mide inestabilidad generativa, no duda clínica (AUROC 0.655, p = 0.054). Y cuesta 10× por imagen.

**La lección conjunta:** la señal cross-modal (interna) captura información que la señal de output (logits, texto generado, confianza declarada) **no puede ver**, porque el desacuerdo entre modalidades ocurre **antes de la generación** — en los hidden states, no en lo que el modelo "dice". Los métodos de output solo observan el resultado final de un proceso de colapso a un token; nosotros observamos el proceso. Es la versión empírica del argumento central de la tesis.

---

[⬅️ 06 — Resultados Experimentales](06_Resultados_Experimentales.md) | [➡️ 08 — Discusión y Limitaciones](08_Discusion_y_Limitaciones.md)
