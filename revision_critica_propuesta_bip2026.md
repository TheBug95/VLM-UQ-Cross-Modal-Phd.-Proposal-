# Revisión Crítica de la Propuesta BIP 2026
## *"Cross-Modal Representation Disagreement as a Lightweight Uncertainty Signal for Glaucoma Detection in Medical Vision-Language Models"*

**Fecha de revisión:** 2026-07-17
**Alcance:** `Propuesta BIP 2026.md`, `BIP2026_Dossier_Maestro.md`, `bip2026_experimental_plan.md`, `bip2026_pilares_analysis.md`, y `research/` (dim01–dim10 vía dossier consolidado + verificación directa de claims críticos en dim03, dim04, dim07, dim08; cross_verification; insight; file_analysis).
**Verificación externa:** sitio oficial bipconference.org (2026-07-17).

---

## 0. Veredicto ejecutivo

La propuesta es **publicable en principio y está bien documentada**, pero tiene **3 problemas metodológicos de fondo (P0)** que un revisor competente detectaría de inmediato, **6 problemas graves de diseño experimental (P1)**, y varios **riesgos de logística/timeline (P2)** — entre ellos, que **la premisa de "2 semanas" posiblemente está mal fundamentada**: el deadline oficial de BIP 2026 **no está publicado** y el patrón histórico sugiere mediados de agosto.

Top 5 hallazgos:

1. **La KL propuesta no mide lo que el paper dice que mide.** El código (dim07) computa la KL sobre hidden states del *prefill* (solo tokens del prompt), y la respuesta generada por el modelo —"lo que el modelo dice"— **nunca entra al cálculo**.
2. **El softmax sobre la dimensión oculta (2560) no tiene semántica probabilística.** El paper inspirador se lo permite porque entrena un LightGBM supervisado encima; aquí se usa el escalar crudo, *training-free*, sin validación de que esa magnitud signifique algo.
3. **El deadline real probablemente NO es el 31 de julio.** La sección "Important dates" del sitio de BIP 2026 está vacía (verificado hoy) y el link de CMT aparece sin URL. BIP 2021 cerró el 20-ago y BIP 2022 el 15-ago. El 31-jul es un deadline **interno/autoimpuesto**. Confirmarlo puede cambiar todas las decisiones de alcance.
4. **El power analysis usa un N que no corresponde a ningún split real.** Los 207 casos de glaucoma son del split de *training*; el test oficial tiene 30–58 positivos, y un split propio 70/15/15 dejaría ~31 positivos en test. El plan estadístico tal cual no se sostiene.
5. **La novedad está amenazada por trabajos citados en la propia investigación** (VIG-TUQ, UMPIRE, VLM-UQBench) y el título promete "Models" en plural evaluando un solo modelo.

---

## 1. Problemas críticos de metodología (P0) — amenazan la validez científica

### P0-1. La señal no incluye la respuesta del modelo

El pipeline (dim07, pasos 4–6) extrae:

```python
prefill = outputs.hidden_states[0]        # solo el prefill
vision_repr = prefill[-1][:, :256, :]     # tokens visuales proyectados
text_repr   = prefill[-1][:, 256:, :]     # tokens DEL PROMPT
```

y luego genera la respuesta aparte (`max_new_tokens=5`) solo para etiquetar correcto/incorrecto. Es decir:

- El lado "texto" de la KL son los hidden states del **prompt** ("Does this image show glaucoma?"), que es **idéntico para todas las imágenes**. Lo único que lo hace variar entre muestras es el condicionamiento por la imagen vía cross-attention.
- Los tokens de la **respuesta generada** ("Yes"/"No", la explicación) — que es lo que el paper describe como "lo que el modelo *dice*"— no participan en la KL.

**Consecuencia:** la métrica compara "features de la imagen" contra "features del prompt condicionadas por la imagen", no "lo que el modelo ve vs. lo que el modelo dice". La narrativa del paper (y el título) queda desalineada con lo que se computa.

**Corrección sugerida (elegir una o ambas):**
1. Incluir los hidden states de los tokens **generados** (al menos el token de respuesta yes/no) en el lado textual.
2. Añadir como ablación una versión *output-space*: KL/JS entre la distribución next-token en la posición de respuesta con imagen vs. sin imagen (esto además conecta con VIG-TUQ, ver P0-4).

### P0-2. Softmax sobre la dimensión oculta ≠ distribución de probabilidad

El código hace `F.softmax(mean_pooled_hidden, dim=-1)`: un softmax sobre las 2560 componentes del vector de activaciones. Esas componentes:

- no son outcomes de una variable categórica (pueden ser negativas, tienen escalas arbitrarias por dimensión, y los LLMs tienen dimensiones con "massive activations" que dominarían el softmax);
- producen una "distribución" cuyo significado probabilístico es nulo.

"Between the Layers" usa el mismo truco, **pero con una diferencia crucial**: aplana la matriz L×L completa y entrena un **LightGBM supervisado** para predecir correctitud — el clasificador aprende qué dimensiones/patrones importan. La propuesta BIP usa el **escalar crudo sin aprendizaje**, de modo que toda la carga de validez recae en una heurística no validada. La propia dim03 advierte dos problemas relacionados que el plan experimental **no recoge**:

- el **modality gap** estructural (los espacios visión/texto están sistemáticamente separados en CLIP-like models): si la KL tiene un piso alto constante por construcción del modelo, todo parecerá "incierto" y la señal medirá arquitectura, no incertidumbre (dim03, §10);
- la temperatura del softmax puede reflejar "artefactos de escala más que verdadero desacuerdo semántico" (dim03, §6).

**Corrección sugerida:**
1. Ablación **logit-lens**: proyectar los pooled states por la matriz de unembedding (LM head) para obtener distribuciones **sobre el vocabulario** ("qué diría la imagen si hablara") y computar la KL ahí. Tiene semántica probabilística real y es barata.
2. Normalización contra el modality gap: reportar la KL relativa a la KL media del dataset (o de un conjunto de referencia), no el valor absoluto.
3. Control negativo: barajar etiquetas (label shuffle) y verificar que la KL **no** predice nada (sanity check del pipeline completo).

### P0-3. Los dos lados de la KL no son modalidades independientes

En Gemma 3 / MedGemma, los tokens de imagen tienen atención bidireccional y los tokens de texto atienden a los de imagen (dim02). En las capas profundas del decoder, los "text features" **ya absorbieron la información visual** — es el propósito de la arquitectura. Por tanto:

- KL(visión || texto) en capas tardías puede ser pequeña en todas partes (convergencia representacional), o reflejar diferencias de posición/profundidad/rol del token más que "desacuerdo".
- La interpretación del Insight 2 (dirección de la KL como diagnóstico del modo de fallo) se debilita: con representaciones contaminadas, la direccionalidad no tiene la lectura limpia que se propone.

**Mitigaciones:** ablación por capas ya existe (úsese también para mostrar *dónde* la señal emerge, no solo *cuál* es mejor); considerar extraer el lado visual desde la **salida del vision tower** (antes del decoder, vía `model.vision_tower()`) proyectada por el multimodal projector, en vez de los hidden states del decoder.

### P0-4. Novedad: el enemigo está citado en la propia carpeta

La propuesta afirma: *"nadie ha publicado cross-modal KL como señal de incertidumbre en MedVLMs para oftalmología"*. Pero la propia investigación (dim03) documenta:

- **VIG-TUQ** (may 2026): JS entre predicciones **con y sin imagen** + attention weights, training-free, single-pass, y demuestra que "predicciones correctas dependen más del input visual" — es conceptualmente el vecino más cercano (desacuerdo con/sin visión como señal de UQ).
- **UMPIRE** (feb 2026): training-free, incoherencia multimodal, reporta AUROC/ECE/AURAC.
- **VLM-UQBench** (feb 2026): tiene categoría explícita de *cross-modal data uncertainty*.

**Acciones:**
1. Reformular la novedad con precisión quirúrgica: *primera señal de UQ **feature-space** (hidden-state cross-modal divergence), single-pass y training-free, aplicada a diagnóstico oftalmológico en un MedVLM*. No reclamar más que eso.
2. **Incluir un baseline estilo VIG-TUQ** (JS entre logits de respuesta con/sin imagen). Si la KL propuesta no lo supera (o no lo iguala con menor costo), el paper pierde su razón de ser; si lo supera, es el resultado más fuerte del paper. De cualquier forma, omitirlo es una debilidad fatal de revisión.

### P0-5. El título dice "Models" (plural); el experimento evalúa uno

Con un solo MedVLM, el claim general sobre "Medical Vision-Language Models" es vulnerable (la propia sección 8.1 del plan admite dependencia de arquitectura). Opciones: (a) añadir un segundo modelo barato de correr con el mismo pipeline (LLaVA-Med-7B es el candidato natural); (b) cambiar el título a singular ("a Medical Vision-Language Model" / nombrar MedGemma); (c) presentar explícitamente el estudio como proof-of-concept single-model. Decidir antes de escribir el abstract.

---

## 2. Problemas graves de diseño experimental (P1)

### P1-1. El power analysis no corresponde a ningún split real

- Los **207 casos de glaucoma son del split de entrenamiento** público (3,500 pacientes). El test oficial tiene **30 (off-site) + 58 (on-site)** positivos.
- Un split propio 70/15/15 sobre los 207 dejaría **~31 positivos en test**, no 207.
- El plan afirma "con N≈200, ΔAUROC de 0.03–0.05 detectable con 80% power" — válido solo si los ~200 positivos están **en el conjunto de evaluación**.
- Además hay que **verificar si los labels de los test sets oficiales son públicos** (file_analysis dice "3,500 training cases released publicly"). Si no lo son, el universo evaluable con GT son los 207 pacientes de train (≈414 imágenes con keywords por ojo, o ~295 pacientes si se suman test sets con label público).

**Corrección:** definir el split ANTES de cualquier cómputo. Recomendación: método training-free ⇒ se puede evaluar sobre **todo el conjunto público etiquetado**, reservando un validation (p.ej. 20% de pacientes, estratificado por G) **solo** para seleccionar configuración (prompt, dirección, T) y calibración; el test se reporta sobre el 80% restante, o se hace repeated split ×5 y se reporta media±std. Rehacer el power analysis con el N real.

### P1-2. Unidad de análisis indefinida (imagen vs. paciente) — y los keywords por ojo resuelven lo que dim05 da por sentado

- dim05 afirma "los labels son por paciente; evaluar por imagen introduce label noise". **Pero dim04 documenta que el CSV trae `Left-Diagnostic-Keywords` y `Right-Diagnostic-Keywords` por ojo** — los labels a nivel imagen son derivables parseando esos keywords.
- El plan nunca define: ¿la unidad es la imagen o el paciente? Si es paciente, ¿cómo se agregan las dos KL (max? mean?)? No hay regla.
- Parsear keywords en texto libre ("glaucoma suspect", "suspicious glaucoma", etc.) es trabajo no presupuestado.

**Corrección:** unidad primaria = **imagen**, con labels derivados de keywords por ojo (reglas de parseo documentadas y auditadas a mano en una muestra); análisis de sensibilidad a nivel paciente (agregación por max KL de ambos ojos). Esto además aumenta el N de positivos.

### P1-3. Brier/ECE sin mapeo KL→probabilidad (riesgo de leakage)

La KL es no acotada; Brier y ECE exigen una probabilidad de error. El plan dice "aplicable si se normaliza a [0,1]" sin protocolo. Min-max o cualquier ajuste hecho **sobre el test set** contamina la evaluación. **Corrección:** Platt/logistic fit **solo en validation** (ver P1-1), o descartar Brier/ECE para la señal UQ y reportar AUROC/AUPRC + curvas accuracy-coverage (AURAC), que no requieren calibración. Con 2 semanas, lo segundo es más seguro.

### P1-4. El set de 9 baselines es irrealizable en 2 semanas — y dos son inviables en absoluto

| Baseline | Problema |
|---|---|
| MC-Dropout | Gemma 3/MedGemma **no tiene dropout activo** por defecto; el plan lo admite pero lo mantiene como "upper bound teórico" — un revisor objetará un baseline no funcional. |
| Semantic Entropy | 10 muestras × clustering NLI × ~7k imágenes: días de GPU, no horas. |
| SAPLMA probe | Requiere labels de correctitud (derivadas del propio modelo) y disciplina estricta train/val/test; riesgo de leakage y de circularidad; con ~200 positivos, probe frágil. |
| Human reader study (n=3 oftalmólogos) | Inviable en 2 semanas y con implicaciones éticas; ya está (correctamente) en la versión journal. Descópeselo del paper. |
| Mahalanobis | Requiere ajustar Gaussianas clase-condicional sobre el train split: ya no es "training-free" y añade un pase extra por todo train. |

**Set mínimo recomendado (4–5, todos baratos y defendibles):** MSP (prob. del token de respuesta), entropía predictiva, TS+entropía, **JS con/sin imagen (estilo VIG-TUQ — obligatorio por P0-4)**, y opcionalmente energy score. SE puede correr en una **submuestra** (p.ej. 500–1000 imágenes) como referencia de techo, declarándolo así.

### P1-5. No hay protocolo de extracción de respuesta

Nada en la carpeta define cómo se obtiene el label "correcto/incorrecto" del modelo:

- ¿Decodificación libre y parseo ("Yes, this image shows…")? Frágil: MedGemma puede responder con matices, disclaimers o **negarse** ("I cannot provide a diagnosis"). ¿Las respuestas no comprometidas cuentan como incorrectas, se excluyen, son una tercera clase? (La tasa de no-compromiso es en sí un dato interesante de UQ: repórtese.)
- **Recomendación fuerte:** constrained/forced decoding binario — comparar logits de "Yes" vs "No" en la primera posición de respuesta. Esto (a) da p(yes) limpia, que hace bien definidos MSP/entropía/Brier del diagnóstico; (b) elimina el parseo; (c) da el punto exacto donde medir las distribuciones output-space de P0-1/P0-4. Es el cambio de mayor valor/hora de todo el plan.

### P1-6. Resultados pre-comprometidos (riesgo de HARKing)

La propuesta escribe el pipeline como: *"→ Sí (t-test p<0.05, AUPRC ~0.72)"*. Fijar el resultado esperado antes de correr nada es un sesgo que además contradice al plan (que designa **AUROC como métrica primaria**, no AUPRC, y Mann-Whitney, no t-test). **Corrección:** escribir media página de **pre-registro interno** antes del primer experimento: endpoint primario (p.ej. AUROC de detección de error de KL(vis‖text), última capa, mean pooling, T=1, prompt P1), comparaciones primarias (vs MSP y entropía), criterio de éxito (DeLong/bootstrap p<0.05) y qué se reporta aunque salga negativo. Con 6 prompts × 4 direcciones × 8 temperaturas × 5 capas hay ~48+ configuraciones: una primaria pre-registrada y el resto exploratorias con corrección BH (el plan ya la contempla — bien).

---

## 3. Riesgos de timeline y logística (P2)

### P2-1. El deadline oficial NO está confirmado — y probablemente es posterior al 31-jul

- Verificado hoy (2026-07-17) en bipconference.org: la sección **"Important dates:" existe pero está vacía**, y el CFP dice "Papers must be submitted through the following system link:" **sin link de CMT**.
- Histórico: BIP 2021 deadline **20-ago** (evento 4–5 nov); BIP 2022 deadline **15-ago** (evento 15–17 nov); BIP 2026 es 11–13 nov.
- La propia dim08 lo marcó como confianza MEDIA: "deadline inferred from user directive and historical cadence".

**Implicación:** toda la arquitectura de la propuesta ("post-hoc, training-free, single-pass, mínimo viable") está optimizada para una restricción de 2 semanas que podría ser ficticia. **Acción inmediata (hoy):** confirmar la fecha real por CMT/comité (sos del comité organizador). Si el deadline real es ~15-ago, se gana margen para: segundo modelo (P0-5), REFUGE, submuestra de SE, y un reader study mínimo informal.

### P2-2. La semana actual es GECCO 2026 (13–17 jul, San José)

El plan "2 semanas" no contabiliza que esta semana está consumida por GECCO (donde además tenés responsabilidades de comité/póster), ni que entre la compilación del dossier (23-jun) y hoy pasaron 3.5 semanas. **Re-baseline obligatorio:** ¿qué está hecho ya? ¿La licencia de MedGemma está aceptada? ¿ODIR-5K está descargado? ¿El "dataset local creado por mí" de la tabla de la propuesta existe ya, y con qué estructura exacta (labels por imagen o por paciente, qué split)? El cronograma debe partir del estado real, no del 23-jun.

### P2-3. Estimación de cómputo subestimada

"Forward pass: 1 día" es optimista: ~7,000 imágenes × (prefill + generación + hidden states) a ~2–4 s/img ≈ **4–8 h solo para el método propuesto**; los baselines multi-pass y las ablaciones (48 configuraciones, aunque reusan activaciones si se cachean bien) multiplican. Con colas, debugging del bug de hidden states (confianza MEDIA, dim02/cross-verif) y reruns, presupuestar **2–4 días de GPU**, no 1. Mitigación: cachear logits + pooled hidden states una sola vez por imagen y derivar todas las variantes offline.

### P2-4. REFUGE y RIM-ONE: descope recomendado

REFUGE requiere registro/DUA (días de espera) y RIM-ONE v3 tiene 159 imágenes (CIs inmanejables). Con 2 semanas: dejar ambos como "future work" o un único experimento cross-dataset **solo si** el deadline real resulta ser agosto (ver P2-1). Nótese que REFUGE tiene GT clínico mucho más sólido que ODIR (cuyo label de glaucoma viene de keywords diagnósticos, sin IOP/OCT/campo visual) — si algún día se expande, REFUGE es la prioridad #1.

### P2-5. Pendientes administrativos sin fecha

Licencia Health AI Developer Foundations de MedGemma (aceptar hoy — ya en el dossier, confirmar que se hizo); términos de uso de ODIR-5K; cuenta CMT; repo anónimo en anonymous.4open.science; y verificar que los términos de MedGemma permiten el uso y la publicación con el framing clínico previsto (incluir disclaimer "no es dispositivo médico / no validado clínicamente").

---

## 4. Puntos no tenidos en cuenta (checklist de huecos)

1. **Confound de calidad de imagen (el favorito de los revisores):** imágenes borrosas/oscuras pueden causar *simultáneamente* KL alta y error → la "UQ" sería solo un detector de blur. El Insight 4 lo menciona, pero el plan experimental no lo convierte en control. **Hacer:** correlación KL vs. varianza Laplaciana (blur), y AUROC de detección de error **dentro del estrato de imágenes de buena calidad**.
2. **Control negativo por label-shuffle** (ver P0-2): 30 minutos de trabajo, blinda el pipeline contra acusaciones de artefacto.
3. **Ablación logit-lens** (KL en espacio de vocabulario vía LM head) — semántica real, casi gratis (ver P0-2).
4. **Normalización por modality gap** (ver P0-2/dim03 §10).
5. **Efecto de cuantización 4-bit vs bf16 en la KL**: si se usa 4-bit por VRAM, medir estabilidad de la KL en una submuestra (el paper inspirador reporta robustez — citarlo).
6. **Análisis de label-noise como hallazgo:** los casos "incorrect + high KL" pueden contener **errores de anotación de ODIR** (labels por keywords, glaucoma unilateral, etc.). Inspeccionar manualmente el top-10 puede convertir una limitación en un resultado narrativo fuerte ("la KL detecta también ruido de etiquetas").
7. **DeLong en Python es incómodo** (pROC es R): presupuestar implementación propia o usar **bootstrap pareado** para la diferencia de AUROC (más simple y defendible).
8. **No confundir AUROC diagnóstico con AUROC de detección de error:** las tablas de SOTA de dim05 (0.79–0.98) son diagnósticas; la contribución es UQ. Mezclarlos en una tabla sería un error fatal de presentación.
9. **Inconsistencias internas entre documentos** (resolver antes de escribir el paper):
   - Métrica primaria: propuesta dice AUPRC (~0.72), plan dice AUROC.
   - Test estadístico: propuesta dice t-test, plan dice Mann-Whitney (las KL son asimétricas/sesgadas ⇒ Mann-Whitney).
   - "MM-ODIR": no existe como dataset separado (dim04) ⇒ usar "ODIR-5K" en el paper.
   - Dossier recomienda "P4 + P1" como si fuera un prompt; son dos ⇒ definir UN prompt primario.
   - Conteo del dataset: dossier/dim04 dicen ~7,000 imágenes; file_analysis y dim04-cuerpo dicen ~10,000 archivos (5,000 pacientes × 2 ojos) ⇒ fijar cifras exactas.
   - `file_analysis.md` tiene specs erróneas de MedGemma (32 capas, hidden 4096 — lo correcto es 34 capas, hidden 2560 según dim02) y nombra mal la conferencia ("Biomedical Imaging and Processing") ⇒ marcarlo como **superseded** para no citarlo por accidente.
10. **Verificar existencia real de TODAS las referencias** antes de subir: varias citas del plan llevan arXiv IDs de 2026 generados durante la investigación automatizada (p.ej. 2603.22299, 2602.09214, 2605.27136, 2603.23953, 2604.08261, 2606.06959). Una referencia alucinada en un paper double-blind es riesgo reputacional serio. Pasar cada [^n] por búsqueda real (DBLP/Scholar/arXiv).
11. **Limpieza de la carpeta `research/`**: hay ~25 archivos `test*.txt` (test.txt, test100.txt, test_chain*.txt, etc.) y un `desktop.ini` de pruebas de tooling del 23-jun. Eliminarlos antes de compartir la carpeta con el tutor o archivarla.
12. **Criterio go/no-go explícito** (no existe en ningún documento): definir qué resultado mínimo hace publicable el paper. Propuesta: *la KL cross-modal supera a MSP y a entropía en AUROC de detección de error con CI bootstrap no solapado, y se mantiene dentro del estrato de imágenes de buena calidad*. Si a la semana 1 el piloto de 50–100 imágenes no muestra separación alguna, pivotar a la variante output-space (P0-1 op. 2) en vez de forzar la feature-space.

---

## 5. Lo que YA está bien cubierto (no repetir trabajo)

- Bug de `output_hidden_states` del vision tower + workaround documentado (dim02).
- Patient-level split para evitar leakage (dim04/dim10).
- Cumplimiento double-blind: repo anónimo, no arXiv pre-notificación, sin logos institucionales (dim08).
- Ethics statement para dataset público de-identificado (dim08).
- Framing bio-inspirado sólido (McGurk, integración multisensorial, dim03/dim08) — genuinamente bien argumentado para el scope de BIP.
- Tabla de costos computacionales como argumento de impacto Global South (Insight 6).
- Checklist de reproducibilidad + STARD-AI (sección 9 del plan).
- Ablación de dirección KL/JSD resuelta correctamente (cross-verif #13).
- Reportar AUROC **y** AUPRC dado el debate NeurIPS 2024 (cross-verif #9).

---

## 6. Plan de acción recomendado (re-baseline a 14 días)

**Día 0 (hoy, 17-jul):** Confirmar deadline real (CMT/comité). Aceptar licencia MedGemma si no está hecho. Re-baseline de estado (¿dataset? ¿código?). Escribir pre-registro de 1 página (P1-6).

**Días 1–2:** Pipeline corregido: constrained decoding yes/no + hidden states de **prefill + tokens generados** + cacheo de activaciones. Piloto en 50–100 imágenes: gate go/no-go (P4-12). Implementar KL feature-space + variante logit-lens + JS con/sin imagen.

**Días 3–5:** Corrida completa ODIR-5K (split según P1-1). Baselines mínimos (P1-4). Control label-shuffle + métrica de blur.

**Días 6–7:** Ablaciones acotadas (dirección KL, capa, logit-lens, prompt ×2–3 máx). Análisis de calidad de imagen y de cuadrantes. Inspección manual top-10 label-noise.

**Días 8–9:** Estadística (bootstrap pareado / DeLong, CIs, effect sizes, BH). Figuras: boxplot KL, PR/ROC de detección de error, accuracy-coverage, cuadrantes.

**Días 10–13:** Redacción 6–8 págs IEEE (estructura ya bosquejada en la propuesta), verificación de referencias (P4-10), formato double-blind, repo anónimo, ethics statement.

**Día 14:** Buffer + submission.

**Si el deadline real resulta ser ~15-ago:** añadir, en este orden: (1) segundo MedVLM (P0-5), (2) REFUGE cross-dataset, (3) SE en submuestra como techo, (4) reader study informal con 1–2 oftalmólogos como análisis cualitativo.

---

*Revisión basada en la totalidad de los archivos de la carpeta BIP 2026 (verificados el 2026-07-17) y verificación externa del sitio oficial de BIP 2026. Los archivos dim01, dim02, dim05, dim06, dim09, dim10 fueron revisados a través de su consolidación en el Dossier Maestro y la cross-verificación; los claims críticos se verificaron directamente en dim03, dim04, dim07 y dim08.*
