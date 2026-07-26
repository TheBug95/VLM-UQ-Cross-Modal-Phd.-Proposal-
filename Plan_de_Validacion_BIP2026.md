# Plan de Validación Pre-Implementación — BIP 2026

**Propósito:** antes de escribir una sola línea del pipeline experimental, verificar
que **cada afirmación técnica** en la que se apoya el diseño es cierta, con código
ejecutable y fuentes autoritativas. La regla es simple: *nada se implementa sobre un
supuesto no verificado*.

**Fecha de verificación:** 22-jul-2026 · **Estado:** verificación documental completa;
scripts listos para correr (requieren GPU y licencia MedGemma para val_04/val_07).

---

## 0. Cómo se hizo la verificación y cómo usar este plan

Se lanzaron 6 frentes de verificación independientes, cada uno contrastando las
afirmaciones del diseño contra la fuente primaria correspondiente (código fuente de
`transformers`, tokenizer real, repositorio HF del dataset, papers originales,
documentación de scipy/sklearn). Cada afirmación recibió un veredicto:

| Veredicto | Significado |
|---|---|
| ✅ **Verificada** | Confirmada contra fuente primaria; se puede implementar tal cual |
| 🔧 **Corregida** | La idea era correcta pero los detalles no; la Sección 2 da la versión correcta |
| ❌ **Refutada** | Era falsa; hay que cambiar el diseño o el discurso del paper |
| ⏳ **Pendiente** | Solo verificable ejecutando (GPU / licencia); cubierta por un script val_* |

La **Sección 3** describe la batería de scripts `Codigo/validacion/val_0*.py`. Cada
script imprime checks `[PASS]/[FAIL]` con criterios explícitos y sale con código 1 si
algo falla. El orden de ejecución importa: val_01 → val_05/val_06 (sin GPU) →
val_02/val_03 (sin GPU pero con red) → val_04 → val_07 (con GPU).

---

## 1. Matriz de verificación de afirmaciones

### Frente A — Arquitectura y tokenizer (MedGemma-4B / Gemma-3-4B)

| # | Afirmación del diseño | Veredicto | Evidencia |
|---|---|---|---|
| A1 | Decoder de 34 capas, hidden size 2560 | ✅ | config del modelo (technical report Gemma 3) |
| A2 | SigLIP-400M: 27 capas, 1152-dim | ✅ | config `vision_config` |
| A3 | Imagen se procesa a 896×896, normalización mean/std=0.5 (rango [-1,1]) y **el processor ya lo hace** | ✅ | `image_processing_siglip.py` — no normalizar manualmente |
| A4 | Pooling AvgPool2d 4×4 → **256 tokens** de imagen, proyección 1152→2560 | ✅ | `modeling_gemma3.py`; 64×64=4096 patches / 16 = 256 |
| A5 | `len(tokenizer)` = 262.144; `config.vocab_size` = 262.208 | ✅ | tokenizer real inspeccionado |
| A6 | IDs: `<image_soft_token>`=262144 (= `config.image_token_index`), `<start_of_image>`=255999, `<end_of_image>`=256000 | ✅ | tokenizer real inspeccionado |
| A7 | yes/no: como el chat template termina en `<start_of_turn>model\n`, el primer token de respuesta **no lleva espacio inicial** → primarios `yes`=4443, `no`=1904; alternativos `Yes`=10784, `No`=3771, `▁yes`=11262, `▁no`=951 | ✅ (verificar empíricamente en val_02) | tokenizer real + chat template |
| A8 | El system prompt se pliega en el primer turno de usuario (no hay rol system) | ✅ | chat template de Gemma 3 |
| A9 | `attention_dropout=0.0` → la ablación MC-Dropout es **inaplicable** | ✅ | config — ya reflejado en la definición |
| A10 | Los hidden states del SigLIP **no** están accesibles vía `generate()` | ✅ | `modeling_gemma3.py` v4.51.3 (issue #42759 / PR #44952). No afecta: ambos lados de la KL viven en el decoder LM |
| A11 | Todo lo anterior es idéntico en el repo gated `google/medgemma-4b-it` | ⏳ 99% | verificado contra espejo `unsloth/gemma-3-4b-it` (mismos pesos/tokenizer); residual 1% se cierra con val_02 tras aceptar licencia |

### Frente B — API de `generate()` (transformers ≥ 4.51.3)

| # | Afirmación | Veredicto | Evidencia |
|---|---|---|---|
| B1 | `outputs.hidden_states` es tupla **por paso de generación**; con `max_new_tokens=1` solo existe `[0]` (prefill) | ✅ | código fuente `generation/utils.py` — **esto corrige lo que decía el AGENTS.md** |
| B2 | El hidden state que condiciona el primer token yes/no es `hidden_states[0][capa][:, -1, :]` (última posición del prefill) | ✅ | consecuencia directa de B1 |
| B3 | `hidden_states[1]` solo existe con `max_new_tokens≥2` (estado *tras* generar el token) | ✅ | queda como ablación, no como primario |
| B4 | Cada tupla por paso tiene 35 entradas: índice 0 = embeddings (escalados por √2560), 1–34 = capas | ✅ | `modeling_gemma3.py` — **no mezclar índice 0 con capas en la KL** |
| B5 | `scores[0]` shape (batch, 262208); en greedy puro = logits crudos (equivalente a forward manual); alternativa `output_logits=True` | ✅ | `generation/utils.py` |
| B6 | VRAM real: ~10–12 GB bf16 / ~4–5 GB NF4 (pesos 8.6 GB) — cifras anteriores "16 GB / 6–8 GB" eran sobreestimadas | 🔧 | medición + cálculo; implica: T4 16 GB va justo en bf16 → recomendar 4-bit en Colab T4, bf16 en L4/A100 |
| B7 | Fijar `transformers>=4.51.3`, `attn_implementation`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 explícito, batch=1 | ✅ | reproducibilidad numérica |

### Frente C — Matemáticas y estadística

| # | Afirmación | Veredicto | Evidencia |
|---|---|---|---|
| C1 | `F.kl_div(log_q, p, reduction='batchmean')` = KL(p‖q); el orden de argumentos decide la dirección | ✅ | docs PyTorch + verificación numérica en val_05 |
| C2 | `scipy.jensenshannon` devuelve **distancia** (√JSD), hay que elevar al cuadrado; cota ln 2 | ✅ | docs SciPy — un error aquí reporta JSDs 40% menores |
| C3 | Mann-Whitney: efecto r = \|Z\|/√N, N = n₁+n₂ | 🔧 **corregida 23-jul-2026** | `mannwhitneyu` **no** expone `zstatistic` (falló en scipy 1.16.3 del usuario): el Z se calcula manualmente con corrección de continuidad y empates — fórmula verificada, reproduce el p-value de scipy con error < 1e-10 (val_06); Fritz et al. 2012 |
| C4 | AUROC/AUPRC aceptan scores continuos (KL cruda OK); Brier exige probabilidades calibradas y sklearn≥1.7 (`y_proba`) | ✅ | docs sklearn |
| C5 | DeLong **no** está en sklearn → paquete `pauc` o R `pROC` | ✅ | val_06 lo detecta |
| C6 | Bootstrap: 9.999 remuestreos, **preferir BCa** sobre percentile con N pequeño | 🔧 | `scipy.stats.bootstrap(method='BCa')` |
| C7 | p-value asintótico de `spearmanr` no fiable con n=69 → `scipy.stats.permutation_test`; Kendall tau-b como sensibilidad | 🔧 | docs SciPy (fiable solo n>500) |
| C8 | Shapiro-Wilk solo diagnóstico (p dudoso n>5000) | ✅ | no se usa para decisiones |

### Frente D — Dataset y licencias

| # | Afirmación | Veredicto | Evidencia |
|---|---|---|---|
| D1 | `load_dataset("TheBug95/MM-ODIR-129")` **sí funciona** (imagefolder + `metadata.jsonl` por split); el visor web roto es fallo de conversión parquet del servidor | ❌→✅ **refuta mi análisis anterior** | verificado ejecutando `datasets==5.0.0` |
| D2 | Conteos 77/26/26 = 129; 60 Normal / 69 Pathological | ✅ **confirmado ejecutando val_03 (22-jul-2026)** | splits exactos; imágenes 2048×1536 RGB decodifican con PIL |
| D2b | El grado de severidad existe pero se llama **`cup_to_disc_ratio`** y es **ordinal entero 0–4** (no continuo), no nulo en 70 muestras (los 69 Pathological + 1 Normal) | ✅ confirmado ejecutando | H4 corre sobre esos 69 con Spearman; ya mapeado en AGENTS.md y val_03 |
| D2c | No hay columna `patient_id` en `metadata.jsonl` | ⚠️ | viene de `split.json`; verificar manualmente ausencia de solape de pacientes entre splits |
| D3 | `split.json` expone `doctor_name` (**PII del anotador**) → no redistribuir en ningún artefacto | ✅⚠️ | inspección del repo |
| D4 | ODIR-5K **no tiene licencia formal publicada** → citar Li et al. 2021 (arXiv:2102.07978, DOI 10.1007/978-3-030-71058-3_11) + URL del challenge, sin afirmar licencia permisiva | 🔧 | búsqueda en el repo/challenge |
| D5 | HAI-DEF (MedGemma): gated con aprobación automática; **no** restringe publicar benchmarks de investigación; incluir disclaimer "not for clinical use" y citar Sellergren et al. arXiv:2507.05201 | ✅ | términos HAI-DEF |
| D6 | Usar checkpoint MedGemma **v1.0.1** (fix del end-of-image token, 09-jul-2025) | ✅ | model card / changelog |

### Frente E — Literatura VLM-UQ (posicionamiento y números que irán al paper)

| # | Afirmación previa | Veredicto | Corrección |
|---|---|---|---|
| E1 | VIG-TUQ (mayo 2026) es single-pass | 🔧 | Es **arXiv:2605.27136**; su score JSD requiere **2º pass sin imagen** — solo su score de atención es single-pass. Nos favorece: nuestra KL sí es single-pass |
| E2 | UMPIRE es comparable directo | 🔧 | **arXiv:2602.24195**, multi-sample → no es single-pass; nos favorece en tabla de costos |
| E3 | VLM-UQBench | ✅ | arXiv:2602.09214 (600 muestras VizWiz, 9 métodos) |
| E4 | Expert-CFG (ICCV 2025) AUC>0.8 textual | ✅ | confirmado; es expert-in-the-loop con Phi-3.5-Vision |
| E5 | "Between the Layers" reporta AUPRC ~0.82 | 🔧 **importante** | **arXiv:2603.22299** (Badash, Belinkov, Freiman): el 0.82 es el promedio **bajo cuantización 4-bit**, no in-distribution (in-dist hay paridad o −1.4 a −1.8 pp vs. probing); el +2.86 es **solo en Llama-3.1-8B**; y el método **no es training-free** (LightGBM supervisado) → nuestra versión training-free es la diferenciación |
| E6 | VOLMO: MedGemma-4B glaucoma F1 63.74%, 27B 32.78%, DR 94.25% (4B) | ✅ | números exactos verificados |
| E7 | Uncertainty-Gated Glaucoma | ✅ | medRxiv DOI 10.64898/2026.04.17.26351127 (claim 100% sens. existe, con confounds) |
| E8 | Faltaban competidores directos | 🔧 | Añadir a related work: **Semantic Entropy Probes** (Kossen et al. 2024, arXiv:2406.15927) e **INSIDE** (Chen et al., ICLR 2024) |

### Frente F — Referencias clásicas y marco bio-inspirado

| # | Afirmación | Veredicto | Evidencia |
|---|---|---|---|
| F1 | Hendrycks & Gimpel ICLR 2017 (arXiv:1610.02136); Guo ICML 2017 (PMLR v70:1321–1330); Liu NeurIPS 2020 (33:21464–21475); Kuhn ICLR 2023 (arXiv:2302.09664); Farquhar Nature 630:625–630; Azaria & Mitchell Findings EMNLP 2023:967–976 | ✅ | volúmenes/páginas exactos |
| F2 | DeLong Biometrics 44(3):837–845; McGurk & MacDonald Nature 264:746–748; Botvinick Psych Review 108(3):624–652; STARD-AI Nat Med 31(10):3283–3289 (ojo: Author Correction 13-jul-2026) | ✅ | verificados |
| F3 | "AUPRC más informativo que AUROC en desbalance" | 🔧 **reformular** | McDermott et al. NeurIPS 2024 (37:44102–44163, arXiv:2401.06091) muestran que la ventaja de AUPRC puede ser artefacto in-distribution → citar junto con Saito & Rehmsmeier y **reportar ambas métricas** |
| F4 | Cadena bio-inspirada defendible | 🔧 | McGurk (fenómeno) + **Ernst & Bülthoff 2004 TiCS 8(4):162–169** (integración Bayesiana) + Botvinick 2001 (monitoreo de conflicto) + **Yeung et al. 2004 Psych Review 111(4):931–959** (conflicto→detección de errores) |

---

## 2. Correcciones que ya se aplicaron / quedan pendientes

**Ya aplicadas en `Codigo/AGENTS.md` (esta sesión):**
1. §6.2 y §11: corregido el error crítico — el hidden state de respuesta sale de
   `hidden_states[0][capa+1][:, -1, :]` (prefill), no de `hidden_states[1]`
   (inexistente con `max_new_tokens=1`). p_text primario = última posición del prefill
   (mantiene el claim single-pass); la variante con `max_new_tokens=2` queda como ablación.
2. IDs verificados de tokens (imagen 262144/255999/256000; yes=4443, no=1904 y variantes).
3. Convención de indexación de capas: índice de tupla 1–34 (índice 0 = embeddings, excluido de la KL).

**Aplicadas tras la ejecución de la batería por el usuario (23-jul-2026):**
- **C3/Z de Mann-Whitney:** `mannwhitneyu` no expone `zstatistic` (falló en scipy 1.16.3) → Z manual con corrección de continuidad y empates, verificado contra el p-value de scipy (< 1e-10). Corregido en val_06 (PASS), Definición Paso 5, val_01 y requirements.txt.
- **KL=inf por massive activations (val_07):** las activaciones de capas tardías de Gemma (10²–10³) colapsan `softmax` float32 a ceros exactos (2559/2560) → `log(0)` → KL=inf. Corregido: KL/JSD siempre con `F.log_softmax` en float64 (verificado: KL finita, KL(p‖p)=0). Propagado a val_07, AGENTS.md (`src/uncertainty.py`) y la Definición (nota numérica en §2).
- **Confirmaciones empíricas de val_07 (parcial, antes del fix):** 256 tokens de imagen por muestra ✅, P(yes)+P(no)=1 ✅, 4.3 s/muestra → 258 inferencias ≈ 18.5 min ✅, VRAM pico **5.57 GB en 4-bit** ✅ (confirma la estimación B6 de ~4–5 GB NF4). Queda re-correr val_07 tras el fix para cerrar V-PIL-4/5.
- **Aviso benigno:** `[transformers] Deprecated: processor.image_token...` no nos afecta — la máscara usa `config.image_token_index`, no `processor.image_token`.

**Pendientes de propagar a los documentos narrativos** (decisión del usuario, §6):
4. VIG-TUQ = arXiv:2605.27136 y no es single-pass en su score JSD.
5. "Between the Layers" = arXiv:2603.22299: el 0.82 AUPRC es bajo 4-bit, no in-distribution.
6. Reformular el claim AUPRC-vs-AUROC según McDermott et al. 2024.
7. Añadir citas: Semantic Entropy Probes, INSIDE, Ernst & Bülthoff 2004, Yeung et al. 2004.
8. Licencia ODIR-5K: citar paper + URL sin afirmar licencia; advertencia PII de `doctor_name`.
9. VRAM realista: ~10–12 GB bf16 / ~4–5 GB NF4 (4-bit recomendado en T4).
10. Estadística: BCa en bootstrap; permutación para Spearman (n=69); JSD = jensenshannon².

---

## 3. Batería de scripts de validación (`Codigo/validacion/`)

Todos imprimen checks `[PASS]/[FAIL]` por afirmación y terminan con código de salida
0 (PASS) o 1 (FAIL). Orden recomendado: primero los que no necesitan GPU.

| Script | Qué valida | Requiere | Mapea a sanity check de la definición |
|---|---|---|---|
| `val_01_environment.py` | Versiones (transformers≥4.51.3, scipy≥1.11), GPU/VRAM, flags de determinismo | nada | prerrequisito de todos |
| `val_02_tokenizer.py` | IDs de imagen y yes/no, 256 tokens de imagen, chat template, plegado de system prompt | red (y licencia MedGemma, o espejo Gemma-3) | #3 (tokens de imagen), #4 (tokens yes/no) |
| `val_03_dataset.py` | load_dataset funciona, 77/26/26, campos, balance 60/69, imágenes decodifican, sin solape de pacientes | red | #1 (dataset cargable), base de H4 |
| `val_05_metrics.py` | Convención de `F.kl_div` (dirección), JSD = distancia², AUROC/AUPRC con scores continuos, Brier | nada | #5 (KL ≥ 0 y bien orientada) |
| `val_06_stats.py` | Z manual de Mann-Whitney (verificado contra scipy), permutación para Spearman, bootstrap BCa, disponibilidad de DeLong, potencia esperada con 60/69 | nada | #7 (análisis ejecutable) |
| `val_04_generate_api.py` | **El más crítico**: estructura real de `outputs.hidden_states` y `scores`, máscara de imagen, equivalencia scores↔forward, variante max_new_tokens=2, VRAM pico | GPU + modelo | #2 (hidden states), #3, #4 |
| `val_07_pilot.py` | Integración de extremo a extremo en 5 imágenes reales: extracción de u(x) en las 18 variantes, reproducibilidad (2 ejecuciones idénticas), tiempo/VRAM y proyección a 129×2 | GPU + modelo + dataset | #6 (pipeline completo), #8 (presupuesto) |

**Qué hacer si algo falla:**
- `val_01` FAIL → actualizar la librería indicada; no seguir.
- `val_02` FAIL → el tokenizer de MedGemma difiere del espejo: congelar los IDs
  *reales* que imprime el script en `src/inference.py` (para eso está el script).
- `val_04` FAIL en V-API-1/2 → la versión de transformers cambió la estructura:
  actualizar la convención de extracción y el AGENTS.md antes de seguir.
- `val_07` FAIL en reproducibilidad → exportar `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  fijar TF32 explícito y repetir; si persiste, documentar tolerancia en el paper.

---

## 4. Checklist Go/No-Go antes de implementar

- [ ] Licencia HAI-DEF aceptada en HF (repo `google/medgemma-4b-it` accesible)
- [ ] `val_01` PASS
- [ ] `val_02` PASS contra el repo gated (no solo el espejo)
- [x] `val_03` PASS — **ejecutado 22-jul-2026**: 77/26/26, 60 Normal/69 Pathological, `cup_to_disc_ratio` ordinal 0–4 confirmado (queda verificación manual de solape de pacientes vía `split.json`)
- [ ] `val_05` + `val_06` PASS
- [ ] `val_04` PASS (convenciones de extracción confirmadas en GPU real)
- [ ] `val_07` PASS con proyección de tiempo < 1 h para 258 inferencias
- [x] Correcciones de la Sección 2 propagadas a `Revision_Propuesta_BIP2026.md` y `Definicion_Experimental_Minima_BIP2026.md` (v2.1, 22-jul-2026; copia en raíz sincronizada)

Con los 8 ítems en verde, la implementación parte de **cero supuestos sin verificar**.

---

## 5. Decisiones que necesitan tu confirmación

1. **p_text primario** — Ya decidido por defecto y reflejado en AGENTS.md: última
   posición del prefill (`hidden_states[0][capa][:, -1, :]`), porque es el estado que
   *produce* la distribución yes/no y mantiene el claim "single-pass". La alternativa
   (estado tras generar el token, con `max_new_tokens=2`) queda como ablación.
   ¿Confirmas?
2. **Propagación de correcciones numéricas** (ítems 4–10 de la Sección 2) a
   `Revision_Propuesta_BIP2026.md` y `Definicion_Experimental_Minima_BIP2026.md`:
   ¿lo hago ahora o lo dejamos para la escritura del paper? (Recomiendo ahora: son
   pocos cambios y evitan que los documentos sigan citando números incorrectos.)

---

## 6. Fuentes primarias usadas en la verificación

- Código fuente `transformers` v4.51.3: `modeling_gemma3.py`, `generation/utils.py`
  (estructura de `GenerateDecoderOnlyOutput.hidden_states` / `scores`).
- Tokenizer y config reales de `unsloth/gemma-3-4b-it` (espejo del tokenizer de
  MedGemma-4B según technical report Gemma 3); `google/medgemma-4b-it` gated (401 sin licencia).
- Repositorio HF `TheBug95/MM-ODIR-129` (ejecutado con `datasets==5.0.0`).
- Documentación PyTorch (`F.kl_div`), SciPy (`jensenshannon`, `mannwhitneyu`,
  `spearmanr`, `permutation_test`, `bootstrap`), scikit-learn (`roc_auc_score`,
  `average_precision_score`, `brier_score_loss`).
- Papers: arXiv:2605.27136 (VIG-TUQ), 2602.24195 (UMPIRE), 2602.09214 (VLM-UQBench),
  2603.22299 (Between the Layers), 2406.15927 (Semantic Entropy Probes), 2401.06091
  (McDermott et al.), 2102.07978 (ODIR), 2507.05201 (MedGemma), 1610.02136
  (Hendrycks & Gimpel), 2302.09664 (Semantic Entropy), 264:746–748 Nature (McGurk),
  Biometrics 44(3):837–845 (DeLong), TiCS 8(4):162–169 (Ernst & Bülthoff),
  Psych Review 108(3):624–652 (Botvinick) y 111(4):931–959 (Yeung et al.).
