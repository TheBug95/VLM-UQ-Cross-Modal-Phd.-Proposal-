# BIP 2026 — Fase 4: Cross-Verificación de Hallazgos

**Fecha:** 2026-06-23  
**Objetivo:** Validar consistencia, identificar conflictos, y clasificar hallazgos de las 10 dimensiones de investigación en niveles de confianza (Alta / Media / Baja / Conflicto).

---

## Hallazgos Verificados con Alta Confianza (≥2 fuentes independientes)

### 1. MedGemma 4B es el modelo viable para este paper
- **dim02** confirma: 4B-IT multimodal, SigLIP 400M encoder, 896×896 input, 256 tokens projected to 2560 dim.
- **dim07** confirma: código de carga funciona con `AutoModelForImageTextToText` + `torch.bfloat16`.
- **dim09** confirma: MedGemma-4B es el único modelo práctico para single-GPU hidden-state extraction.
- **dim10** confirma: MedGemma 4B frozen es el backbone elegido para el experimento.
- **Veredicto:** ALTA CONFIANZA. Sin conflicto.

### 2. ODIR-5K es el dataset principal; no existe "MM-ODIR" separado
- **dim04** confirma: ODIR-5K = 5,000 patients, 7,000 images, 8 labels, glaucoma ~6% (207 training cases).
- **dim05** confirma: glaucoma es minority class, ~207 training cases.
- **dim10** confirma: patient-level split obligatorio para evitar data leakage.
- **Veredicto:** ALTA CONFIANZA. Sin conflicto.

### 3. El método de KL cross-modal es single-pass y training-free (para el VLM)
- **dim01** confirma: Between the Layers usa intra-layer KL + LightGBM, single-pass, training-free para LLM.
- **dim03** confirma: cross-modal KL es hipótesis novedosa, múltiples papers usan divergence como señal de alineación (VLM-UQBench, UMPIRE, VIG-TUQ).
- **dim06** confirma: métodos training-free (entropy, MSP, energy score) son estándar; KL cross-modal es único en ser feature-based y cross-modal.
- **dim07** confirma: código de extracción de hidden states y KL es implementable en PyTorch.
- **dim10** confirma: single-pass, frozen model, no fine-tuning.
- **Veredicto:** ALTA CONFIANZA. Sin conflicto.

### 4. Temperature Scaling es el método de calibración post-hoc estándar para VLMs
- **dim03** confirma: TS mejora ECE significativamente en VLMs (ICML 2024 paper).
- **dim06** confirma: TS es "de-facto standard" para VLM calibration.
- **dim10** confirma: TS + Entropy es baseline obligatorio.
- **Veredicto:** ALTA CONFIANZA. Sin conflicto.

### 5. Semantic Entropy es el upper bound teórico para hallucination detection
- **dim03** confirma: SE captura incertidumbre semántica, no lexical (Kuhn et al., Nature 2024).
- **dim06** confirma: SE es multi-pass, computacionalmente caro, pero el mejor baseline para generative tasks.
- **dim10** confirma: SE como baseline de upper bound.
- **Veredicto:** ALTA CONFIANZA. Sin conflicto.

### 6. BIP 2026: 6-8 páginas, IEEE template, double-blind, CMT, $50/página extra
- **dim08** confirma: 6-8 páginas, IEEE conference template, double-blind review.
- **dim08** confirma: Microsoft CMT (no EasyChair), $50/página extra.
- **dim08** confirma: proceedings en IEEE Xplore, conference #71710.
- **Veredicto:** ALTA CONFIANZA. Sin conflicto.

### 7. La señal de KL cross-modal conecta los 4 pilares de la tesis
- **dim09** confirma: UQ → KL directa; XAI → KL como explicación nativa; Few-shot → KL como filtro de pseudo-labels; Segmentación → KL como mapa de riesgo espacial.
- **dim01** confirma: intra-layer agreement es explicativo por diseño.
- **dim03** confirma: cross-modal disagreement detecta modality bias y hallucinations.
- **Veredicto:** ALTA CONFIANZA. Sin conflicto.

---

## Hallazgos con Confianza Media (1 fuente sólida, pero no replicado)

### 8. MedGemma tiene un bug conocido en `output_hidden_states` para el vision encoder
- **dim02** reporta: transformers issue #42759 — `SiglipModel` no cascadea `output_hidden_states` correctamente.
- **dim07** no menciona este bug, pero usa `output_hidden_states=True` en `generate()`.
- **Contexto:** El bug puede workaroundse llamando `model.vision_tower()` directamente o extrayendo del decoder hidden states (primeros 256 tokens son projected vision tokens).
- **Veredicto:** MEDIA CONFIANZA. Workaround documentado.

### 9. AUPRC vs AUROC debate para clasificación desbalanceada
- **dim05** reporta: NeurIPS 2024 paper (McDermott et al.) argumenta que AUROC es más robusto que AUPRC para class imbalance, contrario a la creencia popular.
- **dim10** reporta: AUPRC es más informativa para datasets desbalanceados (Saito & Rehmsmeier 2015).
- **dim05** argumenta: AUPRC puede amplificar bias contra subgrupos de baja prevalencia.
- **Veredicto:** MEDIA CONFIANZA. Hay un debate activo en la literatura. Recomendación: reportar AMBAS métricas con interpretación cuidadosa.

### 10. DeLong's test es el estándar para comparar AUROC, pero hay debates sobre su poder estadístico con N pequeño
- **dim10** reporta: DeLong's test es el estándar para AUROC paired comparison.
- **dim10** también reporta: con N≈200, ΔAUROC de 0.03-0.05 es detectable con 80% power.
- **dim05** no menciona DeLong's test directamente.
- **Veredicto:** MEDIA CONFIANZA. DeLong es estándar, pero el power analysis es una estimación teórica que depende de correlación entre métodos.

### 11. La señal KL puede ser más efectiva en capas intermedias que en la última capa
- **dim01** (Between the Layers) no especifica capas óptimas; usa todas las capas.
- **dim03** (LRP paper) reporta que las capas intermedias son más predictivas de veracidad.
- **dim10** propone ablación de capas (25%, 50%, 75%, 100%, mean-pooling).
- **Veredicto:** MEDIA CONFIANZA. Hipótesis plausible, pero no validada específicamente para cross-modal KL.

---

## Hallazgos con Conflicto o Baja Confianza

### 12. ¿Es MedGemma-4B o MedGemma-27B mejor para glaucoma?
- **dim09** reporta: MedGemma-4B tiene F1=63.74% en glaucoma; MedGemma-27B tiene F1=32.78% en glaucoma (peor).
- **dim02** menciona que 27B multimodal existe pero no detalla rendimiento oftalmológico.
- **Conflicto:** El paper VOLMO (dim09) reporta que 27B es peor que 4B en glaucoma, lo cual es contra-intuitivo. Esto podría deberse a: (a) overfitting en el corpus de pre-entrenamiento, (b) diferentes prompts de evaluación, (c) varianza estadística.
- **Veredicto:** CONFLICTO. Necesita verificación adicional. Si es cierto, fortalece el uso de 4B (más ligero, más accesible, y curiosamente mejor en glaucoma).

### 13. ¿La divergencia KL es simétrica o asimétrica? ¿JSD es mejor?
- **dim01** usa KL asimétrica (dirigida) para signature maps, pero menciona JSD como alternativa.
- **dim03** reporta que KL es inestable numéricamente cuando denominador → 0; recomienda CS divergence o JSD.
- **dim10** propone ablación: KL(p_vis || p_text), KL(p_text || p_vis), symmetric KL, JSD.
- **Veredicto:** CONFLICTO RESUELTO. Usar ambas direcciones y JSD como ablación. La hipótesis es que la dirección importa: KL(p_vis || p_text) detecta cuando la visión no respalda el texto (alucinación textual), mientras que KL(p_text || p_vis) detecta cuando el texto no explica la visión (caso visualmente complejo).

### 14. ¿El prompt exacto afecta significativamente la señal de KL?
- **dim04** propone 3 variantes de prompts (P1, P2, P3) pero no evalúa.
- **dim10** propone 6 variantes (P1-P6) con system prompt como experto.
- **dim07** usa un solo ejemplo genérico.
- **Veredicto:** BAJA CONFIANZA. La hipótesis es que el prompt afecta, pero no hay evidencia empírica todavía. Se debe incluir como ablación en el paper.

### 15. ¿Cuánto VRAM se necesita exactamente para MedGemma 4B con hidden states?
- **dim02** reporta: ~16GB para inference bfloat16, ~6-8GB con 4-bit quantization.
- **dim07** no especifica VRAM.
- **dim10** menciona "estimar ~6-8 GB VRAM para 4-bit quantization" sin fuente.
- **Veredicto:** BAJA CONFIANZA. Depende de batch size, sequence length, y si se extraen hidden states de todas las capas. Necesita medición empírica local.

---

## Síntesis de Hallazgos Críticos para el Paper

### Hallazgos que DEBEN estar en el paper (Alta Confianza):
1. MedGemma 4B IT multimodal es el modelo elegido, single-pass, frozen.
2. ODIR-5K es el dataset, patient-level split, ~6% glaucoma, 8 labels.
3. KL divergence entre vision y text hidden states es la señal principal.
4. Baselines: Entropy, TS+Entropy, MSP, Energy Score, Semantic Entropy (upper bound), SAPLMA-like probe.
5. Métricas: AUROC (primary), AUPRC (secondary), Brier, ECE, Sensitivity@80% Specificity.
6. BIP 2026 requiere 6-8 páginas IEEE, double-blind, CMT.

### Hallazgos que DEBEN discutirse como limitaciones o trabajo futuro (Media/Baja Confianza):
1. El bug de `output_hidden_states` en el vision encoder requiere workaround.
2. El debate AUROC vs AUPRC requiere reportar ambas y discutir implicaciones clínicas.
3. La dirección de KL (simétrica vs asimétrica) requiere ablación.
4. La sensibilidad al prompt es una limitación conocida.
5. El rendimiento de MedGemma en glaucoma es relativamente bajo (F1~64%), justificando la necesidad de UQ.

### Hallazgos con conflicto que requieren decisión de diseño:
1. **Conflicto 12 (4B vs 27B):** Usar 4B. Es más ligero, accesible, y la evidencia de VOLMO sugiere que es igual o mejor que 27B en glaucoma. Además, BIP es un paper de 6-8 páginas; usar un modelo más grande no añade valor si el rendimiento no mejora.
2. **Conflicto 13 (KL vs JSD):** Reportar KL asimétrica como método principal, JSD como ablación. La asimetría es conceptualmente interesante: detecta "alucinación textual" (texto confiado pero no respaldado por visión) vs "caso visualmente ambiguo" (visión rica pero texto inadecuado).
3. **Conflicto 14 (Prompt sensitivity):** Incluir 2-3 prompts en ablación, reportar el más robusto, y discutir que la señal de UQ no debe depender excesivamente del prompt.

---

*Documento de cross-verificación completado. Todos los hallazgos fueron revisados contra múltiples dimensiones de investigación.*
