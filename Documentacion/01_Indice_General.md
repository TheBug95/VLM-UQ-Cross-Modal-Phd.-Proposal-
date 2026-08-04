# Documentación Doctoral — Proyecto BIP 2026

## Cross-Modal Representation Disagreement as a Lightweight Uncertainty Signal for Glaucoma Detection in Medical Vision-Language Models

**Autor:** Miguel Guillermo Abreu Cárdenas — Doctorado en Inteligencia Artificial aplicada a Oftalmología
**Tutor:** Saúl Calderón Ramírez
**Fecha de esta documentación:** 4 de agosto de 2026 (actualizada con el análisis de calibración estilo FUSE §5.2)
**Estado del experimento:** ✅ Completo, con todos los baselines ejecutados, verificación independiente aprobada (19/19 checks) y validación sintética de calibración (6/6 checks)

---

## Resumen ejecutivo

Este proyecto demuestra que cuando MedGemma-4B (un Vision-Language Model médico basado en Gemma 3) se equivoca al clasificar una imagen de fondo de ojo como glaucoma/normal, el **desacuerdo entre su representación visual interna y su representación textual** — medido por la divergencia KL entre los hidden states de los tokens de imagen y el estado que condiciona el token de respuesta, en la última capa del decoder — es significativamente mayor que cuando acierta (AUROC = 0.661, IC 95% [0.522, 0.772], p = 0.006). La señal es **training-free**, **single-pass** (costo computacional 1×) y **cross-modal**: tres propiedades que esta técnica de Uncertainty Quantification (UQ) reúne simultáneamente. Además, la **fusión parameter-free por ranks** `rank(KL) + rank(1−MSP)` — contribución original del autor — mejora la detección de errores a AUROC = 0.698 y, sobre todo, reduce el riesgo de selective prediction (Excess-AURC = 0.407 frente a 0.670 de la KL sola), habilitando un esquema clínico de triage en tres zonas donde el 30.2% menos incierto de la cohorte no contiene ningún error del modelo.

---

## Tabla de contenidos

| # | Documento | Contenido |
|---|-----------|-----------|
| 01 | [Índice General](01_Indice_General.md) | Este documento: navegación, resumen, estado del avance, glosario |
| 02 | [Marco Teórico](02_Marco_Teorico.md) | VLMs, UQ en deep learning, UQ para LLMs/VLMs, métodos cross-modales y contrastivos, dilución espacial, glaucoma |
| 03 | [Hipótesis y Diseño Experimental](03_Hipotesis_y_Diseno_Experimental.md) | H1–H4, variables, protocolo, prompts congelados, métricas |
| 04 | [Arquitectura Técnica](04_Arquitectura_Tecnica.md) | MedGemma-4B, extracción de representaciones, KL/JSD/coseno, 8 poolings, señal combinada |
| 05 | [Implementación Software](05_Implementacion_Software.md) | Stack, módulos `src/`, formatos de datos, reproducibilidad |
| 06 | [Resultados Experimentales](06_Resultados_Experimentales.md) | **Documento central:** H1–H3, accuracy-coverage, zona verde, baselines multi-pass, generalización, calibración (Fig 10 + T5) |
| 07 | [Ablaciones y Análisis Profundo](07_Ablaciones_y_Analisis_Profundo.md) | Tipo de divergencia, pooling, temperatura, capa, prompt, baselines multi-pass |
| 08 | [Discusión y Limitaciones](08_Discusion_y_Limitaciones.md) | Interpretación teórica, posicionamiento vs. estado del arte, contribuciones, limitaciones, riesgos, future work |
| 09 | [Dataset MM-ODIR-129](09_Dataset_MM_ODIR_129.md) | Origen, estructura, anotaciones, artefactos, estadística, ética |
| 10 | [Guía de Reproducibilidad](10_Guia_Reproducibilidad.md) | Requisitos, instalación, HuggingFace, comandos, validaciones |
| 11 | [Conclusiones y Próximos Pasos](11_Conclusiones_y_Proximos_Pasos.md) | Contribuciones, verificación de hipótesis, roadmap doctoral |
| 12 | [Verificación y Validación](12_Verificacion_y_Validacion.md) | Verificación independiente (val_08), robustez cross-GPU, zona verde, validación de calibración (val_09), lecciones aprendidas |

**Figuras:** todas en [`assets/`](assets/) (copiadas desde `figures/` del proyecto; están en inglés por convención de publicación).

---

## Estado actual del avance (checklist)

| Componente                                       | Estado | Detalle                                                                                    |
| ------------------------------------------------ | ------ | ------------------------------------------------------------------------------------------ |
| Definición experimental congelada (v2)           | ✅      | `Definicion_Experimental_Minima_BIP2026.md`                                                |
| Dataset descargado + tabla maestra               | ✅      | 129 imágenes, `data/master_table.csv`, flag `has_annotation_artifact`                      |
| Pipeline de inferencia (`src/inference.py`)      | ✅      | 129 imgs × 2 prompts = 258 inferencias deterministas                                       |
| 97 variantes de señal evaluadas                  | ✅      | KL/JSD/coseno × 8 poolings × 3 τ + kl_prompt                                               |
| Sanity checks del piloto (8/8)                   | ✅      | Piloto de 20 imágenes                                                                      |
| Baselines de igual costo (1×)                    | ✅      | entropy, 1−MSP, energy                                                                     |
| Baseline verbalized confidence (P5, 2×)          | ✅      | 129 inferencias extra — señal degenerada                                                   |
| Baseline self-consistency (SC, 10×)              | ✅      | 50 imgs × 10 muestras × 2 prompts a T=1.5                                                  |
| Señal combinada `rank(KL)+rank(1−MSP)`           | ✅      | Contribución original — mejor señal del estudio                                            |
| Evaluación estadística completa                  | ✅      | BCa bootstrap, Mann-Whitney, Spearman H4, AURC/Excess-AURC, Monte Carlo CV                 |
| Análisis de calibración (FUSE §5.2)                | ✅      | Platt en train, ECE, correlaciones, Brier, TPR@FPR; val_09 6/6 PASS                                    |
| Figuras del paper (Fig 2–10 + heatmaps)           | ✅      | En `assets/` (resincronizadas con `figures/` el 04-ago)                                                                               |
| Tablas T1–T5                                     | ✅      | Resultados, ablaciones, comparativa, costo-beneficio, calibración                                       |
| Verificación independiente (val_08)              | ✅      | 19/19 checks PASS                                                                          |
| Robustez cross-GPU                               | ✅      | Logits bitwise idénticos; Spearman 0.964                                                   |
| Análisis de robustez por artefactos de anotación | ⏳      | **Pendiente obligatorio** antes de submission (ver [§8.4](08_Discusion_y_Limitaciones.md)) |
| Redacción del paper BIP 2026                     | ⏳      | Siguiente fase                                                                             |

---

## Los números que importan (referencia rápida)

| Pregunta                                     | Respuesta                                                                                   |
| -------------------------------------------- | ------------------------------------------------------------------------------------------- |
| ¿Funciona la señal KL cross-modal?           | Sí, modestamente: AUROC 0.661 [0.522, 0.772], p = 0.006                                     |
| ¿Supera al baseline gratis (1−MSP)?          | Por poco sola (+0.037); claramente en combinación (0.698)                                   |
| Configuración ganadora                       | `kl_t_v`, capa 34, τ = 1, pooling `max`                                                     |
| ¿Sirve para triage?                          | Sí: derivando el 50% más incierto, accuracy 79.8% → 89.1%; zona verde del 30.2% sin errores |
| ¿Correlaciona con la severidad del glaucoma? | No — H4 rechazada (rho = +0.001, p = 0.99)                                                  |
| ¿Generaliza a pacientes nuevos?              | Estimación Monte Carlo: KL sola ~0.58–0.66; combinación 0.648–0.698                         |
| ¿Está calibrada la señal?                        | KL: ECE 0.087, cal. Pearson/Spearman 0.748/0.789 — la mejor calibrada entre las 1×; la combinación discrimina más pero calibra peor (ECE 0.145). Evidencia secundaria (N=129, [§6.13](06_Resultados_Experimentales.md)) |
| ¿Baselines multi-pass (2×/10×)?              | Dominados por nuestra señal 1× en el frente de Pareto                                       |

---

## Glosario de términos técnicos

| Término | Definición en este proyecto |
|---|---|
| **VLM (Vision-Language Model)** | Modelo que procesa conjuntamente imágenes y texto; aquí, MedGemma-4B-it |
| **MedGemma-4B** | VLM médico de Google Research (~4B parámetros), Gemma 3 + vision encoder MedSigLIP |
| **hidden state** | Vector de activación interna (2560-dim) de una capa del decoder para una posición de token |
| **prefill** | Pasada del modelo sobre el prompt completo (imagen + texto) antes de generar |
| **p_vis** | Distribución obtenida del pooling de los 256 hidden states de los tokens de imagen (capa 34) |
| **p_text** | Distribución obtenida del hidden state de la última posición del prefill (el estado que condiciona la respuesta) |
| **KL divergence** | Divergencia de Kullback-Leibler $D_{KL}(p \Vert q) = \sum_i p_i \ln(p_i / q_i)$; asimétrica |
| **kl_t_v / kl_v_t** | KL(texto‖imagen) / KL(imagen‖texto) — las dos direcciones de la señal |
| **JSD** | Divergencia de Jensen-Shannon; versión simétrica de KL, acotada en $\ln 2$ |
| **pooling** | Estrategia para resumir los 256 tokens de imagen en un solo vector (mean, max, roi, attn, topk, normw, rollout, headspec) |
| **τ (temperatura)** | Factor que aplana/afila la softmax al convertir hidden states en distribuciones |
| **u(x)** | Señal de incertidumbre de una imagen $x$; más alto = más sospechoso de error |
| **θ (umbral de triage)** | Punto de corte sobre u(x): los casos con u(x) > θ se derivan al especialista, los demás se auto-responden. Se define como **percentil de la cohorte** (ej.: derivar el 20% más incierto ⇒ θ = percentil 80), nunca como valor absoluto de nats. Ver [§6.6](06_Resultados_Experimentales.md#6.6-accuracy-coverage-aplicación-clínica-y-zona-verde) |
| **MSP (Maximum Softmax Probability)** | $\max(p_{yes}, p_{no})$; el baseline 1−MSP mide la duda del modelo |
| **entropy (answer)** | Entropía binaria de $(p_{yes}, p_{no})$ |
| **energy** | $-\mathrm{logsumexp}(\ell_{yes}, \ell_{no})$; baseline de igual costo |
| **Verbalized Confidence (P5)** | Baseline 2×: preguntarle al modelo su confianza (0–100) en un segundo turno |
| **Self-Consistency (SC)** | Baseline 10×: muestrear 10 respuestas a T=1.5 y medir su dispersión |
| **frac_other** | Fracción de muestras SC que generan un token fuera del formato yes/no |
| **AUROC** | Área bajo la curva ROC para detectar errores del modelo (0.5 = azar) |
| **AUPRC** | Área bajo la curva Precision-Recall |
| **AURC / Excess-AURC** | Área bajo la curva riesgo-cobertura (selective prediction); Excess normalizado: 0 = oracle, 1 = azar |
| **ECE (Expected Calibration Error)** | Media ponderada de $\bar{u}_{bin} - error_{empírico,bin}$ (en valor absoluto); mide la calibración de la señal (0 = perfecta) |
| **Platt scaling** | Sigmoide de 1 feature que mapea $u(x) \rightarrow P(error)$; aquí se ajusta SOLO con el split train |
| **Brier score** | Error cuadrático medio entre la probabilidad calibrada y el error observado |
| **TPR@FPR** | Tasa de errores detectados permitiendo una tasa fija de falsas alarmas (5/10/20%) |
| **reliability diagram** | Curva de $u(x)$ calibrada vs. frecuencia empírica de error por bins (Fig 10) |
| **BCa bootstrap** | Intervalos de confianza por bootstrap corregido por sesgo y acelerado (9.999 remuestreos) |
| **Mann-Whitney U** | Test no paramétrico de diferencia de distribuciones (errores vs. aciertos); $r$ = effect size |
| **accuracy-coverage** | Accuracy del modelo reteniendo solo el X% menos incierto de casos |
| **zona verde** | Cola de menor incertidumbre del ranking donde el modelo no comete errores |
| **rank fusion** | Combinar señales sumando sus posiciones (ranks) en vez de sus valores crudos |
| **winsorización (KL)** | Techo numérico de la KL en $\ln(1/\varepsilon) = 23.03$ por el clamp $\varepsilon = 10^{-10}$ |
| **massive activations** | Activaciones de magnitud extrema en modelos Gemma que colapsan la softmax en baja precisión |
| **greedy decoding** | Generación determinista (siempre el token más probable); `do_sample=False` |
| **single-pass / multi-pass** | Costo de la señal en pasadas del modelo: 1×, 2×, 10× |
| **oracle (roi)** | Pooling con máscaras del disco óptico: cota superior no desplegable (usa información privilegiada) |
| **Monte Carlo CV** | Validación cruzada por 200 splits aleatorios estratificados |
| **MM-ODIR-129** | Dataset: 129 imágenes de fondo de ojo de ODIR-5K re-anotadas por oftalmólogos de Costa Rica |
| **CDR (cup-to-disc ratio)** | Proporción copa/disco; signo cardinal del glaucoma; grading ordinal 0–4 |
| **HAI-DEF** | Licencia de Google para modelos de salud (requerida para MedGemma) |

---

> ➡️ Siguiente: [02 — Marco Teórico](02_Marco_Teorico.md)
