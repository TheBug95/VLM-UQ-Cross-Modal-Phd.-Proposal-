---
title: Incertidumbre Cross-Modal en Detección de Glaucoma
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# Incertidumbre Cross-Modal en Detección de Glaucoma

Dashboard interactivo del experimento: detección de errores de un modelo de
lenguaje con visión médico (MedGemma-4B) en detección de glaucoma sobre 129
fondos de ojo, usando el desacuerdo entre sus representaciones visual y textual
(divergencia KL sobre hidden states) como señal de incertidumbre
**training-free y single-pass** (costo 1×).

## Pestañas

- **🏠 Panorama** — la historia del experimento, métricas headline y la galería
  interactiva de las 129 imágenes con la ficha de cada caso.
- **📊 Explorador de señales** — ROC/PR, boxplots y distribuciones
  recalculados en vivo para todas las variantes de la señal (KL en ambas
  direcciones, JSD, coseno) y los baselines de igual costo.
- **🗺️ Mapas de pooling** — heatmaps 16×16 de los pesos por token visual de
  las 8 técnicas de pooling, superpuestos conceptualmente al fundus. Requiere
  generar `pooling_maps.csv` con una pasada extra de GPU
  (`python -m src.extract_pooling_maps` — ver las instrucciones en el propio tab).
- **🚑 Simulador de triage** — mueve el slider de cobertura y observa cómo
  sube la accuracy al derivar los casos más inciertos al oftalmólogo.
- **📑 Resultados y tablas** — las tablas del paper con sus cifras exactas,
  calibración y la hipótesis H4.
- **🤖 Asistente IA** — responde preguntas sobre los datos y explica los
  conceptos. Para activar el modo IA completo, añade el secret `HF_TOKEN`
  (Settings → Variables and secrets). Sin token funciona en modo local con las
  cifras y el glosario del estudio.

## ⚠️ Disclaimer

Herramienta de investigación. No es un dispositivo médico ni está validada
para uso clínico. N=129: los intervalos de confianza son anchos y los
resultados se reportan como evidencia sugestiva.

*El dataset de imágenes se cita de forma anonimizada por revisión doble ciego.*
