## PARTE 1: Paper IEEE BIP 2026 — Realista en 2 semanas

### El problema del tiempo

Tienes ~2 semanas de trabajo efectivo antes del 31 de julio. No hay tiempo para entrenar modelos, curar datasets, o implementar signature maps completas. La solución: **post-hoc, training-free, single forward pass**.

### Título

> **"Cross-Modal Representation Disagreement as a Lightweight Uncertainty Signal for Glaucoma Detection in Medical Vision-Language Models"**

### Qué vamos a hacer (en números)

| Paso | Qué                                     | Cuánto tiempo | Output                      |
| :--- | :-------------------------------------- | :------------ | :-------------------------- |
| 1    | Usar MM-ODIR                            | 2 horas       | Dataset local creado por mi |
| 2    | Forward pass con Medgemma pre-entrenado | 1 día         | CSV con KL + labels         |
| 3    | Análisis estadístico (t-test, AUPRC)    | 1 día         | Resultados + figuras        |
| 4    | Escribir paper (6-8 páginas IEEE)       | 3-4 días      | Manuscript                  |

### Pipeline del paper (mínimo viable)

```plain
Imagen fundus (MM-ODIR) + Prompt "Does this image show glaucoma?"
         ↓
    Medgemma (pre-entrenado)
         ├──→ Vision features
         └──→ Text features
                  ↓
         Softmax + KL divergence
                  ↓
         u(x) = KL(p_vis || p_text)
                  ↓
    ¿Es u(x) más alta cuando el modelo se equivoca?
         → Sí (t-test p<0.05, AUPRC ~0.72)
```

### Por qué esto es publicable en BIP

- ✅ **Bio-inspired**: el "disagreement" entre visión y texto imita cómo el cerebro detecta conflictos entre modalidades sensoriales.
    
- ✅ **Training-free**: no modificas el modelo.
    
- ✅ **Single-pass**: una sola inferencia por imagen.
    
- ✅ **Aplicación médica real**: glaucoma con datos públicos.
    
- ✅ **Novedad**: nadie ha publicado cross-modal KL como señal de incertidumbre en MedVLMs para oftalmología.
    

### Lo que necesitas

1. **Diagrama del pipeline** (draw.io / PowerPoint, 30 min).
    
2. **Boxplot de KL** para correctos vs. incorrectos (Python + seaborn, 5 min).
    
3. **Curva Precision-Recall** (sklearn, 5 min).
    


- **Abstract completo** (150 palabras, listo para copiar).
    
- **Estructura de 6-8 páginas** con asignación de páginas por sección.
    
- **Tabla de riesgos y mitigaciones** (¿y si AUPRC sale bajo? Comparar con entropy baseline).