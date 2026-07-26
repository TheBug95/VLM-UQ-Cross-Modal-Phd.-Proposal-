# Análisis de Articulación: BIP 2026 y los Cuatro Pilares de la Tesis Doctoral

**Fecha:** 2026-06-23  
**Autor:** Agente de Investigación Académica — Miguel Guillermo Abreu Cárdenas  
**Tutor:** Saúl Calderón Ramírez  
**Contexto:** Doctorado en Inteligencia Artificial Aplicada a Oftalmología  
**Paper Analizado:** *Cross-Modal Representation Disagreement as a Lightweight Uncertainty Signal for Glaucoma Detection in Medical Vision-Language Models* (Propuesta BIP 2026)

---

## Resumen Ejecutivo

La propuesta de paper para la **BIP 2026** — titulada *Cross-Modal Representation Disagreement as a Lightweight Uncertainty Signal for Glaucoma Detection in Medical Vision-Language Models* — constituye un nodo articulador fundamental dentro de la tesis doctoral de Miguel Guillermo Abreu Cárdenas. Este trabajo propone utilizar la **divergencia KL entre representaciones cross-modales** (imagen y texto) en modelos de visión-lenguaje médicos (Medical VLMs) como una señal de incertidumbre *ligera* (*lightweight*) para la detección de glaucoma. A diferencia de los métodos tradicionales de estimación de incertidumbre (UQ) que requieren múltiples forward passes (MC-Dropout, ensembles profundos) o modificaciones arquitectónicas invasivas, el método propuesto opera en **un solo pase (*single-shot*)** y explota la estructura multimodal inherente de los VLMs.

Este documento analiza sistemáticamente cómo el paper BIP 2026 se articula con los **cuatro pilares** de la tesis doctoral: (1) estimación de incertidumbre (UQ), (2) explicabilidad (XAI), (3) aprendizaje few-shot, y (4) segmentación de disco/copa óptica. Además, se propone un **roadmap de tesis** que secuencia lógicamente los papers y proyectos existentes, se identifican **conexiones cross-pilares** facilitadas por el método KL cross-modal, y se señalan las **brechas de investigación** que permanecen abiertas tras BIP 2026.

---

## 1. Contexto: El Ecosistema de Trabajo Previo

Antes de analizar la articulación pilar por pilar, es necesario contextualizar el corpus de trabajo previo sobre el cual se asienta BIP 2026:

- **Revisión sistemática de ~50 papers VLM-oftalmología:** Proporciona el mapeo del estado del arte en modelos fundacionales oftalmológicos (EyeCLIP, RetiZero, MM-Retinal V2, VOLMO, MedGemma), identificando que la mayoría de los VLMs médicos operan bajo el supuesto de que las representaciones visuales y textuales están adecuadamente alineadas, lo cual no siempre es cierto en escenarios de distribución desplazada o patologías raras [^8][^16][^18].
- **"Between the Layers Lies the Truth" (arXiv:2603.22299):** Demuestra que la incertidumbre en LLMs puede capturarse mediante patrones de acuerdo *cross-layer* en representaciones internas, usando un único forward pass. Este principio de *internal agreement* como proxy de incertidumbre es el antecedente conceptual directo del método KL cross-modal propuesto en BIP 2026 [^4].
- **ophthalmo_capture:** Herramienta de etiquetado de imágenes médicas con backend FastAPI/Cloud Run. Constituye la infraestructura de recolección de datos que habilita la generación de pares imagen-texto de calidad para entrenar y evaluar VLMs oftalmológicos.
- **Módulo de segmentación clásica (OpenCV/scikit-image):** Implementa segmentación de disco y copa óptica mediante métodos tradicionales (thresholding, morfología, transformada de Hough), sirviendo como *baseline* interpretable y como fuente de biomarcadores morfológicos (CDR, VCDR) que pueden integrarse en pipelines multimodales.
- **GECCO 2026 (MC-Dropout UQ + Workshop CMA-ES MedGemma):** Trabajo previo que establece la experiencia del doctorando en UQ mediante MC-Dropout para glaucoma, y la exploración de modelos fundacionales médicos como MedGemma mediante optimización evolutiva [^13][^14].
- **IEEE CBMS 2026 (artículo publicado):** Artículo empírico previo que valida técnicas de diagnóstico automatizado en oftalmología.
- **CIARP 2026 (en preparación):** Paper que probablemente extiende aspectos de segmentación o clasificación en imágenes médicas latinoamericanas.

Este ecosistema posiciona a BIP 2026 como la **primera incursión formal del doctorando en el territorio de los Medical VLMs** aplicados a la UQ para glaucoma, fusionando su experiencia previa en incertidumbre (MC-Dropout) con su infraestructura de datos (ophthalmo_capture) y su conocimiento del dominio oftalmológico.

---

## 2. Pilar 1: Estimación de Incertidumbre (UQ)

### 2.1. Articulación con BIP 2026

El pilar de **estimación de incertidumbre** es el más directamente abordado por el paper BIP 2026. La propuesta de utilizar la **divergencia KL entre la distribución de representaciones visuales y la distribución de representaciones textuales** dentro de un VLM médico constituye un paradigma novedoso de UQ *lightweight*.

Tradicionalmente, la UQ en glaucoma se ha abordado mediante:
- **MC-Dropout:** Requiere múltiples inferencias estocásticas (costoso computacionalmente) [^5][^7].
- **Deep Ensembles:** Múltiples modelos entrenados con diferentes inicializaciones (costoso en memoria y entrenamiento) [^7].
- **Bayesian Neural Networks (BNN):** Aproximaciones variacionales como SWAG o Bayes by Backprop, que modifican el proceso de entrenamiento [^6].
- **Evidential Deep Learning (DEC):** Modelos que predicten parámetros de distribuciones de evidencia [^6].

El método propuesto en BIP 2026 difiere radicalmente: en lugar de agregar estocasticidad al modelo o entrenar múltiples modelos, **cuantifica la incertidumbre a partir de la inconsistencia interna entre dos modalidades**. Si el encoder visual y el encoder textual de un VLM médico generan representaciones que difieren significativamente en el espacio latente compartido —medido por KL—, esto indica que el modelo no tiene una comprensión coherente y robusta del caso clínico. Este desacuerdo cross-modal puede deberse a:
- Imágenes de baja calidad o artefactos de adquisición.
- Patologías atípicas o subclínicas no bien representadas en el espacio de entrenamiento.
- Desajuste entre la descripción textual y los hallazgos visuales (hallucination textual o visual).
- Casos cercanos a la frontera de decisión (*decision boundary*).

La evidencia reciente respalda esta intuición: los VLMs clínicos frecuentemente exhiben **modality bias**, favoreciendo priores lingüísticos sobre evidencia visual, generando predicciones plausibles pero alucinadas [^1]. Además, se ha demostrado que VLMs pueden producir respuestas de alta confianza incluso en ausencia de contenido visual significativo, sugiriendo que la confianza del modelo no está anclada en la modalidad imagen [^1]. En este contexto, el desacuerdo cross-modal funciona como un **detector de alineación fallida**, una señal de alerta clínica fundamental.

### 2.2. Extensiones Futuras en UQ

BIP 2026 abre múltiples líneas de extensión dentro del pilar UQ:

1. **Calibración de la señal KL:** Investigar cómo calibrar la magnitud de la divergencia KL en términos de probabilidad predictiva (Expected Calibration Error, ECE). Los trabajos previos en UQ para glaucoma han mostrado que la calibración es tan importante como la discriminación [^7].
2. **Combinación con métodos de muestreo:** Utilizar la señal KL como *triage* para decidir cuándo activar métodos costosos como MC-Dropout o ensembles. Esta estrategia de *adaptive uncertainty estimation* reduce el costo computacional promedio manteniendo alta confiabilidad en casos críticos [^14].
3. **Modelado de distribuciones no gaussianas:** Trabajos recientes como MAP (CVPR 2023) modelan representaciones multimodales como distribuciones gaussianas en lugar de puntos, capturando incertidumbre semántica de manera más rica [^3]. La divergencia KL podría extenderse a espacios de distribuciones (e.g., divergencia de Wasserstein, Fréchet Distance) para capturar incertidumbres de orden superior [^3].
4. **UQ en segmentación:** Extender el método KL cross-modal a tareas de segmentación (ver Pilar 4), donde la incertidumbre espacial puede mapearse a regiones específicas de la retina.

### 2.3. Trabajos Previos del Usuario en UQ

El doctorando ya posee experiencia sólida en este pilar:
- **GECCO 2026:** Trabajo con MC-Dropout para UQ en glaucoma, lo que proporciona un *baseline* metodológico y experimental contra el cual comparar el método KL [^5].
- **"Between the Layers Lies the Truth":** Establece que los patrones de acuerdo interno (*cross-layer agreement*) son proxies transferibles de incertidumbre. BIP 2026 generaliza este principio de *intra-modal* (cross-layer) a *inter-modal* (cross-modal) [^4].
- **IEEE CBMS 2026:** Artículo publicado que valida técnicas de diagnóstico, probablemente incluyendo aspectos de confiabilidad predictiva.

---

## 3. Pilar 2: Explicabilidad (XAI)

### 3.1. Articulación con BIP 2026

Aunque el paper BIP 2026 no propone explícitamente un método de XAI, el **desacuerdo cross-modal KL constituye inherentemente una señal explicativa**. En el contexto médico, especialmente en oftalmología, la explicabilidad no es un lujo regulatorio sino una necesidad clínica: los oftalmólogos necesitan saber *por qué* un modelo está inseguro antes de decidir si escalar el caso a un especialista o solicitar pruebas adicionales.

La divergencia KL entre modalidades puede desagregarse para proporcionar **explicaciones direccionales**:
- **KL alta con baja coherencia visual:** El modelo textual "alucina" un diagnóstico que no se condice con la imagen. Esto señala un *language prior bias*.
- **KL alta con baja coherencia textual:** La imagen contiene hallazgos atípicos que el vocabulario clínico disponible no captura adecuadamente. Esto indica un *vocabulary gap* o patología emergente/rara.
- **KL alta con ambas modalidades coherentes individualmente pero discordantes entre sí:** Sugiere ambigüedad intrínseca del caso (e.g., glaucoma sospechoso con CDR en el límite, sin defecto de campo visual confirmado).

Trabajos recientes en XAI para glaucoma han utilizado Grad-CAM, SHAP, y adversarial examples para visualizar qué regiones retinianas influyen en la decisión del modelo [^11][^12]. Sin embargo, estos métodos explican *dónde* el modelo mira, pero no *por qué* el modelo duda. La señal KL cross-modal añade una dimensión explicativa cualitativa: **la incertidumbre se convierte en una explicación estructural del modelo mismo**.

Además, el paper *Uncertainty-Gated Glaucoma Screening* (2026) muestra que el routing de casos inciertos a un sistema multi-agente basado en MedGemma mejora drásticamente la sensibilidad (100% en casos flagged) [^14]. Esto valida indirectamente que la señal de incertidumbre, cuando se hace *explicable* y *accionable*, tiene valor clínico directo.

### 3.2. Extensiones Futuras en XAI

1. **Attribution maps derivadas de KL:** Desarrollar técnicas de *gradient-based attribution* sobre la divergencia KL para generar *heatmaps* que muestren qué píxeles de la imagen y qué tokens del texto contribuyen más al desacuerdo cross-modal. Esto fusionaría UQ con XAI visual.
2. **Razonamiento estructurado con VLM:** Utilizar el valor de KL como *prompt* para inducir al VLM a generar un razonamiento explicativo del desacuerdo. Trabajos como VOLMO y EVLF-FM demuestran que los VLMs médicos pueden generar diagnósticos diferenciales y planes de tratamiento justificados [^15][^19]. Integrar la señal de incertidumbre en este *chain-of-thought* clínico es una extensión natural.
3. **Concept Bottleneck Models:** El trabajo del doctorando con Concept Whitening (aunque reportado como no exitoso en el contexto de glaucoma) apunta hacia la necesidad de explicaciones basadas en conceptos clínicos (CDR, notching, PPA). La divergencia KL podría calcularse no solo a nivel de representación global, sino a nivel de *concept embeddings* alineados con biomarcadores oftalmológicos.

### 3.3. Trabajos Previos del Usuario en XAI

- **Revisión sistemática (~50 papers):** Mapea exhaustivamente las técnicas de XAI aplicadas a VLMs en oftalmología, incluyendo Grad-CAM, SHAP, attention rollout, y saliency maps [^10][^11].
- **Segmentación clásica (OpenCV/scikit-image):** Aunque no es un método de XAI *post-hoc*, la segmentación de OD/OC proporciona biomarcadores interpretables por diseño (CDR, área, diámetros), funcionando como un modelo *white-box* complementario a los VLMs *black-box*.
- **"Between the Layers Lies the Truth":** El principio de *cross-layer agreement* ya es inherente

mente explicativo: las capas internas del modelo que "discrepan" entre sí son precisamente aquellas que el clínico debería inspeccionar. BIP 2026 extiende esta idea al espacio cross-modal: la discrepancia entre lo que "ve" el encoder visual y lo que "entiende" el encoder textual es una explicación *native* del modelo [^4].

---

## 4. Pilar 3: Aprendizaje Few-Shot

### 4.1. Articulación con BIP 2026

Los VLMs, especialmente en el dominio médico, se promueven precisamente por su capacidad de **adaptación few-shot**: un modelo preentrenado en millones de pares imagen-texto puede especializarse para glaucoma con apenas decenas o cientos de ejemplos etiquetados [^8][^9]. Sin embargo, el aprendizaje few-shot en oftalmología enfrenta dos problemas críticos:

1. **Distribuciones de clase altamente desbalanceadas:** El glaucoma es una condición de baja prevalencia en datasets de screening. Los support sets few-shot contienen categorías subrepresentadas, lo que degrada el rendimiento en regímenes de bajo shot [^9].
2. **Etiquetado ruidoso o ambiguo:** En oftalmología, la etiqueta "glaucoma" no siempre es binaria. Existen casos de "glaucoma sospechoso" o "sospechoso de glaucoma" que introducen incertidumbre en los ejemplos de soporte.

El método KL cross-modal de BIP 2026 se articula con few-shot de manera dual:

- **Filtrado de pseudo-labels:** En escenarios semi-supervisados (como los explorados en *Semi-Supervised Few-Shot Adaptation of Vision-Language Models*), los datos no etiquetados se pseudo-etiquetan a partir de priores textuales. La señal KL puede servir como **filtro de calidad**: rechazar pseudo-labels generados a partir de pares imagen-texto con alta divergencia KL, mitigando el *confirmation bias* del modelo [^9].
- **Selección de support sets:** Dado un pool de imágenes candidatas para few-shot, la divergencia KL puede guiar la selección de ejemplos que maximicen la cobertura del espacio de representación y minimicen la redundancia. Los ejemplos con KL moderada (no cero, no extrema) suelen ser informativos porque representan casos que el modelo aún no domina completamente.
- **Detección de failure modes:** Los VLMs few-shot en oftalmología pueden sufrir *modality collapse* donde el texto domina completamente la predicción ignorando la imagen. La KL alta en el conjunto de adaptación es una señal temprana de este failure mode, permitiendo activar técnicas de *modality-balancing* (e.g., *modality dropout*, *contrastive unimodal pretraining*) [^1].

EyeCLIP y RetiZero demuestran que los VLMs oftalmológicos preentrenados con suficiente volumen de datos pueden alcanzar rendimiento competitivo en zero-shot y few-shot en glaucoma [^8][^16]. BIP 2026 añade una capa de **diagnóstico metacognitivo**: no solo predice, sino que evalúa cuán confiable es su propia adaptación few-shot.

### 4.2. Extensiones Futuras en Few-Shot

1. **Prompt learning con UQ:** Los métodos de *context optimization* (CoOp, CoCoOp) adaptan VLMs aprendiendo *prompt vectors* en el espacio de embeddings textuales. La divergencia KL puede incorporarse como término de regularización en la optimización del prompt, penalizando prompts que generen representaciones textuales excesivamente divergentes de las visuales.
2. **Meta-learning de incertidumbre:** Utilizar la señal KL como métrica de *task difficulty* en meta-learning (MAML, ProtoNet). Tareas (datasets) con alta KL promedio son intrínsecamente más difíciles y requieren más shots o mayor regularización.
3. **Cross-modal data augmentation:** En few-shot, la augmentación es crítica. La señal KL puede guiar augmentaciones que preserven la alineación cross-modal (e.g., solo rotaciones que no alteren la anatomía clínica), evitando *semantic drift*.

### 4.3. Trabajos Previos del Usuario en Few-Shot

- **GECCO 2026 (Workshop CMA-ES MedGemma):** La optimización evolutiva de prompts para MedGemma constituye una forma de adaptación few-shot por búsqueda automática. La experiencia adquirida en este trabajo proporciona un *testbed* para validar la señal KL como guía de búsqueda [^13].
- **Revisión sistemática:** Mapea la literatura de VLMs few-shot en oftalmología, identificando que el rendimiento en zero-shot de modelos generales (CLIP, BioMedCLIP) es insuficiente para glaucoma, mientras que modelos específicos del dominio (EyeCLIP, RetiZero) logran F1 > 90% [^8][^16].
- **CIARP 2026:** Potencialmente aborda clasificación en escenarios de datos limitados en el contexto latinoamericano, donde el few-shot y el semi-supervisado son especialmente relevantes.

---

## 5. Pilar 4: Segmentación de Disco/Copa Óptica (OD/OC)

### 5.1. Articulación con BIP 2026

La segmentación de disco óptico (OD) y copa óptica (OC) es el **biomarcador estructural fundamental** para el diagnóstico de glaucoma. El *vertical cup-to-disc ratio* (VCDR) y otros parámetros morfológicos (área, diámetros, notching, peripapillary atrophy) constituyen el estándar de oro clínico. BIP 2026, al ser un paper de clasificación/deteción basado en VLMs, no aborda directamente la segmentación. Sin embargo, la **señal KL cross-modal puede integrarse en pipelines de segmentación** de manera novedosa:

- **Segmentación guiada por texto (*text-guided segmentation*):** Los VLMs como MedGemma y EVLF-FM pueden generar *segmentation masks* a partir de prompts textuales (e.g., "segment the optic disc"). La divergencia KL entre la representación del prompt y la representación de la imagen puede utilizarse como **señal de confianza espacial**: regiones de la imagen donde la KL local es alta corresponden a zonas donde el modelo no está seguro de la correspondencia texto-imagen, sugiriendo bordes ambiguos o patologías disruptivas [^19].
- **Segmentación clásica como ancla:** El módulo de segmentación clásica del doctorando (OpenCV/scikit-image) proporciona segmentaciones *interpretables por diseño* y biomarcadores cuantitativos. Estos biomarcadores pueden **enriquecer el prompt textual** del VLM: en lugar de describir la imagen genéricamente, el prompt incluye "CDR=0.75, disc area=2.5 mm²". La divergencia KL entre el embedding de este prompt enriquecido y la imagen se reduce en casos donde el VLM "entiende" estos biomarcadores, y permanece alta cuando el VLM ignora la información numérica o la interpreta incorrectamente [^11].
- **Uncertainty-aware segmentation:** Trabajos recientes han demostrado que la visualización de incertidumbre (entropy maps) en segmentaciones de capas retinianas (OCT) permite identificar regiones problemáticas y focalizar la atención del clínico [^10]. La KL cross-modal puede reinterpretarse como una *uncertainty map global* que, combinada con segmentaciones locales, produce un **perfil de riesgo estructural** del paciente.

### 5.2. Extensiones Futuras en Segmentación

1. **Cross-modal segmentation evaluation:** En lugar de evaluar segmentación solo con métricas de overlap (Dice, IoU), evaluar la **alineación entre la representación de la máscara segmentada y la representación textual de la estructura**. Esto penaliza segmentaciones geométricamente correctas pero semánticamente inconsistentes (e.g., un VLM que segmenta "cup" pero predice "healthy disc").
2. **Federated segmentation:** ophthalmo_capture puede recolectar segmentaciones de múltiples centros. La KL cross-modal puede servir como métrica de consenso entre las descripciones textuales de diferentes expertos y las segmentaciones automáticas, cuantificando la *inter-expert agreement* en el espacio latente.
3. **Segmentación de estructuras emergentes:** BIP 2026 se centra en glaucoma, pero la segmentación de otras estructuras (fovea, vasculatura, drusen) puede beneficiarse del mismo framework. La señal KL puede indicar cuándo un VLM está extrapolando fuera de su conocimiento anatómico (e.g., prompt "segment lamina cribrosa" en un modelo no entrenado para OCT de alta resolución).

### 5.3. Trabajos Previos del Usuario en Segmentación

- **Módulo de segmentación clásica (OpenCV/scikit-image):** Proporciona un *baseline* robusto, rápido y completamente interpretable. Funciona como oráculo de bajo costo para generar biomarcadores que alimentan el pipeline cross-modal de BIP 2026.
- **CIARP 2026:** Paper en preparación que probablemente extiende la segmentación a nuevos datasets o modalidades latinoamericanas.
- **Revisión sistemática:** Incluye análisis de segmentación OD/OC en VLMs y CNNs, identificando que los métodos de *polar transformation* y *multi-label deep networks* son el estado del arte para segmentación simultánea de OD y OC [^11].

---

## 6. Conexiones Cross-Pilares: La Divergencia KL como Hilo Conductor

El método de divergencia KL cross-modal no es una contribución aislada al Pilar 1 (UQ), sino un **hilo conductor que teje conexiones entre los cuatro pilares**. A continuación se detallan estas conexiones cross-pilares:

### 6.1. UQ → XAI: La Incertidumbre como Explicación Nativa

Como se discutió en la Sección 3, la magnitud y dirección de la divergencia KL son inherentemente explicativas. A diferencia de los métodos de XAI *post-hoc* (Grad-CAM, LIME) que requieren cálculos adicionales sobre un modelo entrenado, la KL es una **medida *native* de la arquitectura VLM**. No se explica *qué* predice el modelo, sino *por qué* el modelo no está seguro. Esta es una forma de *epistemic XAI* complementaria a la *attribution XAI* tradicional.

### 6.2. UQ → Few-Shot: La Incertidumbre como Curador de Datos

En few-shot, la calidad de los ejemplos de soporte domina sobre la cantidad. La KL cross-modal puede funcionar como **curador automático**: seleccionar ejemplos que maximicen la reducción esperada de la divergencia promedio del conjunto. Esto es una forma de *active learning* en el espacio de representaciones multimodales. Además, en semi-supervised learning, la KL filtra pseudo-labels ruidosos, mitigando el ciclo de confirmación de errores que afecta a los métodos few-shot autónomos [^9].

### 6.3. UQ → Segmentación: La Incertidumbre como Mapa de Riesgo Espacial

En segmentación, la KL global del VLM puede descomponerse en contribuciones locales mediante técnicas de *attention attribution* o *token-level divergence*. Esto genera un **mapa de riesgo espacial** donde las regiones de alta contribución a la KL son precisamente las regiones anatómicas ambiguas (e.g., borde borroso entre copa y disco, presencia de PPA que distorsiona la anatomía). Este mapa puede guiar al clínico en la revisión manual o al sistema en la adquisición de vistas adicionales.

### 6.4. XAI → Few-Shot: Explicaciones que Mejoran la Adaptación

Los métodos de *prompt learning* few-shot pueden beneficiarse de *explanatory prompts*. En lugar de optimizar un vector de prompt genérico, optimizar un prompt que incluya una explicación de la decisión. La KL cross-modal puede servir como métrica de coherencia: un prompt explicativo es válido si la divergencia entre su embedding y el de la imagen es baja.

### 6.5. Segmentación → UQ: Biomarcadores como Anclas de Certeza

Los biomarcadores extraídos de la segmentación clásica (CDR, VCDR) actúan como **anclas de certeza clínica**. Cuando el VLM predice "glaucoma" y el CDR segmentado es 0.8, la KL entre el prompt "glaucoma con CDR alto" y la imagen debería ser baja. Si la KL es alta a pesar de la coherencia biomarcador-predicción, esto indica una *shortcut learning* o *spurious correlation* en el VLM que el sistema de UQ ha detectado.

---

## 7. Roadmap de Tesis: Secuencia Lógica de Papers y Proyectos

Con BIP 2026 como punto de partida, se propone el siguiente roadmap de tesis, articulando los 4 pilares en una narrativa coherente:

### Fase 1: Fundamentos y Estado del Arte (Completada)
- **Revisión sistemática de ~50 papers VLM-oftalmología:** Establece el mapa cognitivo del dominio y justifica la necesidad de UQ, XAI, few-shot y segmentación en este espacio.
- **IEEE CBMS 2026 (publicado):** Valida técnicas de diagnóstico automatizado en un contexto clínico específico, sentando las bases empíricas.
- **Módulo de segmentación clásica + ophthalmo_capture:** Infraestructura tecnológica y *baseline* interpretable para biomarcadores estructurales.

### Fase 2: UQ Intra-Modal y Lightweight (Completada / En curso)
- **"Between the Layers Lies the Truth" (arXiv:2603.22299):** Demuestra que la UQ puede ser *single-shot* y basada en acuerdo interno (*cross-layer*). Esto resuelve la objeción de costo computacional de MC-Dropout.
- **GECCO 2026 (MC-Dropout UQ + CMA-ES MedGemma):** Establece la experiencia en UQ tradicional y en la interacción con modelos fundacionales médicos, contrastando métodos costosos con métodos lightweight.

### Fase 3: UQ Inter-Modal y Cross-Pillar (Punto Actual — BIP 2026)
- **BIP 2026 — *Cross-Modal Representation Disagreement*:** Generaliza la UQ de intra-modal a inter-modal. Este es el **nodo central del roadmap**. Articula UQ con XAI (la discrepancia como explicación), con few-shot (la discrepancia como filtro), y con segmentación (la discrepancia como mapa de riesgo).

### Fase 4: Integración Multimodal y Clinical Deployment (Futura inmediata)
- **CIARP 2026 (en preparación):** Extender la segmentación y/o clasificación a datasets latinoamericanos, probablemente con recursos limitados, validando la utilidad del framework en escenarios de few-shot reales.
- **Paper "KL-Guided Segmentation" (propuesto):** Extender la señal KL a segmentación OD/OC text-guidada, integrando el módulo clásico con un VLM médico. Publicación objetivo: MICCAI o ISBI 2027.
- **Paper "Uncertainty-Aware Few-Shot Adaptation" (propuesto):** Desarrollar el algoritmo de filtrado de pseudo-labels y selección de support sets basado en KL. Publicación objetivo: CVPR / ICCV / NeurIPS Medical AI Workshop 2027.

### Fase 5: Síntesis Doctoral y Clinical AI System (Futura a largo plazo)
- **Tesis Doctoral — Capítulo de Síntesis:** Integrar todos los papers en un marco teórico unificado: *"A Multimodal Uncertainty Framework for Trustworthy Glaucoma AI"*.
- **Despliegue de ophthalmo_capture v2.0:** Incorporar el pipeline de UQ/XAI en la herramienta de etiquetado, permitiendo que los clínicos vean no solo la predicción del modelo, sino también la señal de incertidumbre y las explicaciones cross-modales en tiempo real.
- **Validación clínica prospectiva:** Evaluación del sistema completo en un cohorte clínico real, midiendo no solo métricas de ML (AUC, ECE) sino métricas de *human-AI interaction*: confianza del clínico, tiempo de decisión, tasa de escalamiento a especialista.

---

## 8. Brechas Identificadas: Qué Queda por Investigar

A pesar de la solidez del roadmap, BIP 2026 deja abiertas brechas significativas en cada pilar:

### 8.1. Brechas en UQ

- **Falta de calibración probabilística:** La divergencia KL es una distancia en el espacio de representaciones, pero no está calibrada como probabilidad predictiva. Falta investigar cómo transformar KL en un ECE confiable, especialmente en datasets de glaucoma donde la prevalencia es baja (<5% en screening) y el costo de falsos negativos es extremadamente alto [^7].
- **Invariancia a arquitecturas:** La señal KL depende de la estructura específica del VLM (e.g., CLIP-style dual encoder vs. Flamingo-style multimodal decoder). Es necesario investigar si el método es robusto a través de arquitecturas distintas (MedGemma, EyeCLIP, LLaVA-Med) [^13].
- **Incertidumbre en OOD extremo:** El trabajo de DRUE (2026) muestra que la detección OOD en glaucoma es crucial cuando los datos de test vienen de datasets con características de adquisición radicalmente distintas (PAPILA vs. ACRIMA vs. HAM10000) [^6]. La KL cross-modal necesita validación en escenarios OOD extremos, donde una de las modalidades puede estar completamente ausente o corrupta.

### 8.2. Brechas en XAI

- **Escasa validación con oftalmólogos:** La mayoría de los trabajos de XAI en glaucoma (incluyendo las propuestas de BIP 2026) no incluyen estudios de usabilidad con clínicos. Falta un *user study* que evalúe si las explicaciones derivadas de KL son realmente útiles para la toma de decisiones [^12].
- **Falta de estandarización de conceptos:** La comunidad no ha acordado un *ontology* de conceptos oftalmológicos para VLMs. El trabajo con Concept Whitening del doctorando evidencia que forzar conceptos predefinidos no es trivial. Se necesita una taxonomía de conceptos multimodales (imagen + texto) para glaucoma [^11].
- **XAI para errores del modelo:** Los métodos actuales explican predicciones correctas. Falta investigar cómo la KL cross-modal puede explicar *por qué* el modelo se equivoca (e.g., un falso positivo debido a un *language prior* de "glaucoma sospechoso" en un paciente miope).

### 8.3. Brechas en Few-Shot

- **Generalización cross-dataset:** Los resultados de few-shot en oftalmología frecuentemente no generalizan entre datasets debido a diferencias de adquisición (cámara, resolución, campo de visión). La señal KL necesita validación como predictor de *transferability* few-shot [^8].
- **Escalabilidad de pseudo-labels:** El trabajo de Semi-Supervised Few-Shot Adaptation muestra que la calidad de pseudo-labels es el cuello de botella [^9]. La KL puede filtrar pseudo-labels, pero ¿cuál es la tasa de descarte óptima? ¿Existe un *threshold* universal o depende del dataset?
- **Integración con VLMs de razonamiento:** Modelos como VOLMO-2B y EVLF-FM generan razonamientos estructurados [^15][^19]. Falta investigar cómo el few-shot se adapta cuando el output no es solo una clase, sino un *differential diagnosis* textual. La KL entre el razonamiento generado y la imagen es un espacio de investigación inexplorado.

### 8.4. Brechas en Segmentación

- **Segmentación en VLMs generativos:** Los VLMs como MedGemma y EVLF-FM pueden generar texto descriptivo, pero su capacidad para generar *segmentation masks* de alta resolución es limitada (usualmente operan a 224×224 o 448×448). La integración con segmentación clásica a resolución nativa (1024×1024 o superior) es una bre técnica [^19].
- **Evaluación multimodal de segmentación:** Las métricas tradicionales (Dice, IoU) no capturan la coherencia semántica. Falta una métrica que evalúe si la segmentación es *clínicamente coherente* con la descripción textual del VLM. La KL cross-modal podría inspirar tal métrica.
- **Segmentación de estructuras finas:** La copa óptica en glaucoma temprano puede ser extremadamente sutil. Los métodos clásicos (thresholding) y los VLMs pueden fallar en casos de CDR < 0.5 con glaucoma real (false negatives críticos). La señal de incertidumbre espacial derivada de KL podría *flaggear* estos casos para revisión humana prioritaria.

---

## 9. Conclusiones

El paper *Cross-Modal Representation Disagreement as a Lightweight Uncertainty Signal for Glaucoma Detection in Medical Vision-Language Models* (BIP 2026) representa un **punto de inflexión metodológico** en la tesis doctoral de Miguel Guillermo Abreu Cárdenas. Al proponer que la divergencia KL entre modalidades sea una señal de incertidumbre, el trabajo no solo resuelve el problema de costo computacional de los métodos tradicionales (MC-Dropout, ensembles), sino que **crea un puente natural entre los cuatro pilares** de la tesis.

La KL es, simultáneamente:
- Una **señal de UQ** (Pilar 1) que detecta casos ambiguos y alucinaciones cross-modales.
- Una **explicación nativa** (Pilar 2) que indica *por qué* el modelo duda y en qué dirección falla la alineación.
- Un **curador de datos** (Pilar 3) que filtra pseudo-labels y selecciona ejemplos few-shot informativos.
- Un **mapa de riesgo espacial** (Pilar 4) que, combinado con segmentación clásica, prioriza regiones anatómicas ambiguas.

El roadmap propuesto — con BIP 2026 como nodo central — articula lógicamente los trabajos previos (MC-Dropout, cross-layer agreement, segmentación clásica, revisión sistemática) con las extensiones futuras (segmentación text-guidada, few-shot adaptativo, despliegue clínico). Las brechas identificadas (calibración, validación clínica, métricas multimodales de segmentación, generalización cross-dataset) proporcionan un programa de investigación concreto para los próximos 18-24 meses del doctorado.

En síntesis, BIP 2026 no es un paper aislado: es la **manifestación de una visión de tesis** donde la incertidumbre no es un problema a eliminar, sino una **señal a explotar** para hacer los sistemas de IA en oftalmología más confiables, explicables, eficientes y clínicamente útiles.

---

## Referencias

[^1]: Clinical decision-making relies on the integrated analysis of medical images and associated textual information. Vision-Language Models often exhibit modality bias favoring language priors over visual inputs, leading to unsafe predictions. arXiv:2508.00171, 2025.

[^2]: Cross-modal linkage risk in clinical vision-language models. arXiv:2606.02276, 2026.

[^3]: Ji, Y. et al. MAP: Multimodal Uncertainty-Aware Vision-Language Pre-Training Model. CVPR 2023.

[^4]: Badash, Z. et al. Between the Layers Lies the Truth: Compact, Per-Instance Uncertainty via Cross-Layer Agreement. arXiv:2603.22299, 2026.

[^5]: Is Uncertainty Quantification a Viable Alternative to Learned Deferral? Evaluation on glaucoma fundus images. arXiv:2508.02319, 2025.

[^6]: Robust Uncertainty Estimation under Distribution Shift via Difference Reconstruction. Evaluation on Glaucoma-Light V2, PAPILA, ACRIMA. arXiv:2601.19341, 2026.

[^7]: A framework for robust glaucoma detection: A confidence-aware deep uncertainty quantification approach. Engineering Applications of Artificial Intelligence, 2025.

[^8]: Shi, D. et al. EyeCLIP: A multimodal visual-language foundation model for computational ophthalmology. Nature Medicine / arXiv:2409.06644, 2025.

[^9]: Semi-Supervised Few-Shot Adaptation of Vision-Language Models. arXiv:2603.02959, 2026.

[^10]: Explainable AI (XAI) in Image Segmentation in Medicine, Industry, and Beyond: A Survey. arXiv:2405.01636, 2024.

[^11]: Deep Learning for Ophthalmology: The State-of-the-Art and Emerging Applications. arXiv:2501.04073, 2025.

[^12]: Explainable AI for glaucoma detection and classification. Springer, 2026.

[^13]: Sellergren, A. et al. MedGemma: Technical Report. arXiv:2507.05201, 2025.

[^14]: Uncertainty-Gated Glaucoma Screening: Combining Semi-Supervised Classification with Multi-Agent Large Language Model Deliberation. medRxiv, 2026.

[^15]: VOLMO: Versatile and Open Large Models for Ophthalmology. arXiv:2603.23953, 2026.

[^16]: RetiZero: Common and Rare Fundus Diseases Identification Using Vision-Language Foundation Model with Knowledge of Over 400 Diseases. arXiv:2406.09317, 2024.

[^17]: Beyond CLIP: Knowledge-Enhanced Multimodal Transformers for Cross-Modal Alignment in Diabetic Retinopathy Diagnosis. arXiv:2512.19663, 2025.

[^18]: MM-Retinal V2: Transfer an Elite Knowledge Spark into Fundus Vision-Language Pretraining. arXiv:2501.15798, 2025.

[^19]: EVLF-FM: A multimodal vision-language foundation model with fine-grain explainability. arXiv:2509.24231, 2025.

[^20]: Malik, M. H. et al. A hybrid Transformer-CNN framework for uncertainty-guided semi-supervised multiclass eye disease classification with enhanced interpretability. Computerized Medical Imaging and Graphics, 2026.
