# AUROC (Area Under the Receiver Operating Characteristic Curve): 

* Esta métrica mide la capacidad general del modelo para discriminar entre casos positivos y casos negativos. 
* Funciona de tal forma que cuando el modelo escoge un caso positivo al azar y un caso negativo al azar, le de una puntuación mayor al caso positivo.
* Va de 0.5 a 1.0, siendo 0.5 que el modelo esta trabajando al azar, literalmente esta adivinando y 1.0 separación perfecta de los casos positivos de los negativos

# AUPRC  (Area Under the Precision-Recall Curve)

* Evalúa si las alertas positivas del modelo son reales
* Es la precisión del modelo con la que detecta la clase de interés (casos positivos)
* Grafica la precision (% de aciertos reales entre todo lo que predijo el modelo) frente al recall (% )

# AURC (Area under Risk-Coverage Curve)

* Es la métrica que mide la capacidad que tiene el modelo de abstenerse o rechazar el dar una respuesta cuando tiene dudas 
* El modelo ordena sus predicciones de mayor a menor confianza. 
* COBERTURA: Es el porcentaje de datos que el modelo decide responder 
* RIESGO: Es la precision del modelo con respecto a los datos que respondió

# Recall@K: 
* Me dice que proporcion de elementos relevantes se ha logrado incluir en los primeros K resultados. Esta es una técnica que se utiliza en sistemas de recomendación y sistemas basados en búsquedas  

# Prueba estadística U Mann-Withney: 
- Es la prueba equivalente no paramétrica de la prueba t-test para grupos independientes 

cantidad de parametos, tiempo de inferencia(Benchmark de modelos)

Sparse Autoencoders que son?

Diferentes tipos de Pooling en redes neuronales-Saber las que implementé y buscar otras para saber como funcionan e implementarlas para ver si funcxionan mejor