# Lazy promt

actúa como un ingeniero de prompt, se quiere desarrolllar un proyecto de clase para el curso de
Deep learning de la Universidad NAcional de Colombia, tu tarea es crear un flujo de trabajo
completo que permita el desarrollo del mismo, el proyecto es crear un modelo de redes neuronales
para pronósticar el precio de bolsa de energía de colombia (el dataset se encuentra en el portal
SIMEM), es importarte que tengas en cuenta el ciclo de vida de la análitica que es algo así:
obtención de datos, transformación, modelado, resultados, presentación. ten en cuenta metodología
DevOps, Scrum u otra que consideres apropiada, también ten presente que se cuenta con un
repositorio Github que posee la siguiente estructura de carpetas

```text
├── config/              # Parámetros del proyecto (rutas, hiperparámetros, credenciales de ejemplo).
├── data/
│   ├── raw/             # Datos originales e inmutables.
│   ├── interim/         # Datos en transformación, resultados parciales de limpieza.
│   ├── processed/       # Datos limpios y listos para modelado.
│   └── external/        # Datos de terceros o fuentes externas.
├── docs/                # Documentación del proyecto (Sphinx/MkDocs), guías y referencias.
├── models/              # Modelos entrenados y serializados (ej. .pkl, .h5).
│   ├── trained/         # Modelos finales listos para inferencia.
│   ├── checkpoints/     # Puntos de guardado durante entrenamiento.
├── notebooks/           # Jupyter Notebooks para exploración y visualización (nombrados secuencialmente).
├── src/                 # Código fuente principal del proyecto.
│   ├── __init__.py
│   ├── data/            # Scripts para ingesta y transformación de datos.
│   ├── utils/           # Funciones utilitarias y helpers.
│   ├── features/        # Scripts de feature engineering.
│   ├── models/          # Scripts para entrenamiento, optimización y predicción.
│   └── visualization/   # Generación de gráficos y pósteres digitales.
├── tests/               # Pruebas unitarias y de integración.
├── dashboards/          # Aplicaciones interactivas.
├── reports/             # Resultados y comunicación.
│   ├── figures/         # imágenes generadas por scripts/notebooks (versionadas o regenerables).
├── scripts/             # Automatización.
├── .gitignore           # Archivos ignorados por Git.
├── environment.yml      # Dependencias para Conda.
├── requirements.txt     # Dependencias para pip.
└── README.md            # Este archivo.
```

## prompt Gemini

El siguiente flujo de trabajo integra el ciclo de vida de la analítica de datos, principios de
**MLOps/DevOps** y una gestión ágil basada en **Scrum** adaptada a un proyecto académico de
posgrado de la **Universidad Nacional de Colombia**.

Para construir el proyecto de manera modular, trazable y sin deuda técnica, utiliza esta secuencia
de **Prompt Chaining** (encadenamiento de prompts). Cada fase está diseñada para ejecutar un
_Sprint_ específico sobre tu estructura de repositorio.

---

### Marco Metodológico del Flujo (CRISP-DM + Scrum + MLOps)

- **Gestión Ágil (Scrum):** El desarrollo se divide en **5 Sprints** alineados con las etapas
  analíticas. Cada prompt actúa como el _Sprint Backlog_ técnico.
- **Ingeniería y DevOps:** Se prioriza el código modular en `src/` sobre los _notebooks_
  monolíticos, pruebas automatizadas en `tests/`, configuración centralizada en `config/` y
  documentación continua con **MkDocs**.

---

## Fase 0: Configuración de Infraestructura y DevOps (Sprint 0)

**Objetivo:** Establecer las bases del repositorio, entornos reproducibles, calidad de código y
automatización inicial.

> **Prompt 0 (Copiar y ejecutar):** "Actúa como un Ingeniero de MLOps y Arquitecto de Software
> Senior en Python. Estamos iniciando un proyecto de Deep Learning para la Universidad Nacional de
> Colombia cuyo objetivo es pronosticar el precio de bolsa de energía en Colombia utilizando datos
> del portal SIMEM. Basándote estrictamente en nuestra estructura de repositorio (`config/`,
> `src/`, `tests/`, `docs/`, `scripts/`, etc.), genera los archivos base para garantizar
> reproducibilidad y calidad de código bajo prácticas DevOps:
>
> 1. **`config/config.yaml`**: Archivo maestro con rutas relativas a `data/`, parámetros de
>    conexión a la API/portal SIMEM, ventanas temporales, semillas aleatorias (`seed: 42`) y
>    parámetros base de entrenamiento.
> 2. **`requirements.txt` y `environment.yml`**: Dependencias exactas enfocadas en series de tiempo
>    y Deep Learning (PyTorch o TensorFlow/Keras, pandas, polars, scikit-learn, optuna, pyyaml,
>    pytest, ruff, mkdocs-material, streamlit).
> 3. **`.github/workflows/ci.yml`**: Pipeline de GitHub Actions que ejecute `ruff`
>    (linter/formatter) y `pytest` en cada _push_ o _pull request_ a `main`.
> 4. **`scripts/setup_env.sh`**: Script de automatización para inicializar el entorno y verificar
>    la estructura de carpetas.
>
> Asegúrate de que todo el código siga estándares PEP 8, tipado estático (`typing`) y principios de
> código limpio."

---

## Fase 1: Obtención y Exploración de Datos (Sprint 1)

**Objetivo:** Construir el _pipeline_ de extracción reproducible desde SIMEM hacia `data/raw/` y
realizar el Análisis Exploratorio de Datos (EDA) específico del mercado eléctrico colombiano.

> **Prompt 1 (Copiar y ejecutar):** "Actúa como un Ingeniero de Datos especializado en el Mercado
> de Energía Mayorista (MEM) de Colombia. Vamos a desarrollar el **Sprint 1: Obtención y
> Exploración de Datos** dentro de nuestra estructura de proyecto. Considerando el dominio del
> problema (estacionalidad horaria/diaria, fenómenos macroclimáticos como El Niño/La Niña, aportes
> hídricos, demanda nacional y generación térmica/hidráulica):
>
> 1. **`src/data/make_dataset.py`**: Escribe un script modular que descargue o cargue los datos
>    históricos del precio de bolsa nacional y variables exógenas clave desde el portal SIMEM
>    (usando su API pública o lectura estructurada), guardando los archivos inmutables con control
>    de versiones por fecha en `data/raw/`. Incluye manejo de excepciones, _logging_ estructurado y
>    lectura de parámetros desde `config/config.yaml`.
> 2. **`notebooks/01_eda_simem_precio_bolsa.ipynb`**: Diseña la estructura detallada (celdas de
>    código y markdown) de un notebook exploratorio que analice: valores faltantes, valores
>    atípicos (picos de escasez), estacionariedad (pruebas ADF/KPSS), autocorrelación (ACF/PACF) y
>    descomposición estacional del precio de bolsa. Las figuras clave deben exportarse
>    automáticamente a `reports/figures/`.
> 3. **`tests/test_data_ingestion.py`**: Crea pruebas unitarias con `pytest` para validar el
>    esquema de datos, tipos de columnas y ausencia de duplicados temporales tras la ingesta."

---

## Fase 2: Transformación e Ingeniería de Características (Sprint 2)

**Objetivo:** Limpiar los datos, evitar fuga de información (_data leakage_) y construir tensores
secuenciales listos para redes neuronales.

Opción 1

> **Prompt 2 (Copiar y ejecutar):** "Actúa como un Científico de Datos Senior experto en Series de
> Tiempo. Ejecutemos el **Sprint 2: Transformación y Feature Engineering**. Utilizando los datos
> crudos en `data/raw/` y siguiendo las reglas definidas en `config/config.yaml`:
>
> 1. **`src/data/preprocess.py`**: Implementa la limpieza de datos, imputación causal (sin mirar al
>    futuro) y alineación temporal horaria/diaria, guardando el resultado intermedio en
>    `data/interim/`.
> 2. **`src/features/build_features.py`**: Crea un pipeline de ingeniería de características que
>    genere:
>
> - Rezagos (_lags_) autorregresivos óptimos y ventanas móviles (media, desviación estándar,
>   cuantiles).
> - Codificación cíclica de variables temporales ($\sin/\cos$ para hora del día, día de la semana y
>   mes).
> - Indicadores de dominio energético (ej. brecha demanda-aportes hídricos si están disponibles).
> - Escalado (`RobustScaler` o `StandardScaler`) ajustado **estrictamente sobre el conjunto de
>   entrenamiento** (serializando el escalador en `models/trained/scaler.pkl` para inferencia
>   futura).
> - Generación de ventanas deslizantes 3D `(muestras, pasos_tiempo, características)` almacenadas
>   en `data/processed/`.
>
> 3. **`tests/test_features.py`**: Implementa pruebas unitarias que garanticen que no existe _data
>    leakage_ entre los conjuntos de _train_, _validation_ y _test_ (split cronológico estricto)."

Opción 2

> Actúa como un Científico de Datos Senior especializado en series de tiempo y en modelado de
> sistemas energéticos. Necesito que implementes el **Sprint 2: Transformación y Feature
> Engineering** de este proyecto, siguiendo estrictamente las reglas definidas en
> `config/config.yaml`. El objetivo es dejar listos, de forma reproducible y libre de fuga de
> información, los datos transformados para entrenar modelos de pronóstico.

> Los datos crudos están en `data/raw/`. Trabaja sobre ellos respetando el orden cronológico en
> todo momento, porque cualquier transformación que use información futura invalidaría el
> pronóstico.

> Implementa lo siguiente:

> 1. **`src/data/preprocess.py`**: limpieza de datos, imputación causal (cada valor faltante se
>    rellena únicamente con información disponible en el pasado, nunca con valores posteriores) y
>    alineación temporal a frecuencia horaria y diaria. Guarda el resultado intermedio en
>    `data/interim/`.

> 2. **`src/features/build_features.py`**: un pipeline de ingeniería de características que genere:
>
> - Rezagos (_lags_) autorregresivos y ventanas móviles (media, desviación estándar y cuantiles).
>   Selecciona los rezagos a partir de evidencia del propio dato (por ejemplo, autocorrelación o
>   autocorrelación parcial) y justifica la elección en comentarios del código.
> - Codificación cíclica de variables temporales mediante seno y coseno para hora del día, día de
>   la semana y mes.
> - Indicadores de dominio energético, como la brecha entre demanda y aportes hídricos, siempre que
>   las columnas correspondientes existan en los datos.
> - Escalado con `RobustScaler` o `StandardScaler`, ajustado **únicamente sobre el conjunto de
>   entrenamiento**. Serializa el escalador en `models/trained/scaler.pkl` para la inferencia
>   futura.
> - Generación de ventanas deslizantes tridimensionales con forma
>   `(muestras, pasos_tiempo, características)`, almacenadas en `data/processed/`.

> 3. **`tests/test_features.py`**: pruebas unitarias que verifiquen la ausencia de _data leakage_
>    entre los conjuntos de entrenamiento, validación y prueba. El particionado debe ser
>    cronológico estricto. Incluye pruebas que confirmen que el escalador no fue ajustado con datos
>    de validación ni de prueba, que la imputación no usa valores futuros y que las ventanas
>    deslizantes no cruzan las fronteras entre particiones.

> Restricciones de implementación: usa Python con las bibliotecas estándar del ecosistema
> científico (pandas, numpy, scikit-learn, pytest). Mantén el código modular, con funciones
> pequeñas y documentadas. Si alguna columna mencionada en `config/config.yaml` no existe en
> `data/raw/`, no la inventes: registra la omisión de forma explícita y continúa con las demás.
> Antes de dar por terminado el trabajo, ejecuta las pruebas y confirma que pasan.

---

## Fase 3: Modelado de Deep Learning y Optimización (Sprint 3)

**Objetivo:** Diseñar, entrenar, regularizar y optimizar arquitecturas neuronales para pronóstico
de series de tiempo.

Opción 1

> **Prompt 3 (Copiar y ejecutar):** "Actúa como un Investigador en Deep Learning. Vamos a
> desarrollar el **Sprint 3: Modelado, Entrenamiento y Optimización**. Sobre los tensores
> procesados en `data/processed/`, implementa una arquitectura modular en `src/models/` que permita
> comparar un modelo base estadístico/ML contra arquitecturas profundas (ej. **LSTM/GRU con
> mecanismo de atención**, **TCN - Temporal Convolutional Networks**, o **N-BEATS/TFT**):
>
> 1. **`src/models/architectures.py`**: Define al menos dos arquitecturas de redes neuronales en
>    PyTorch (o TensorFlow/Keras) bien estructuradas, incluyendo capas de _Dropout_, normalización
>    y funciones de pérdida robustas ante picos de precio (ej. _Huber Loss_ o _Quantile Loss_ para
>    intervalos de predicción).
> 2. **`src/models/train_model.py`**: Script de entrenamiento configurado desde
>    `config/config.yaml` que incluya: _Early Stopping_, programación de tasa de aprendizaje
>    (_Learning Rate Scheduler_), guardado de los mejores pesos en `models/checkpoints/`, y
>    registro de métricas de entrenamiento/validación.
> 3. **`src/models/optimize.py`**: Implementa una búsqueda de hiperparámetros con **Optuna** (tasa
>    de aprendizaje, tamaño de ventana, unidades ocultas, _dropout_) que guarde los mejores
>    parámetros y exporte el modelo campeón final a `models/trained/`."

Opción 2

> Actúa como un investigador senior en Deep Learning especializado en pronóstico de series
> temporales financieras. Tu tarea es implementar el **Sprint 3: Modelado, Entrenamiento y
> Optimización** de un proyecto de predicción de precios, partiendo de tensores ya procesados
> ubicados en `data/processed/`.

> El objetivo final es construir un marco modular en `src/models/` que compare de forma justa un
> modelo base estadístico/ML contra arquitecturas profundas (por ejemplo, LSTM/GRU con mecanismo de
> atención, TCN — Temporal Convolutional Networks, o N-BEATS/TFT), de modo que se pueda identificar
> el modelo campeón con evidencia.

> Trabaja con PyTorch (o TensorFlow/Keras si el código existente ya lo usa; mantén la coherencia
> con el resto del proyecto). Entrega tres componentes:

> 1. **`src/models/architectures.py`**: Define al menos tres arquitecturas de redes neuronales bien
>    estructuradas y modulares, además del modelo base estadístico/ML para la comparación. Incluye
>    capas de _Dropout_ y normalización donde corresponda. Usa una función de pérdida robusta ante
>    picos de precio, como _Huber Loss_, y considera _Quantile Loss_ para producir intervalos de
>    predicción si el objetivo lo requiere. Justifica brevemente en comentarios la elección de cada
>    arquitectura frente a la naturaleza de los datos.

> 2. **`src/models/train_model.py`**: Script de entrenamiento parametrizado desde
>    `config/config.yaml` (no hardcodees hiperparámetros). Debe incluir _Early Stopping_ sobre la
>    pérdida de validación, un _Learning Rate Scheduler_, guardado de los mejores pesos en
>    `models/checkpoints/`, y registro de métricas de entrenamiento y validación por época. Respeta
>    el orden temporal de los datos: la validación debe ser posterior al entrenamiento, sin fuga de
>    información del futuro.

> 3. **`src/models/optimize.py`**: Búsqueda de hiperparámetros con **Optuna** sobre tasa de
>    aprendizaje, tamaño de ventana, unidades ocultas y _dropout_. Guarda los mejores parámetros
>    encontrados y exporta el modelo campeón final a `models/trained/`. La selección del campeón
>    debe basarse en el rendimiento de validación, no de prueba.

> Antes de escribir código, revisa la estructura real de `data/processed/` y `config/config.yaml`
> para adaptar las rutas, formas de los tensores (ventana, horizonte, número de variables) y
> nombres de parámetros a lo que ya existe. Mantén el alcance acotado a estos tres archivos y a la
> configuración que necesiten; no reescribas el preprocesamiento.

> Entrega código completo y ejecutable, con docstrings breves y manejo de semillas aleatorias para
> reproducibilidad. Incluye un ejemplo mínimo de uso (comando de ejecución) al final de cada
> script.

---

## Fase 4: Evaluación de Resultados e Interpretabilidad (Sprint 4)

**Objetivo:** Evaluar el rendimiento fuera de muestra bajo métricas financieras y energéticas, y
explicar las predicciones del modelo.

> **Prompt 4 (Copiar y ejecutar):** "Actúa como un Analista Cuantitativo del Sector Energético.
> Ejecutemos el **Sprint 4: Evaluación de Resultados e Interpretabilidad**. Carga el modelo campeón
> desde `models/trained/` y evalúalo sobre el conjunto de prueba (_out-of-sample_) en
> `data/processed/`:
>
> 1. **`src/models/predict_model.py`**: Genera las predicciones en la escala original (COP/kWh)
>    invirtiendo la transformación con `models/trained/scaler.pkl` y calcula métricas clave: **MAE,
>    RMSE, MAPE, sMAPE** y métricas direccionales (precisión en la tendencia de subida/bajada del
>    precio). Incluye comparación directa contra un modelo _Baseline_ (ej. persistencia / Naïve
>    estacional).
> 2. **`src/visualization/visualize.py`**: Genera gráficos de alta resolución (300 DPI) listos para
>    publicación académica en `reports/figures/`:
>
> - Serie real vs. pronosticada (incluyendo zoom en eventos de alta volatilidad).
> - Análisis de residuales y distribución de errores por hora del día.
> - Importancia de características mediante **SHAP** o gradientes integrados para explicar qué
>   variables impulsan el precio de bolsa.
>
> 3. **`notebooks/02_evaluacion_errores.ipynb`**: Notebook de análisis post-mortem enfocado en
>    identificar en qué condiciones del mercado (horas pico, baja hidrología) falla el modelo."

Opción 2

> Actúa como un analista cuantitativo senior del sector energético especializado en mercados
> eléctricos mayoristas y en machine learning aplicado a series temporales. Tu tarea es ejecutar la
> evaluación de resultados e interpretabilidad (Sprint 4) de un modelo de pronóstico del precio de
> bolsa de energía eléctrica en Colombia, expresado en COP/kWh, sobre un conjunto de prueba
> out-of-sample.

> Trabaja sobre los archivos y rutas del proyecto que te indico, asumiendo que tienes acceso al
> repositorio, y ten en cuenta que el modelo campeón y el escalador ya fueron entrenados y
> persistidos previamente:

> - Modelo campeón: `models/trained/` (carga el modelo ganador del entrenamiento previo).
> - Escalador: `models/trained/scaler.pkl`, necesario para invertir la transformación y devolver
>   las predicciones y los valores reales a la escala original COP/kWh.
> - Datos de prueba (out-of-sample): `data/processed/`. No uses ninguna observación de este
>   conjunto durante el entrenamiento o la selección del modelo.

> **1. `src/models/predict_model.py`: predicciones y métricas**

> - Genera las predicciones sobre el conjunto de prueba e invierte la transformación con
>   `scaler.pkl` para trabajar en COP/kWh.
> - Calcula **MAE, RMSE, MAPE y sMAPE**. Documenta cómo tratas los valores cercanos a cero en MAPE,
>   ya que el precio puede aproximarse a cero o ser negativo en algunas horas; elige y justifica
>   una variante robusta (por ejemplo, sMAPE o MAPE con denominador acotado).
> - Calcula métricas direccionales: la precisión en la dirección del movimiento del precio
>   (subida/bajada) respecto al periodo anterior, con su matriz de confusión o tasa de acierto.
> - Compara el modelo contra un **baseline de persistencia / Naïve estacional** (por ejemplo, el
>   precio de la misma hora del día anterior o de la semana anterior, según la estacionalidad de
>   los datos). Reporta las mismas métricas para ambos y la mejora relativa del modelo campeón.
> - Guarda las métricas en un formato tabular reproducible.

> **2. `src/visualization/visualize.py`: gráficos para publicación**

> Genera figuras a **300 DPI**, con etiquetas, unidades (COP/kWh) y leyendas legibles, listas para
> un artículo académico, en `reports/figures/`:

> - Serie real vs. pronosticada en el conjunto de prueba, con un zoom adicional sobre episodios de
>   alta volatilidad.
> - Análisis de residuales (serie temporal de errores e histograma/distribución) y distribución de
>   errores agrupada por hora del día.
> - Importancia de características mediante **SHAP** o gradientes integrados, que explique qué
>   variables impulsan el precio de bolsa. Si usas SHAP, indica el tipo de explicador empleado
>   según el modelo y el tamaño de la muestra para mantener el cálculo tratable.

> **3. `notebooks/02_evaluacion_errores.ipynb`: análisis post-mortem**

> Construye un notebook que identifique en qué condiciones del mercado falla el modelo. Analiza al
> menos:

> - Horas pico frente a horas valle, comparando el error en cada régimen.
> - Periodos de baja hidrología, si la base de datos incluye variables hidrológicas o de aportes;
>   si no las incluye, indícalo explícitamente y utiliza el proxy disponible.
> - Episodios de alta volatilidad y eventos extremos, cuantificando cuánto del error total proviene
>   de ellos.

> Concluye con hallazgos accionables sobre dónde y por qué el modelo degrada su desempeño.

> **Criterios de calidad**

> - Reporta las métricas con su contexto: cuántas observaciones, qué periodo cubre el conjunto de
>   prueba y cuál es el baseline.
> - Si algún archivo, columna o ruta no existe o no coincide con lo descrito, detén ese paso,
>   explica la discrepancia con precisión y continúa con lo que sea posible sin inventar datos.
> - Mantén el código modular, reproducible y con semillas fijas donde aplique.
> - Presenta un resumen final breve con las métricas principales frente al baseline y los hallazgos
>   más relevantes del análisis de errores.

---

## Fase 5: Presentación, Despliegue y Documentación Académica (Sprint 5)

**Objetivo:** Consolidar los entregables finales para la sustentación en la Universidad Nacional de
Colombia: aplicación interactiva, documentación técnica y reporte ejecutivo.

> **Prompt 5 (Copiar y ejecutar):** "Actúa como un Líder Técnico y Divulgador Científico. Vamos a
> cerrar el proyecto con el **Sprint 5: Presentación, Despliegue y Documentación**. Con base en
> todos los artefactos generados en las fases anteriores:
>
> 1. **`dashboards/app.py`**: Desarrolla una aplicación interactiva en **Streamlit** que consuma el
>    modelo en `models/trained/` y permita al usuario seleccionar un horizonte de pronóstico,
>    visualizar la predicción del precio de bolsa con sus intervalos de confianza, explorar
>    métricas de desempeño y consultar la explicabilidad de variables.
> 2. **`docs/` y `README.md**`: Redacta la documentación técnica completa utilizando **MkDocs**
>    (`docs/index.md`, `docs/architecture.md`, `docs/methodology.md`) y actualiza el `README.md`
>    principal con instrucciones claras de instalación, ejecución de pipelines (`scripts/`) y
>    resumen de resultados.
> 3. **`reports/informe_final_unal.md` (o estructura LaTeX/póster)**: Genera el borrador técnico
>    del artículo/informe final del curso de Deep Learning, estructurado en: _Resumen, Introducción
>    al Mercado Eléctrico Colombiano, Metodología y Arquitectura Neuronal, Diseño Experimental,
>    Resultados y Discusión, Conclusiones y Trabajo Futuro_, referenciando las figuras almacenadas
>    en `reports/figures/`."

Opción 2

> Actúa como un Líder Técnico senior con experiencia en ciencia de datos y aprendizaje profundo, y
> como divulgador científico capaz de comunicar resultados técnicos con rigor y claridad. Vamos a
> cerrar el proyecto con el **Sprint 5: Presentación, Despliegue y Documentación**, a partir de los
> artefactos generados en las fases anteriores del proyecto de pronóstico del precio de bolsa del
> mercado eléctrico colombiano. Tu objetivo es producir los tres entregables descritos abajo,
> coherentes entre sí y con los resultados reales del modelo entrenado.

> Los artefactos previos del proyecto (modelo entrenado en `models/trained/`, métricas, figuras en
> `reports/figures/`, scripts en `scripts/` y código de entrenamiento) están disponibles en el
> repositorio. Antes de redactar, revisa estos archivos y basa cada afirmación en lo que realmente
> contienen: cifras, nombres de variables, arquitectura y resultados. No inventes métricas, fechas
> ni parámetros que no aparezcan en los artefactos; si algún dato no está disponible, indícalo
> explícitamente.

> **1. `dashboards/app.py` (aplicación en Streamlit)** Desarrolla una aplicación interactiva que
> cargue el modelo desde `models/trained/` y permita al usuario:
>
> - Seleccionar el horizonte de pronóstico.
> - Visualizar la predicción del precio de bolsa junto con sus intervalos de confianza.
> - Explorar las métricas de desempeño del modelo.
> - Consultar la explicabilidad de las variables de entrada.

> Mantén el código limpio y ejecutable, con la carga del modelo y de los datos separada de la
> lógica de la interfaz, y con un manejo claro de errores si falta algún artefacto.

> **2. Documentación con MkDocs y `README.md`** Redacta la documentación técnica completa en
> `docs/index.md`, `docs/architecture.md` y `docs/methodology.md`, y actualiza el `README.md`
> principal con:
>
> - Instrucciones claras de instalación y de ejecución de los pipelines en `scripts/`.
> - Un resumen de resultados que refleje las métricas reales del proyecto.

> La documentación debe ser precisa para un lector técnico que quiera reproducir el trabajo.

> **3. Informe final del curso de Deep Learning (`reports/informe_final_unal.md`, o bien estructura
> LaTeX/póster si es lo más adecuado)** Genera el borrador técnico del artículo final con las
> secciones: Resumen, Introducción al Mercado Eléctrico Colombiano, Metodología y Arquitectura
> Neuronal, Diseño Experimental, Resultados y Discusión, y Conclusiones y Trabajo Futuro.
> Referencia las figuras almacenadas en `reports/figures/` en las secciones correspondientes,
> indicando qué muestra cada una. Mantén un tono académico, con explicaciones accesibles para una
> audiencia no especialista en la introducción y el resumen.

> Entrega cada uno de los tres artefactos completo y listo para usarse, en Markdown o Python según
> corresponda, y asegúrate de que la terminología, las cifras y los nombres de archivos sean
> consistentes en los tres.

---

### Recomendaciones de Ejecución en tu Flujo de Trabajo

- **Iteración incremental:** No ejecutes todos los prompts a la vez. Ejecuta el **Prompt 0**,
  valida que los archivos funcionen en tu entorno local y haz el primer `git commit`. Avanza al
  siguiente _Sprint_ solo cuando los `tests/` del _Sprint_ actual pasen en verde.
- **Control de contexto:** Cuando pases del Prompt $N$ al Prompt $N+1$, adjunta únicamente los
  nombres de columnas reales obtenidas de SIMEM y la arquitectura seleccionada para mantener
  coherencia estricta entre scripts.
- **Automatización final:** Al terminar el Sprint 3, pide generar un archivo
  `scripts/run_pipeline.sh` (o `Makefile`) que encadene `make_dataset.py` $\rightarrow$
  `preprocess.py` $\rightarrow$ `build_features.py` $\rightarrow$ `train_model.py` $\rightarrow$
  `predict_model.py` en un solo comando.
