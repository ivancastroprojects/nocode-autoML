# TokiiAI ML - Plataforma de Aprendizaje Automático Accesible

![Captura de pantalla de la aplicación TokiiAI](https://raw.githubusercontent.com/ivancastroprojects/nocode-autoML/desarrollo/web/static/img/screenshot.png)

## Descripción General

TokiiAI ML es una plataforma de **AutoML (Aprendizaje Automático Automatizado)** que revoluciona la forma en que los usuarios, con o sin experiencia en ciencia de datos, pueden aprovechar el poder del machine learning. Nuestro sistema combina la facilidad de uso de una interfaz web intuitiva con potentes capacidades de backend, permitiendo a los usuarios cargar datasets, entrenar múltiples modelos simultáneamente, optimizar sus hiperparámetros, evaluar métricas y realizar predicciones de manera eficiente.

Inspirada en la simplicidad de soluciones como Amazon SageMaker Canvas, pero con un enfoque en la **transparencia, personalización y el aprendizaje del usuario**, TokiiAI ML se construye sobre un stack tecnológico moderno y open-source.

## Flujo de Trabajo Simplificado

1.  **Carga y Activa tu Dataset**: Sube tus propios archivos CSV o utiliza los que ya están almacenados. El sistema analizará los datos y los preparará para el siguiente paso.
2.  **Configura y Lanza el Entrenamiento**: Selecciona tu variable objetivo, las características a utilizar y los algoritmos que deseas probar. Con un solo clic, puedes lanzar un proceso de entrenamiento completo que incluye la optimización de hiperparámetros.
3.  **Evalúa y Predice**: Una vez finalizado el entrenamiento, la plataforma te presenta un resumen de los modelos entrenados y sus métricas. Puedes ver los detalles de cada uno, comparar su rendimiento y utilizarlos para hacer nuevas predicciones.

## Arquitectura y Tecnologías

La plataforma se ha diseñado con un enfoque modular, utilizando un stack tecnológico robusto y escalable.

-   **Backend**:
    -   **Python**: Lenguaje principal para toda la lógica de negocio y machine learning.
    -   **Flask**: Micro-framework web para servir la API y la interfaz de usuario.
    -   **Scikit-learn**: El corazón del motor de ML, utilizado para el preprocesamiento, entrenamiento y evaluación de modelos.
    -   **Pandas & NumPy**: Para la manipulación y procesamiento eficiente de datos.

-   **Frontend**:
    -   **HTML5 & Jinja2**: Para la estructura semántica y la renderización de las plantillas.
    -   **CSS3**: Estilos personalizados para una interfaz limpia y moderna.
    -   **JavaScript (Vanilla)**: Para la interactividad en tiempo real, como la actualización de logs de entrenamiento sin recargar la página.

-   **Comunicación y Persistencia**:
    -   **MQTT (Simulado)**: La arquitectura está preparada para la comunicación asíncrona a través de un broker MQTT, permitiendo desacoplar la interfaz del motor de entrenamiento. Actualmente, funciona en modo simulación para un despliegue local sencillo.
    -   **Serialización (Pickle/Joblib)**: Los modelos entrenados, preprocesadores y metadatos se serializan y guardan en disco para su posterior uso en predicciones.

-   **Contenerización**:
    -   **Docker**: La aplicación está completamente contenerizada, asegurando un entorno de ejecución consistente y facilitando el despliegue en cualquier sistema compatible con Docker.

## Características Principales

-   **Interfaz Web Intuitiva**: Un flujo de trabajo guiado que abstrae la complejidad del proceso de machine learning.
-   **Entrenamiento de Múltiples Modelos**: Capacidad para seleccionar y entrenar varios algoritmos de Scikit-learn en paralelo.
-   **Preprocesamiento Automático**: Limpieza de datos, imputación de valores faltantes, escalado de características y codificación de variables categóricas de forma automática.
-   **Optimización de Hiperparámetros**: Integración de estrategias como `RandomizedSearchCV` y `Optuna` para encontrar la mejor configuración para tus modelos.
-   **Visualización de Resultados**: Gráficos de evaluación y métricas claras para entender y comparar el rendimiento de los modelos.
-   **Explicabilidad (Próximamente)**: Planificado para integrar técnicas como SHAP para entender las decisiones de los modelos.
-   **Sistema de Predicción Interactivo**: Una vez entrenado, puedes usar cualquier modelo para realizar predicciones introduciendo los valores de las características.

## Uso Local

Para ejecutar la plataforma en tu entorno local, sigue estos pasos:

1.  **Clonar el repositorio**:
    ```bash
    git clone <URL-del-repositorio>
    cd <nombre-del-repositorio>
    ```

2.  **Crear y activar un entorno virtual**:
    ```bash
    python -m venv .venv
    # En Windows
    .venv\Scripts\activate
    # En macOS/Linux
    source .venv/bin/activate
    ```

3.  **Instalar las dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar la aplicación web**:
    ```bash
    python web/app.py
    ```

    Abre tu navegador y ve a `http://127.0.0.1:5000`.

## Próximos Pasos

Nuestro objetivo es seguir mejorando la plataforma. Las futuras mejoras incluyen:

-   **Integración Cloud**: Desarrollar integraciones nativas con AWS, Azure y GCP.
-   **Capacidades Avanzadas**: Añadir soporte para problemas de NLP y Visión por Computadora.
-   **MLOps Completo**: Fortalecer las capacidades de monitoreo de modelos en producción y detección de *drift*.
-   **Comunidad**: Expandir la comunidad de desarrolladores y usuarios a través de una mejor documentación y programas de divulgación.

## Contribuciones

¡Las contribuciones son bienvenidas! Si quieres mejorar la plataforma, por favor, abre un *issue* o envía un *pull request*. 