# Análisis de Morfología Mitocondrial

## 📊 Descripción del Proyecto

Este proyecto analiza métricas morfológicas de mitocondrias para identificar patrones y diferencias entre grupos de estudio (Control vs ELA). Utilizamos técnicas de análisis exploratorio, reducción dimensional (PCA) y deep learning (Autoencoder) para visualizar el espacio latente y detectar posibles clusterizaciones.

### Métricas Analizadas

Los datos contienen las siguientes métricas por mitocondria:

- **N mitocondrias**: Número de mitocondrias analizadas
- **IsoVol (SUMA/PROM)**: Volumen isométrico total y promedio
- **Surface (SUMA/PROM)**: Superficie total y promedio
- **Length (SUMA/PROM)**: Longitud total y promedio
- **RoughSph (SUMA/PROM)**: Índice de rugosidad/esfericidad total y promedio
- **Variables demográficas**: Age, Sex, Group (CT/ELA), Participant

## 🎯 Objetivos

1. **Análisis Exploratorio**: Examinar distribuciones y diferencias entre grupos (CT vs ELA), sexos y participantes
2. **PCA (Análisis de Componentes Principales)**: Reducir dimensionalidad y visualizar la varianza explicada
3. **Autoencoder**: Entrenar una red neuronal para comprimir la información y explorar el espacio latente
4. **Visualización**: Identificar si existe clusterización natural de los datos según características morfológicas

## 🏗️ Estructura del Proyecto

```
mitochondrial-morphology/
│
├── data/
│   └── data.csv                    # Dataset original
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Carga y preprocesamiento de datos
│   ├── exploratory_analysis.py     # Análisis exploratorio (EDA)
│   ├── pca_analysis.py             # Implementación del PCA
│   ├── autoencoder.py              # Arquitectura del Autoencoder (PyTorch)
│   └── utils.py                    # Funciones auxiliares
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb   # EDA detallado
│   ├── 02_pca_analysis.ipynb           # Análisis PCA
│   └── 03_autoencoder_training.ipynb   # Entrenamiento del autoencoder
│
├── models/
│   └── .gitkeep                    # Modelos entrenados guardados aquí
│
├── logs/
│   └── .gitkeep                    # Logs de TensorBoard (Lightning)
│
├── config/
│   └── config.yaml                 # Configuración del proyecto
│
├── app.py                          # Aplicación Streamlit principal
├── requirements.txt                # Dependencias del proyecto
├── .gitignore                      # Archivos a ignorar en Git
└── README.md                       # Este archivo
```

### Justificación de la Estructura

- **`src/`**: Contiene módulos reutilizables para análisis y modelado, facilitando la separación de lógica
- **`notebooks/`**: Análisis exploratorios paso a paso, útiles para documentación y experimentación
- **`models/`**: Almacena checkpoints del autoencoder entrenado
- **`logs/`**: PyTorch Lightning genera logs automáticamente para TensorBoard
- **`config/`**: Centraliza parámetros (rutas, hiperparámetros) en un solo archivo
- **`app.py`**: Interfaz interactiva Streamlit que integra todos los análisis

## 🚀 Instalación y Uso

### Prerrequisitos

- Python 3.8+
- pip o conda

### 1. Clonar el Repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd mitochondrial-morphology
```

### 2. Crear Entorno Virtual (Recomendado)

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la Aplicación Streamlit

```bash
streamlit run app.py
```

La aplicación se abrirá en tu navegador (por defecto: `http://localhost:8501`)

### 5. Ver Logs de TensorBoard (Opcional)

Durante el entrenamiento del autoencoder, puedes monitorear el progreso en tiempo real:

```bash
tensorboard --logdir=logs/
```

## 📈 Estrategia de Análisis

### Fase 1: Análisis Exploratorio de Datos (EDA)

**Objetivo**: Comprender la distribución y relaciones de las métricas

**Técnicas**:
- Estadísticas descriptivas por grupo (CT vs ELA)
- Visualizaciones:
  - Distribuciones (histogramas, boxplots) por grupo y sexo
  - Matrices de correlación
  - Pairplots para variables clave
- Pruebas estadísticas (t-test, ANOVA) para diferencias entre grupos

**Herramientas**: Pandas, Seaborn, Plotly (para interactividad en Streamlit)

### Fase 2: PCA (Reducción Dimensional)

**Objetivo**: Identificar las componentes principales que explican la mayor varianza

**Proceso**:
1. Normalización de features (StandardScaler)
2. Aplicar PCA y visualizar varianza explicada (scree plot)
3. Proyectar datos en 2D/3D (PC1 vs PC2 vs PC3)
4. Colorear por grupo, sexo y participante para identificar patrones

**Interpretación**: 
- ¿Se separan los grupos CT y ELA en el espacio PCA?
- ¿Qué métricas contribuyen más a cada componente?

### Fase 3: Autoencoder con PyTorch Lightning

**Objetivo**: Aprender una representación comprimida del espacio latente

**Arquitectura Propuesta**:
```
Input (8 features) → Encoder (Dense layers) → Latent Space (2-3D) → Decoder → Output (8 features)
```

**Configuración**:
- **Framework**: PyTorch Lightning (simplifica entrenamiento, logging automático)
- **Loss**: MSE (Mean Squared Error) para reconstrucción
- **Optimizer**: Adam
- **Logging**: TensorBoard nativo de Lightning (`TensorBoardLogger`)
- **Callbacks**: Early Stopping, ModelCheckpoint

**Visualización**:
- Proyección del espacio latente en 2D/3D
- Comparar con PCA: ¿El autoencoder captura estructura no lineal?
- Visualizar reconstrucciones vs datos originales

### Fase 4: Integración en Streamlit

**Componentes de la App**:

1. **Página de inicio**: Descripción del proyecto y dataset
2. **EDA Interactivo**:
   - Selector de métricas y grupos
   - Gráficos interactivos (Plotly)
3. **PCA Visualization**:
   - Sliders para seleccionar componentes
   - Scatter plots coloreados por grupo/sexo
4. **Autoencoder Dashboard**:
   - Visualización del espacio latente
   - Métricas de entrenamiento (integradas desde TensorBoard)
   - Comparación de reconstrucciones
5. **Insights y Conclusiones**:
   - Resumen de hallazgos
   - Recomendaciones

**Ventaja**: Todo nativo en Streamlit, sin necesidad de exportar imágenes estáticas

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**: Lenguaje base
- **Streamlit**: Framework para la aplicación web interactiva
- **PyTorch**: Framework de deep learning
- **PyTorch Lightning**: Wrapper para simplificar entrenamiento y logging
- **TensorBoard**: Visualización de métricas de entrenamiento (integrado con Lightning)
- **Pandas & NumPy**: Manipulación de datos
- **Scikit-learn**: PCA, normalización, métricas
- **Plotly & Seaborn**: Visualizaciones interactivas y estáticas
- **Matplotlib**: Gráficos complementarios

## 📊 Dataset

- **Formato**: CSV
- **Filas**: Observaciones de mitocondrias individuales
- **Columnas**: 12 (métricas morfológicas + variables demográficas)
- **Grupos**: CT (Control) y ELA (Esclerosis Lateral Amiotrófica)

## 🔍 Preguntas de Investigación

1. ¿Existen diferencias morfológicas significativas entre grupos CT y ELA?
2. ¿Las métricas de superficie, volumen y longitud están correlacionadas?
3. ¿El PCA revela separación natural entre grupos?
4. ¿El autoencoder captura patrones no lineales que el PCA no detecta?
5. ¿Hay clusterización por participante o características demográficas?

## 🤝 Contribuciones

Este es un proyecto de investigación. Las sugerencias y mejoras son bienvenidas.

## 📝 Licencia

[Especificar licencia si aplica]

## 👤 Autor

Daniel - Análisis de morfología mitocondrial

---

**Nota**: Este proyecto utiliza PyTorch Lightning para el entrenamiento del autoencoder, lo que permite una integración nativa con TensorBoard para monitorear métricas en tiempo real, que luego se visualizan directamente en la aplicación Streamlit.
