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
│   ├── pca_analysis.py             # Implementación del PCA
│   ├── autoencoder.py              # Arquitectura del Autoencoder (PyTorch Lightning)
│   └── utils.py                    # Funciones auxiliares y visualización
│
├── pages/
│   ├── 1_📊_EDA.py                 # Página de Análisis Exploratorio
│   ├── 2_🎯_PCA.py                 # Página de Análisis PCA
│   └── 3_🤖_Autoencoder.py         # Página de Entrenamiento y Visualización
│
├── scripts/
│   └── train_autoencoder.py        # Script para entrenar el autoencoder
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
├── app.py                          # Aplicación Streamlit principal (home)
├── requirements.txt                # Dependencias del proyecto
├── .gitignore                      # Archivos a ignorar en Git
└── README.md                       # Este archivo
```

### Justificación de la Estructura

- **`src/`**: Módulos reutilizables para análisis y modelado (backend lógico)
- **`pages/`**: Páginas de Streamlit - arquitectura multi-page nativa de Streamlit
- **`scripts/`**: Scripts Python ejecutables (ej: entrenamiento del autoencoder)
- **`models/`**: Checkpoints del autoencoder entrenado (generados por PyTorch Lightning)
- **`logs/`**: Logs de TensorBoard generados automáticamente por PyTorch Lightning
- **`config/`**: Archivo YAML centralizado con todos los parámetros del proyecto
- **`app.py`**: Página principal de Streamlit (home), punto de entrada de la aplicación

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

**Navegación**: La aplicación usa la arquitectura multi-page de Streamlit:
- **Home (app.py)**: Página principal con descripción del proyecto
- **📊 EDA**: Análisis exploratorio interactivo
- **🎯 PCA**: Visualización de componentes principales  
- **🤖 Autoencoder**: Entrenamiento y exploración del espacio latente

### 5. Entrenar el Autoencoder

Puedes entrenar el autoencoder de dos formas:

**Opción A - Desde la interfaz web**:
1. Ejecuta la app: `streamlit run app.py`
2. Ve a la página "🤖 Autoencoder"
3. Haz clic en "🚀 Entrenar Autoencoder"

**Opción B - Desde la terminal**:
```bash
python scripts/train_autoencoder.py
```

### 5. Ver Logs de TensorBoard (Opcional)

Durante el entrenamiento del autoencoder, PyTorch Lightning genera logs automáticamente. 
Para visualizarlos en tiempo real:

```bash
tensorboard --logdir=logs/
```

Abre tu navegador en `http://localhost:6006` para ver métricas de entrenamiento, gráficos de pérdida, y más.

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

**Arquitectura Multi-Page de Streamlit**:

La aplicación utiliza la estructura nativa de múltiples páginas de Streamlit:

1. **Home (app.py)**: 
   - Descripción del proyecto y dataset
   - Métricas generales
   - Vista previa de los datos

2. **📊 EDA (pages/1_📊_EDA.py)**:
   - Selección interactiva de métricas y grupos
   - Gráficos de distribución (box, violin, histogram)
   - Matriz de correlación interactiva
   - Pruebas estadísticas automáticas (t-test/ANOVA)
   - Scatter plot matrix
   - Análisis por edad y participante

3. **🎯 PCA (pages/2_🎯_PCA.py)**:
   - Configuración dinámica del número de componentes
   - Scree plot de varianza explicada
   - Proyecciones 2D y 3D interactivas (Plotly)
   - Análisis de loadings (contribución de variables)
   - Colorización por grupo/sexo/participante
   - Exportación de resultados

4. **🤖 Autoencoder (pages/3_🤖_Autoencoder.py)**:
   - Interfaz para entrenar el modelo desde la web
   - Visualización del espacio latente 2D/3D
   - Comparación de reconstrucciones
   - Métricas de error (MSE, MAE, RMSE)
   - Comparación conceptual con PCA
   - Exportación del espacio latente
   - Instrucciones para TensorBoard

**Ventajas de esta arquitectura**:
- ✅ Todo nativo en Streamlit (sin necesidad de frameworks adicionales)
- ✅ Navegación automática mediante sidebar
- ✅ Cache de datos para mejor rendimiento
- ✅ Visualizaciones interactivas con Plotly
- ✅ Entrenamiento del modelo integrado en la UI
- ✅ Logs nativos de PyTorch Lightning visibles en TensorBoard

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
