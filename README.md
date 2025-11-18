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
│   ├── 2_�_Entrenar_Modelo.py     # Página de Entrenamiento con TensorBoard
│   └── 3_🤖_Autoencoder.py         # Página de Visualización de Modelos
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
- **� Entrenar Modelo**: Entrenar VAE/LSTM-VAE con TensorBoard en tiempo real
- **🤖 Autoencoder**: Visualización del espacio latente y métricas

### 5. Entrenar el Autoencoder

Puedes entrenar el autoencoder de dos formas:

**Opción A - Desde la interfaz web (Recomendado)**:
1. Ejecuta la app: `streamlit run app.py`
2. Ve a la página "🎓 Entrenar Modelo"
3. Selecciona el tipo de modelo:
   - **VAE Estándar**: Agrega mediciones por participante (mean pooling)
   - **LSTM-VAE**: Preserva variabilidad intra-participante (secuencias completas)
4. Configura hiperparámetros (epochs, learning rate, batch size, patience)
5. Haz clic en "🚀 Iniciar Entrenamiento"
6. **TensorBoard se abre automáticamente en la misma página** mostrando métricas en tiempo real

**Opción B - Desde la terminal**:
```bash
# VAE estándar
python scripts/train_autoencoder.py

# LSTM-VAE (preserva variabilidad intra-participante)
python scripts/train_autoencoder.py --lstm
```

### 6. Ver Logs de TensorBoard

**Durante el entrenamiento desde Streamlit**: TensorBoard se muestra automáticamente en un iframe embebido.

**Manualmente** (opcional):
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

### Fase 3: Variational Autoencoder (VAE) con PyTorch Lightning

**Objetivo**: Aprender una representación probabilística comprimida del espacio latente

**Dos Arquitecturas Disponibles**:

#### 1. VAE Estándar (Mean Pooling)
```
Input (8 features agregadas) → Encoder [64, 32] → Latent 8D (μ, σ) → Decoder [32, 64] → Output (8 features)
                                                        ↓
                                                 Classifier [16] → CT/ELA
```
- **Ventaja**: Rápido, simple, interpretable
- **Desventaja**: Pierde variabilidad intra-participante

#### 2. LSTM-VAE (Sequences)
```
Input (secuencias 4-36 mediciones × 8 features) → Bidirectional LSTM Encoder (2 capas, hidden=64)
                                                        ↓
                                                 Latent 16D (μ, σ)
                                                        ↓
                                          Decoder LSTM (2 capas, hidden=64)
                                                        ↓
                                          Output (secuencias reconstruidas)
                                                        ↓
                                          Classifier [32, 16] → CT/ELA
```
- **Ventaja**: Preserva variabilidad intra-participante, mayor capacidad
- **Desventaja**: Más lento, más parámetros (~205k vs ~6k)

**Configuración**:
- **Framework**: PyTorch Lightning (simplifica entrenamiento, logging automático)
- **Loss**: Reconstrucción + KL Divergence + Clasificación
- **Optimizer**: Adam con learning rate configurable
- **Logging**: TensorBoard embebido en Streamlit (tiempo real)
- **Callbacks**: Early Stopping, ModelCheckpoint, LearningRateMonitor

**Monitoreo en Tiempo Real**:
- TensorBoard se muestra **dentro de Streamlit** durante el entrenamiento
- Métricas: loss, accuracy, KL divergence, reconstruction error
- Visualizaciones: curvas de aprendizaje, histogramas de pesos

**Visualización**:
- Proyección del espacio latente en 2D/3D por grupo (CT/ELA)
- Métricas de clasificación (accuracy, confusion matrix)
- Comparar reconstrucciones vs datos originales
- Identificar si la variabilidad intra-participante mejora la clasificación

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

3. **� Entrenar Modelo (pages/2_�_Entrenar_Modelo.py)** ⭐ **NUEVO**:
   - Selección de tipo de modelo (VAE estándar vs LSTM-VAE)
   - Configuración interactiva de hiperparámetros:
     - Max epochs, learning rate, batch size, early stopping patience
   - **TensorBoard embebido en tiempo real** durante el entrenamiento
   - Visualización de métricas: loss, accuracy, KL divergence
   - Ver entrenamientos anteriores y comparar runs
   - Guías contextuales sobre arquitecturas y hiperparámetros
   - Todo integrado - no necesitas abrir terminales adicionales

4. **🤖 Autoencoder (pages/3_🤖_Autoencoder.py)**:
   - Carga de modelos entrenados (VAE o LSTM-VAE)
   - Detección automática del tipo de modelo
   - Visualización del espacio latente 2D/3D (Plotly interactivo)
   - Métricas de clasificación (accuracy, confusion matrix)
   - Análisis de reconstrucciones
   - Comparación conceptual con PCA
   - Exportación del espacio latente
   - Ver logs históricos de TensorBoard

**Ventajas de esta arquitectura**:
- ✅ Todo nativo en Streamlit (sin necesidad de frameworks adicionales)
- ✅ Navegación automática mediante sidebar
- ✅ Cache de datos para mejor rendimiento
- ✅ Visualizaciones interactivas con Plotly
- ✅ **TensorBoard embebido en tiempo real** - sin abrir ventanas adicionales
- ✅ Entrenamiento del modelo integrado en la UI
- ✅ Comparación fácil entre VAE estándar y LSTM-VAE
- ✅ Logs nativos de PyTorch Lightning visibles en TensorBoard
- ✅ Workflow completo: configurar → entrenar → monitorear → visualizar

## 🆕 Características Destacadas

### TensorBoard en Tiempo Real

La nueva página de entrenamiento incluye **TensorBoard embebido** que muestra métricas en tiempo real:

- 📊 **Curvas de aprendizaje**: Loss y accuracy (train/validation)
- 📈 **KL Divergence**: Regularización del espacio latente
- 🔍 **Reconstruction Loss**: Calidad de reconstrucción
- 🎯 **Classification Metrics**: Accuracy de CT vs ELA
- 📉 **Learning Rate**: Evolución durante entrenamiento

**Sin necesidad de:**
- Abrir terminales adicionales
- Ejecutar comandos TensorBoard manualmente
- Cambiar entre ventanas

**Todo en una sola interfaz web integrada.**

### Dos Modelos para Comparar

1. **VAE Estándar (Mean Pooling)**:
   - Agrega múltiples mediciones por participante
   - ~6,700 parámetros
   - Entrenamiento rápido (~2-5 min)
   - Baseline sólido

2. **LSTM-VAE (Sequences)**:
   - Preserva variabilidad intra-participante
   - ~205,850 parámetros
   - Entrenamiento más lento (~5-15 min)
   - Captura patrones temporales/secuenciales

**Pregunta de Investigación**: ¿La variabilidad intra-participante mejora la clasificación CT vs ELA?

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
6. **¿La variabilidad intra-participante (LSTM-VAE) mejora la clasificación vs mean pooling (VAE estándar)?** ⭐

## 📚 Documentación Adicional

- **`LSTM_VAE_ARCHITECTURE.md`**: Guía técnica detallada de la arquitectura LSTM-VAE
- **`docs/TRAINING_GUIDE.md`**: Guía completa de entrenamiento con TensorBoard
- **`TENSORBOARD_INTEGRATION_SUMMARY.md`**: Resumen de integración y características
- **`test_lstm_vae.py`**: Script de validación de la implementación LSTM-VAE
- **`test_tensorboard_integration.py`**: Test de integración de TensorBoard

## 🤝 Contribuciones

Este es un proyecto de investigación. Las sugerencias y mejoras son bienvenidas.

## 📝 Licencia

[Especificar licencia si aplica]

## 👤 Autor

Daniel - Análisis de morfología mitocondrial

---

**Nota**: Este proyecto utiliza PyTorch Lightning para el entrenamiento del autoencoder, con **TensorBoard embebido en Streamlit** para monitorear métricas en tiempo real. La integración completa permite entrenar, monitorear y visualizar modelos sin salir del navegador.

## 🎯 Inicio Rápido

```bash
# 1. Clonar repositorio
git clone https://github.com/DanZangrando/mitochondrial-morphology.git
cd mitochondrial-morphology

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar aplicación
streamlit run app.py

# 4. Entrenar modelo
# Ir a página "🎓 Entrenar Modelo" en el navegador
# Seleccionar tipo de modelo y configurar hiperparámetros
# Click en "🚀 Iniciar Entrenamiento"
# TensorBoard se abre automáticamente mostrando métricas en tiempo real

# 5. Visualizar resultados
# Ir a página "🤖 Autoencoder"
# Cargar modelo entrenado
# Explorar espacio latente y métricas
```
