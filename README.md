# Análisis de Morfología Mitocondrial - CT vs ELA# Análisis de Morfología Mitocondrial



## 📊 Descripción del Proyecto## 📊 Descripción del Proyecto



Proyecto de clasificación supervisada que utiliza **deep learning con LSTM** para distinguir participantes Control (CT) vs Esclerosis Lateral Amiotrófica (ELA) basándose en métricas morfológicas mitocondriales.Este proyecto analiza métricas morfológicas de mitocondrias para identificar patrones y diferencias entre grupos de estudio (Control vs ELA). Utilizamos técnicas de análisis exploratorio, reducción dimensional (PCA) y deep learning (Autoencoder) para visualizar el espacio latente y detectar posibles clusterizaciones.



**Características principales:**### Métricas Analizadas

- **Dataset pequeño**: 20 participantes (10 CT, 10 ELA) con secuencias de longitud variable (4-36 mediciones)

- **K-Fold Cross-Validation**: Entrenamiento robusto con validación cruzada estratificadaLos datos contienen las siguientes métricas por mitocondria:

- **LSTM Bidireccional**: Captura patrones en secuencias de mediciones

- **Clasificación binaria**: CT (clase 0) vs ELA (clase 1)- **N mitocondrias**: Número de mitocondrias analizadas

- **IsoVol (SUMA/PROM)**: Volumen isométrico total y promedio

### 📐 Métricas Morfológicas Analizadas- **Surface (SUMA/PROM)**: Superficie total y promedio

- **Length (SUMA/PROM)**: Longitud total y promedio

**8 features de entrada** (todas agregadas SUMA/PROM por mitocondria):- **RoughSph (SUMA/PROM)**: Índice de rugosidad/esfericidad total y promedio

- **IsoVol**: Volumen isométrico- **Variables demográficas**: Age, Sex, Group (CT/ELA), Participant

- **Surface**: Área de superficie

- **Length**: Longitud## 🎯 Objetivos

- **RoughSph**: Índice de rugosidad/esfericidad

1. **Análisis Exploratorio**: Examinar distribuciones y diferencias entre grupos (CT vs ELA), sexos y participantes

**Variables demográficas** (NO usadas como input del modelo):2. **PCA (Análisis de Componentes Principales)**: Reducir dimensionalidad y visualizar la varianza explicada

- Age, Sex, Group (CT/ELA), Participant, n_mitochondrias3. **Autoencoder**: Entrenar una red neuronal para comprimir la información y explorar el espacio latente

4. **Visualización**: Identificar si existe clusterización natural de los datos según características morfológicas

## 🎯 Objetivos

## 🏗️ Estructura del Proyecto

1. **Clasificación Supervisada**: Predecir correctamente participantes CT vs ELA usando solo morfología mitocondrial

2. **Análisis Exploratorio**: Examinar distribuciones y diferencias entre grupos```

3. **Reducción Dimensional**: Visualizar patrones con PCA (lineal) y explorar separabilidadmitochondrial-morphology/

4. **K-Fold Cross-Validation**: Obtener métricas robustas (mean ± std) para datasets pequeños│

├── data/

## 🏗️ Estructura del Proyecto│   └── data.csv                    # Dataset original

│

```├── src/

mitochondrial-morphology/│   ├── __init__.py

││   ├── data_loader.py              # Carga y preprocesamiento de datos

├── data/│   ├── pca_analysis.py             # Implementación del PCA

│   └── data.csv                    # Dataset (306 muestras, 20 participantes)│   ├── autoencoder.py              # Arquitectura del Autoencoder (PyTorch Lightning)

││   └── utils.py                    # Funciones auxiliares y visualización

├── src/│

│   ├── __init__.py├── pages/

│   ├── data_loader.py              # Carga y preprocesamiento│   ├── 1_📊_EDA.py                 # Página de Análisis Exploratorio

│   ├── pca_analysis.py             # Análisis PCA│   ├── 2_�_Entrenar_Modelo.py     # Página de Entrenamiento con TensorBoard

│   ├── classifier.py               # LSTM Classifier (PyTorch Lightning)│   └── 3_🤖_Autoencoder.py         # Página de Visualización de Modelos

│   └── utils.py                    # Funciones auxiliares│

│├── scripts/

├── pages/│   └── train_autoencoder.py        # Script para entrenar el autoencoder

│   ├── 1_📊_EDA.py                 # Análisis Exploratorio│

│   ├── 2_🎯_PCA.py                 # PCA (participante + individual)├── models/

│   ├── 3_🎓_Entrenar_Modelo.py     # Entrenamiento (simple split o K-Fold)│   └── .gitkeep                    # Modelos entrenados guardados aquí

│   └── 4_🤖_Clasificador.py        # Visualización de resultados│

│├── logs/

├── scripts/│   └── .gitkeep                    # Logs de TensorBoard (Lightning)

│   └── train_classifier.py         # Script de entrenamiento (CLI)│

│├── config/

├── models/│   └── config.yaml                 # Configuración del proyecto

│   ├── *.ckpt                      # Modelos entrenados (simple split)│

│   └── kfold_K/                    # Modelos K-Fold + summary.json├── app.py                          # Aplicación Streamlit principal (home)

│├── requirements.txt                # Dependencias del proyecto

├── logs/├── .gitignore                      # Archivos a ignorar en Git

│   └── lstm_classifier/            # TensorBoard logs└── README.md                       # Este archivo

│```

├── config/

│   └── config.yaml                 # Configuración del proyecto### Justificación de la Estructura

│

├── app.py                          # Punto de entrada Streamlit- **`src/`**: Módulos reutilizables para análisis y modelado (backend lógico)

├── README.md                       # Este archivo- **`pages/`**: Páginas de Streamlit - arquitectura multi-page nativa de Streamlit

└── requirements.txt                # Dependencias Python- **`scripts/`**: Scripts Python ejecutables (ej: entrenamiento del autoencoder)

```- **`models/`**: Checkpoints del autoencoder entrenado (generados por PyTorch Lightning)

- **`logs/`**: Logs de TensorBoard generados automáticamente por PyTorch Lightning

## 🚀 Quick Start- **`config/`**: Archivo YAML centralizado con todos los parámetros del proyecto

- **`app.py`**: Página principal de Streamlit (home), punto de entrada de la aplicación

### 1. Instalar dependencias

```bash## 🚀 Instalación y Uso

# Crear entorno virtual (recomendado)

python -m venv venv### Prerrequisitos

source venv/bin/activate  # Linux/Mac

# venv\Scripts\activate  # Windows- Python 3.8+

- pip o conda

# Instalar dependencias

pip install -r requirements.txt### 1. Clonar el Repositorio

```

```bash

### 2. Ejecutar la aplicacióngit clone <URL_DEL_REPOSITORIO>

```bashcd mitochondrial-morphology

streamlit run app.py```

```

### 2. Crear Entorno Virtual (Recomendado)

La aplicación se abrirá en `http://localhost:8501`

```bash

### 3. Navegaciónpython -m venv venv

source venv/bin/activate  # En Windows: venv\Scripts\activate

**Páginas disponibles:**```



1. **📊 EDA (Análisis Exploratorio)**### 3. Instalar Dependencias

   - Estadísticas descriptivas

   - Distribuciones por grupo (CT vs ELA)```bash

   - Correlaciones entre variablespip install -r requirements.txt

   - Análisis por participante```



2. **🎯 PCA (Análisis de Componentes Principales)**### 4. Ejecutar la Aplicación Streamlit

   - PCA a nivel participante (agregado por participant)

   - PCA a nivel individual (todas las mediciones)```bash

   - Visualización 2D/3D interactivastreamlit run app.py

   - Varianza explicada```



3. **🎓 Entrenar Modelo**La aplicación se abrirá en tu navegador (por defecto: `http://localhost:8501`)

   - **Train/Val Split Simple** (80/20)

   - **K-Fold Cross-Validation** (K=3-10, recomendado K=5)**Navegación**: La aplicación usa la arquitectura multi-page de Streamlit:

   - Configuración de hiperparámetros:- **Home (app.py)**: Página principal con descripción del proyecto

     - Max epochs, batch size, learning rate- **📊 EDA**: Análisis exploratorio interactivo

     - Hidden dim, num layers, dropout- **� Entrenar Modelo**: Entrenar VAE/LSTM-VAE con TensorBoard en tiempo real

     - Early stopping, checkpointing- **🤖 Autoencoder**: Visualización del espacio latente y métricas

   - Resultados en tiempo real

### 5. Entrenar el Autoencoder

4. **🤖 Clasificador (Resultados)**

   - Accuracy, confusion matrixPuedes entrenar el autoencoder de dos formas:

   - Classification report (precision, recall, F1)

   - Distribución de probabilidades**Opción A - Desde la interfaz web (Recomendado)**:

   - Análisis por participante1. Ejecuta la app: `streamlit run app.py`

   - Métricas de entrenamiento (loss, accuracy)2. Ve a la página "🎓 Entrenar Modelo"

3. Selecciona el tipo de modelo:

## 🧠 Arquitectura del Modelo   - **VAE Estándar**: Agrega mediciones por participante (mean pooling)

   - **LSTM-VAE**: Preserva variabilidad intra-participante (secuencias completas)

### LSTM Classifier4. Configura hiperparámetros (epochs, learning rate, batch size, patience)

5. Haz clic en "🚀 Iniciar Entrenamiento"

```6. **TensorBoard se abre automáticamente en la misma página** mostrando métricas en tiempo real

Input: (batch, seq_len, 8)  [8 features morfométricas]

           ↓**Opción B - Desde la terminal**:

Bidirectional LSTM (2 layers, hidden_dim=64)```bash

           ↓# VAE estándar

Concatenate [forward_hidden, backward_hidden]python scripts/train_autoencoder.py

           ↓

Fully Connected: 128 → ReLU → Dropout(0.3)# LSTM-VAE (preserva variabilidad intra-participante)

           ↓python scripts/train_autoencoder.py --lstm

Fully Connected: 64 → ReLU → Dropout(0.3)```

           ↓

Fully Connected: 32 → ReLU → Dropout(0.3)### 6. Ver Logs de TensorBoard

           ↓

Output: (batch, 2)  [logits para CT/ELA]**Durante el entrenamiento desde Streamlit**: TensorBoard se muestra automáticamente en un iframe embebido.

```

**Manualmente** (opcional):

**Características:**```bash

- **Input variable**: Acepta secuencias de longitud variable (4-36 mediciones/participante)tensorboard --logdir=logs/

- **Bidireccional**: Captura patrones en ambas direcciones```

- **Regularización**: Dropout para prevenir overfitting

- **Loss**: Cross EntropyAbre tu navegador en `http://localhost:6006` para ver métricas de entrenamiento, gráficos de pérdida, y más.

- **Optimizer**: Adam con learning rate scheduler (ReduceLROnPlateau)

- **Parámetros**: ~147K trainable params## 📈 Estrategia de Análisis



## 📊 Dataset### Fase 1: Análisis Exploratorio de Datos (EDA)



- **Total muestras**: 306 mediciones**Objetivo**: Comprender la distribución y relaciones de las métricas

- **Participantes**: 20 (10 CT, 10 ELA)

- **Distribución por grupo**:**Técnicas**:

  - CT: 167 samples, 10 participants- Estadísticas descriptivas por grupo (CT vs ELA)

  - ELA: 139 samples, 10 participants- Visualizaciones:

- **Secuencias**: Longitud variable por participante (min: 4, max: 36)  - Distribuciones (histogramas, boxplots) por grupo y sexo

- **Features**: 8 métricas morfológicas (IsoVol, Surface, Length, RoughSph - SUMA/PROM)  - Matrices de correlación

  - Pairplots para variables clave

## 🔬 K-Fold Cross-Validation- Pruebas estadísticas (t-test, ANOVA) para diferencias entre grupos



### ¿Por qué K-Fold para 20 participantes?**Herramientas**: Pandas, Seaborn, Plotly (para interactividad en Streamlit)



**Problema con train/val split simple:**### Fase 2: PCA (Reducción Dimensional)

- Solo 16 participantes para entrenar (1 modelo)

- Sensible al split aleatorio (puede ser "fácil" o "difícil" por suerte)**Objetivo**: Identificar las componentes principales que explican la mayor varianza

- 1 métrica (ej: 75% accuracy) → ¿es representativa?

**Proceso**:

**Solución con K-Fold (K=5):**1. Normalización de features (StandardScaler)

- Entrena 5 modelos independientes2. Aplicar PCA y visualizar varianza explicada (scree plot)

- Cada participante se valida exactamente 1 vez3. Proyectar datos en 2D/3D (PC1 vs PC2 vs PC3)

- Métricas robustas: **Mean ± Std** (ej: 60% ± 12%)4. Colorear por grupo, sexo y participante para identificar patrones

- Refleja mejor la generalización real

**Interpretación**: 

**Ejemplo con K=5:**- ¿Se separan los grupos CT y ELA en el espacio PCA?

```- ¿Qué métricas contribuyen más a cada componente?

Fold 1: Train [16 participants] → Val [4 participants]

Fold 2: Train [16 participants] → Val [4 participants] (diferentes)### Fase 3: Variational Autoencoder (VAE) con PyTorch Lightning

Fold 3: Train [16 participants] → Val [4 participants] (diferentes)

Fold 4: Train [16 participants] → Val [4 participants] (diferentes)**Objetivo**: Aprender una representación probabilística comprimida del espacio latente

Fold 5: Train [16 participants] → Val [4 participants] (diferentes)

**Dos Arquitecturas Disponibles**:

Resultado final: Val Accuracy = Mean(Fold1, ..., Fold5) ± Std

```#### 1. VAE Estándar (Mean Pooling)

```

### Cuándo usar cada modoInput (8 features agregadas) → Encoder [64, 32] → Latent 8D (μ, σ) → Decoder [32, 64] → Output (8 features)

                                                        ↓

| Modo | Cuándo usar | Ventajas | Desventajas |                                                 Classifier [16] → CT/ELA

|------|-------------|----------|-------------|```

| **Simple Split** | Testing rápido, datasets grandes (N>100) | Rápido (1 modelo) | Menos robusto para N pequeño |- **Ventaja**: Rápido, simple, interpretable

| **K-Fold** | Datasets pequeños (N<100), reportes científicos | Métricas robustas, menos sesgo | Más lento (K modelos) |- **Desventaja**: Pierde variabilidad intra-participante



## 🎓 Entrenamiento#### 2. LSTM-VAE (Sequences)

```

### Opción 1: Desde Streamlit (Recomendado)Input (secuencias 4-36 mediciones × 8 features) → Bidirectional LSTM Encoder (2 capas, hidden=64)

                                                        ↓

1. Ejecutar `streamlit run app.py`                                                 Latent 16D (μ, σ)

2. Ir a página **🎓 Entrenar Modelo**                                                        ↓

3. Seleccionar modo:                                          Decoder LSTM (2 capas, hidden=64)

   - **Train/Val Split Simple** (rápido)                                                        ↓

   - **K-Fold Cross-Validation** (robusto)                                          Output (secuencias reconstruidas)

4. Configurar hiperparámetros                                                        ↓

5. Click en **🚀 Iniciar Entrenamiento**                                          Classifier [32, 16] → CT/ELA

6. Ver resultados en página **🤖 Clasificador**```

- **Ventaja**: Preserva variabilidad intra-participante, mayor capacidad

### Opción 2: Desde terminal- **Desventaja**: Más lento, más parámetros (~205k vs ~6k)



```bash**Configuración**:

# Entrenamiento con K-Fold (K=5)- **Framework**: PyTorch Lightning (simplifica entrenamiento, logging automático)

python scripts/train_classifier.py- **Loss**: Reconstrucción + KL Divergence + Clasificación

- **Optimizer**: Adam con learning rate configurable

# Ver métricas con TensorBoard- **Logging**: TensorBoard embebido en Streamlit (tiempo real)

tensorboard --logdir logs/lstm_classifier- **Callbacks**: Early Stopping, ModelCheckpoint, LearningRateMonitor

```

**Monitoreo en Tiempo Real**:

## 📈 Interpretación de Resultados- TensorBoard se muestra **dentro de Streamlit** durante el entrenamiento

- Métricas: loss, accuracy, KL divergence, reconstruction error

### Métricas K-Fold- Visualizaciones: curvas de aprendizaje, histogramas de pesos



**Ejemplo:** Val Accuracy: 60% ± 12%**Visualización**:

- Proyección del espacio latente en 2D/3D por grupo (CT/ELA)

- **Mean (60%)**: Accuracy promedio esperada en nuevos participantes- Métricas de clasificación (accuracy, confusion matrix)

- **Std (12%)**: Variabilidad entre folds- Comparar reconstrucciones vs datos originales

  - **Std < 10%**: Modelo estable ✅- Identificar si la variabilidad intra-participante mejora la clasificación

  - **Std > 15%**: Alta variabilidad (normal con N=20) ⚠️

### Fase 4: Integración en Streamlit

### Accuracy Guidelines

**Arquitectura Multi-Page de Streamlit**:

| Accuracy | Interpretación |

|----------|----------------|La aplicación utiliza la estructura nativa de múltiples páginas de Streamlit:

| **>70%** | ✅ Excelente - Hay señal morfométrica clara entre CT/ELA |

| **60-70%** | ⚡ Bueno - Hay separabilidad, puede mejorar con tuning |1. **Home (app.py)**: 

| **50-60%** | ⚠️ Regular - Poco mejor que azar, revisar features |   - Descripción del proyecto y dataset

| **<50%** | ❌ Malo - Modelo no está aprendiendo |   - Métricas generales

   - Vista previa de los datos

### Outputs guardados

2. **📊 EDA (pages/1_📊_EDA.py)**:

**Simple Split:**   - Selección interactiva de métricas y grupos

```   - Gráficos de distribución (box, violin, histogram)

models/   - Matriz de correlación interactiva

├── lstm_classifier-epoch=XX-val_acc=X.XXXX.ckpt   - Pruebas estadísticas automáticas (t-test/ANOVA)

└── ...   - Scatter plot matrix

   - Análisis por edad y participante

logs/lstm_classifier/

└── version_0/3. **� Entrenar Modelo (pages/2_�_Entrenar_Modelo.py)** ⭐ **NUEVO**:

    └── events.out.tfevents...   - Selección de tipo de modelo (VAE estándar vs LSTM-VAE)

```   - Configuración interactiva de hiperparámetros:

     - Max epochs, learning rate, batch size, early stopping patience

**K-Fold:**   - **TensorBoard embebido en tiempo real** durante el entrenamiento

```   - Visualización de métricas: loss, accuracy, KL divergence

models/kfold_5/   - Ver entrenamientos anteriores y comparar runs

├── fold1-epoch=XX-val_acc=X.XXXX.ckpt   - Guías contextuales sobre arquitecturas y hiperparámetros

├── fold2-epoch=XX-val_acc=X.XXXX.ckpt   - Todo integrado - no necesitas abrir terminales adicionales

├── ...

└── summary.json  ← Métricas agregadas + hiperparámetros4. **🤖 Autoencoder (pages/3_🤖_Autoencoder.py)**:

   - Carga de modelos entrenados (VAE o LSTM-VAE)

logs/lstm_classifier_kfold_5/   - Detección automática del tipo de modelo

├── fold_1/   - Visualización del espacio latente 2D/3D (Plotly interactivo)

├── fold_2/   - Métricas de clasificación (accuracy, confusion matrix)

└── ...   - Análisis de reconstrucciones

```   - Comparación conceptual con PCA

   - Exportación del espacio latente

## 🛠️ Stack Tecnológico   - Ver logs históricos de TensorBoard



| Tecnología | Propósito |**Ventajas de esta arquitectura**:

|-----------|-----------|- ✅ Todo nativo en Streamlit (sin necesidad de frameworks adicionales)

| **PyTorch** | Deep learning framework |- ✅ Navegación automática mediante sidebar

| **PyTorch Lightning** | Training loop, callbacks, logging |- ✅ Cache de datos para mejor rendimiento

| **Streamlit** | Web app interactiva |- ✅ Visualizaciones interactivas con Plotly

| **TensorBoard** | Visualización de métricas |- ✅ **TensorBoard embebido en tiempo real** - sin abrir ventanas adicionales

| **scikit-learn** | PCA, cross-validation, metrics |- ✅ Entrenamiento del modelo integrado en la UI

| **Plotly** | Gráficos interactivos 3D |- ✅ Comparación fácil entre VAE estándar y LSTM-VAE

| **Pandas/NumPy** | Manipulación de datos |- ✅ Logs nativos de PyTorch Lightning visibles en TensorBoard

- ✅ Workflow completo: configurar → entrenar → monitorear → visualizar

## ⚙️ Hiperparámetros Recomendados

## 🆕 Características Destacadas

### Para empezar (baseline)

### TensorBoard en Tiempo Real

```yaml

hidden_dim: 64La nueva página de entrenamiento incluye **TensorBoard embebido** que muestra métricas en tiempo real:

num_layers: 2

dropout: 0.3- 📊 **Curvas de aprendizaje**: Loss y accuracy (train/validation)

learning_rate: 1e-3- 📈 **KL Divergence**: Regularización del espacio latente

batch_size: 16- 🔍 **Reconstruction Loss**: Calidad de reconstrucción

max_epochs: 50-100- 🎯 **Classification Metrics**: Accuracy de CT vs ELA

```- 📉 **Learning Rate**: Evolución durante entrenamiento



### Si el modelo no converge**Sin necesidad de:**

- Abrir terminales adicionales

```yaml- Ejecutar comandos TensorBoard manualmente

learning_rate: 5e-3  # Aumentar- Cambiar entre ventanas

dropout: 0.2         # Reducir

hidden_dim: 128      # Aumentar capacidad**Todo en una sola interfaz web integrada.**

```

### Dos Modelos para Comparar

### Si hay overfitting (train_acc >> val_acc)

1. **VAE Estándar (Mean Pooling)**:

```yaml   - Agrega múltiples mediciones por participante

dropout: 0.4-0.5     # Aumentar regularización   - ~6,700 parámetros

hidden_dim: 32       # Reducir capacidad   - Entrenamiento rápido (~2-5 min)

num_layers: 1        # Simplificar arquitectura   - Baseline sólido

val_split: 0.3       # Más datos de validación

```2. **LSTM-VAE (Sequences)**:

   - Preserva variabilidad intra-participante

### Si hay underfitting (ambos accuracy bajos)   - ~205,850 parámetros

   - Entrenamiento más lento (~5-15 min)

```yaml   - Captura patrones temporales/secuenciales

hidden_dim: 128-256  # Aumentar capacidad

num_layers: 3        # Más profundidad**Pregunta de Investigación**: ¿La variabilidad intra-participante mejora la clasificación CT vs ELA?

dropout: 0.2         # Menos regularización

max_epochs: 100-200  # Más tiempo de entrenamiento## 🛠️ Tecnologías Utilizadas

```

- **Python 3.8+**: Lenguaje base

## 🐛 Troubleshooting- **Streamlit**: Framework para la aplicación web interactiva

- **PyTorch**: Framework de deep learning

**Error: Module not found**- **PyTorch Lightning**: Wrapper para simplificar entrenamiento y logging

```bash- **TensorBoard**: Visualización de métricas de entrenamiento (integrado con Lightning)

pip install -r requirements.txt- **Pandas & NumPy**: Manipulación de datos

```- **Scikit-learn**: PCA, normalización, métricas

- **Plotly & Seaborn**: Visualizaciones interactivas y estáticas

**CUDA out of memory**- **Matplotlib**: Gráficos complementarios

```python

# En train_classifier.py, reducir batch_size## 📊 Dataset

batch_size: 8  # o 4

```- **Formato**: CSV

- **Filas**: Observaciones de mitocondrias individuales

**Model not converging**- **Columnas**: 12 (métricas morfológicas + variables demográficas)

- Aumentar learning_rate a 5e-3- **Grupos**: CT (Control) y ELA (Esclerosis Lateral Amiotrófica)

- Reducir dropout a 0.2

- Aumentar max_epochs a 100-200## 🔍 Preguntas de Investigación



**High variance across folds**1. ¿Existen diferencias morfológicas significativas entre grupos CT y ELA?

- Normal con N=202. ¿Las métricas de superficie, volumen y longitud están correlacionadas?

- Probar diferentes random_state3. ¿El PCA revela separación natural entre grupos?

- Aumentar early_stopping_patience4. ¿El autoencoder captura patrones no lineales que el PCA no detecta?

- Considerar feature engineering5. ¿Hay clusterización por participante o características demográficas?

6. **¿La variabilidad intra-participante (LSTM-VAE) mejora la clasificación vs mean pooling (VAE estándar)?** ⭐

**Puerto 8501 ocupado**

```bash## 📚 Documentación Adicional

streamlit run app.py --server.port 8502

```- **`LSTM_VAE_ARCHITECTURE.md`**: Guía técnica detallada de la arquitectura LSTM-VAE

- **`docs/TRAINING_GUIDE.md`**: Guía completa de entrenamiento con TensorBoard

## 📝 Próximos Pasos- **`TENSORBOARD_INTEGRATION_SUMMARY.md`**: Resumen de integración y características

- **`test_lstm_vae.py`**: Script de validación de la implementación LSTM-VAE

1. **Feature Engineering**: Explorar ratios, combinaciones no lineales- **`test_tensorboard_integration.py`**: Test de integración de TensorBoard

2. **Ensemble**: Promediar predicciones de los K modelos

3. **Interpretabilidad**: SHAP values, attention weights## 🤝 Contribuciones

4. **Más datos**: Si es posible, aumentar N participantes

5. **Transfer learning**: Pre-entrenar en datasets similaresEste es un proyecto de investigación. Las sugerencias y mejoras son bienvenidas.



---## 📝 Licencia



**Última actualización**: Noviembre 2025[Especificar licencia si aplica]


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
