# Quick Start Guide - Mitochondrial Morphology Analysis# Quick Start Guide - Mitochondrial Morphology Analysis



## 🚀 Inicio Rápido## 🚀 Inicio Rápido



### 1. Instalar dependencias### 1. Instalar dependencias

```bash```bash

# Crear entorno virtual (recomendado)pip install -r requirements.txt

python -m venv venv```

source venv/bin/activate  # Linux/Mac

# venv\Scripts\activate  # Windows### 2. Ejecutar la aplicación

```bash

# Instalar dependenciasstreamlit run app.py

pip install -r requirements.txt```

```

### 3. Navegación

### 2. Ejecutar la aplicación- **Home**: Vista general del proyecto y dataset

```bash- **📊 EDA**: Análisis exploratorio interactivo

streamlit run app.py- **🎯 PCA**: Reducción dimensional con PCA

```- **🤖 Autoencoder**: Entrenamiento y visualización del espacio latente



La aplicación se abrirá en `http://localhost:8501`### 4. Entrenar el Autoencoder



### 3. Navegación**Opción A**: Desde la interfaz web (página Autoencoder)

- **Home**: Vista general del proyecto y dataset

- **📊 EDA**: Análisis exploratorio interactivo**Opción B**: Desde terminal

- **🎯 PCA**: Reducción dimensional con PCA```bash

- **🎓 Entrenar Modelo**: Entrenamiento de LSTM Classifier con K-Fold CVpython scripts/train_autoencoder.py

- **🤖 Clasificador**: Visualización de resultados```



### 4. Entrenar el Clasificador### 5. Ver logs de entrenamiento

```bash

**Opción A**: Desde la interfaz web (página 🎓 Entrenar Modelo)tensorboard --logdir=logs/

- Selecciona modo: Train/Val Split Simple o **K-Fold CV (recomendado)**```

- Configura hiperparámetros (hidden_dim, learning_rate, dropout, etc.)Luego abre: http://localhost:6006

- Click en "🚀 Iniciar Entrenamiento"

- Ve métricas en tiempo real con TensorBoard## 📊 Estructura del Dataset



**Opción B**: Desde terminal- **Observaciones**: 385 mediciones de mitocondrias

```bash- **Participantes**: 20 (10 CT, 10 ELA)

# Entrenar con K-Fold Cross-Validation (K=5 por defecto)- **Grupos**: 

python scripts/train_classifier.py  - CT (Control): 195 observaciones

```  - ELA (Esclerosis Lateral Amiotrófica): 190 observaciones

- **Métricas**: 8 features morfológicas (IsoVol, Surface, Length, RoughSph)

### 5. Ver logs de entrenamiento- **Variables demográficas**: Age, Sex, Participant

```bash

tensorboard --logdir=logs/lstm_classifier## 🎯 Objetivos

```

Luego abre: http://localhost:60061. **EDA**: Identificar diferencias estadísticas entre grupos CT y ELA

2. **PCA**: Visualizar estructura de datos en espacio reducido

## 📊 Estructura del Dataset3. **Autoencoder**: Capturar relaciones no lineales y explorar clusterización



- **Observaciones**: 306 mediciones de mitocondrias## 📝 Configuración

- **Participantes**: 20 (10 CT, 10 ELA)

- **Grupos**: Edita `config/config.yaml` para modificar:

  - CT (Control): 167 observaciones- Arquitectura del autoencoder

  - ELA (Esclerosis Lateral Amiotrófica): 139 observaciones- Hiperparámetros de entrenamiento

- **Métricas**: 8 features morfológicas (IsoVol, Surface, Length, RoughSph - SUMA/PROM)- Número de componentes PCA

- **Variables demográficas**: Age, Sex, Participant (no usadas como input del modelo)- Rutas de datos



## 🎯 Objetivos## 🛠️ Stack Tecnológico



1. **EDA**: Identificar diferencias estadísticas entre grupos CT y ELA- **Streamlit**: Framework web interactivo

2. **PCA**: Visualizar estructura de datos en espacio reducido- **PyTorch + Lightning**: Deep learning con logging automático

3. **LSTM Classifier**: Clasificación supervisada CT vs ELA usando secuencias de longitud variable- **TensorBoard**: Visualización de métricas de entrenamiento

4. **K-Fold CV**: Obtener métricas robustas (mean ± std) para dataset pequeño (N=20)- **Plotly**: Gráficos interactivos 3D

- **Scikit-learn**: PCA y preprocesamiento

## 🧠 Modelo- **Pandas/NumPy**: Manipulación de datos



**LSTM Bidirectional Classifier**:## ⚡ Tips

- Input: (batch, seq_len, 8) - secuencias de longitud variable (4-36 mediciones/participante)

- 2 capas LSTM bidireccionales (hidden_dim=64)- Los datos se cachean automáticamente en Streamlit (mejor rendimiento)

- Classifier head: FC 128→64→32→2 con Dropout- Usa `Ctrl+C` en terminal para detener la app

- Loss: Cross Entropy- Los modelos entrenados se guardan en `models/`

- Optimizer: Adam + ReduceLROnPlateau- Los logs se generan automáticamente en `logs/`

- ~147K parámetros entrenables- Todos los gráficos son interactivos (zoom, pan, hover)



## 📝 Configuración## 🐛 Troubleshooting



Edita `config/config.yaml` para modificar:**Error: Module not found**

- Arquitectura del clasificador```bash

- Hiperparámetros de entrenamientopip install -r requirements.txt

- Callbacks (early stopping, checkpointing)```

- Número de componentes PCA

- Rutas de datos**No se encuentra el dataset**

- Verifica que `data/data.csv` existe

## 🔬 K-Fold Cross-Validation

**Error al entrenar**

**¿Por qué K-Fold para N=20?**- Verifica instalación de PyTorch: `pip install torch pytorch-lightning`



Con solo 20 participantes, un train/val split simple (16/4) es muy sensible al azar. K-Fold entrena K modelos independientes y reporta métricas robustas:**Puerto 8501 ocupado**

```bash

**K=5 (recomendado)**:streamlit run app.py --server.port 8502

- 5 modelos independientes```

- Cada participante validado exactamente 1 vez
- Métricas: **Val Accuracy = Mean ± Std** (ej: 60% ± 12%)

**Interpretación**:
- Mean: Accuracy esperada en nuevos participantes
- Std < 10%: Modelo estable ✅
- Std > 15%: Alta variabilidad (normal con N=20) ⚠️

## 🛠️ Stack Tecnológico

- **Streamlit**: Framework web interactivo
- **PyTorch + Lightning**: Deep learning con logging automático
- **TensorBoard**: Visualización de métricas de entrenamiento
- **Plotly**: Gráficos interactivos 3D
- **Scikit-learn**: PCA, cross-validation, metrics
- **Pandas/NumPy**: Manipulación de datos

## ⚡ Tips

- Usa **K-Fold CV** para obtener métricas robustas con dataset pequeño
- Los datos se cachean automáticamente en Streamlit (mejor rendimiento)
- Los modelos entrenados se guardan en `models/`
- K-Fold guarda summary.json con métricas agregadas y hiperparámetros
- Los logs se generan automáticamente en `logs/`
- Todos los gráficos son interactivos (zoom, pan, hover)

## 🐛 Troubleshooting

**Error: Module not found**
```bash
pip install -r requirements.txt
```

**No se encuentra el dataset**
- Verifica que `data/data.csv` existe

**CUDA out of memory**
- Reduce batch_size (ej: 8 o 4)

**Modelo no converge**
- Aumenta learning_rate (5e-3)
- Reduce dropout (0.2)
- Aumenta max_epochs (100-200)

**Puerto 8501 ocupado**
```bash
streamlit run app.py --server.port 8502
```

---

Ver `README.md` para documentación completa.
