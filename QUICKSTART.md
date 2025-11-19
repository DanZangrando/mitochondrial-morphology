# Quick Start Guide - Mitochondrial Morphology Analysis# Quick Start Guide - Mitochondrial Morphology Analysis# Quick Start Guide - Mitochondrial Morphology Analysis



## 🚀 Inicio Rápido



### 1. Instalar dependencias## 🚀 Inicio Rápido## 🚀 Inicio Rápido



```bash

pip install -r requirements.txt

```### 1. Instalar dependencias### 1. Instalar dependencias



### 2. Ejecutar la aplicación```bash```bash



```bash# Crear entorno virtual (recomendado)pip install -r requirements.txt

streamlit run app.py

```python -m venv venv```



### 3. Navegaciónsource venv/bin/activate  # Linux/Mac



La aplicación se abrirá en `http://localhost:8501` con 3 páginas:# venv\Scripts\activate  # Windows### 2. Ejecutar la aplicación



- **📊 EDA**: Análisis exploratorio interactivo```bash

- **🎯 PCA**: Reducción dimensional con PCA

- **🎯 Entrenar Clasificador**: Entrenamiento + Evaluación integrada# Instalar dependenciasstreamlit run app.py



### 4. Entrenar el Clasificadorpip install -r requirements.txt```



**Opción A - Desde la interfaz web (Recomendado)**:```



1. Ve a la página "🎯 Entrenar Clasificador"### 3. Navegación

2. Configura hiperparámetros en el sidebar:

   - Hidden Dim (64 por defecto)### 2. Ejecutar la aplicación- **Home**: Vista general del proyecto y dataset

   - Learning Rate (1e-3)

   - Dropout (0.3)```bash- **📊 EDA**: Análisis exploratorio interactivo

   - Epochs (50)

   - Batch Size (16)streamlit run app.py- **🎯 PCA**: Reducción dimensional con PCA

3. Click en "🚀 Iniciar Entrenamiento"

4. Ve el progreso en tiempo real```- **🤖 Autoencoder**: Entrenamiento y visualización del espacio latente

5. **Resultados se muestran automáticamente** al terminar



**Opción B - Desde terminal**:

La aplicación se abrirá en `http://localhost:8501`### 4. Entrenar el Autoencoder

```bash

python scripts/train_classifier.py

```

### 3. Navegación**Opción A**: Desde la interfaz web (página Autoencoder)

### 5. Ver logs de entrenamiento

- **Home**: Vista general del proyecto y dataset

```bash

tensorboard --logdir=logs/lstm_classifier- **📊 EDA**: Análisis exploratorio interactivo**Opción B**: Desde terminal

```

- **🎯 PCA**: Reducción dimensional con PCA```bash

Luego abre: http://localhost:6006

- **🎓 Entrenar Modelo**: Entrenamiento de LSTM Classifier con K-Fold CVpython scripts/train_autoencoder.py

## 📊 Estructura del Dataset

- **🤖 Clasificador**: Visualización de resultados```

- **Observaciones**: 306 mediciones de mitocondrias

- **Participantes**: 20 (10 CT, 10 ELA)

- **Grupos**: 

  - CT (Control): 167 observaciones### 4. Entrenar el Clasificador### 5. Ver logs de entrenamiento

  - ELA (Esclerosis Lateral Amiotrófica): 139 observaciones

- **Métricas**: 8 features morfológicas (IsoVol, Surface, Length, RoughSph - SUMA/PROM)```bash

- **Variables demográficas**: Age, Sex, Participant (no usadas en el modelo)

**Opción A**: Desde la interfaz web (página 🎓 Entrenar Modelo)tensorboard --logdir=logs/

## 🎯 Objetivos

- Selecciona modo: Train/Val Split Simple o **K-Fold CV (recomendado)**```

1. **EDA**: Identificar diferencias estadísticas entre grupos CT y ELA

2. **PCA**: Visualizar estructura de datos en espacio reducido- Configura hiperparámetros (hidden_dim, learning_rate, dropout, etc.)Luego abre: http://localhost:6006

3. **LSTM Classifier**: Clasificar participantes CT vs ELA con significancia estadística

- Click en "🚀 Iniciar Entrenamiento"

## 📈 Evaluación con P-Value

- Ve métricas en tiempo real con TensorBoard## 📊 Estructura del Dataset

### Doble Evaluación



Al entrenar, verás **dos matrices de confusión**:

**Opción B**: Desde terminal- **Observaciones**: 385 mediciones de mitocondrias

1. **Validación** (azul):

   - Solo participantes de validación```bash- **Participantes**: 20 (10 CT, 10 ELA)

   - Métricas reales de generalización

   - **Incluye p-value del test binomial**# Entrenar con K-Fold Cross-Validation (K=5 por defecto)- **Grupos**: 



2. **Dataset Completo** (verde):python scripts/train_classifier.py  - CT (Control): 195 observaciones

   - Todos los participantes (train + val)

   - Solo para referencia```  - ELA (Esclerosis Lateral Amiotrófica): 190 observaciones



### Interpretación del P-Value- **Métricas**: 8 features morfológicas (IsoVol, Surface, Length, RoughSph)



El test binomial evalúa: ¿El modelo clasifica mejor que el azar (50%)?### 5. Ver logs de entrenamiento- **Variables demográficas**: Age, Sex, Participant



- **p < 0.05**: ✅ Significativo - el modelo aprende```bash

- **p ≥ 0.05**: ⚠️ No significativo - puede ser azar

tensorboard --logdir=logs/lstm_classifier## 🎯 Objetivos

**Símbolos**:

- `***`: p < 0.001 (altamente significativo)```

- `**`: p < 0.01 (muy significativo)

- `*`: p < 0.05 (significativo)Luego abre: http://localhost:60061. **EDA**: Identificar diferencias estadísticas entre grupos CT y ELA

- `ns`: p ≥ 0.05 (no significativo)

2. **PCA**: Visualizar estructura de datos en espacio reducido

## 📝 Configuración

## 📊 Estructura del Dataset3. **Autoencoder**: Capturar relaciones no lineales y explorar clusterización

Edita `config/config.yaml` para modificar:



- Arquitectura del clasificador

- Hiperparámetros de entrenamiento- **Observaciones**: 306 mediciones de mitocondrias## 📝 Configuración

- Callbacks (early stopping, checkpointing)

- Número de componentes PCA- **Participantes**: 20 (10 CT, 10 ELA)

- Rutas de datos

- **Grupos**: Edita `config/config.yaml` para modificar:

## 🛠️ Stack Tecnológico

  - CT (Control): 167 observaciones- Arquitectura del autoencoder

- **Streamlit**: Framework web interactivo

- **PyTorch + Lightning**: Deep learning con logging automático  - ELA (Esclerosis Lateral Amiotrófica): 139 observaciones- Hiperparámetros de entrenamiento

- **TensorBoard**: Visualización de métricas de entrenamiento

- **Plotly**: Gráficos interactivos modernos (confusion matrices)- **Métricas**: 8 features morfológicas (IsoVol, Surface, Length, RoughSph - SUMA/PROM)- Número de componentes PCA

- **Scikit-learn**: Normalización, stratification, métricas

- **SciPy**: Test estadístico binomial- **Variables demográficas**: Age, Sex, Participant (no usadas como input del modelo)- Rutas de datos

- **Pandas/NumPy**: Manipulación de datos



## ⚡ Tips

## 🎯 Objetivos## 🛠️ Stack Tecnológico

- Los datos se cachean automáticamente en Streamlit (mejor rendimiento)

- Usa `Ctrl+C` en terminal para detener la app

- Los modelos entrenados se guardan en `models/`

- Cada modelo guarda metadata con participantes train/val en JSON1. **EDA**: Identificar diferencias estadísticas entre grupos CT y ELA- **Streamlit**: Framework web interactivo

- Los logs se generan automáticamente en `logs/`

- Todos los gráficos son interactivos (zoom, pan, hover)2. **PCA**: Visualizar estructura de datos en espacio reducido- **PyTorch + Lightning**: Deep learning con logging automático

- Las matrices de confusión tienen gradientes de color modernos

3. **LSTM Classifier**: Clasificación supervisada CT vs ELA usando secuencias de longitud variable- **TensorBoard**: Visualización de métricas de entrenamiento

## 🐛 Troubleshooting

4. **K-Fold CV**: Obtener métricas robustas (mean ± std) para dataset pequeño (N=20)- **Plotly**: Gráficos interactivos 3D

**Error: Module not found**

```bash- **Scikit-learn**: PCA y preprocesamiento

pip install -r requirements.txt

```## 🧠 Modelo- **Pandas/NumPy**: Manipulación de datos



**No se encuentra el dataset**

- Verifica que `data/data.csv` existe

**LSTM Bidirectional Classifier**:## ⚡ Tips

**Error al entrenar**

- Verifica instalación de PyTorch: `pip install torch pytorch-lightning`- Input: (batch, seq_len, 8) - secuencias de longitud variable (4-36 mediciones/participante)

- Verifica scipy: `pip install scipy`

- 2 capas LSTM bidireccionales (hidden_dim=64)- Los datos se cachean automáticamente en Streamlit (mejor rendimiento)

**Puerto 8501 ocupado**

```bash- Classifier head: FC 128→64→32→2 con Dropout- Usa `Ctrl+C` en terminal para detener la app

streamlit run app.py --server.port 8502

```- Loss: Cross Entropy- Los modelos entrenados se guardan en `models/`



**CUDA out of memory**- Optimizer: Adam + ReduceLROnPlateau- Los logs se generan automáticamente en `logs/`

- Reduce batch_size (ej: 8 o 4)

- ~147K parámetros entrenables- Todos los gráficos son interactivos (zoom, pan, hover)

**Modelo no converge**

- Aumenta learning_rate (5e-3)

- Reduce dropout (0.2)

- Aumenta max_epochs (100-200)## 📝 Configuración## 🐛 Troubleshooting



**Error con scipy.stats.binom_test**

- El código usa `binomtest` (versión moderna)

- Si tienes scipy < 1.7, actualiza: `pip install --upgrade scipy`Edita `config/config.yaml` para modificar:**Error: Module not found**



## 🎯 Flujo de Trabajo Típico- Arquitectura del clasificador```bash



1. **Exploración inicial**: Página EDA para entender distribuciones- Hiperparámetros de entrenamientopip install -r requirements.txt

2. **Análisis PCA**: Ver si hay separación lineal CT vs ELA

3. **Entrenar modelo**: Página Entrenar Clasificador- Callbacks (early stopping, checkpointing)```

4. **Evaluar significancia**: Revisar p-value en matriz de validación

5. **Ajustar hiperparámetros** si es necesario- Número de componentes PCA

6. **Repetir con diferentes seeds** para robustez

- Rutas de datos**No se encuentra el dataset**

## 📊 Métricas Clave

- Verifica que `data/data.csv` existe

Al evaluar el modelo, observa:

## 🔬 K-Fold Cross-Validation

- **Accuracy de validación**: ¿>70%? ✅ Excelente

- **P-value**: ¿<0.05? ✅ Significativo estadísticamente**Error al entrenar**

- **Matriz de confusión**: ¿Balanceada entre CT/ELA?

- **Per-class accuracy**: ¿Una clase mucho peor que otra?**¿Por qué K-Fold para N=20?**- Verifica instalación de PyTorch: `pip install torch pytorch-lightning`



## 🗂️ Archivos Generados



Después del entrenamiento:Con solo 20 participantes, un train/val split simple (16/4) es muy sensible al azar. K-Fold entrena K modelos independientes y reporta métricas robustas:**Puerto 8501 ocupado**



``````bash

models/

├── lstm_classifier-epoch=XX-val_acc=0.XXXX.ckpt        # Pesos del modelo**K=5 (recomendado)**:streamlit run app.py --server.port 8502

└── lstm_classifier-epoch=XX-val_acc=0.XXXX_metadata.json  # Metadata

- 5 modelos independientes```

logs/lstm_classifier/

└── version_0/- Cada participante validado exactamente 1 vez

    └── events.out.tfevents...  # Logs de TensorBoard- Métricas: **Val Accuracy = Mean ± Std** (ej: 60% ± 12%)

```

**Interpretación**:

El archivo metadata.json contiene:- Mean: Accuracy esperada en nuevos participantes

- Lista de participantes train/val- Std < 10%: Modelo estable ✅

- Hiperparámetros usados- Std > 15%: Alta variabilidad (normal con N=20) ⚠️

- Val split y random state

- Best score obtenido## 🛠️ Stack Tecnológico



Esto permite **reproducibilidad completa** del experimento.- **Streamlit**: Framework web interactivo

- **PyTorch + Lightning**: Deep learning con logging automático

---- **TensorBoard**: Visualización de métricas de entrenamiento

- **Plotly**: Gráficos interactivos 3D

Ver `README.md` para documentación completa.- **Scikit-learn**: PCA, cross-validation, metrics

- **Pandas/NumPy**: Manipulación de datos

## 🔗 Links Útiles

## ⚡ Tips

- **TensorBoard**: http://localhost:6006 (después de `tensorboard --logdir=logs/`)

- **Streamlit**: http://localhost:8501- Usa **K-Fold CV** para obtener métricas robustas con dataset pequeño

- **Documentación PyTorch Lightning**: https://lightning.ai/docs/pytorch/- Los datos se cachean automáticamente en Streamlit (mejor rendimiento)

- **Plotly Docs**: https://plotly.com/python/- Los modelos entrenados se guardan en `models/`

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
