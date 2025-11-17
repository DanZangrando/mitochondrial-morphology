# Quick Start Guide - Mitochondrial Morphology Analysis

## 🚀 Inicio Rápido

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar la aplicación
```bash
streamlit run app.py
```

### 3. Navegación
- **Home**: Vista general del proyecto y dataset
- **📊 EDA**: Análisis exploratorio interactivo
- **🎯 PCA**: Reducción dimensional con PCA
- **🤖 Autoencoder**: Entrenamiento y visualización del espacio latente

### 4. Entrenar el Autoencoder

**Opción A**: Desde la interfaz web (página Autoencoder)

**Opción B**: Desde terminal
```bash
python scripts/train_autoencoder.py
```

### 5. Ver logs de entrenamiento
```bash
tensorboard --logdir=logs/
```
Luego abre: http://localhost:6006

## 📊 Estructura del Dataset

- **Observaciones**: 385 mediciones de mitocondrias
- **Participantes**: 20 (10 CT, 10 ELA)
- **Grupos**: 
  - CT (Control): 195 observaciones
  - ELA (Esclerosis Lateral Amiotrófica): 190 observaciones
- **Métricas**: 8 features morfológicas (IsoVol, Surface, Length, RoughSph)
- **Variables demográficas**: Age, Sex, Participant

## 🎯 Objetivos

1. **EDA**: Identificar diferencias estadísticas entre grupos CT y ELA
2. **PCA**: Visualizar estructura de datos en espacio reducido
3. **Autoencoder**: Capturar relaciones no lineales y explorar clusterización

## 📝 Configuración

Edita `config/config.yaml` para modificar:
- Arquitectura del autoencoder
- Hiperparámetros de entrenamiento
- Número de componentes PCA
- Rutas de datos

## 🛠️ Stack Tecnológico

- **Streamlit**: Framework web interactivo
- **PyTorch + Lightning**: Deep learning con logging automático
- **TensorBoard**: Visualización de métricas de entrenamiento
- **Plotly**: Gráficos interactivos 3D
- **Scikit-learn**: PCA y preprocesamiento
- **Pandas/NumPy**: Manipulación de datos

## ⚡ Tips

- Los datos se cachean automáticamente en Streamlit (mejor rendimiento)
- Usa `Ctrl+C` en terminal para detener la app
- Los modelos entrenados se guardan en `models/`
- Los logs se generan automáticamente en `logs/`
- Todos los gráficos son interactivos (zoom, pan, hover)

## 🐛 Troubleshooting

**Error: Module not found**
```bash
pip install -r requirements.txt
```

**No se encuentra el dataset**
- Verifica que `data/data.csv` existe

**Error al entrenar**
- Verifica instalación de PyTorch: `pip install torch pytorch-lightning`

**Puerto 8501 ocupado**
```bash
streamlit run app.py --server.port 8502
```
