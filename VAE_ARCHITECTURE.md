# Arquitectura VAE para Morfología Mitocondrial

## 🎯 Objetivo

Utilizar un **Variational Autoencoder (VAE)** con clasificación integrada para:

1. Aprender representaciones latentes de métricas morfológicas mitocondriales
2. Clasificar participantes en grupos CT (Control) vs ELA (Esclerosis Lateral Amiotrófica)
3. Explorar el espacio latente para encontrar patrones discriminativos

## 🏗️ Arquitectura

### Componentes Principales

```
Input (8 features)
    ↓
┌─────────────────────┐
│   Encoder Network   │
│   [64] → [32]       │
│   BatchNorm + ReLU  │
│   + Dropout(0.2)    │
└─────────────────────┘
    ↓           ↓
   [μ]       [log σ²]  ← Latent parameters
    └─────┬─────┘
          ↓
    Reparameterization
    z = μ + σ * ε
          ↓
    ┌─────┴─────┐
    ↓           ↓
┌────────────┐ ┌──────────────┐
│  Decoder   │ │ Classifier   │
│  [32]→[64] │ │    [16]      │
│  →8 feat   │ │  →2 classes  │
└────────────┘ └──────────────┘
```

### Detalles Técnicos

**Encoder:**
- Input: 8 features (métricas morfológicas estandarizadas)
- Hidden: [64, 32] con BatchNorm + ReLU + Dropout(0.2)
- Output: μ (mean) y log σ² (log variance) del espacio latente (8D)

**Reparameterization Trick:**
```python
z = μ + σ * ε, donde ε ~ N(0, 1)
```
Permite backpropagation a través de muestras aleatorias.

**Decoder:**
- Input: Latent vector z (8D)
- Hidden: [32, 64] con BatchNorm + ReLU + Dropout(0.2)
- Output: Reconstrucción de las 8 features

**Classifier:**
- Input: Latent vector z (8D)
- Hidden: [16] con BatchNorm + ReLU + Dropout(0.2)
- Output: Logits para 2 clases (CT=0, ELA=1)

## 📊 Manejo de Datos

### Problema: Múltiples Medidas por Participante

Cada participante tiene **n medidas distintas** (observaciones independientes de mitocondrias). 

**Solución implementada:**
```python
# Agregación por participante (mean pooling)
data_agg = data.groupby('Participant').agg({
    'PROM IsoVol': 'mean',
    'PROM Surface': 'mean',
    # ... resto de features
})
```

**Ventajas:**
- ✅ Evita data leakage (medidas del mismo participante no aparecen en train y val)
- ✅ Cada sample representa un participante completo
- ✅ Clasificación a nivel de participante (más interpretable)

**Alternativas consideradas:**
- ❌ Usar todas las medidas individualmente → data leakage
- ❌ Attention mechanism → complejidad innecesaria para este dataset

### Split Train/Validation

```python
# Split por participantes (no por medidas)
train_participants = 80% 
val_participants = 20%
```

Esto garantiza que todas las medidas de un participante están en train O en val, nunca en ambos.

## 🔢 Función de Pérdida

El VAE combina 3 componentes de pérdida:

### 1. Reconstruction Loss (MSE)
```python
L_recon = MSE(x_reconstructed, x_original)
```
Qué tan bien el decoder reconstruye el input.

### 2. KL Divergence Loss
```python
L_KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```
Regulariza el espacio latente hacia N(0,1). Permite:
- Espacios latentes continuos
- Interpolación suave
- Generación de nuevos datos

### 3. Classification Loss (Cross-Entropy)
```python
L_class = CrossEntropy(predicted_class, true_class)
```
Entrena el clasificador para predecir CT vs ELA.

### Pérdida Total
```python
L_total = L_recon + α * L_KL + β * L_class

α = 0.001  # KL weight (bajo para evitar "posterior collapse")
β = 1.0    # Classification weight
```

## 🎓 Entrenamiento

### Hiperparámetros

```yaml
batch_size: 16          # Pequeño porque son participantes agregados
learning_rate: 0.0005   # Bajo para estabilidad del VAE
max_epochs: 200         # VAEs necesitan más épocas
early_stopping: 20      # Paciencia aumentada
optimizer: AdamW        # Con weight_decay=0.01
scheduler: ReduceLROnPlateau
```

### Callbacks

1. **EarlyStopping**: Para en `val_loss` con patience=20
2. **ModelCheckpoint**: Guarda top-3 modelos por `val_loss`
3. **LearningRateMonitor**: Registra LR en TensorBoard

### Métricas Monitoreadas

Durante entrenamiento se registran:
- `train_loss`, `val_loss` (total)
- `train_recon`, `val_recon` (reconstrucción)
- `train_kl`, `val_kl` (KL divergence)
- `train_class_loss`, `val_class_loss` (clasificación)
- `train_acc`, `val_acc` (accuracy de clasificación)

## 📈 Interpretación

### Espacio Latente

El espacio latente 8D captura:
- Patrones morfológicos mitocondriales
- Información discriminativa entre CT/ELA
- Estructura probabilística (μ, σ) en lugar de puntos fijos

**Visualización:**
- Proyecciones 2D/3D usando primeras dimensiones latentes
- Colorear por: Group, Prediction, Correcto, Sex
- Buscar clusterización de grupos

### Clasificación

Si el modelo alcanza **accuracy > 70%**, indica que:
- Las métricas morfológicas tienen poder discriminativo
- El espacio latente captura diferencias entre CT/ELA
- Hay patrones subyacentes en la morfología mitocondrial

### Comparación con PCA

| Aspecto | PCA | VAE |
|---------|-----|-----|
| Tipo | Lineal | No lineal |
| Complejidad | Baja | Alta |
| Interpretabilidad | Alta | Media |
| Clasificación | No | Sí (integrada) |
| Generación | No | Sí |
| Mejor para | EDA rápido | Análisis profundo |

## 🚀 Uso

### Entrenar

```bash
# Opción 1: Desde terminal
python scripts/train_autoencoder.py

# Opción 2: Desde Streamlit
streamlit run app.py
# → Ir a página "🤖 Autoencoder"
# → Click "🚀 Entrenar Autoencoder"
```

### Monitorear

```bash
tensorboard --logdir=logs/
# Abrir http://localhost:6006
```

### Evaluar

```bash
streamlit run app.py
# → Página "🤖 Autoencoder"
# → Seleccionar modelo entrenado
# → Ver accuracy, matriz de confusión, espacio latente
```

## 📚 Referencias

**Inspiración:**
- Nature Methods paper on quantized VAEs for biological data
- VAE original paper: Kingma & Welling (2013)
- PyTorch Lightning documentation

**Implementación:**
- `src/autoencoder.py`: Código del modelo VAE
- `scripts/train_autoencoder.py`: Script de entrenamiento
- `pages/3_🤖_Autoencoder.py`: Interfaz Streamlit
- `config/config.yaml`: Configuración de hiperparámetros

## 💡 Próximos Pasos

**Mejoras posibles:**
1. **β-VAE**: Aumentar peso de KL para espacio latente más disentangled
2. **Attention mechanism**: Para ponderar medidas por participante
3. **Conditional VAE**: Condicionar en Sex o Age
4. **Ensemble**: Combinar múltiples VAEs entrenados
5. **Transfer learning**: Pre-entrenar en dataset más grande

**Análisis adicionales:**
1. Importancia de features (gradient-based)
2. Interpolación en espacio latente
3. Generación de muestras sintéticas
4. Análisis de dimensiones latentes individuales
