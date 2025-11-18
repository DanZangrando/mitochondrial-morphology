# 🎉 TensorBoard en Tiempo Real - Implementación Completa

## ✅ Resumen de Cambios

Se ha implementado exitosamente la integración de TensorBoard directamente en la aplicación Streamlit, permitiendo monitorear el entrenamiento en tiempo real sin necesidad de abrir una terminal adicional.

## 🚀 Características Nuevas

### 1. Nueva Página de Entrenamiento (`pages/2_🎓_Entrenar_Modelo.py`)

**Funcionalidades:**
- ✅ Selección de tipo de modelo (VAE estándar vs LSTM-VAE)
- ✅ Configuración de hiperparámetros interactivos
- ✅ Inicio de entrenamiento con un click
- ✅ TensorBoard embebido en iframe durante el entrenamiento
- ✅ Visualización de métricas en tiempo real
- ✅ Ver entrenamientos anteriores
- ✅ Guías y tips de entrenamiento

**Hiperparámetros Configurables:**
- Max Epochs (10-500)
- Learning Rate (0.00001-0.01)
- Batch Size (2-32, auto-ajustado por modelo)
- Early Stopping Patience (5-50)

### 2. Script de Entrenamiento Mejorado

**Archivo:** `scripts/train_autoencoder.py`

**Cambios:**
- ✅ Parámetros opcionales `max_epochs` y `batch_size`
- ✅ Override de configuración desde Streamlit
- ✅ Compatible con llamadas desde UI

### 3. Página de Autoencoder Actualizada

**Archivo:** `pages/3_🤖_Autoencoder.py`

**Mejoras:**
- ✅ Imports para TensorBoard embebido
- ✅ Soporte para subprocess y threading
- ✅ Visualización de logs históricos

### 4. Documentación Completa

**Archivo:** `docs/TRAINING_GUIDE.md`

**Contenido:**
- Guía paso a paso de entrenamiento
- Comparación VAE vs LSTM-VAE
- Interpretación de métricas
- Solución de problemas
- Tips de hiperparámetros
- Workflow completo

### 5. Script de Test

**Archivo:** `test_tensorboard_integration.py`

**Funcionalidad:**
- Verifica dependencias
- Prueba inicio de TensorBoard
- Valida conexión HTTP
- Detecta logs existentes

## 📦 Dependencias Agregadas

```bash
pip install streamlit-tensorboard
```

**Ya incluidas:**
- `tensorboard` (PyTorch Lightning dependency)
- `requests` (para verificar conexión)

## 🎯 Cómo Usar

### Opción 1: Interfaz Streamlit (Recomendado)

```bash
# 1. Iniciar la aplicación
streamlit run app.py

# 2. Navegar a: 🎓 Entrenar Modelo

# 3. Configurar y entrenar:
#    - Seleccionar tipo de modelo
#    - Ajustar hiperparámetros
#    - Click en "🚀 Iniciar Entrenamiento"
#    - Ver TensorBoard en tiempo real

# 4. Evaluar resultados:
#    - Ir a: 🤖 Autoencoder
#    - Cargar modelo entrenado
#    - Visualizar espacio latente
```

### Opción 2: Terminal (Tradicional)

```bash
# VAE estándar
python scripts/train_autoencoder.py

# LSTM-VAE
python scripts/train_autoencoder.py --lstm

# Ver logs
tensorboard --logdir=logs/
```

## 📊 Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────┐
│  Streamlit App (app.py)                             │
│  ┌───────────────────────────────────────────────┐  │
│  │  🏠 Home Page                                 │  │
│  │  - Dataset overview                           │  │
│  │  - Interactive filters                        │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
│  ┌───────────────────────────────────────────────┐  │
│  │  📊 EDA Page                                  │  │
│  │  - Statistical analysis                       │  │
│  │  - Visualizations                             │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
│  ┌───────────────────────────────────────────────┐  │
│  │  🎓 Train Model Page (NEW!)                   │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │  Configuration                          │  │  │
│  │  │  - Model type selection                 │  │  │
│  │  │  - Hyperparameters                      │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  │                                                │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │  Training Process                       │  │  │
│  │  │  ┌───────────────────────────────────┐  │  │  │
│  │  │  │  TensorBoard (port 6006)          │  │  │  │
│  │  │  │  - Loss curves                    │  │  │  │
│  │  │  │  - Accuracy metrics               │  │  │  │
│  │  │  │  - Learning rate                  │  │  │  │
│  │  │  └───────────────────────────────────┘  │  │  │
│  │  │                     ▲                   │  │  │
│  │  │                     │ HTTP iframe       │  │  │
│  │  │                     │                   │  │  │
│  │  │  ┌─────────────────▼─────────────────┐  │  │  │
│  │  │  │  train_autoencoder.py             │  │  │  │
│  │  │  │  - PyTorch Lightning              │  │  │  │
│  │  │  │  - TensorBoardLogger              │  │  │  │
│  │  │  │  - Model training loop            │  │  │  │
│  │  │  └───────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  │                                                │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │  View Previous Runs                     │  │  │
│  │  │  - Compare multiple trainings           │  │  │
│  │  │  - Filter by model type                 │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
│  ┌───────────────────────────────────────────────┐  │
│  │  🤖 Autoencoder Page                          │  │
│  │  - Load trained models                        │  │
│  │  - Visualize latent space                     │  │
│  │  - Classification metrics                     │  │
│  │  - Reconstruction quality                     │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘

         ▼                           ▼
┌────────────────────┐    ┌─────────────────────┐
│  logs/             │    │  models/            │
│  ├─ vae_classifier │    │  ├─ vae-*.ckpt      │
│  └─ lstm_vae_*     │    │  └─ lstm_vae-*.ckpt │
└────────────────────┘    └─────────────────────┘
```

## 🔄 Flujo de Trabajo

```
Usuario abre Streamlit
    │
    ├─> Página Entrenar Modelo
    │       │
    │       ├─> Selecciona tipo de modelo (VAE / LSTM-VAE)
    │       │
    │       ├─> Configura hiperparámetros
    │       │
    │       ├─> Click "Iniciar Entrenamiento"
    │       │       │
    │       │       ├─> Limpia procesos TensorBoard previos
    │       │       │
    │       │       ├─> Inicia TensorBoard en puerto 6006
    │       │       │
    │       │       ├─> Muestra iframe con TensorBoard
    │       │       │
    │       │       ├─> Ejecuta train_vae()
    │       │       │       │
    │       │       │       ├─> Carga datos
    │       │       │       ├─> Prepara dataloaders
    │       │       │       ├─> Inicializa modelo
    │       │       │       ├─> Entrena con PyTorch Lightning
    │       │       │       │       │
    │       │       │       │       └─> Logs a TensorBoard (cada batch)
    │       │       │       │               │
    │       │       │       │               └─> Usuario ve métricas en tiempo real
    │       │       │       │
    │       │       │       └─> Guarda mejor modelo en models/
    │       │       │
    │       │       └─> Muestra success message + confetti 🎉
    │       │
    │       └─> Opción: Ver entrenamientos anteriores
    │
    └─> Página Autoencoder
            │
            ├─> Carga modelo entrenado (*.ckpt)
            │
            ├─> Genera predicciones
            │
            ├─> Visualiza espacio latente (2D/3D)
            │
            ├─> Muestra métricas de clasificación
            │
            └─> Analiza reconstrucciones
```

## 🎨 Capturas de Pantalla Conceptuales

### Página de Entrenamiento

```
┌────────────────────────────────────────────────────────┐
│  🎓 Entrenar Modelo VAE                                │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ⚙️ Configuración de Entrenamiento                     │
│  ┌──────────────────────┐  ┌────────────────────────┐ │
│  │ Tipo de Modelo       │  │ Hiperparámetros        │ │
│  │ ◉ VAE Estándar       │  │ Max Epochs: [100    ]  │ │
│  │ ○ LSTM-VAE           │  │ Learning Rate: 0.001   │ │
│  │                      │  │ Batch Size: [16     ]  │ │
│  │ Arquitectura:        │  │ Patience: [15       ]  │ │
│  │ Input: 8 features    │  │                        │ │
│  │ Encoder: [64, 32]    │  │                        │ │
│  │ Latent: 8D           │  │                        │ │
│  │ Decoder: [32, 64]    │  │                        │ │
│  └──────────────────────┘  └────────────────────────┘ │
│                                                        │
│  [    🚀 Iniciar Entrenamiento    ]    [ 🧹 Limpiar ] │
│                                                        │
│  📊 Monitoreo en Tiempo Real                           │
│  ┌──────────────────────────────────────────────────┐ │
│  │                                                  │ │
│  │  TensorBoard                                     │ │
│  │  ┌──────────────────────────────────────────┐   │ │
│  │  │ Loss                                      │   │ │
│  │  │   ╱──────────                             │   │ │
│  │  │  ╱        ╲                               │   │ │
│  │  │ ╱          ───────                        │   │ │
│  │  │                                           │   │ │
│  │  │ Accuracy                                  │   │ │
│  │  │              ╱──────────                  │   │ │
│  │  │            ╱                              │   │ │
│  │  │          ╱                                │   │ │
│  │  └──────────────────────────────────────────┘   │ │
│  │                                                  │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  ✅ Entrenamiento completado!                          │
│  📦 Modelo guardado en: models/vae-epoch=45.ckpt       │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 📈 Métricas Monitoreadas

### Durante el Entrenamiento

**TensorBoard muestra:**
- `train_loss`: Pérdida total en entrenamiento
- `val_loss`: Pérdida total en validación
- `train_acc`: Accuracy en entrenamiento
- `val_acc`: **Métrica principal** - Accuracy en validación
- `train_recon_loss`: Pérdida de reconstrucción
- `val_recon_loss`: Pérdida de reconstrucción en validación
- `train_kl_loss`: KL Divergence en entrenamiento
- `val_kl_loss`: KL Divergence en validación
- `train_class_loss`: Pérdida de clasificación
- `val_class_loss`: Pérdida de clasificación en validación
- `lr-Adam`: Learning rate del optimizador

### Interpretación

| Métrica | Bueno | Aceptable | Malo |
|---------|-------|-----------|------|
| `val_acc` | >70% | 50-70% | <50% |
| `val_loss` | <0.5 | 0.5-1.0 | >1.0 |
| Gap train-val | <10% | 10-30% | >30% |
| `kl_loss` | 0.01-0.1 | <0.01 o >0.1 | Extremos |

## 🛠️ Solución de Problemas Comunes

### Problema 1: TensorBoard no se muestra

**Síntoma:** Iframe vacío o error de conexión

**Soluciones:**
```bash
# Verificar puerto
lsof -i :6006

# Matar procesos
pkill -f tensorboard

# Reiniciar Streamlit
streamlit run app.py
```

### Problema 2: Error durante entrenamiento

**Síntoma:** Excepción en Python durante training

**Diagnóstico:**
```bash
# Ver logs detallados
python scripts/train_autoencoder.py --lstm

# Verificar datos
python -c "import pandas as pd; print(pd.read_csv('data/data.csv').info())"

# Test completo
python test_tensorboard_integration.py
```

### Problema 3: GPU Out of Memory

**Soluciones:**
- Reducir batch size (16 → 8 → 4)
- Entrenar en CPU (más lento pero funciona)
- Usar VAE estándar en lugar de LSTM

## 📚 Archivos Clave

| Archivo | Propósito | Líneas |
|---------|-----------|--------|
| `pages/2_🎓_Entrenar_Modelo.py` | Nueva página de entrenamiento | ~450 |
| `pages/3_🤖_Autoencoder.py` | Visualización de modelos (actualizado) | ~630 |
| `scripts/train_autoencoder.py` | Script de entrenamiento (mejorado) | ~240 |
| `docs/TRAINING_GUIDE.md` | Documentación completa | ~250 |
| `test_tensorboard_integration.py` | Test de integración | ~150 |

## ✨ Ventajas del Nuevo Sistema

### Antes (Terminal)
```bash
# Terminal 1
python scripts/train_autoencoder.py --lstm

# Terminal 2  
tensorboard --logdir=logs/

# Navegador
# Abrir http://localhost:6006

# Problema: 3 ventanas, cambio manual entre ellas
```

### Ahora (Streamlit)
```bash
# Terminal 1
streamlit run app.py

# Todo lo demás en el navegador:
# - Configuración
# - Entrenamiento
# - Monitoreo
# - Visualización

# Ventaja: Todo integrado, flujo continuo
```

## 🎯 Próximos Pasos Sugeridos

1. **Entrenar Ambos Modelos:**
   ```bash
   # Desde Streamlit UI:
   # - Entrenar VAE estándar
   # - Entrenar LSTM-VAE
   # - Comparar val_acc
   ```

2. **Comparar Resultados:**
   - Latent space clustering
   - Classification accuracy
   - Reconstruction quality

3. **Analizar Variabilidad:**
   - ¿LSTM-VAE tiene mayor accuracy?
   - Si sí → variabilidad intra-participante es informativa
   - Si no → mean pooling captura la información esencial

4. **Optimizar Hiperparámetros:**
   - Experimentar con learning rate
   - Probar diferentes arquitecturas
   - Ajustar KL weight

## 🎉 Conclusión

Has implementado exitosamente:

✅ **Nueva página de entrenamiento** con UI intuitiva  
✅ **TensorBoard embebido** en tiempo real  
✅ **Configuración flexible** de hiperparámetros  
✅ **Documentación completa** y guías  
✅ **Script de testing** para validar instalación  
✅ **Workflow integrado** desde training hasta visualización  

**El sistema está listo para entrenar y comparar modelos VAE vs LSTM-VAE!** 🚀

---

**Comandos Útiles:**

```bash
# Iniciar app
streamlit run app.py

# Entrenar manualmente
python scripts/train_autoencoder.py [--lstm]

# Ver TensorBoard standalone
tensorboard --logdir=logs/

# Test integración
python test_tensorboard_integration.py

# Limpiar logs
rm -rf logs/*/

# Limpiar modelos
rm models/*.ckpt
```

**URLs:**
- Streamlit: http://localhost:8501
- TensorBoard: http://localhost:6006
- GitHub: https://github.com/DanZangrando/mitochondrial-morphology
