"""
Page 4: LSTM Classifier Training
Train and evaluate LSTM classifier using train/validation split
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import sys
import os
import json
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import binomtest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier import LSTMClassifier
from src.data_loader import MitochondriaDataLoader

# Page config
st.set_page_config(page_title="Entrenar Clasificador", page_icon="🎯", layout="wide")

st.title("🎯 Entrenar Clasificador LSTM")

st.markdown("""
## Clasificación CT vs ELA con LSTM Bidireccional

Este clasificador utiliza un modelo LSTM bidireccional para distinguir entre participantes Control (CT) y ELA.

### 🧠 Arquitectura del Modelo

- **Entrada**: 8 variables morfométricas mitocondriales (secuencias de longitud variable)
- **LSTM Bidireccional**: Captura patrones temporales en ambas direcciones
- **Classifier Head**: Capas fully connected para clasificación binaria
- **Salida**: Probabilidades CT vs ELA

### 📊 Train/Val Split

El dataset se divide en:
- **Train set**: Para entrenar el modelo (ajustar pesos)
- **Validation set**: Para evaluar desempeño y evitar overfitting

El split se hace a **nivel de participante** (no por muestra individual) para evitar data leakage.
""")

# Load data info
@st.cache_data
def load_data_info():
    loader = MitochondriaDataLoader()
    X_scaled, data = loader.prepare_features(standardize=True)
    return X_scaled, data

X_scaled, data = load_data_info()

# Dataset info
st.markdown("---")
st.markdown("## 📊 Dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Muestras", len(data))

with col2:
    st.metric("Participantes", data['Participant'].nunique())

with col3:
    ct_count = (data.groupby('Participant')['Group'].first() == 'CT').sum()
    st.metric("CT", ct_count)

with col4:
    ela_count = (data.groupby('Participant')['Group'].first() == 'ELA').sum()
    st.metric("ELA", ela_count)

# Training configuration
st.markdown("---")
st.markdown("## ⚙️ Configuración del Entrenamiento")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Train/Val Split")
    
    val_split = st.slider(
        "Validación Split",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Proporción de participantes para validación"
    )
    
    n_participants = data['Participant'].nunique()
    n_val = int(n_participants * val_split)
    n_train = n_participants - n_val
    
    st.info(f"""
    📊 **Distribución de Datos**:
    - Train: {n_train} participantes ({(1-val_split)*100:.0f}%)
    - Validation: {n_val} participantes ({val_split*100:.0f}%)
    - Random state: 42 (reproducible)
    - Split estratificado por grupo (CT/ELA)
    """)

with col2:
    st.markdown("### 🧠 Hiperparámetros del Modelo")
    
    hidden_dim = st.slider(
        "Hidden Dimension",
        min_value=32,
        max_value=256,
        value=64,
        step=32,
        help="Dimensión oculta del LSTM"
    )
    
    num_layers = st.slider(
        "Número de Capas LSTM",
        min_value=1,
        max_value=4,
        value=2,
        help="Número de capas LSTM apiladas"
    )
    
    dropout = st.slider(
        "Dropout",
        min_value=0.0,
        max_value=0.5,
        value=0.3,
        step=0.05,
        help="Dropout para regularización"
    )

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏋️ Entrenamiento")
    
    max_epochs = st.slider(
        "Máximo de Épocas",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Máximo número de épocas (early stopping puede detener antes)"
    )
    
    batch_size = st.slider(
        "Batch Size",
        min_value=4,
        max_value=32,
        value=16,
        step=4,
        help="Número de secuencias por batch"
    )
    
    learning_rate = st.select_slider(
        "Learning Rate",
        options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        value=1e-3,
        help="Tasa de aprendizaje inicial"
    )

with col2:
    st.markdown("### ⚙️ Callbacks")
    
    es_patience = st.slider(
        "Early Stopping Patience",
        min_value=5,
        max_value=30,
        value=15,
        help="Épocas sin mejora antes de detener"
    )
    
    es_min_delta = st.number_input(
        "Early Stopping Min Delta",
        min_value=0.0001,
        max_value=0.01,
        value=0.001,
        step=0.0001,
        format="%.4f",
        help="Mínima mejora para considerar progreso"
    )
    
    ckpt_monitor = st.selectbox(
        "Checkpoint Monitor",
        ["val_acc", "val_loss"],
        index=0,
        help="Métrica para guardar mejores modelos"
    )
    
    save_top_k = st.slider(
        "Guardar Mejores N Modelos",
        min_value=1,
        max_value=5,
        value=3,
        help="Número de mejores checkpoints a guardar (según val_acc)"
    )

# Train button
st.markdown("---")

if st.button("🚀 Iniciar Entrenamiento", type="primary", use_container_width=True):
    
    st.markdown("## 🏋️ Entrenando Modelo...")
    
    try:
        # Import simple training function
        from scripts.train_classifier import train_classifier
        
        # Prepare arguments
        kwargs = {
            'max_epochs': max_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'val_split': val_split,
            'early_stopping_patience': es_patience,
            'early_stopping_min_delta': es_min_delta,
            'checkpoint_monitor': ckpt_monitor,
            'checkpoint_mode': 'max' if ckpt_monitor == 'val_acc' else 'min',
            'checkpoint_save_top_k': save_top_k,
        }
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train model
        status_text.text("Entrenando... Esto puede tomar varios minutos.")
        
        best_model_path = train_classifier(**kwargs)
        
        progress_bar.progress(100)
        status_text.text("✅ Entrenamiento completado!")
        
        # Success message
        st.success(f"✅ Entrenamiento completado!")
        
        # Store in session state
        st.session_state['best_model_path'] = best_model_path
        st.session_state['val_split'] = val_split
        
    except Exception as e:
        st.error(f"❌ Error durante el entrenamiento: {str(e)}")
        st.exception(e)

# Results section
st.markdown("---")
st.markdown("## 📊 Resultados y Evaluación")

# Find available models
simple_models = sorted(glob.glob("models/*.ckpt"), key=os.path.getmtime, reverse=True)

if not simple_models:
    st.info("👆 Entrena un modelo para ver los resultados aquí")
    st.stop()

# Model selection
selected_model = st.selectbox(
    "Seleccionar Modelo",
    simple_models,
    format_func=lambda x: f"{os.path.basename(x)} ({os.path.getmtime(x)})",
    help="Modelos entrenados con split simple (más recientes primero)"
)

# Load model
@st.cache_resource
def load_model(model_path):
    model = LSTMClassifier.load_from_checkpoint(model_path, map_location='cpu')
    model.eval()
    return model

with st.spinner("Cargando modelo..."):
    model = load_model(selected_model)

st.success(f"✅ Modelo cargado: `{os.path.basename(selected_model)}`")

# Extract val_split from model hparams
val_participants_from_model = None
train_participants_from_model = None
model_val_split = None

try:
    if hasattr(model, 'hparams'):
        if 'val_split' in model.hparams:
            model_val_split = model.hparams['val_split']
            st.success(f"✅ Modelo guardó **val_split={model_val_split:.0%}**")
        if 'val_participants' in model.hparams:
            val_participants_from_model = model.hparams['val_participants']
            st.success(f"✅ Modelo guardó participantes de validación: {sorted(val_participants_from_model)}")
        if 'train_participants' in model.hparams:
            train_participants_from_model = model.hparams['train_participants']
            st.info(f"📊 Participantes de entrenamiento: {sorted(train_participants_from_model)}")
    else:
        st.warning("⚠️ Modelo antiguo sin información de participantes guardada")
except Exception as e:
    st.warning(f"⚠️ No se pudo leer información del modelo: {str(e)}")

# Val split slider for evaluation
default_val_split = model_val_split if model_val_split is not None else 0.2

# If model has participants saved, use them directly
if val_participants_from_model is not None and train_participants_from_model is not None:
    st.success("✅ Usando participantes guardados del entrenamiento")
    train_participants = np.array(train_participants_from_model)
    val_participants = np.array(val_participants_from_model)
    eval_val_split = len(val_participants) / (len(train_participants) + len(val_participants))
else:
    # Manual split reproduction
    st.warning("⚠️ Modelo sin participantes guardados. Reproduciendo split manualmente:")
    
    eval_val_split = st.slider(
        "Val Split usado en entrenamiento",
        min_value=0.1,
        max_value=0.5,
        value=default_val_split,
        step=0.05,
        help="Debe coincidir con el val_split usado durante el entrenamiento"
    )
    
    # Split participants (same way as training)
    random_state = 42
    participants = data['Participant'].unique()
    
    train_participants, val_participants = train_test_split(
        participants,
        test_size=eval_val_split,
        random_state=random_state,
        stratify=[data[data['Participant'] == p]['Group'].iloc[0] for p in participants]
    )

# Evaluation mode - now always evaluate on both
st.markdown("---")
st.markdown("## 📊 Evaluación del Modelo")

st.info(f"""
**Participantes de Validación**: {sorted([int(p) for p in val_participants])} ({len(val_participants)} participantes)

**Participantes de Entrenamiento**: {sorted([int(p) for p in train_participants])} ({len(train_participants)} participantes)

A continuación se muestran dos evaluaciones:
1. **Solo Validación**: Métricas reales del modelo (participantes no vistos)
2. **Dataset Completo**: Incluye train + val (para referencia)
""")

# Generate predictions for VALIDATION SET
st.markdown("### 🎯 Evaluación 1: Solo Validación (Métricas Reales)")

with st.spinner("Generando predicciones sobre validación..."):
    val_predictions = []
    val_true_labels = []
    val_participants_list = []
    val_probabilities_list = []
    
    with torch.no_grad():
        for participant in val_participants:
            participant_data = data[data['Participant'] == participant]
            sequence = X_scaled[participant_data.index]
            true_label = participant_data['Group'].iloc[0]
            
            seq_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            length = torch.LongTensor([len(sequence)])
            
            logits = model(seq_tensor, length)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
            pred_class = torch.argmax(logits, dim=1).item()
            
            val_predictions.append('CT' if pred_class == 0 else 'ELA')
            val_true_labels.append(true_label)
            val_participants_list.append(participant)
            val_probabilities_list.append(probs)

val_results_df = pd.DataFrame({
    'Participant': val_participants_list,
    'True_Label': val_true_labels,
    'Prediction': val_predictions,
    'Prob_CT': [p[0] for p in val_probabilities_list],
    'Prob_ELA': [p[1] for p in val_probabilities_list],
    'Correct': [t == p for t, p in zip(val_true_labels, val_predictions)]
})

val_accuracy = val_results_df['Correct'].mean()

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Val Accuracy", f"{val_accuracy:.2%}", help="Accuracy sobre participantes de validación únicamente")

with col2:
    st.success(f"✅ Este es el desempeño **real** del modelo sobre {len(val_participants)} participantes no vistos durante entrenamiento.")

# Confusion Matrix - Validation Only with modern visualization
st.markdown("#### 📊 Matriz de Confusión (Validación)")

val_cm = confusion_matrix(val_results_df['True_Label'], val_results_df['Prediction'], labels=['CT', 'ELA'])

# Calculate p-value using binomial test
n_correct = val_results_df['Correct'].sum()
n_total = len(val_results_df)
# H0: accuracy = 0.5 (random chance for binary classification)
p_value = binomtest(n_correct, n_total, 0.5, alternative='greater').pvalue

# Create modern plotly heatmap with annotations
fig_val = go.Figure()

# Add heatmap
fig_val.add_trace(go.Heatmap(
    z=val_cm,
    x=['CT', 'ELA'],
    y=['CT', 'ELA'],
    colorscale=[
        [0, '#f0f9ff'],      # Very light blue
        [0.3, '#bae6fd'],    # Light blue
        [0.6, '#38bdf8'],    # Medium blue
        [1, '#0284c7']       # Dark blue
    ],
    showscale=True,
    colorbar=dict(
        title=dict(text="Count", side="right"),
        tickmode="linear",
        tick0=0,
        dtick=1
    ),
    hovertemplate='<b>Real: %{y}</b><br>Predicho: %{x}<br>Casos: %{z}<extra></extra>',
    text=val_cm,
    texttemplate='<b>%{text}</b>',
    textfont={"size": 24, "color": "white", "family": "Arial Black"}
))

# Add annotations with percentages
for i in range(2):
    for j in range(2):
        percentage = (val_cm[i, j] / val_cm[i].sum() * 100) if val_cm[i].sum() > 0 else 0
        fig_val.add_annotation(
            x=j,
            y=i,
            text=f"{percentage:.1f}%",
            showarrow=False,
            font=dict(size=14, color="white"),
            yshift=-15
        )

fig_val.update_layout(
    title={
        'text': f"<b>Matriz de Confusión - Validación</b><br><sub>p-value = {p_value:.4f} {'✓' if p_value < 0.05 else ''}</sub>",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18}
    },
    xaxis_title="<b>Predicción</b>",
    yaxis_title="<b>Valor Real</b>",
    height=500,
    font=dict(size=14),
    xaxis=dict(side='bottom', tickfont=dict(size=14)),
    yaxis=dict(tickfont=dict(size=14)),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig_val, use_container_width=True)

# Interpretation of p-value
if p_value < 0.001:
    significance = "***"
    interpretation = "Altamente significativo (p < 0.001)"
    color = "green"
elif p_value < 0.01:
    significance = "**"
    interpretation = "Muy significativo (p < 0.01)"
    color = "green"
elif p_value < 0.05:
    significance = "*"
    interpretation = "Significativo (p < 0.05)"
    color = "blue"
else:
    significance = "ns"
    interpretation = "No significativo (p ≥ 0.05)"
    color = "orange"

st.markdown(f"""
**📊 Significancia Estadística {significance}**

- **p-value**: {p_value:.4f}
- **Interpretación**: {interpretation}
- **H₀**: El modelo clasifica al azar (accuracy = 50%)
- **H₁**: El modelo clasifica mejor que el azar (accuracy > 50%)

{'✅ El modelo clasifica significativamente mejor que el azar.' if p_value < 0.05 else '⚠️ No hay evidencia suficiente de que el modelo clasifique mejor que el azar.'}
""")

# Per-class metrics for validation
col1, col2, col3 = st.columns(3)

val_ct_correct = val_cm[0, 0]
val_ct_total = val_cm[0, 0] + val_cm[0, 1]
val_ela_correct = val_cm[1, 1]
val_ela_total = val_cm[1, 0] + val_cm[1, 1]

with col1:
    st.metric("Accuracy CT (Val)", f"{val_ct_correct/val_ct_total:.1%}" if val_ct_total > 0 else "N/A")

with col2:
    st.metric("Accuracy ELA (Val)", f"{val_ela_correct/val_ela_total:.1%}" if val_ela_total > 0 else "N/A")

with col3:
    st.metric("Overall Accuracy (Val)", f"{val_accuracy:.1%}")

# Generate predictions for FULL DATASET
st.markdown("---")
st.markdown("### 📈 Evaluación 2: Dataset Completo (Train + Val)")

st.warning("⚠️ Esta evaluación incluye participantes de entrenamiento. Se muestra solo como referencia.")

with st.spinner("Generando predicciones sobre dataset completo..."):
    all_predictions = []
    all_true_labels = []
    all_participants_list = []
    all_probabilities_list = []
    all_set_type = []
    
    all_participants = data['Participant'].unique()
    
    with torch.no_grad():
        for participant in all_participants:
            participant_data = data[data['Participant'] == participant]
            sequence = X_scaled[participant_data.index]
            true_label = participant_data['Group'].iloc[0]
            
            seq_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            length = torch.LongTensor([len(sequence)])
            
            logits = model(seq_tensor, length)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
            pred_class = torch.argmax(logits, dim=1).item()
            
            all_predictions.append('CT' if pred_class == 0 else 'ELA')
            all_true_labels.append(true_label)
            all_participants_list.append(participant)
            all_probabilities_list.append(probs)
            all_set_type.append('Validación' if participant in val_participants else 'Entrenamiento')

all_results_df = pd.DataFrame({
    'Participant': all_participants_list,
    'Set': all_set_type,
    'True_Label': all_true_labels,
    'Prediction': all_predictions,
    'Prob_CT': [p[0] for p in all_probabilities_list],
    'Prob_ELA': [p[1] for p in all_probabilities_list],
    'Correct': [t == p for t, p in zip(all_true_labels, all_predictions)]
})

all_accuracy = all_results_df['Correct'].mean()

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Accuracy Completo", f"{all_accuracy:.2%}", help="Accuracy sobre todos los participantes (train + val)")

with col2:
    train_in_full = all_results_df[all_results_df['Set'] == 'Entrenamiento']['Correct'].mean()
    st.info(f"Train accuracy: {train_in_full:.1%} | Val accuracy: {val_accuracy:.1%}")

# Confusion Matrix - Full Dataset with modern visualization
st.markdown("#### 📊 Matriz de Confusión (Dataset Completo)")

all_cm = confusion_matrix(all_results_df['True_Label'], all_results_df['Prediction'], labels=['CT', 'ELA'])

# Create modern plotly heatmap for full dataset
fig_all = go.Figure()

# Add heatmap
fig_all.add_trace(go.Heatmap(
    z=all_cm,
    x=['CT', 'ELA'],
    y=['CT', 'ELA'],
    colorscale=[
        [0, '#f0fdf4'],      # Very light green
        [0.3, '#bbf7d0'],    # Light green
        [0.6, '#4ade80'],    # Medium green
        [1, '#16a34a']       # Dark green
    ],
    showscale=True,
    colorbar=dict(
        title=dict(text="Count", side="right"),
        tickmode="linear",
        tick0=0,
        dtick=1
    ),
    hovertemplate='<b>Real: %{y}</b><br>Predicho: %{x}<br>Casos: %{z}<extra></extra>',
    text=all_cm,
    texttemplate='<b>%{text}</b>',
    textfont={"size": 24, "color": "white", "family": "Arial Black"}
))

# Add annotations with percentages
for i in range(2):
    for j in range(2):
        percentage = (all_cm[i, j] / all_cm[i].sum() * 100) if all_cm[i].sum() > 0 else 0
        fig_all.add_annotation(
            x=j,
            y=i,
            text=f"{percentage:.1f}%",
            showarrow=False,
            font=dict(size=14, color="white"),
            yshift=-15
        )

fig_all.update_layout(
    title={
        'text': "<b>Matriz de Confusión - Dataset Completo</b><br><sub>(Train + Val - Solo referencia)</sub>",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18}
    },
    xaxis_title="<b>Predicción</b>",
    yaxis_title="<b>Valor Real</b>",
    height=500,
    font=dict(size=14),
    xaxis=dict(side='bottom', tickfont=dict(size=14)),
    yaxis=dict(tickfont=dict(size=14)),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig_all, use_container_width=True)

# Per-class metrics for full dataset
col1, col2, col3 = st.columns(3)

all_ct_correct = all_cm[0, 0]
all_ct_total = all_cm[0, 0] + all_cm[0, 1]
all_ela_correct = all_cm[1, 1]
all_ela_total = all_cm[1, 0] + all_cm[1, 1]

with col1:
    st.metric("Accuracy CT (Full)", f"{all_ct_correct/all_ct_total:.1%}" if all_ct_total > 0 else "N/A")

with col2:
    st.metric("Accuracy ELA (Full)", f"{all_ela_correct/all_ela_total:.1%}" if all_ela_total > 0 else "N/A")

with col3:
    st.metric("Overall Accuracy (Full)", f"{all_accuracy:.1%}")

# Detailed results table - Combined view
st.markdown("---")
st.markdown("### 📋 Resultados Detallados por Participante")

# Use all_results_df for complete view
display_df = all_results_df.copy()
display_df['Prob_CT'] = display_df['Prob_CT'].apply(lambda x: f"{x:.1%}")
display_df['Prob_ELA'] = display_df['Prob_ELA'].apply(lambda x: f"{x:.1%}")
display_df['Status'] = display_df['Correct'].apply(lambda x: "✅" if x else "❌")
display_df = display_df.drop('Correct', axis=1)
display_df.columns = ['Participante', 'Conjunto', 'Etiqueta Real', 'Predicción', 'Prob CT', 'Prob ELA', 'Estado']

# Highlight validation rows
def highlight_validation(row):
    if row['Conjunto'] == 'Validación':
        return ['background-color: #e8f4f8'] * len(row)
    return [''] * len(row)

st.dataframe(
    display_df.style.apply(highlight_validation, axis=1),
    use_container_width=True, 
    hide_index=True
)

st.caption("💡 Las filas resaltadas en azul son participantes de **validación** (no vistos durante entrenamiento)")

# Classification report - VALIDATION ONLY
st.markdown("### 📈 Reporte de Clasificación (Validación)")

val_report = classification_report(val_results_df['True_Label'], val_results_df['Prediction'], output_dict=True)
val_report_df = pd.DataFrame(val_report).transpose()

# Format and display
report_display = val_report_df[['precision', 'recall', 'f1-score', 'support']].copy()
report_display.columns = ['Precisión', 'Recall', 'F1-Score', 'Soporte']
report_display['Precisión'] = report_display['Precisión'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
report_display['Recall'] = report_display['Recall'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
report_display['F1-Score'] = report_display['F1-Score'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
report_display['Soporte'] = report_display['Soporte'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")

st.dataframe(report_display, use_container_width=True)

st.info("📊 Este reporte se calculó usando **solo los participantes de validación**")

# Footer
st.markdown("---")
st.info(f"""
📁 **Modelo guardado en**: `{selected_model}`

 **TensorBoard**: Visualiza las curvas de entrenamiento ejecutando:
```bash
tensorboard --logdir logs/lstm_classifier
```

💡 **Tip**: Para experimentos científicos, asegúrate de reportar métricas sobre el conjunto de **validación** únicamente.
""")
