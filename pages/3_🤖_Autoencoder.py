"""
Page 3: Variational Autoencoder (VAE) with Classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MitochondriaDataLoader
from src.autoencoder import MitochondriaVAE, LSTMVariationalAutoencoder, ParticipantDataset, ParticipantSequenceDataset, collate_sequences
from src.utils import calculate_reconstruction_error

# Page config
st.set_page_config(page_title="VAE + Clasificación", page_icon="🤖", layout="wide")

st.title("🤖 VAE - Espacio Latente y Clasificación de Grupos")

# Load data
@st.cache_data
def load_and_prepare_data():
    loader = MitochondriaDataLoader()
    X_scaled, data = loader.prepare_features(standardize=True)
    return X_scaled, data, loader

X_scaled, data, loader = load_and_prepare_data()

# Sidebar
st.sidebar.header("Configuración")

# Training section
st.markdown("## 🎓 Entrenamiento del Modelo")

st.markdown("""
El **Variational Autoencoder (VAE)** es un modelo generativo que aprende una representación 
probabilística en el espacio latente. A diferencia de un autoencoder estándar, el VAE:

- Aprende una distribución (μ, σ) en lugar de puntos fijos
- Usa el **reparametrization trick** para hacer el proceso diferenciable
- Incluye **KL divergence** en la pérdida para regularizar el espacio latente

**Arquitectura VAE + Clasificador**:
- Input: 8 features → Encoder → μ & σ (Latent 8D) → Decoder → Reconstruction
- Clasificador: Latent → Hidden → 2 clases (CT/ELA)

**Manejo de múltiples medidas**: Cada participante tiene n medidas que se agregan (mean pooling) 
antes de entrenar, evitando data leakage.
""")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Iniciar Entrenamiento")
    
    st.info("""
    **Antes de entrenar**, asegúrate de tener las dependencias instaladas:
    
    ```bash
    pip install -r requirements.txt
    ```
    """)
    
    if st.button("🚀 Entrenar Autoencoder", type="primary"):
        with st.spinner("Entrenando modelo... Esto puede tomar varios minutos."):
            try:
                # Import training function
                from scripts.train_autoencoder import train_autoencoder
                
                # Train model
                best_model_path = train_autoencoder()
                
                st.success(f"✓ Entrenamiento completado!")
                st.success(f"Modelo guardado en: {best_model_path}")
                st.balloons()
                
            except Exception as e:
                st.error(f"Error durante el entrenamiento: {e}")
                st.info("Puedes entrenar el modelo manualmente ejecutando: `python scripts/train_autoencoder.py`")

with col2:
    st.markdown("### Configuración Actual")
    
    import yaml
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    ae_config = config['autoencoder']
    
    st.markdown(f"""
    - **Input dim**: {ae_config['architecture']['input_dim']}
    - **Encoder**: {ae_config['architecture']['encoder_layers']}
    - **Latent dim**: {ae_config['architecture']['latent_dim']}
    - **Decoder**: {ae_config['architecture']['decoder_layers']}
    - **Batch size**: {ae_config['training']['batch_size']}
    - **Max epochs**: {ae_config['training']['max_epochs']}
    - **Learning rate**: {ae_config['training']['learning_rate']}
    """)

# Model loading section
st.markdown("---")
st.markdown("## 📊 Visualización del Espacio Latente")

# Find available models
model_files = glob.glob("models/*.ckpt")

if not model_files:
    st.warning("⚠️ No se encontraron modelos entrenados. Entrena un modelo primero.")
    st.info("También puedes entrenar manualmente: `python scripts/train_autoencoder.py`")
else:
    # Select model
    selected_model = st.selectbox(
        "Seleccionar modelo",
        options=model_files,
        format_func=lambda x: os.path.basename(x)
    )
    
    try:
        # Detect model type and load accordingly
        @st.cache_resource
        def load_model(model_path):
            # Try to detect model type from filename
            is_lstm = 'lstm' in os.path.basename(model_path).lower()
            
            if is_lstm:
                st.info("🔬 Detectado: LSTM-VAE (preserva secuencias)")
                model = LSTMVariationalAutoencoder.load_from_checkpoint(model_path)
            else:
                st.info("📊 Detectado: VAE estándar (con agregación)")
                model = MitochondriaVAE.load_from_checkpoint(model_path)
            
            model.eval()
            return model, is_lstm
        
        model, is_lstm = load_model(selected_model)
        
        st.success(f"✓ Modelo cargado: {os.path.basename(selected_model)}")
        
        # Prepare data according to model type
        feature_cols = loader.get_feature_columns()
        
        if is_lstm:
            # LSTM-VAE: use sequences
            st.markdown("**Modo**: Usando secuencias completas por participante")
            
            # Create sequence dataset
            participant_dataset = ParticipantSequenceDataset(
                data,
                feature_cols,
                include_labels=True
            )
            
            # Get predictions for LSTM model
            with torch.no_grad():
                all_latents = []
                all_preds = []
                all_labels = []
                all_participants = []
                
                for idx in range(len(participant_dataset)):
                    seq, length, label = participant_dataset[idx]
                    seq = seq.unsqueeze(0)  # Add batch dimension
                    length = torch.tensor([length.item()])  # Ensure it's a proper tensor
                    
                    # Get latent and prediction
                    mu = model.get_latent(seq, length)
                    _, _, _, class_logits = model(seq, length)
                    
                    all_latents.append(mu.cpu().numpy())
                    all_preds.append(torch.argmax(class_logits, dim=1).cpu().numpy())
                    all_labels.append(label.cpu().numpy() if label.dim() > 0 else np.array([label.item()]))
                    all_participants.append(participant_dataset.participants[idx])
                
                latent = np.vstack(all_latents)
                predictions = np.concatenate(all_preds)
                true_labels = np.concatenate(all_labels)
                participants_list = all_participants
            
            reconstructed = None  # LSTM reconstruction is per-sequence, skip for now
            X_agg = None
            
        else:
            # Standard VAE: use aggregation
            st.markdown("**Modo**: Usando agregación (mean pooling) por participante")
            
            participant_dataset = ParticipantDataset(
                data,
                feature_cols,
                aggregation='mean',
                include_labels=True
            )
            
            # Get predictions
            with torch.no_grad():
                all_latents = []
                all_recons = []
                all_preds = []
                all_labels = []
                
                for features, label in participant_dataset:
                    features = features.unsqueeze(0)  # Add batch dimension
                    recon_x, mu, logvar, class_logits = model(features)
                    
                    all_latents.append(mu.cpu().numpy())
                    all_recons.append(recon_x.cpu().numpy())
                    all_preds.append(torch.argmax(class_logits, dim=1).cpu().numpy())
                    all_labels.append(label.cpu().numpy())
                
                latent = np.vstack(all_latents)
                reconstructed = np.vstack(all_recons)
                predictions = np.concatenate(all_preds)
                true_labels = np.concatenate(all_labels)
                
            # Get aggregated features for comparison
            agg_data = data.groupby('Participant').agg({col: 'mean' for col in feature_cols}).reset_index()
            X_agg = agg_data[feature_cols].values
        
        # Classification metrics
        st.markdown("---")
        st.markdown("## 🎯 Resultados de Clasificación")
        
        col1, col2 = st.columns(2)
        
        with col1:
            accuracy = (predictions == true_labels).mean()
            st.metric("Accuracy", f"{accuracy:.2%}")
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, predictions)
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicho", y="Verdadero", color="Count"),
                x=['CT', 'ELA'],
                y=['CT', 'ELA'],
                text_auto=True,
                title='Matriz de Confusión',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # Classification report
            class_names = ['CT', 'ELA']
            report = classification_report(
                true_labels,
                predictions,
                target_names=class_names,
                output_dict=True
            )
            
            st.markdown("### Reporte de Clasificación")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)
            
            # Prediction distribution
            pred_dist = pd.DataFrame({
                'Verdadero': true_labels,
                'Predicho': predictions
            })
            pred_dist['Correcto'] = pred_dist['Verdadero'] == pred_dist['Predicho']
            
            st.markdown(f"""
            - **Correctos**: {pred_dist['Correcto'].sum()} / {len(pred_dist)}
            - **Incorrectos**: {(~pred_dist['Correcto']).sum()} / {len(pred_dist)}
            """)
        
        # Sidebar controls for visualization
        color_by = st.sidebar.selectbox(
            "Colorear por",
            options=['Group', 'Sex', 'Participant'],
            index=0
        )
        
        # Latent space visualization
        st.markdown("---")
        st.markdown("## 📊 Visualización del Espacio Latente")
        
        # Prepare latent space data with participant info
        latent_df = pd.DataFrame(
            latent,
            columns=[f'Latent_{i+1}' for i in range(latent.shape[1])]
        )
        
        # Get participant info
        if is_lstm:
            # For LSTM, we already have participants_list
            latent_df['Participant'] = participants_list
        else:
            # For standard VAE, get from aggregated data
            agg_data = data.groupby('Participant').agg({col: 'mean' for col in feature_cols}).reset_index()
            latent_df['Participant'] = agg_data['Participant'].values
        
        latent_df = latent_df.merge(
            data[['Participant', 'Group', 'Sex']].drop_duplicates(),
            on='Participant',
            how='left'
        )
        latent_df['Prediction'] = ['CT' if p == 0 else 'ELA' for p in predictions]
        latent_df['Correcto'] = predictions == true_labels
        
        color_option = st.selectbox(
            "Colorear por:",
            ['Group', 'Prediction', 'Correcto', 'Sex']
        )
        
        # 2D projection using first 2 latent dims
        st.markdown("### Proyección 2D del Espacio Latente")
        
        fig_2d = px.scatter(
            latent_df,
            x='Latent_1',
            y='Latent_2',
            color=color_option,
            hover_data=['Participant', 'Group', 'Prediction'],
            title=f'Espacio Latente 2D (coloreado por {color_option})',
            template='plotly_white',
            opacity=0.7
        )
        
        fig_2d.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
        st.plotly_chart(fig_2d, use_container_width=True)
        
        # 3D projection
        st.markdown("### Proyección 3D del Espacio Latente")
        
        fig_3d = px.scatter_3d(
            latent_df,
            x='Latent_1',
            y='Latent_2',
            z='Latent_3',
            color=color_option,
            hover_data=['Participant', 'Group', 'Prediction'],
            title=f'Espacio Latente 3D (coloreado por {color_option})',
            template='plotly_white',
            opacity=0.7
        )
        
        fig_3d.update_traces(marker=dict(size=5, line=dict(width=0.5, color='white')))
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Reconstruction quality (only for standard VAE)
        if not is_lstm and reconstructed is not None and X_agg is not None:
            st.markdown("---")
            st.markdown("## 🔧 Calidad de Reconstrucción")
            
            errors = calculate_reconstruction_error(X_agg, reconstructed)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MSE", f"{errors['MSE']:.6f}")
            
            with col2:
                st.metric("MAE", f"{errors['MAE']:.6f}")
            
            with col3:
                st.metric("RMSE", f"{errors['RMSE']:.6f}")
            
            with col4:
                st.metric("R²", f"{errors.get('R2', 0):.4f}")
            
            st.markdown("""
            **Interpretación**: 
            - MSE (Mean Squared Error): Error cuadrático medio
            - MAE (Mean Absolute Error): Error absoluto medio
            - RMSE (Root Mean Squared Error): Raíz del error cuadrático medio
            
            Valores más bajos indican mejor reconstrucción.
            """)
            
            # Sample reconstructions (participant level)
            st.markdown("### Ejemplos de Reconstrucción (por Participante)")
            
            n_samples = st.slider("Número de participantes a mostrar", 1, min(10, len(latent_df)), 5)
            sample_indices = np.random.choice(len(X_agg), n_samples, replace=False)
            
            for idx in sample_indices:
                participant_id = latent_df.iloc[idx]['Participant']
                group = latent_df.iloc[idx]['Group']
                pred = latent_df.iloc[idx]['Prediction']
                correct = latent_df.iloc[idx]['Correcto']
                
                status = "✓ Correcto" if correct else "✗ Incorrecto"
                
                with st.expander(f"Participante {participant_id} - Real: {group}, Predicho: {pred} ({status})"):
                    df_comparison = pd.DataFrame({
                        'Feature': feature_cols,
                        'Original': X_agg[idx],
                        'Reconstructed': reconstructed[idx],
                        'Error': np.abs(X_agg[idx] - reconstructed[idx])
                    })
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig_comp = go.Figure()
                        
                        fig_comp.add_trace(go.Bar(
                            name='Original',
                            x=df_comparison['Feature'],
                            y=df_comparison['Original'],
                            marker_color='steelblue'
                        ))
                        
                        fig_comp.add_trace(go.Bar(
                            name='Reconstructed',
                            x=df_comparison['Feature'],
                            y=df_comparison['Reconstructed'],
                            marker_color='orange'
                        ))
                        
                        fig_comp.update_layout(
                            barmode='group',
                            title='Original vs Reconstruido',
                            template='plotly_white',
                            height=350
                        )
                        
                        st.plotly_chart(fig_comp, use_container_width=True)
                
                with col2:
                    st.dataframe(df_comparison.round(4), use_container_width=True, height=350)
        
        # Comparison with PCA
        st.markdown("---")
        st.markdown("## 🔬 VAE vs PCA: ¿Cuál es mejor?")
        
        st.markdown("""
        ### Ventajas del VAE
        
        - **No lineal**: Captura relaciones complejas que PCA (método lineal) no puede detectar
        - **Clasificación integrada**: Predice el grupo directamente desde el espacio latente
        - **Espacio latente estructurado**: La pérdida KL regulariza el espacio, haciéndolo más interpretable
        - **Manejo de participantes**: Agregación explícita de múltiples medidas por participante
        
        ### Interpretación de Resultados
        
        1. **Alta accuracy (>70%)**: Las métricas morfológicas tienen poder discriminativo entre CT/ELA
        2. **Clusterización visible**: Los grupos forman clusters en el espacio latente
        3. **Baja reconstrucción error**: El modelo captura bien la información de las features
        
        ### ¿Cuándo preferir VAE?
        
        - Cuando buscas clasificación además de reducción dimensional
        - Si hay relaciones no lineales entre variables
        - Cuando necesitas un modelo generativo (puedes samplear nuevos datos)
        
        ### ¿Cuándo preferir PCA?
        
        - Para análisis exploratorio rápido
        - Cuando la interpretabilidad lineal es importante
        - Si el dataset es pequeño (menos overfitting)
        """)
        
        # Export latent space
        st.markdown("---")
        
        if st.button("Exportar Espacio Latente"):
            latent_data = pd.DataFrame(
                latent,
                columns=[f'Latent_{i+1}' for i in range(latent.shape[1])]
            )
            latent_data['Group'] = data['Group'].values
            latent_data['Sex'] = data['Sex'].values
            latent_data['Participant'] = data['Participant'].values
            
            csv = latent_data.to_csv(index=False)
            st.download_button(
                label="Descargar CSV",
                data=csv,
                file_name="autoencoder_latent_space.csv",
                mime="text/csv"
            )
            st.success("✓ Espacio latente listo para descargar")
    
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.info("Asegúrate de que el modelo fue entrenado correctamente.")

# TensorBoard section
st.markdown("---")
st.markdown("## 📈 Monitoreo con TensorBoard")

st.markdown("""
Para visualizar las métricas de entrenamiento en tiempo real, ejecuta en tu terminal:

```bash
tensorboard --logdir=logs/
```

Luego abre tu navegador en `http://localhost:6006` para ver:
- Curvas de pérdida (train/val)
- Learning rate schedule
- Arquitectura del modelo
- Y más...
""")

st.info("💡 TensorBoard se integra nativamente con PyTorch Lightning, registrando automáticamente todas las métricas durante el entrenamiento.")
