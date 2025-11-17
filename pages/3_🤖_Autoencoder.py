"""
Page 3: Autoencoder Training and Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MitochondriaDataLoader
from src.autoencoder import MitochondriaAutoencoder
from src.utils import calculate_reconstruction_error

# Page config
st.set_page_config(page_title="Autoencoder", page_icon="🤖", layout="wide")

st.title("🤖 Autoencoder - Espacio Latente")

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
El autoencoder aprende una representación comprimida (espacio latente) de los datos 
mediante una red neuronal que intenta reconstruir la entrada original.

**Arquitectura**:
- Input: 8 features → Encoder → Latent Space (3D) → Decoder → Output: 8 features
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
        # Load model
        @st.cache_resource
        def load_model(model_path):
            model = MitochondriaAutoencoder.load_from_checkpoint(model_path)
            model.eval()
            return model
        
        model = load_model(selected_model)
        
        st.success(f"✓ Modelo cargado: {os.path.basename(selected_model)}")
        
        # Encode data
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            latent = model.encode(X_tensor).numpy()
            reconstructed = model(X_tensor).numpy()
        
        # Sidebar controls for visualization
        color_by = st.sidebar.selectbox(
            "Colorear por",
            options=['Group', 'Sex', 'Participant'],
            index=0
        )
        
        # 2D Latent space
        st.markdown("### Proyección 2D del Espacio Latente")
        
        df_latent = pd.DataFrame({
            'Latent 1': latent[:, 0],
            'Latent 2': latent[:, 1],
            color_by: data[color_by]
        })
        
        fig_2d = px.scatter(
            df_latent,
            x='Latent 1',
            y='Latent 2',
            color=color_by,
            title=f'Espacio Latente 2D (coloreado por {color_by})',
            template='plotly_white',
            opacity=0.7
        )
        
        fig_2d.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
        st.plotly_chart(fig_2d, use_container_width=True)
        
        # 3D Latent space
        st.markdown("### Proyección 3D del Espacio Latente")
        
        df_latent_3d = pd.DataFrame({
            'Latent 1': latent[:, 0],
            'Latent 2': latent[:, 1],
            'Latent 3': latent[:, 2],
            color_by: data[color_by]
        })
        
        fig_3d = px.scatter_3d(
            df_latent_3d,
            x='Latent 1',
            y='Latent 2',
            z='Latent 3',
            color=color_by,
            title=f'Espacio Latente 3D (coloreado por {color_by})',
            template='plotly_white',
            opacity=0.7
        )
        
        fig_3d.update_traces(marker=dict(size=4, line=dict(width=0.3, color='white')))
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Reconstruction quality
        st.markdown("---")
        st.markdown("## 🎯 Calidad de Reconstrucción")
        
        errors = calculate_reconstruction_error(X_scaled, reconstructed)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MSE", f"{errors['MSE']:.6f}")
        
        with col2:
            st.metric("MAE", f"{errors['MAE']:.6f}")
        
        with col3:
            st.metric("RMSE", f"{errors['RMSE']:.6f}")
        
        st.markdown("""
        **Interpretación**: 
        - MSE (Mean Squared Error): Error cuadrático medio
        - MAE (Mean Absolute Error): Error absoluto medio
        - RMSE (Root Mean Squared Error): Raíz del error cuadrático medio
        
        Valores más bajos indican mejor reconstrucción.
        """)
        
        # Sample reconstructions
        st.markdown("### Ejemplos de Reconstrucción")
        
        n_samples = st.slider("Número de muestras a mostrar", 1, 10, 5)
        sample_indices = np.random.choice(len(X_scaled), n_samples, replace=False)
        
        for idx in sample_indices:
            with st.expander(f"Muestra {idx} - Grupo: {data.iloc[idx]['Group']}, Sexo: {data.iloc[idx]['Sex']}"):
                df_comparison = pd.DataFrame({
                    'Feature': loader.get_feature_columns(),
                    'Original': X_scaled[idx],
                    'Reconstructed': reconstructed[idx],
                    'Error': np.abs(X_scaled[idx] - reconstructed[idx])
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
        st.markdown("## 🔬 Comparación: Autoencoder vs PCA")
        
        st.markdown("""
        ### ¿Por qué usar ambos?
        
        - **PCA**: Método lineal, rápido, interpretable. Encuentra direcciones de máxima varianza.
        - **Autoencoder**: Método no lineal, puede capturar relaciones complejas que PCA no detecta.
        
        ### ¿Qué buscar?
        
        1. **Clusterización**: ¿Los grupos se separan mejor en el espacio latente del autoencoder?
        2. **Estructuras no lineales**: ¿El autoencoder revela patrones que PCA no muestra?
        3. **Calidad de reconstrucción**: Errores bajos indican que el modelo captura bien la información.
        
        ### Interpretación
        
        Si observas clusterización clara en el espacio latente (puntos del mismo grupo cercanos entre sí),
        sugiere que las métricas morfológicas tienen poder discriminativo entre grupos CT y ELA.
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
