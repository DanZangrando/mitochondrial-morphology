"""
Page 2: PCA Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MitochondriaDataLoader
from src.pca_analysis import MitochondriaPCA

# Page config
st.set_page_config(page_title="PCA", page_icon="🎯", layout="wide")

st.title("🎯 Análisis de Componentes Principales (PCA)")

# Load data
@st.cache_data
def load_and_prepare_data():
    loader = MitochondriaDataLoader()
    X_scaled, data = loader.prepare_features(standardize=True)
    return X_scaled, data, loader

X_scaled, data, loader = load_and_prepare_data()

# Sidebar controls
st.sidebar.header("Configuración PCA")

n_components = st.sidebar.slider(
    "Número de Componentes",
    min_value=2,
    max_value=min(8, X_scaled.shape[1]),
    value=3
)

color_by = st.sidebar.selectbox(
    "Colorear puntos por",
    options=['Group', 'Sex', 'Participant'],
    index=0
)

# Perform PCA
@st.cache_data
def perform_pca(X, n_comp):
    pca = MitochondriaPCA(n_components=n_comp)
    pca.fit_transform(X)
    return pca

pca = perform_pca(X_scaled, n_components)

# Main content
st.markdown("## Varianza Explicada")

# Explained variance plot
fig_variance = pca.plot_explained_variance()
st.plotly_chart(fig_variance, use_container_width=True)

# Show explained variance values
col1, col2, col3 = st.columns(3)

explained_var = pca.get_explained_variance()

with col1:
    st.metric("PC1 Varianza", f"{explained_var[0]*100:.2f}%")

with col2:
    st.metric("PC2 Varianza", f"{explained_var[1]*100:.2f}%")

with col3:
    if n_components >= 3:
        st.metric("PC3 Varianza", f"{explained_var[2]*100:.2f}%")
    else:
        st.metric("Total (PC1+PC2)", f"{sum(explained_var)*100:.2f}%")

st.info(f"**Varianza total explicada por {n_components} componentes**: {sum(explained_var)*100:.2f}%")

# PCA Loadings
st.markdown("---")
st.markdown("## Contribución de Variables (Loadings)")

loadings = pca.get_loadings(loader.get_feature_columns())

col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(loadings.round(3), use_container_width=True)

with col2:
    st.markdown("""
    **Interpretación**:
    
    Los loadings muestran cómo cada variable original contribuye a cada componente principal.
    
    - Valores altos (positivos o negativos) indican mayor contribución
    - El signo indica la dirección de la contribución
    """)

# 2D PCA Plot
st.markdown("---")
st.markdown("## Proyección 2D: PC1 vs PC2")

labels = data[color_by]
fig_2d = pca.plot_2d(labels, color_by)
st.plotly_chart(fig_2d, use_container_width=True)

# 3D PCA Plot
if n_components >= 3:
    st.markdown("---")
    st.markdown("## Proyección 3D: PC1 vs PC2 vs PC3")
    
    fig_3d = pca.plot_3d(labels, color_by)
    st.plotly_chart(fig_3d, use_container_width=True)

# Component analysis
st.markdown("---")
st.markdown("## Análisis de Componentes Individuales")

selected_pc = st.selectbox(
    "Seleccionar Componente Principal",
    options=[f"PC{i+1}" for i in range(n_components)],
    index=0
)

pc_idx = int(selected_pc[2:]) - 1

# Show top contributors
st.markdown(f"### Variables que más contribuyen a {selected_pc}")

loadings_pc = loadings.iloc[:, pc_idx].abs().sort_values(ascending=False)

col1, col2 = st.columns([1, 2])

with col1:
    st.dataframe(
        pd.DataFrame({
            'Variable': loadings_pc.index,
            'Contribución Abs': loadings_pc.values
        }).round(3),
        use_container_width=True,
        height=300
    )

with col2:
    import plotly.graph_objects as go
    
    fig_contrib = go.Figure(go.Bar(
        x=loadings_pc.values,
        y=loadings_pc.index,
        orientation='h',
        marker=dict(color='steelblue')
    ))
    
    fig_contrib.update_layout(
        title=f"Contribuciones a {selected_pc}",
        xaxis_title="Contribución Absoluta",
        yaxis_title="Variable",
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig_contrib, use_container_width=True)

# Insights
st.markdown("---")
st.markdown("## 💡 Interpretación y Hallazgos")

st.markdown("""
### ¿Qué nos dice el PCA?

El Análisis de Componentes Principales (PCA) reduce la dimensionalidad de los datos mientras retiene 
la mayor cantidad de variabilidad posible. Cada componente principal es una combinación lineal de las 
variables originales.

**Preguntas clave a responder**:

1. **¿Se separan los grupos CT y ELA?** Observa si hay clustering en las proyecciones 2D/3D
2. **¿Cuánta varianza capturan los primeros componentes?** Idealmente >70-80% con 2-3 componentes
3. **¿Qué métricas son más importantes?** Las variables con loadings altos en PC1 y PC2 son las más influyentes
4. **¿Hay outliers?** Puntos alejados del cluster principal pueden ser casos atípicos

### Siguientes pasos

Si el PCA muestra separación limitada, el **Autoencoder** puede capturar relaciones no lineales 
que el PCA (método lineal) no detecta. Visita la página de Autoencoder para entrenar el modelo.
""")

# Export option
st.markdown("---")

if st.button("Exportar Datos PCA"):
    pca_data = pd.DataFrame(
        pca.components,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    pca_data['Group'] = data['Group'].values
    pca_data['Sex'] = data['Sex'].values
    pca_data['Participant'] = data['Participant'].values
    
    csv = pca_data.to_csv(index=False)
    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name="pca_results.csv",
        mime="text/csv"
    )
    st.success("✓ Datos PCA listos para descargar")
