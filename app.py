"""
Mitochondrial Morphology Analysis - Main Streamlit App
"""

import streamlit as st
import pandas as pd
from src.data_loader import MitochondriaDataLoader

# Page configuration
st.set_page_config(
    page_title="Mitochondrial Morphology Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<p class="main-header">🧬 Mitochondrial Morphology Analysis</p>', unsafe_allow_html=True)

# Introduction
st.markdown("""
## Bienvenido al Análisis de Morfología Mitocondrial

Esta aplicación interactiva permite analizar métricas morfológicas de mitocondrias para 
identificar patrones y diferencias entre grupos de estudio (Control vs ELA).

### 🎯 Objetivos del Análisis

1. **Análisis Exploratorio de Datos (EDA)**: Examinar distribuciones y diferencias estadísticas
2. **Reducción Dimensional (PCA)**: Visualizar la estructura de los datos en componentes principales
3. **Autoencoder**: Explorar el espacio latente aprendido por deep learning

### 📊 Dataset

El dataset contiene métricas morfológicas de mitocondrias individuales:
""")

# Load and display data overview
try:
    # Direct load without data_loader for more robustness
    data = pd.read_csv('data/data.csv')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Observaciones", len(data))
    
    with col2:
        st.metric("Participantes", data['Participant'].nunique())
    
    with col3:
        ct_count = len(data[data['Group'] == 'CT'])
        st.metric("Grupo Control (CT)", ct_count)
    
    with col4:
        ela_count = len(data[data['Group'] == 'ELA'])
        st.metric("Grupo ELA", ela_count)
    
    # Interactive data table
    st.markdown("### 📋 Explorador de Datos Interactivo")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_group = st.multiselect(
            "Filtrar por Grupo",
            options=['CT', 'ELA'],
            default=['CT', 'ELA']
        )
    
    with col2:
        filter_sex = st.multiselect(
            "Filtrar por Sexo",
            options=['Male', 'Female'],
            default=['Male', 'Female']
        )
    
    with col3:
        filter_participant = st.multiselect(
            "Filtrar por Participante",
            options=sorted(data['Participant'].unique()),
            default=[]
        )
    
    # Apply filters
    filtered_data = data.copy()
    
    if filter_group:
        filtered_data = filtered_data[filtered_data['Group'].isin(filter_group)]
    
    if filter_sex:
        filtered_data = filtered_data[filtered_data['Sex'].isin(filter_sex)]
    
    if filter_participant:
        filtered_data = filtered_data[filtered_data['Participant'].isin(filter_participant)]
    
    st.info(f"Mostrando {len(filtered_data)} de {len(data)} observaciones")
    
    # Interactive table
    st.dataframe(
        filtered_data,
        use_container_width=True,
        height=400,
        column_config={
            "Participant": st.column_config.NumberColumn("ID Participante", format="%d"),
            "N mitocondrias": st.column_config.NumberColumn("N Mitocondrias", format="%d"),
            "PROM IsoVol": st.column_config.NumberColumn("Prom IsoVol", format="%.3f"),
            "PROM Surface": st.column_config.NumberColumn("Prom Surface", format="%.3f"),
            "PROM Length": st.column_config.NumberColumn("Prom Length", format="%.3f"),
            "PROM RoughSph": st.column_config.NumberColumn("Prom RoughSph", format="%.3f"),
            "Age": st.column_config.NumberColumn("Edad", format="%d años"),
        }
    )
    
    # Download button
    csv = filtered_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar datos filtrados (CSV)",
        data=csv,
        file_name="mitochondria_filtered_data.csv",
        mime="text/csv",
    )
    
    # Feature description
    st.markdown("### 🔍 Métricas Analizadas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Métricas Morfológicas:**
        - **IsoVol**: Volumen isométrico (SUMA y PROMEDIO)
        - **Surface**: Superficie mitocondrial (SUMA y PROMEDIO)
        - **Length**: Longitud mitocondrial (SUMA y PROMEDIO)
        - **RoughSph**: Índice de rugosidad/esfericidad (SUMA y PROMEDIO)
        """)
    
    with col2:
        st.markdown("""
        **Variables Demográficas:**
        - **Group**: CT (Control) o ELA (Esclerosis Lateral Amiotrófica)
        - **Sex**: Sexo del participante (Male/Female)
        - **Age**: Edad del participante
        - **Participant**: Identificador del participante
        - **N mitocondrias**: Número de mitocondrias por observación
        """)
    
    # Statistics summary
    st.markdown("### 📈 Resumen Estadístico por Grupo")
    
    feature_cols = ['N mitocondrias', 'PROM IsoVol', 'PROM Surface', 'PROM Length', 'PROM RoughSph']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Grupo Control (CT)**")
        ct_stats = data[data['Group'] == 'CT'][feature_cols].describe().T[['mean', 'std']]
        ct_stats.columns = ['Media', 'Desv. Est.']
        st.dataframe(ct_stats.round(3), use_container_width=True)
    
    with col2:
        st.markdown("**Grupo ELA**")
        ela_stats = data[data['Group'] == 'ELA'][feature_cols].describe().T[['mean', 'std']]
        ela_stats.columns = ['Media', 'Desv. Est.']
        st.dataframe(ela_stats.round(3), use_container_width=True)
    
    # Navigation guide
    st.markdown("---")
    st.markdown("### 🚀 Cómo Usar esta Aplicación")
    
    st.markdown("""
    Utiliza el **menú lateral** para navegar entre las diferentes páginas:
    
    1. **📊 EDA - Análisis Exploratorio**: Visualiza distribuciones, correlaciones y pruebas estadísticas
    2. **🎯 PCA - Análisis de Componentes**: Explora la reducción dimensional con PCA
    3. **🤖 Autoencoder**: Entrena y visualiza el espacio latente del autoencoder
    
    Cada página contiene visualizaciones interactivas y análisis detallados.
    """)

except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.info("Asegúrate de que el archivo `data/data.csv` existe y está correctamente formateado.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>Análisis de Morfología Mitocondrial | Powered by Streamlit + PyTorch Lightning</p>
</div>
""", unsafe_allow_html=True)
