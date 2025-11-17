"""
Page 1: Exploratory Data Analysis (EDA)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MitochondriaDataLoader
from src.utils import (
    create_distribution_plot,
    create_correlation_heatmap,
    perform_statistical_tests
)

# Page config
st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

st.title("📊 Análisis Exploratorio de Datos (EDA)")

# Load data
@st.cache_data
def load_data():
    loader = MitochondriaDataLoader()
    return loader.load_data(), loader

data, loader = load_data()

# Sidebar controls
st.sidebar.header("Controles de Visualización")

# Select metric
feature_cols = loader.get_feature_columns()
selected_metric = st.sidebar.selectbox(
    "Seleccionar Métrica",
    options=feature_cols,
    index=0
)

# Select grouping
group_by = st.sidebar.radio(
    "Agrupar por",
    options=['Group', 'Sex'],
    index=0
)

# Select plot type
plot_type = st.sidebar.selectbox(
    "Tipo de Gráfico",
    options=['box', 'violin', 'histogram'],
    index=0
)

# Main content
st.markdown("## Distribuciones por Grupo")

# Distribution plot
fig = create_distribution_plot(data, selected_metric, group_by, plot_type)
st.plotly_chart(fig, use_container_width=True)

# Statistical test
st.markdown("### 🔬 Prueba Estadística")

col1, col2 = st.columns([2, 1])

with col1:
    test_results = perform_statistical_tests(data, selected_metric, group_by)
    
    st.markdown(f"**Test aplicado**: {test_results['test']}")
    st.markdown(f"**Estadístico**: {test_results['statistic']:.4f}")
    st.markdown(f"**P-valor**: {test_results['p_value']:.4f}")
    
    if test_results['significant']:
        st.success("✓ Diferencia significativa detectada (p < 0.05)")
    else:
        st.info("Diferencia no significativa (p >= 0.05)")

with col2:
    if 'group1_mean' in test_results:
        groups = data[group_by].unique()
        st.metric(f"Media {groups[0]}", f"{test_results['group1_mean']:.4f}")
        st.metric(f"Media {groups[1]}", f"{test_results['group2_mean']:.4f}")

# Correlation heatmap
st.markdown("---")
st.markdown("## Matriz de Correlación")

fig_corr = create_correlation_heatmap(data, feature_cols)
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("""
**Interpretación**: Los valores cercanos a 1 o -1 indican correlación fuerte (positiva o negativa).
Valores cercanos a 0 indican poca correlación entre variables.
""")

# Summary statistics
st.markdown("---")
st.markdown("## Estadísticas Resumidas")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Por Grupo")
    group_summary = data.groupby('Group')[feature_cols].agg(['mean', 'std', 'count'])
    st.dataframe(group_summary.round(3), use_container_width=True)

with col2:
    st.markdown("### Por Sexo")
    sex_summary = data.groupby('Sex')[feature_cols].agg(['mean', 'std', 'count'])
    st.dataframe(sex_summary.round(3), use_container_width=True)

# Scatter plot matrix (sample)
st.markdown("---")
st.markdown("## Scatter Plot Matrix (Muestra)")

with st.expander("Ver gráfico de dispersión entre variables"):
    # Sample data to avoid overplotting
    sample_data = data.sample(n=min(300, len(data)), random_state=42)
    
    selected_features = st.multiselect(
        "Seleccionar variables para el scatter plot",
        options=feature_cols,
        default=feature_cols[:4]
    )
    
    if len(selected_features) >= 2:
        fig_scatter = px.scatter_matrix(
            sample_data,
            dimensions=selected_features,
            color='Group',
            title="Scatter Plot Matrix (Muestra de 300 observaciones)",
            template='plotly_white',
            opacity=0.6
        )
        fig_scatter.update_traces(diagonal_visible=False, showupperhalf=False)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Selecciona al menos 2 variables")

# Age distribution
st.markdown("---")
st.markdown("## Distribución por Edad")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        data,
        x='Age',
        color='Group',
        barmode='overlay',
        title='Distribución de Edad por Grupo',
        template='plotly_white',
        opacity=0.7
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_age_box = px.box(
        data,
        x='Group',
        y='Age',
        color='Group',
        title='Edad por Grupo',
        template='plotly_white'
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

# Participant overview
st.markdown("---")
st.markdown("## Overview de Participantes")

participant_summary = data.groupby(['Participant', 'Group', 'Sex', 'Age']).size().reset_index(name='N_observaciones')
participant_summary = participant_summary.sort_values('Participant')

st.dataframe(participant_summary, use_container_width=True, height=400)

st.markdown(f"**Total de participantes**: {participant_summary['Participant'].nunique()}")
st.markdown(f"- CT: {len(participant_summary[participant_summary['Group']=='CT'])} participantes")
st.markdown(f"- ELA: {len(participant_summary[participant_summary['Group']=='ELA'])} participantes")
