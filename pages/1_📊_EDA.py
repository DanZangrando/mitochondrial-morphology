"""
Page 1: Exploratory Data Analysis (EDA)
Refactored to integrate with dynamic variable selection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MitochondriaDataLoader

# Page config
st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

# --- Helper Functions ---

def perform_statistical_test(data, metric, group_col):
    """Performs normality test and appropriate statistical test (t-test or Mann-Whitney)."""
    groups = data[group_col].unique()
    if len(groups) != 2:
        return None
    
    g1 = data[data[group_col] == groups[0]][metric].dropna()
    g2 = data[data[group_col] == groups[1]][metric].dropna()
    
    # Normality (Shapiro-Wilk) - requires N >= 3
    _, p_norm1 = stats.shapiro(g1) if len(g1) >= 3 else (None, 1.0)
    _, p_norm2 = stats.shapiro(g2) if len(g2) >= 3 else (None, 1.0)
    is_normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)
    
    # Test
    if is_normal:
        stat, p_val = stats.ttest_ind(g1, g2)
        test_name = "t-test (Paramétrico)"
    else:
        stat, p_val = stats.mannwhitneyu(g1, g2)
        test_name = "Mann-Whitney U (No Paramétrico)"
        
    # Effect Size (Cohen's d)
    pooled_std = np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) / (len(g1)+len(g2)-2))
    cohens_d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0
    
    return {
        'test': test_name,
        'p_value': p_val,
        'significant': p_val < 0.05,
        'is_normal': is_normal,
        'cohens_d': cohens_d,
        'group1': groups[0], 'mean1': g1.mean(), 'std1': g1.std(), 'n1': len(g1),
        'group2': groups[1], 'mean2': g2.mean(), 'std2': g2.std(), 'n2': len(g2)
    }

# --- Main App ---

st.title("📊 Análisis Exploratorio de Datos (EDA)")

# 1. Load Data
loader = MitochondriaDataLoader()
data_raw = loader.load_data()
feature_cols = loader.get_feature_columns()

# Prepare Data Levels
# Level 1: Individual Measurements (Raw)
data_individual = data_raw.copy()

# Level 2: Participant Aggregated
participant_agg = data_raw.groupby(['Participant', 'Group', 'Sex', 'Age']).agg({
    **{col: 'mean' for col in feature_cols}
}).reset_index()

# Add n_mitochondrias to participant level
n_measurements = data_raw.groupby('Participant').size().reset_index(name='n_mitochondrias')
participant_agg = participant_agg.merge(n_measurements, on='Participant')

# 2. Header & Context
st.markdown("---")
st.info(f"**Variables Activas ({len(feature_cols)}):** {', '.join(feature_cols)}")

# 3. Sidebar Controls
st.sidebar.header("⚙️ Configuración del Análisis")

analysis_level = st.sidebar.radio(
    "Nivel de Análisis",
    ["Participante (Promedios)", "Individual (Todas las mitocondrias)"],
    index=0,
    help="Participante: N=20 (Estadísticamente válido). Individual: N=~300 (Exploratorio)."
)

comparison_type = st.sidebar.radio(
    "Comparar por",
    ["Grupo (CT vs ELA)", "Sexo (Male vs Female)"],
    index=0
)

# Determine active dataset and columns
if "Participante" in analysis_level:
    data_active = participant_agg
    available_metrics = feature_cols + ['n_mitochondrias']
    level_name = "Participante"
else:
    data_active = data_individual
    available_metrics = feature_cols
    level_name = "Individual"

# --- Filters ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌪️ Filtrar Datos")
all_groups = sorted(data_active['Group'].unique())
all_sex = sorted(data_active['Sex'].unique())

selected_groups = st.sidebar.multiselect("Grupos", all_groups, default=all_groups)
selected_sex = st.sidebar.multiselect("Sexos", all_sex, default=all_sex)

# Apply filters
data_active = data_active[
    data_active['Group'].isin(selected_groups) & 
    data_active['Sex'].isin(selected_sex)
]

if data_active.empty:
    st.error("⚠️ No hay datos disponibles con los filtros seleccionados.")
    st.stop()

st.sidebar.info(f"**N = {len(data_active)}**")

group_col = 'Group' if "Grupo" in comparison_type else 'Sex'

# 4. Tabs Layout
tab1, tab2, tab3 = st.tabs(["📈 Análisis Univariado", "🔗 Análisis Multivariado", "📋 Datos Crudos"])

# --- Tab 1: Univariate ---
with tab1:
    st.subheader(f"Análisis de Distribución - Nivel {level_name}")
    
    col_sel, col_plot = st.columns([1, 3])
    
    with col_sel:
        selected_metric = st.selectbox("Seleccionar Métrica", available_metrics)
        plot_type = st.radio("Tipo de Gráfico", ["Box Plot", "Violin Plot", "Histograma"])
    
    with col_plot:
        # Dynamic Plotting
        # Simplify: Color by the same variable we are grouping by to avoid confusion
        color_col = group_col 
        
        if plot_type == "Box Plot":
            fig = px.box(data_active, x=group_col, y=selected_metric, color=color_col, points="all", 
                         title=f"{selected_metric} por {group_col}")
        elif plot_type == "Violin Plot":
            fig = px.violin(data_active, x=group_col, y=selected_metric, color=color_col, box=True, points="all",
                            title=f"{selected_metric} por {group_col}")
        else:
            fig = px.histogram(data_active, x=selected_metric, color=group_col, barmode="overlay", opacity=0.7,
                               title=f"Distribución de {selected_metric} por {group_col}")
            
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical Stats
    st.markdown("#### 🧪 Resultados Estadísticos")
    res = perform_statistical_test(data_active, selected_metric, group_col)
    
    if res:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Media {res['group1']}", f"{res['mean1']:.3f} (±{res['std1']:.3f})")
        c2.metric(f"Media {res['group2']}", f"{res['mean2']:.3f} (±{res['std2']:.3f})")
        c3.metric("P-Valor", f"{res['p_value']:.4f}", delta="Significativo" if res['significant'] else "No sig.")
        c4.metric("Cohen's d", f"{res['cohens_d']:.3f}")
        
        st.caption(f"Test usado: **{res['test']}**. Normalidad: {'✅' if res['is_normal'] else '⚠️'}")
    else:
        st.warning("No hay suficientes datos para realizar tests estadísticos.")

# --- Tab 2: Multivariate ---
with tab2:
    st.subheader("Correlaciones y Relaciones")
    
    # Correlation Matrix
    st.markdown("#### Matriz de Correlación")
    if level_name == "Participante":
        corr_cols = feature_cols + ['n_mitochondrias']
    else:
        corr_cols = feature_cols
        
    corr_matrix = data_active[corr_cols].corr()
    
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title=f"Correlación de Pearson (Nivel {level_name})")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Scatter Plot
    st.markdown("#### Scatter Plot Interactivo")
    c1, c2 = st.columns(2)
    x_axis = c1.selectbox("Eje X", available_metrics, index=0)
    y_axis = c2.selectbox("Eje Y", available_metrics, index=1 if len(available_metrics) > 1 else 0)
    
    fig_scatter = px.scatter(data_active, x=x_axis, y=y_axis, color=group_col, 
                             hover_data=['Participant', 'Age'], trendline="ols",
                             title=f"{x_axis} vs {y_axis}")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- Tab 3: Raw Data ---
with tab3:
    st.subheader(f"Datos: {level_name}")
    st.dataframe(data_active, use_container_width=True)
    
    csv = data_active.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Descargar CSV",
        csv,
        f"data_{level_name.lower()}.csv",
        "text/csv"
    )
