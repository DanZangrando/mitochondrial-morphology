"""
Page 1: Exploratory Data Analysis (EDA)
Análisis con nivel de participante (estadísticamente válido) 
y nivel de mediciones individuales (exploratorio)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MitochondriaDataLoader

# Page config
st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

st.title("📊 Análisis Exploratorio de Datos (EDA)")

st.markdown("""
Este análisis utiliza **dos niveles complementarios**:

1. **🎯 Análisis por Participante** (PRINCIPAL): 
   - Estadísticamente válido para comparaciones entre grupos
   - Cada participante = promedio de sus mediciones
   - Tests estadísticos formales

2. **🔍 Análisis de Mediciones Individuales** (EXPLORATORIO):
   - Visualiza variabilidad intra-participante
   - Explora patrones y outliers
   - NO usar para inferencia estadística formal
""")

# Load data
@st.cache_data
def load_data():
    loader = MitochondriaDataLoader()
    data = loader.load_data()
    
    # Aggregate by participant (mean)
    feature_cols = loader.get_feature_columns()
    
    # Group by participant and aggregate (groupby keys are automatically preserved)
    participant_agg = data.groupby(['Participant', 'Group', 'Sex', 'Age']).agg({
        **{col: 'mean' for col in feature_cols}
    }).reset_index()
    
    # Count number of measurements (mitochondria) per participant
    n_measurements = data.groupby('Participant').size().reset_index(name='n_mitochondrias')
    participant_agg = participant_agg.merge(n_measurements, on='Participant')
    
    return data, participant_agg, loader

data_individual, data_participant, loader = load_data()
feature_cols = loader.get_feature_columns()

# Dataset info
st.markdown("---")
st.markdown("## 📋 Información del Dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Participantes", data_participant['Participant'].nunique())
    
with col2:
    st.metric("Mediciones Totales", len(data_individual))
    
with col3:
    ct_count = (data_participant['Group'] == 'CT').sum()
    st.metric("CT", ct_count)
    
with col4:
    ela_count = (data_participant['Group'] == 'ELA').sum()
    st.metric("ELA", ela_count)

# Sidebar controls
st.sidebar.header("⚙️ Controles de Análisis")

analysis_level = st.sidebar.radio(
    "Nivel de Análisis",
    options=['Por Participante (Válido estadísticamente)', 'Por Mediciones Individuales (Exploratorio)'],
    index=0
)

# Add n_mitochondrias to metrics only for participant-level analysis
if 'Participante' in analysis_level:
    available_metrics = feature_cols + ['n_mitochondrias']
else:
    available_metrics = feature_cols

selected_metric = st.sidebar.selectbox(
    "Métrica a Analizar",
    options=available_metrics,
    index=0
)

comparison_type = st.sidebar.radio(
    "Tipo de Comparación",
    options=['Grupo (CT vs ELA)', 'Sexo (F vs M)'],
    index=0
)

# Select appropriate dataset
if 'Participante' in analysis_level:
    data_to_use = data_participant
    data_label = "Participantes"
else:
    data_to_use = data_individual
    data_label = "Mediciones Individuales"

group_col = 'Group' if 'Grupo' in comparison_type else 'Sex'

# Statistical testing function with normality check
def perform_complete_statistical_test(data, metric, group_col):
    """
    Performs normality test first, then appropriate parametric/non-parametric test
    """
    groups = data[group_col].unique()
    
    if len(groups) != 2:
        return None
    
    group1_data = data[data[group_col] == groups[0]][metric].dropna()
    group2_data = data[data[group_col] == groups[1]][metric].dropna()
    
    # Normality tests (Shapiro-Wilk)
    _, p_norm1 = stats.shapiro(group1_data) if len(group1_data) >= 3 else (None, 1.0)
    _, p_norm2 = stats.shapiro(group2_data) if len(group2_data) >= 3 else (None, 1.0)
    
    is_normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)
    
    # Select appropriate test
    if is_normal:
        # Parametric: Independent t-test
        statistic, p_value = stats.ttest_ind(group1_data, group2_data)
        test_name = "t-test (paramétrico)"
    else:
        # Non-parametric: Mann-Whitney U
        statistic, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        test_name = "Mann-Whitney U (no paramétrico)"
    
    # Effect size (Cohen's d)
    mean1, mean2 = group1_data.mean(), group2_data.mean()
    std1, std2 = group1_data.std(), group2_data.std()
    pooled_std = np.sqrt(((len(group1_data)-1)*std1**2 + (len(group2_data)-1)*std2**2) / (len(group1_data)+len(group2_data)-2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    return {
        'test': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'normality': {
            groups[0]: {'p_value': p_norm1, 'is_normal': p_norm1 > 0.05 if p_norm1 else None},
            groups[1]: {'p_value': p_norm2, 'is_normal': p_norm2 > 0.05 if p_norm2 else None}
        },
        'is_normal_overall': is_normal,
        'group1': groups[0],
        'group2': groups[1],
        'group1_mean': mean1,
        'group2_mean': mean2,
        'group1_std': std1,
        'group2_std': std2,
        'group1_median': group1_data.median(),
        'group2_median': group2_data.median(),
        'group1_n': len(group1_data),
        'group2_n': len(group2_data),
        'cohens_d': cohens_d,
        'effect_size': 'pequeño' if abs(cohens_d) < 0.5 else ('mediano' if abs(cohens_d) < 0.8 else 'grande')
    }

# Main analysis
st.markdown("---")
st.markdown(f"## 📊 Análisis: {analysis_level}")

st.info(f"""
**Nivel actual**: {data_label}  
**N = {len(data_to_use)}**  
**Comparación**: {comparison_type}
""")

# Distribution plots
st.markdown("### Distribución de la Métrica")

plot_type = st.radio(
    "Tipo de visualización",
    options=['Box Plot', 'Violin Plot', 'Histogram'],
    horizontal=True,
    key="plot_type_main"
)

if plot_type == 'Box Plot':
    fig = px.box(
        data_to_use,
        x=group_col,
        y=selected_metric,
        color=group_col,
        points='all',
        title=f'{selected_metric} por {group_col}',
        template='plotly_white',
        labels={selected_metric: selected_metric, group_col: group_col}
    )
elif plot_type == 'Violin Plot':
    fig = px.violin(
        data_to_use,
        x=group_col,
        y=selected_metric,
        color=group_col,
        box=True,
        points='all',
        title=f'{selected_metric} por {group_col}',
        template='plotly_white'
    )
else:  # Histogram
    fig = px.histogram(
        data_to_use,
        x=selected_metric,
        color=group_col,
        barmode='overlay',
        title=f'Distribución de {selected_metric}',
        template='plotly_white',
        opacity=0.7
    )

st.plotly_chart(fig, use_container_width=True)

# Statistical testing
st.markdown("---")
st.markdown("### 🧪 Análisis Estadístico")

test_results = perform_complete_statistical_test(data_to_use, selected_metric, group_col)

if test_results:
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("#### Tests de Normalidad (Shapiro-Wilk)")
        
        norm_data = []
        for group in [test_results['group1'], test_results['group2']]:
            norm_info = test_results['normality'][group]
            norm_data.append({
                'Grupo': group,
                'p-valor': f"{norm_info['p_value']:.4f}" if norm_info['p_value'] else 'N/A',
                'Normal?': '✓ Sí' if norm_info['is_normal'] else '✗ No'
            })
        
        st.dataframe(pd.DataFrame(norm_data), use_container_width=True, hide_index=True)
        
        if test_results['is_normal_overall']:
            st.success("✓ Ambos grupos siguen distribución normal → Test paramétrico")
        else:
            st.warning("⚠ Al menos un grupo no es normal → Test no paramétrico")
    
    with col2:
        st.markdown(f"#### {test_results['group1']}")
        st.metric("N", test_results['group1_n'])
        st.metric("Media", f"{test_results['group1_mean']:.4f}")
        st.metric("Mediana", f"{test_results['group1_median']:.4f}")
        st.metric("Std", f"{test_results['group1_std']:.4f}")
    
    with col3:
        st.markdown(f"#### {test_results['group2']}")
        st.metric("N", test_results['group2_n'])
        st.metric("Media", f"{test_results['group2_mean']:.4f}")
        st.metric("Mediana", f"{test_results['group2_median']:.4f}")
        st.metric("Std", f"{test_results['group2_std']:.4f}")
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Test Aplicado", test_results['test'])
    
    with col2:
        st.metric("Estadístico", f"{test_results['statistic']:.4f}")
    
    with col3:
        p_display = f"{test_results['p_value']:.4f}" if test_results['p_value'] >= 0.0001 else "< 0.0001"
        st.metric("P-valor", p_display)
    
    with col4:
        st.metric("Tamaño Efecto", f"{test_results['effect_size']}")
        st.caption(f"Cohen's d = {test_results['cohens_d']:.3f}")
    
    if test_results['significant']:
        st.success("✅ **Diferencia estadísticamente significativa** (p < 0.05)")
    else:
        st.info("ℹ️ **Diferencia no significativa** (p ≥ 0.05)")
    
    # Interpretation
    st.markdown("#### 📖 Interpretación")
    
    diff = abs(test_results['group1_mean'] - test_results['group2_mean'])
    higher_group = test_results['group1'] if test_results['group1_mean'] > test_results['group2_mean'] else test_results['group2']
    
    interpretation = f"""
    - El grupo **{higher_group}** muestra valores {'significativamente ' if test_results['significant'] else ''}más altos
    - Diferencia de medias: **{diff:.4f}**
    - Tamaño del efecto: **{test_results['effect_size']}** (|d| = {abs(test_results['cohens_d']):.3f})
    """
    
    if test_results['significant']:
        interpretation += f"\n- Esta diferencia es **estadísticamente significativa** (p = {test_results['p_value']:.4f})"
    else:
        interpretation += f"\n- Esta diferencia **no es estadísticamente significativa** (p = {test_results['p_value']:.4f})"
    
    st.markdown(interpretation)


# Correlation heatmap
st.markdown("---")
st.markdown("## 🔗 Matriz de Correlación")

st.info("⚠️ Esta correlación se calcula a nivel de **participante** para evitar pseudoreplicación")

# Calculate correlation (include n_mitochondrias)
corr_cols = feature_cols + ['n_mitochondrias']
corr_matrix = data_participant[corr_cols].corr()

fig_corr = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu_r',
    zmid=0,
    text=corr_matrix.values.round(2),
    texttemplate='%{text}',
    textfont={"size": 10},
    colorbar=dict(title="Correlación")
))

fig_corr.update_layout(
    title="Matriz de Correlación (Nivel Participante + N° Mitocondrias)",
    template='plotly_white',
    height=650,
    xaxis={'side': 'bottom'},
    yaxis={'autorange': 'reversed'}
)

st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("""
**Interpretación**: 
- Valores cercanos a **+1**: Correlación positiva fuerte
- Valores cercanos a **-1**: Correlación negativa fuerte  
- Valores cercanos a **0**: Sin correlación
- **n_mitochondrias**: Número de mitocondrias medidas por participante
""")

# Summary statistics by participant
st.markdown("---")
st.markdown("## 📈 Estadísticas Resumidas (Nivel Participante)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Por Grupo (CT vs ELA)")
    summary_cols = feature_cols + ['n_mitochondrias']
    group_summary = data_participant.groupby('Group')[summary_cols].agg(['mean', 'std', 'count'])
    st.dataframe(group_summary.round(4), use_container_width=True)

with col2:
    st.markdown("### Por Sexo (F vs M)")
    sex_summary = data_participant.groupby('Sex')[summary_cols].agg(['mean', 'std', 'count'])
    st.dataframe(sex_summary.round(4), use_container_width=True)

# Age analysis
st.markdown("---")
st.markdown("## 👤 Análisis por Edad y Sexo")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.box(
        data_participant,
        x='Group',
        y='Age',
        color='Group',
        points='all',
        title='Distribución de Edad por Grupo',
        template='plotly_white'
    )
    st.plotly_chart(fig_age, use_container_width=True)
    
    # Age comparison test
    age_test = perform_complete_statistical_test(data_participant, 'Age', 'Group')
    if age_test:
        st.markdown(f"""
        **Test de edad**: {age_test['test']}  
        **P-valor**: {age_test['p_value']:.4f}  
        **Resultado**: {'Diferencia significativa' if age_test['significant'] else 'Sin diferencia significativa'}
        """)

with col2:
    fig_sex = px.histogram(
        data_participant,
        x='Sex',
        color='Group',
        barmode='group',
        title='Distribución de Sexo por Grupo',
        template='plotly_white'
    )
    st.plotly_chart(fig_sex, use_container_width=True)
    
    # Sex distribution
    sex_group_counts = data_participant.groupby(['Group', 'Sex']).size().reset_index(name='count')
    st.dataframe(sex_group_counts, use_container_width=True, hide_index=True)

# Participant details table
st.markdown("---")
st.markdown("## 👥 Detalles por Participante")

st.markdown("""
Cada fila representa un **participante único**. Los valores de las métricas son **promedios** 
de todas las mitocondrias medidas para ese participante.
""")

# Create display dataframe
display_cols = ['Participant', 'Group', 'Sex', 'Age', 'n_mitochondrias'] + feature_cols
participant_display = data_participant[display_cols].sort_values(['Group', 'Participant'])

st.dataframe(
    participant_display.round(4),
    use_container_width=True,
    height=500
)

# Summary statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Participantes", len(data_participant))

with col2:
    ct_part = (data_participant['Group'] == 'CT').sum()
    st.metric("Participantes CT", ct_part)

with col3:
    ela_part = (data_participant['Group'] == 'ELA').sum()
    st.metric("Participantes ELA", ela_part)

with col4:
    total_mito = data_participant['n_mitochondrias'].sum()
    st.metric("Total Mitocondrias", total_mito)

# Multi-comparison analysis
st.markdown("---")
st.markdown("## 🔬 Análisis Multi-Variable")

with st.expander("Ver todas las comparaciones estadísticas"):
    st.markdown("### Comparación de TODAS las métricas")
    
    comparison_choice = st.radio(
        "Tipo de comparación",
        options=['Grupo (CT vs ELA)', 'Sexo (F vs M)'],
        key="multi_comp"
    )
    
    comp_col = 'Group' if 'Grupo' in comparison_choice else 'Sex'
    
    results_list = []
    
    # Include n_mitochondrias in the analysis
    metrics_to_test = feature_cols + ['n_mitochondrias']
    
    for metric in metrics_to_test:
        test_res = perform_complete_statistical_test(data_participant, metric, comp_col)
        if test_res:
            results_list.append({
                'Métrica': metric,
                'Test': test_res['test'],
                'P-valor': test_res['p_value'],
                'Significativo': '✓' if test_res['significant'] else '✗',
                'Cohen\'s d': test_res['cohens_d'],
                'Tamaño Efecto': test_res['effect_size'],
                f"Media {test_res['group1']}": test_res['group1_mean'],
                f"Media {test_res['group2']}": test_res['group2_mean']
            })
    
    results_df = pd.DataFrame(results_list)
    
    # Sort by p-value
    results_df = results_df.sort_values('P-valor')
    
    st.dataframe(results_df.round(4), use_container_width=True, hide_index=True)
    
    # Count significant differences
    n_significant = (results_df['Significativo'] == '✓').sum()
    st.markdown(f"""
    **Resumen**: {n_significant} de {len(metrics_to_test)} métricas muestran diferencias estadísticamente significativas (p < 0.05)
    """)
    
    # Bonferroni correction warning
    bonferroni_threshold = 0.05 / len(metrics_to_test)
    st.warning(f"""
    ⚠️ **Corrección de Bonferroni**: Con {len(metrics_to_test)} comparaciones, 
    el umbral ajustado sería p < {bonferroni_threshold:.4f} para controlar el error tipo I familiar.
    """)

# Scatter plot analysis
st.markdown("---")
st.markdown("## 📍 Análisis de Dispersión")

with st.expander("Ver relaciones entre variables"):
    col1, col2 = st.columns(2)
    
    scatter_cols = feature_cols + ['n_mitochondrias']
    
    with col1:
        x_var = st.selectbox("Variable X", options=scatter_cols, index=0, key="scatter_x")
    
    with col2:
        y_var = st.selectbox("Variable Y", options=scatter_cols, index=1 if len(scatter_cols) > 1 else 0, key="scatter_y")
    
    fig_scatter = px.scatter(
        data_participant,
        x=x_var,
        y=y_var,
        color='Group',
        size='n_mitochondrias',
        hover_data=['Participant', 'Sex', 'Age'],
        title=f'{x_var} vs {y_var} (nivel participante)',
        template='plotly_white',
        opacity=0.7
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Calculate correlation by group
    for group in data_participant['Group'].unique():
        group_data = data_participant[data_participant['Group'] == group]
        corr, p_val = stats.pearsonr(group_data[x_var], group_data[y_var])
        st.markdown(f"**{group}**: r = {corr:.3f}, p = {p_val:.4f}")

# Export option
st.markdown("---")
st.markdown("## 💾 Exportar Datos Agregados")

if st.button("📥 Descargar datos a nivel participante (CSV)"):
    csv = data_participant.to_csv(index=False)
    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name="mitochondria_participant_level.csv",
        mime="text/csv"
    )
