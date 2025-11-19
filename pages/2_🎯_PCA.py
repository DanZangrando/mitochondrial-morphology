"""
Page 2: PCA Analysis
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
from sklearn.decomposition import PCA
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MitochondriaDataLoader

# Page config
st.set_page_config(page_title="PCA", page_icon="🎯", layout="wide")

st.title("🎯 Análisis de Componentes Principales (PCA)")

st.markdown("""
Este análisis utiliza **dos niveles complementarios**:

1. **🎯 Análisis por Participante** (PRINCIPAL): 
   - Estadísticamente válido para comparaciones entre grupos
   - Cada participante = promedio de sus mediciones morfológicas
   - PCA sobre N=20 muestras independientes
   - **Incluye n_mitochondrias** como característica adicional

2. **🔍 Análisis de Mediciones Individuales** (EXPLORATORIO):
   - Visualiza variabilidad intra-participante en el espacio reducido
   - PCA sobre N=306 mediciones (asume independencia)
   - Solo métricas morfológicas (sin n_mitochondrias)
   - NO usar para inferencia estadística formal

### Variables incluidas en PCA:

**Nivel Participante**:
- 8 métricas morfológicas (promedio por participante)
- **n_mitochondrias** (número total de mitocondrias medidas)
- **NO incluye**: Group, Sex, Age (son labels para visualización posterior)

**Nivel Individual**:
- 8 métricas morfológicas solamente
- **NO incluye**: n_mitochondrias (no tiene sentido a nivel de medición individual)
- **NO incluye**: Group, Sex, Age (labels para visualización)

### ¿Por qué no incluir Group/Sex/Age en PCA?

El PCA es un método de **reducción de dimensionalidad no supervisado**. Incluir las etiquetas 
de clase o variables demográficas sesgaría el análisis. Queremos que PCA descubra la estructura 
natural de los datos morfológicos, y **luego** visualizar cómo se agrupan por Group/Sex/Age.
""")

# Load data
@st.cache_data
def load_data():
    loader = MitochondriaDataLoader()
    data = loader.load_data()
    
    # Get feature columns (morphological metrics only)
    feature_cols = loader.get_feature_columns()
    
    # Aggregate by participant (mean of morphological features)
    participant_agg = data.groupby(['Participant', 'Group', 'Sex', 'Age']).agg({
        **{col: 'mean' for col in feature_cols}
    }).reset_index()
    
    # Count number of measurements (mitochondria) per participant
    n_measurements = data.groupby('Participant').size().reset_index(name='n_mitochondrias')
    participant_agg = participant_agg.merge(n_measurements, on='Participant')
    
    return data, participant_agg, loader, feature_cols

data_individual, data_participant, loader, feature_cols = load_data()

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
st.sidebar.header("⚙️ Configuración PCA")

analysis_level = st.sidebar.radio(
    "Nivel de Análisis",
    options=['Por Participante (Válido estadísticamente)', 'Por Mediciones Individuales (Exploratorio)'],
    index=0
)

# Select dataset and features
if 'Participante' in analysis_level:
    data_to_use = data_participant
    data_label = "Participantes"
    # Include n_mitochondrias for participant-level analysis
    pca_features = feature_cols + ['n_mitochondrias']
    st.sidebar.info(f"✅ **Características PCA**: {len(pca_features)} variables (8 morfológicas + n_mitochondrias)")
else:
    data_to_use = data_individual
    data_label = "Mediciones Individuales"
    # Only morphological features for individual measurements
    pca_features = feature_cols
    st.sidebar.info(f"✅ **Características PCA**: {len(pca_features)} variables morfológicas")

n_components = st.sidebar.slider(
    "Número de Componentes",
    min_value=2,
    max_value=min(len(pca_features), 5),
    value=3
)

color_by = st.sidebar.selectbox(
    "Colorear puntos por",
    options=['Group', 'Sex', 'Participant'],
    index=0
)

# Standardize features for PCA
from sklearn.preprocessing import StandardScaler

@st.cache_data
def perform_pca_analysis(data, features, n_comp, level):
    """Perform PCA and return results"""
    # Extract feature matrix
    X = data[features].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=n_comp)
    components = pca.fit_transform(X_scaled)
    
    # Create results dataframe
    pca_df = pd.DataFrame(
        components,
        columns=[f'PC{i+1}' for i in range(n_comp)]
    )
    
    # Add metadata
    pca_df['Participant'] = data['Participant'].values
    pca_df['Group'] = data['Group'].values
    pca_df['Sex'] = data['Sex'].values
    
    return pca, pca_df, X_scaled, scaler

pca, pca_df, X_scaled, scaler = perform_pca_analysis(
    data_to_use, 
    pca_features, 
    n_components,
    analysis_level
)

# Main Analysis
st.markdown("---")
st.markdown(f"## 📊 Análisis PCA: {analysis_level}")

st.info(f"""
**Nivel actual**: {data_label}  
**N = {len(data_to_use)}**  
**Características en PCA**: {', '.join(pca_features)}  
**Componentes extraídos**: {n_components}
""")

# Explained variance
st.markdown("### Varianza Explicada")

explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

col1, col2 = st.columns(2)

with col1:
    # Bar plot
    fig_var = go.Figure()
    
    fig_var.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(n_components)],
        y=explained_var * 100,
        name='Individual',
        marker_color='steelblue'
    ))
    
    fig_var.update_layout(
        title='Varianza Explicada por Componente',
        xaxis_title='Componente Principal',
        yaxis_title='Varianza Explicada (%)',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig_var, use_container_width=True)

with col2:
    # Cumulative plot
    fig_cum = go.Figure()
    
    fig_cum.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(n_components)],
        y=cumulative_var * 100,
        mode='lines+markers',
        marker=dict(size=10),
        line=dict(color='darkgreen', width=3)
    ))
    
    fig_cum.add_hline(y=80, line_dash="dash", line_color="red", 
                      annotation_text="80% threshold")
    
    fig_cum.update_layout(
        title='Varianza Acumulada',
        xaxis_title='Componente Principal',
        yaxis_title='Varianza Acumulada (%)',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig_cum, use_container_width=True)

# Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("PC1 Varianza", f"{explained_var[0]*100:.2f}%")

with col2:
    st.metric("PC2 Varianza", f"{explained_var[1]*100:.2f}%")

with col3:
    if n_components >= 3:
        st.metric("PC3 Varianza", f"{explained_var[2]*100:.2f}%")
    else:
        st.metric("PC1+PC2", f"{cumulative_var[1]*100:.2f}%")

with col4:
    st.metric("Total Acumulada", f"{cumulative_var[-1]*100:.2f}%")

# Loadings (contribution of each feature)
st.markdown("---")
st.markdown("## 🔍 Contribución de Variables (Loadings)")

st.markdown("""
Los **loadings** muestran cómo cada variable original contribuye a cada componente principal.
- Valores altos (positivos o negativos) indican mayor contribución
- El signo indica la dirección de la contribución
""")

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(n_components)],
    index=pca_features
)

st.dataframe(loadings.round(3), use_container_width=True, height=350)

# 2D Visualization
st.markdown("---")
st.markdown("## 📍 Proyección 2D: PC1 vs PC2")

fig_2d = px.scatter(
    pca_df,
    x='PC1',
    y='PC2',
    color=color_by,
    hover_data=['Participant', 'Group', 'Sex'],
    title=f'PCA 2D - Coloreado por {color_by} ({data_label})',
    template='plotly_white',
    height=600
)

fig_2d.update_traces(marker=dict(size=10, opacity=0.7))

# Add variance to axis labels
fig_2d.update_xaxes(title=f'PC1 ({explained_var[0]*100:.2f}% varianza)')
fig_2d.update_yaxes(title=f'PC2 ({explained_var[1]*100:.2f}% varianza)')

st.plotly_chart(fig_2d, use_container_width=True)

# 3D Visualization
if n_components >= 3:
    st.markdown("---")
    st.markdown("## 🎲 Proyección 3D: PC1 vs PC2 vs PC3")
    
    fig_3d = px.scatter_3d(
        pca_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color=color_by,
        hover_data=['Participant', 'Group', 'Sex'],
        title=f'PCA 3D - Coloreado por {color_by} ({data_label})',
        template='plotly_white',
        height=700
    )
    
    fig_3d.update_traces(marker=dict(size=5, opacity=0.7))
    
    # Add variance to axis labels
    fig_3d.update_layout(
        scene=dict(
            xaxis_title=f'PC1 ({explained_var[0]*100:.2f}%)',
            yaxis_title=f'PC2 ({explained_var[1]*100:.2f}%)',
            zaxis_title=f'PC3 ({explained_var[2]*100:.2f}%)'
        )
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)

# Statistical tests on PCs
st.markdown("---")
st.markdown("## 📊 Tests Estadísticos en Componentes Principales")

st.markdown("""
Aplicamos tests de significancia para determinar si los grupos (CT vs ELA) o sexos (F vs M) 
se diferencian significativamente en cada componente principal.

**Metodología**:
1. Test de normalidad (Shapiro-Wilk) para cada grupo
2. Si ambos grupos son normales → t-test (paramétrico)
3. Si algún grupo no es normal → Mann-Whitney U (no paramétrico)
4. Cálculo de Cohen's d para tamaño del efecto
""")

def perform_pc_statistical_test(pca_data, pc_name, group_col):
    """Perform statistical test on a principal component"""
    groups = pca_data[group_col].unique()
    
    if len(groups) != 2:
        return None
    
    group1_data = pca_data[pca_data[group_col] == groups[0]][pc_name].dropna()
    group2_data = pca_data[pca_data[group_col] == groups[1]][pc_name].dropna()
    
    # Normality tests
    _, p_norm1 = stats.shapiro(group1_data) if len(group1_data) >= 3 else (None, 1.0)
    _, p_norm2 = stats.shapiro(group2_data) if len(group2_data) >= 3 else (None, 1.0)
    
    is_normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)
    
    # Select appropriate test
    if is_normal:
        statistic, p_value = stats.ttest_ind(group1_data, group2_data)
        test_name = "t-test"
    else:
        statistic, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        test_name = "Mann-Whitney U"
    
    # Effect size (Cohen's d)
    mean1, mean2 = group1_data.mean(), group2_data.mean()
    std1, std2 = group1_data.std(), group2_data.std()
    pooled_std = np.sqrt(((len(group1_data)-1)*std1**2 + (len(group2_data)-1)*std2**2) / 
                         (len(group1_data)+len(group2_data)-2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    return {
        'test': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'is_normal': is_normal,
        'group1': groups[0],
        'group2': groups[1],
        'group1_mean': mean1,
        'group2_mean': mean2,
        'cohens_d': cohens_d,
        'effect_size': 'pequeño' if abs(cohens_d) < 0.5 else ('mediano' if abs(cohens_d) < 0.8 else 'grande')
    }

# Select comparison type
comparison_type = st.radio(
    "Tipo de Comparación",
    options=['Grupo (CT vs ELA)', 'Sexo (F vs M)'],
    horizontal=True,
    key="pc_comparison"
)

comp_col = 'Group' if 'Grupo' in comparison_type else 'Sex'

# Run tests for all PCs
pc_test_results = []

for i in range(n_components):
    pc_name = f'PC{i+1}'
    test_res = perform_pc_statistical_test(pca_df, pc_name, comp_col)
    
    if test_res:
        pc_test_results.append({
            'Componente': pc_name,
            'Test': test_res['test'],
            'P-valor': test_res['p_value'],
            'Significativo': '✓' if test_res['significant'] else '✗',
            'Cohen\'s d': test_res['cohens_d'],
            'Tamaño Efecto': test_res['effect_size'],
            f"Media {test_res['group1']}": test_res['group1_mean'],
            f"Media {test_res['group2']}": test_res['group2_mean'],
            'Normal': '✓' if test_res['is_normal'] else '✗'
        })

results_df = pd.DataFrame(pc_test_results)

st.dataframe(results_df.round(4), use_container_width=True, hide_index=True)

# Summary
n_significant = (results_df['Significativo'] == '✓').sum()
st.markdown(f"""
**Resumen**: {n_significant} de {n_components} componentes principales muestran diferencias 
estadísticamente significativas (p < 0.05) entre {comparison_type}.
""")

if n_significant > 0:
    st.success(f"✅ Se encontraron {n_significant} componente(s) con diferencias significativas. "
               f"Esto sugiere que el PCA captura variabilidad relacionada con {comparison_type}.")
else:
    st.warning(f"⚠️ No se encontraron componentes con diferencias significativas. "
               f"El PCA puede no estar capturando variabilidad relacionada con {comparison_type}.")

# Biplot (loadings + scores)
st.markdown("---")
st.markdown("## 🎨 Biplot: Scores + Loadings")

st.markdown("""
El **biplot** combina:
- **Scores**: Posición de cada muestra en el espacio PC1-PC2 (puntos)
- **Loadings**: Dirección e importancia de cada variable original (flechas)

Las flechas largas indican variables con mayor influencia en esos componentes.
""")

# Create biplot
fig_biplot = go.Figure()

# Add scores (scatter points)
for group in pca_df['Group'].unique():
    group_data = pca_df[pca_df['Group'] == group]
    fig_biplot.add_trace(go.Scatter(
        x=group_data['PC1'],
        y=group_data['PC2'],
        mode='markers',
        name=group,
        marker=dict(size=10, opacity=0.6),
        text=group_data['Participant'],
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
    ))

# Add loadings (arrows)
# Scale loadings to fit in the score space
loading_scale = 3.0
for i, feature in enumerate(pca_features):
    fig_biplot.add_trace(go.Scatter(
        x=[0, pca.components_[0, i] * loading_scale],
        y=[0, pca.components_[1, i] * loading_scale],
        mode='lines+text',
        line=dict(color='red', width=2),
        text=['', feature],
        textposition='top center',
        textfont=dict(size=10, color='darkred'),
        showlegend=False,
        hovertemplate=f'<b>{feature}</b><br>Loading PC1: {pca.components_[0, i]:.3f}<br>Loading PC2: {pca.components_[1, i]:.3f}<extra></extra>'
    ))

fig_biplot.update_layout(
    title=f'Biplot: PC1 vs PC2 ({data_label})',
    xaxis_title=f'PC1 ({explained_var[0]*100:.2f}%)',
    yaxis_title=f'PC2 ({explained_var[1]*100:.2f}%)',
    template='plotly_white',
    height=700
)

st.plotly_chart(fig_biplot, use_container_width=True)

# Component interpretation
st.markdown("---")
st.markdown("## 🔬 Interpretación de Componentes")

selected_pc = st.selectbox(
    "Seleccionar Componente para Analizar",
    options=[f'PC{i+1}' for i in range(n_components)],
    index=0,
    key="pc_interpret"
)

pc_idx = int(selected_pc[2:]) - 1

st.markdown(f"### Variables que más contribuyen a {selected_pc}")

# Get absolute loadings for this PC
loadings_pc = loadings.iloc[:, pc_idx].abs().sort_values(ascending=False)

col1, col2 = st.columns([1, 2])

with col1:
    st.dataframe(
        pd.DataFrame({
            'Variable': loadings_pc.index,
            'Contribución (abs)': loadings_pc.values,
            'Loading original': loadings.iloc[:, pc_idx].loc[loadings_pc.index].values
        }).round(3),
        use_container_width=True,
        height=400
    )

with col2:
    fig_contrib = go.Figure(go.Bar(
        x=loadings_pc.values,
        y=loadings_pc.index,
        orientation='h',
        marker=dict(
            color=loadings_pc.values,
            colorscale='Viridis',
            showscale=True
        )
    ))
    
    fig_contrib.update_layout(
        title=f'Contribuciones Absolutas a {selected_pc}',
        xaxis_title='|Loading|',
        yaxis_title='Variable',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig_contrib, use_container_width=True)

# Export
st.markdown("---")
st.markdown("## 💾 Exportar Resultados PCA")

if st.button("📥 Descargar datos PCA (CSV)", key="export_pca"):
    export_df = pca_df.copy()
    
    # Add original features if participant level
    if 'Participante' in analysis_level:
        export_df['Age'] = data_to_use['Age'].values
        export_df['n_mitochondrias'] = data_to_use['n_mitochondrias'].values
    
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name=f"pca_results_{data_label.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )
    st.success("✓ Datos PCA listos para descargar")

# Interpretation guide
st.markdown("---")
st.markdown("## 💡 Guía de Interpretación")

st.markdown("""
### ¿Qué nos dice el PCA?

**1. Varianza Explicada**:
- Si PC1+PC2 > 60%: Los primeros dos componentes capturan la mayoría de la información
- Si PC1+PC2 < 40%: Los datos son muy multidimensionales, considerar más componentes

**2. Separación de Grupos**:
- **Clara separación en PC1/PC2**: Las métricas morfológicas discriminan bien entre CT y ELA
- **No hay separación**: Las métricas morfológicas son similares entre grupos
- **Separación en PC3+**: La diferencia existe pero en dimensiones secundarias

**3. Loadings (Contribuciones)**:
- Variables con loadings altos (>0.3) en PC1/PC2 son las más importantes
- Variables con mismo signo están correlacionadas
- Variables con signo opuesto están anti-correlacionadas

**4. Tests Estadísticos**:
- PCs significativos (p < 0.05) indican diferencias reales entre grupos
- Cohen's d grande (>0.8) indica diferencias sustanciales
- Si ningún PC es significativo, considerar métodos no lineales (LSTM Classifier)

### Nivel Participante vs Individual

**Por Participante (N=20)**:
- ✅ Estadísticamente válido para comparaciones
- ✅ Incluye n_mitochondrias como característica
- ✅ Cada punto es independiente
- ⚠️ Menos puntos, puede ser menos estable

**Por Mediciones Individuales (N=306)**:
- 🔍 Visualiza variabilidad intra-participante
- ⚠️ Asume independencia (pseudoreplicación)
- ⚠️ NO usar para inferencia formal
- ✅ Más puntos, patrones más claros visualmente

### Siguientes Pasos

Si el PCA lineal no muestra separación clara, el **LSTM Classifier** puede:
- Capturar relaciones no lineales en secuencias temporales
- Procesar variabilidad intra-participante (secuencias de longitud variable)
- Clasificar activamente CT vs ELA con métricas robustas (K-Fold CV)
- Manejar el dataset pequeño (N=20) mediante validación cruzada

Ve a la página **🎓 Entrenar Modelo** para entrenar el clasificador con K-Fold CV.
""")
