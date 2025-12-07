"""
Page 2: PCA Analysis
Refactored for clarity and simplicity.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MitochondriaDataLoader

# Page config
st.set_page_config(page_title="PCA", page_icon="🎯", layout="wide")

# --- Helper Functions ---
@st.cache_data
def perform_pca(data, features, n_components):
    """Performs PCA and returns results."""
    X = data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(n_components)])
    
    # Attach metadata if available
    for col in ['Participant', 'Group', 'Sex', 'Age']:
        if col in data.columns:
            pca_df[col] = data[col].values
            
    return pca, pca_df, X_scaled

def perform_pc_test(data, pc_col, group_col):
    """Performs statistical test for a PC between two groups."""
    groups = data[group_col].dropna().unique()
    if len(groups) != 2:
        return None
    
    g1 = data[data[group_col] == groups[0]][pc_col]
    g2 = data[data[group_col] == groups[1]][pc_col]
    
    # Normality
    _, p_norm1 = stats.shapiro(g1) if len(g1) >= 3 else (None, 1.0)
    _, p_norm2 = stats.shapiro(g2) if len(g2) >= 3 else (None, 1.0)
    is_normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)
    
    if is_normal:
        stat, p_val = stats.ttest_ind(g1, g2)
        test_name = "t-test"
    else:
        stat, p_val = stats.mannwhitneyu(g1, g2)
        test_name = "Mann-Whitney"
        
    # Cohen's d
    pooled_std = np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) / (len(g1)+len(g2)-2))
    cohens_d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0
    
    return {
        'PC': pc_col,
        'Test': test_name,
        'P-Value': p_val,
        'Significant': '✅' if p_val < 0.05 else '❌',
        'Cohen\'s d': cohens_d,
        'Group 1': groups[0], 'Mean 1': g1.mean(),
        'Group 2': groups[1], 'Mean 2': g2.mean()
    }

# --- Main App ---

st.title("🎯 Análisis de Componentes Principales (PCA)")

# 1. Load Data
loader = MitochondriaDataLoader()
data_raw = loader.load_data()
feature_cols = loader.get_feature_columns()

# Prepare Data Levels
# Level 1: Individual
data_individual = data_raw.copy()

# Level 2: Participant
participant_agg = data_raw.groupby(['Participant', 'Group', 'Sex', 'Age']).agg({
    **{col: 'mean' for col in feature_cols}
}).reset_index()
n_measurements = data_raw.groupby('Participant').size().reset_index(name='n_mitochondrias')
participant_agg = participant_agg.merge(n_measurements, on='Participant')

# 2. Sidebar Controls
st.sidebar.header("⚙️ Configuración PCA")

analysis_level = st.sidebar.radio(
    "Nivel de Análisis",
    ["Participante (N=20)", "Individual (N=306)"],
    index=0
)

# Determine active dataset
if "Participante" in analysis_level:
    data_active = participant_agg
    pca_features = feature_cols + ['n_mitochondrias']
    level_label = "Participante"
else:
    data_active = data_individual
    pca_features = feature_cols
    level_label = "Individual"

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

st.sidebar.info(f"**N = {len(data_active)}** | **Variables ({len(pca_features)}):** {', '.join(pca_features)}")

n_comps = st.sidebar.slider("Componentes", 2, min(len(pca_features), 5), 3)
color_by = st.sidebar.selectbox("Colorear por", ["Group", "Sex", "Participant"])

# 3. Perform PCA
pca, pca_df, X_scaled = perform_pca(data_active, pca_features, n_comps)
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

# 4. Tabs Layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Varianza", "📍 Proyecciones", "🔍 Loadings", "📋 Datos"])

# --- Tab 1: Variance ---
with tab1:
    st.subheader("Varianza Explicada")
    
    c1, c2 = st.columns(2)
    with c1:
        # Scree Plot
        fig_var = px.bar(
            x=[f'PC{i+1}' for i in range(n_comps)],
            y=explained_var * 100,
            labels={'x': 'Componente', 'y': 'Varianza (%)'},
            title="Varianza por Componente"
        )
        st.plotly_chart(fig_var, use_container_width=True)
        
    with c2:
        # Cumulative Plot
        fig_cum = px.line(
            x=[f'PC{i+1}' for i in range(n_comps)],
            y=cumulative_var * 100,
            markers=True,
            labels={'x': 'Componente', 'y': 'Varianza Acumulada (%)'},
            title="Varianza Acumulada"
        )
        fig_cum.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80%")
        st.plotly_chart(fig_cum, use_container_width=True)
        
    # Metrics
    cols = st.columns(n_comps)
    for i in range(n_comps):
        cols[i].metric(f"PC{i+1}", f"{explained_var[i]*100:.1f}%")

# --- Tab 2: Projections ---
with tab2:
    st.subheader(f"Proyección PCA - {level_label}")
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # 2D Plot
        fig_2d = px.scatter(
            pca_df, x='PC1', y='PC2', color=color_by,
            hover_data=['Participant', 'Group'],
            title=f"PC1 ({explained_var[0]*100:.1f}%) vs PC2 ({explained_var[1]*100:.1f}%)"
        )
        fig_2d.update_traces(marker=dict(size=10, opacity=0.8))
        st.plotly_chart(fig_2d, use_container_width=True)
        
    with c2:
        # 3D Plot (if >= 3 comps)
        if n_comps >= 3:
            fig_3d = px.scatter_3d(
                pca_df, x='PC1', y='PC2', z='PC3', color=color_by,
                hover_data=['Participant'],
                title="Vista 3D (PC1-PC2-PC3)"
            )
            fig_3d.update_traces(marker=dict(size=5))
            fig_3d.update_layout(height=400, margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("Selecciona 3 o más componentes para ver el gráfico 3D.")

    # Statistical Validation
    st.markdown("---")
    st.markdown("### 🧪 Validación Estadística")
    
    if color_by in ['Group', 'Sex']:
        st.caption(f"Evaluando diferencias significativas en los componentes principales según **{color_by}**.")
        stats_results = []
        for i in range(n_comps):
            res = perform_pc_test(pca_df, f'PC{i+1}', color_by)
            if res:
                stats_results.append(res)
        
        if stats_results:
            stats_df = pd.DataFrame(stats_results)
            # Format p-value
            stats_df['P-Value'] = stats_df['P-Value'].apply(lambda x: f"{x:.4f}" if x >= 0.0001 else "< 0.0001")
            stats_df['Cohen\'s d'] = stats_df['Cohen\'s d'].apply(lambda x: f"{x:.3f}")
            stats_df['Mean 1'] = stats_df['Mean 1'].apply(lambda x: f"{x:.3f}")
            stats_df['Mean 2'] = stats_df['Mean 2'].apply(lambda x: f"{x:.3f}")
            
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Interpretation
            sig_pcs = [r['PC'] for r in stats_results if r['Significant'] == '✅']
            if sig_pcs:
                st.success(f"🎉 Se encontraron diferencias significativas en: {', '.join(sig_pcs)}. Esto sugiere que el PCA separa bien los grupos en estas dimensiones.")
            else:
                st.warning("⚠️ No se encontraron diferencias significativas en los componentes principales. Los grupos podrían no ser separables linealmente.")
    else:
        st.info("Selecciona 'Group' o 'Sex' en 'Colorear por' para ver el análisis estadístico.")

# --- Tab 3: Loadings ---
with tab3:
    st.subheader("Contribución de Variables (Loadings)")
    
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_comps)],
        index=pca_features
    )
    
    # Biplot Logic
    st.markdown("#### Biplot (PC1 vs PC2)")
    
    fig_biplot = go.Figure()
    
    # Scores
    for g in pca_df[color_by].unique():
        subset = pca_df[pca_df[color_by] == g]
        fig_biplot.add_trace(go.Scatter(
            x=subset['PC1'], y=subset['PC2'], mode='markers', name=str(g),
            marker=dict(size=8, opacity=0.6)
        ))
        
    # Loadings (Arrows)
    scale = 3.0 # Scaling factor for visibility
    for feat in pca_features:
        l1 = loadings_df.loc[feat, 'PC1'] * scale
        l2 = loadings_df.loc[feat, 'PC2'] * scale
        
        fig_biplot.add_trace(go.Scatter(
            x=[0, l1], y=[0, l2], mode='lines+text',
            text=[None, feat], textposition="top center",
            line=dict(color='red', width=2), showlegend=False
        ))
        
    fig_biplot.update_layout(
        title="Biplot: Scores + Loadings",
        xaxis_title=f"PC1 ({explained_var[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({explained_var[1]*100:.1f}%)",
        height=600
    )
    st.plotly_chart(fig_biplot, use_container_width=True)
    
    # Heatmap of loadings
    st.markdown("#### Heatmap de Loadings")
    fig_heat = px.imshow(
        loadings_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Matriz de Loadings"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Correlation with Original Features
    st.markdown("#### 🔗 Correlación: Variables vs PCs")
    st.caption("Muestra qué variables originales están más correlacionadas con cada Componente Principal.")
    
    # Calculate correlations between original features (X) and PCs
    # We can use the loadings directly as they represent the correlation (if standardized) 
    # or calculate explicitly for clarity. Let's use loadings_df which is (n_features x n_components)
    
    # Sort by PC1 contribution
    loadings_sorted = loadings_df.sort_values(by='PC1', key=abs, ascending=False)
    
    st.dataframe(loadings_sorted.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1), use_container_width=True)

# --- Tab 4: Data ---
with tab4:
    st.subheader("Datos Transformados")
    st.dataframe(pca_df, use_container_width=True)
    
    csv = pca_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Descargar Resultados PCA",
        csv,
        "pca_results.csv",
        "text/csv"
    )
