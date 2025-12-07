import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MitochondriaDataLoader
from src.models import MitoAttentionMIL

# Page config
st.set_page_config(page_title="Entrenar Clasificador", page_icon="🎯", layout="wide")

st.title("🎯 Entrenar Clasificador (Attention-based MIL)")

# --- Sidebar ---
st.sidebar.header("⚙️ Configuración del Proyecto")
loader = MitochondriaDataLoader()
feature_cols = loader.get_feature_columns()
st.sidebar.info(f"**Variables Activas ({len(feature_cols)}):** {', '.join(feature_cols)}")

# Filters (Reuse logic from EDA/PCA)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌪️ Filtrar Dataset")
# We load data to get unique values for filters
data_raw = loader.load_data()
all_groups = sorted(data_raw['Group'].unique())
all_sex = sorted(data_raw['Sex'].unique())

selected_groups = st.sidebar.multiselect("Grupos", all_groups, default=all_groups)
selected_sex = st.sidebar.multiselect("Sexos", all_sex, default=all_sex)

# --- Helper Functions ---
# Removed @st.cache_resource to avoid UnhashableParamError with Tensors. 
# We manage persistence via st.session_state.
def train_model(X, y, lengths, config):
    """Train the model and return history + model."""
    input_dim = X.shape[2]
    model = MitoAttentionMIL(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        nhead=config['nhead'],
        dropout=config['dropout']
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # Split data
    dataset = TensorDataset(X, y, lengths)
    train_size = int(config['train_split'] * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Show split info
    st.info(f"📊 Split: {train_size} entrenamiento | {test_size} testeo")
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds)) # Full batch for eval
    
    import copy
    
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y, batch_len in train_loader:
            optimizer.zero_grad()
            logits, _, _ = model(batch_X, batch_len)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['train_acc'].append(correct / total)
        
        # Eval
        model.eval()
        with torch.no_grad():
            for batch_X, batch_y, batch_len in test_loader:
                logits, _, _ = model(batch_X, batch_len)
                loss = criterion(logits, batch_y)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == batch_y).sum().item() / batch_y.size(0)
                
                history['test_loss'].append(loss.item())
                history['test_acc'].append(acc)
                
                # Checkpoint
                if config.get('save_best', False):
                    if acc > best_acc:
                        best_acc = acc
                        best_epoch = epoch
                        best_model_wts = copy.deepcopy(model.state_dict())
        
        progress_bar.progress((epoch + 1) / config['epochs'])
        msg = f"Epoch {epoch+1}/{config['epochs']} - Loss: {history['train_loss'][-1]:.4f} - Acc: {history['train_acc'][-1]:.4f}"
        if config.get('save_best', False):
            msg += f" (Best Test Acc: {best_acc:.4f})"
        status_text.text(msg)
        
    # Load best model weights if enabled
    if config.get('save_best', False):
        model.load_state_dict(best_model_wts)
        st.success(f"🏆 Mejor modelo guardado del Epoch {best_epoch+1} con Accuracy de Test: {best_acc:.4f}")
    else:
        st.info("ℹ️ Se utilizó el modelo del último epoch.")
        
    return model, history, train_ds, test_ds

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🏗️ Arquitectura", "⚙️ Configuración", "📊 Resultados", "🌌 Espacio Latente"])

# --- Tab 1: Architecture ---
with tab1:
    st.markdown("### Arquitectura del Modelo: Attention-based MIL")
    
    st.graphviz_chart("""
    digraph Model {
        rankdir=TD;
        node [shape=box, style=filled, fillcolor="#f0f2f6", fontname="Sans"];
        edge [fontname="Sans", fontsize=10];

        subgraph cluster_input {
            label = "Input Processing";
            style=dashed;
            Input [label="Input Set\n(Batch, N_Mito, Features)", shape=oval, fillcolor="#e1f5fe"];
            Masking [label="Masking\n(Ignore Padding)", shape=note];
        }

        subgraph cluster_feature {
            label = "Feature Extraction";
            style=filled;
            color="#e8f5e9";
            Projection [label="Linear Projection\n(Features -> Hidden)", fillcolor="#c8e6c9"];
        }

        subgraph cluster_transformer {
            label = "Global Context & Attention";
            style=filled;
            color="#fff3e0";
            MHA [label="Multi-Head Self-Attention\n(Heads = N)", fillcolor="#ffe0b2"];
            AddNorm [label="Add & Norm\n(Residual Connection)", shape=circle, width=1];
        }

        subgraph cluster_pooling {
            label = "Aggregation (MIL)";
            style=filled;
            color="#f3e5f5";
            Pooling [label="Weighted Pooling\n(Attention-based)", fillcolor="#e1bee7"];
            Latent [label="Latent Vector\n(Batch, Hidden)", shape=parallelogram, fillcolor="#d1c4e9"];
        }

        subgraph cluster_classifier {
            label = "Classification Head";
            style=filled;
            color="#ffebee";
            FC1 [label="Linear -> ReLU -> Dropout", fillcolor="#ffcdd2"];
            FC2 [label="Linear (Output)", fillcolor="#ef9a9a"];
            Output [label="Prediction\n(CT vs ELA)", shape=oval, fillcolor="#ff8a80"];
        }

        Input -> Masking;
        Masking -> Projection [label="Valid Instances"];
        Projection -> MHA [label="Embeddings"];
        Projection -> AddNorm [label="Residual"];
        MHA -> AddNorm [label="Attended Features"];
        AddNorm -> Pooling [label="Contextualized Instances"];
        Pooling -> Latent;
        Latent -> FC1;
        FC1 -> FC2;
        FC2 -> Output [label="Logits"];
    }
    """)
    
    st.markdown("""
    **Detalles del Flujo (Attention-based Multiple Instance Learning):**
    1.  **Input**: Conjunto de mitocondrias de un participante. Se usa **Padding** y **Máscaras** para manejar el número variable de mitocondrias.
    2.  **Feature Projection**: Cada mitocondria se proyecta independientemente a un espacio latente.
    3.  **Self-Attention**: Permite que cada mitocondria "mire" a todas las demás para determinar su importancia global y contexto.
    4.  **Pooling**: Colapsa el conjunto en un único vector latente usando los pesos de atención.
    5.  **Clasificador**: Toma el vector latente y decide si es Control (CT) o ELA.
    """)

# --- Tab 2: Configuration ---
with tab2:
    st.subheader("Parámetros de Entrenamiento")
    
    c1, c2, c3 = st.columns(3)
    epochs = c1.number_input("Epochs", 10, 500, 50)
    lr = c2.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
    batch_size = c3.number_input("Batch Size", 1, 64, 8)
    
    c1, c2 = st.columns(2)
    hidden_dim = c1.number_input("Hidden Dimension", 16, 256, 64)
    nhead = c2.number_input("Attention Heads", 1, 8, 4)
    
    train_split = st.slider("Porcentaje de Entrenamiento", 0.5, 0.9, 0.8)
    
    # Checkbox for checkpointing
    save_best = st.checkbox("💾 Guardar el mejor modelo (basado en Test Accuracy)", value=True)
    
    if st.button("🚀 Entrenar Modelo", type="primary"):
        # Load and Filter Data
        # We need to filter at the participant level.
        # get_sequences returns all. We should filter the dataframe first inside get_sequences?
        # Or filter the output tensors.
        # Let's modify get_sequences to accept filtered dataframe or filter indices.
        # Actually, let's just use the filtered dataframe logic here.
        
        # 1. Filter Dataframe
        filtered_df = data_raw[
            data_raw['Group'].isin(selected_groups) & 
            data_raw['Sex'].isin(selected_sex)
        ]
        
        if filtered_df.empty:
            st.error("No hay datos con los filtros seleccionados.")
            st.stop()
            
        # Hack: Inject filtered data into loader temporarily or create new loader instance with filtered data?
        # Loader loads from file.
        # Let's manually filter the sequences returned by get_sequences.
        # But get_sequences re-reads data.
        # Better: Update loader.data with filtered_df
        loader.data = filtered_df
        
        with st.spinner("Generando secuencias y entrenando..."):
            X, y, lengths = loader.get_sequences(feature_cols)
            
            config = {
                'epochs': epochs, 'lr': lr, 'batch_size': batch_size,
                'hidden_dim': hidden_dim, 'nhead': nhead,
                'dropout': 0.1, 'train_split': train_split,
                'save_best': save_best
            }
            
            # Clear cache to force retrain if button clicked
            # train_model.clear() # Removed as we removed @st.cache_resource
            model, history, train_ds, test_ds = train_model(X, y, lengths, config)
            
            st.session_state['model'] = model
            st.session_state['history'] = history
            st.session_state['train_ds'] = train_ds
            st.session_state['test_ds'] = test_ds
            st.session_state['config'] = config
            st.success("✅ Entrenamiento completado!")

# --- Tab 3: Results ---
with tab3:
    if 'model' in st.session_state:
        history = st.session_state['history']
        model = st.session_state['model']
        
        # Loss Curves
        st.subheader("Curvas de Aprendizaje")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=history['train_loss'], name='Train Loss'))
        fig_loss.add_trace(go.Scatter(y=history['test_loss'], name='Test Loss'))
        st.plotly_chart(fig_loss, use_container_width=True)
        
        # Metrics & Confusion Matrix
        st.subheader("Evaluación")
        
        def evaluate_dataset(ds, name):
            loader_eval = DataLoader(ds, batch_size=len(ds))
            X_b, y_b, len_b = next(iter(loader_eval))
            model.eval()
            with torch.no_grad():
                logits, attn, _ = model(X_b, len_b)
                preds = torch.argmax(logits, dim=1)
                
            acc = accuracy_score(y_b, preds)
            f1 = f1_score(y_b, preds, average='weighted')
            cm = confusion_matrix(y_b, preds)
            
            return acc, f1, cm, attn, X_b, y_b, preds
            
        acc_train, f1_train, cm_train, _, _, _, _ = evaluate_dataset(st.session_state['train_ds'], "Train")
        acc_test, f1_test, cm_test, attn_test, X_test, y_test, preds_test = evaluate_dataset(st.session_state['test_ds'], "Test")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Set de Entrenamiento")
            st.metric("Accuracy", f"{acc_train:.4f}")
            st.metric("F1 Score", f"{f1_train:.4f}")
            fig_cm1 = px.imshow(cm_train, text_auto=True, title="Matriz de Confusión (Train)", labels=dict(x="Predicción", y="Real"))
            st.plotly_chart(fig_cm1, use_container_width=True)
            
        with c2:
            st.markdown("#### Set de Testeo")
            st.metric("Accuracy", f"{acc_test:.4f}")
            st.metric("F1 Score", f"{f1_test:.4f}")
            fig_cm2 = px.imshow(cm_test, text_auto=True, title="Matriz de Confusión (Test)", labels=dict(x="Predicción", y="Real"))
            st.plotly_chart(fig_cm2, use_container_width=True)
            
        # Attention Visualization
        st.markdown("---")
        st.subheader("👁️ Visualización de Atención (Test Set)")
        st.caption("Visualiza a qué mitocondrias prestó atención cada cabeza del Transformer.")
        
        # Select a sample from test set
        sample_idx = st.slider("Seleccionar Participante (Índice Test)", 0, len(X_test)-1, 0)
        
        # Get attention for this sample: (Heads, Seq, Seq)
        # We care about the attention of the CLS token or the self-attention map?
        # In self-attention, every token attends to every token.
        # To see "importance", we can look at the average attention received by each token, 
        # or if we had a CLS token, its attention weights.
        # Since we used Mean Pooling, all tokens contributed equally to the pool *after* being contextualized.
        # But the self-attention weights show how tokens relate.
        # Let's visualize the attention map for Head 0 (or select head).
        
        attn_sample = attn_test[sample_idx].cpu().numpy() # (Heads, Seq, Seq)
        n_heads = attn_sample.shape[0]
        
        head_idx = st.selectbox("Seleccionar Cabeza de Atención", range(n_heads))
        
        # Plot Heatmap (Seq x Seq)
        # We need to trim padding
        seq_len = (X_test[sample_idx].sum(dim=1) != 0).sum().item() # Rough check for non-zero rows
        # Better: use lengths from dataset if we stored them properly, but batch_len in eval loop had them.
        # Let's just crop to a reasonable size or show full.
        
        attn_map = attn_sample[head_idx, :seq_len, :seq_len]
        
        fig_attn = px.imshow(
            attn_map, 
            title=f"Mapa de Atención (Cabeza {head_idx}) - Participante {sample_idx}",
            labels=dict(x="Key (Mitochondria)", y="Query (Mitochondria)"),
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_attn, use_container_width=True)
        
        # Global Importance (Sum of Attention received)
        st.markdown("#### 📊 Importancia Global (Simplificada)")
        st.markdown("Este gráfico resume el heatmap anterior sumando cuánta atención recibió cada mitocondria en total.")
        
        # Sum columns (axis 0) to see how much attention each Key received from all Queries
        global_importance = attn_map.sum(axis=0)
        # Normalize to 0-1 for easier reading
        global_importance = global_importance / global_importance.max()
        
        df_imp = pd.DataFrame({
            'Mitocondria (Índice)': range(len(global_importance)),
            'Importancia Relativa': global_importance
        })
        
        fig_imp = px.bar(
            df_imp, x='Mitocondria (Índice)', y='Importancia Relativa',
            title=f"Mitocondrias más Influyentes (Participante {sample_idx})",
            color='Importancia Relativa',
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.info("""
        **Interpretación:**
        *   **Heatmap**: Las **columnas verticales brillantes** indican mitocondrias que son "miradas" por casi todas las demás.
        *   **Gráfico de Barras**: Las barras altas identifican las mitocondrias específicas que el modelo considera **cruciales** para clasificar a este paciente.
        """)

    else:
        st.info("Entrena el modelo para ver resultados.")
        
    # --- Advanced Analysis: Global Attention & Features ---
    if 'model' in st.session_state:
        st.markdown("---")
        st.header("🔬 Análisis Avanzado de Atención")
        
        # Helper to get global attention and features
        def get_global_analysis_data(ds, name):
            loader_eval = DataLoader(ds, batch_size=len(ds)) # Full batch
            X_all, y_all, len_all = next(iter(loader_eval))
            
            model.eval()
            with torch.no_grad():
                logits, attn, _ = model(X_all, len_all)
                preds = torch.argmax(logits, dim=1)
            
            # attn: (Batch, Heads, Seq, Seq)
            # Sum over Heads and Queries to get importance per Key (Mitochondrion)
            # Shape: (Batch, Seq)
            attn_importance = attn.sum(dim=1).sum(dim=1)
            
            # Normalize per participant (0-1)
            # Avoid div by zero
            max_vals = attn_importance.max(dim=1, keepdim=True)[0]
            attn_importance = attn_importance / max_vals.clamp(min=1e-9)
            
            return X_all, y_all, preds, attn_importance
            
        # Get data
        X_train, y_train, preds_train, attn_train = get_global_analysis_data(st.session_state['train_ds'], "Train")
        X_test, y_test, preds_test, attn_test = get_global_analysis_data(st.session_state['test_ds'], "Test")
        
        # 1. Mosaic View (Heatmaps)
        st.subheader("1. Mosaico de Atención Global")
        st.markdown("Visualización de la atención recibida por las mitocondrias de **todos** los participantes. Las filas son participantes, las columnas son sus mitocondrias (ordenadas por volumen).")
        
        def plot_mosaic(attn_tensor, y_true, y_pred, title):
            # attn_tensor: (Batch, Seq)
            attn_np = attn_tensor.cpu().numpy()
            
            # Create labels for Y-axis
            y_labels = []
            for i in range(len(y_true)):
                true_cls = "ELA" if y_true[i]==1 else "CT"
                pred_cls = "ELA" if y_pred[i]==1 else "CT"
                icon = "✅" if true_cls == pred_cls else "❌"
                y_labels.append(f"P{i} ({true_cls}->{pred_cls} {icon})")
                
            fig = px.imshow(
                attn_np,
                labels=dict(x="Mitocondria (Ordenada por Volumen)", y="Participante", color="Importancia"),
                y=y_labels,
                title=title,
                color_continuous_scale="Magma",
                aspect="auto" # Allow non-square pixels
            )
            fig.update_layout(height=max(400, 20 * len(y_true))) # Adjust height
            return fig
            
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_mosaic(attn_train, y_train, preds_train, "Mosaico - Train Set"), use_container_width=True)
        with c2:
            st.plotly_chart(plot_mosaic(attn_test, y_test, preds_test, "Mosaico - Test Set"), use_container_width=True)
            
        # 2. Feature Analysis of High Attention Mitos
        st.subheader("2. ¿Qué hace especial a una mitocondria importante?")
        st.markdown("Análisis de las características físicas de las mitocondrias a las que el modelo presta **mucha atención** (vs. poca atención), considerando solo participantes **clasificados correctamente**.")
        
        # Threshold slider
        attn_threshold = st.slider("Umbral de Atención Alta (0-1)", 0.5, 0.99, 0.8, help="Mitocondrias con importancia normalizada mayor a este valor se consideran 'Importantes'.")
        
        # Combine Train and Test for more robust stats? Or just Train? Or selectable?
        # Let's use ALL correctly classified data
        X_full = torch.cat([X_train, X_test])
        y_full = torch.cat([y_train, y_test])
        preds_full = torch.cat([preds_train, preds_test])
        attn_full = torch.cat([attn_train, attn_test])
        
        # Filter: Correctly Classified AND Non-Padding
        correct_mask = (y_full == preds_full)
        
        high_attn_features = []
        low_attn_features = []
        
        # Iterate over correctly classified participants
        for i in torch.where(correct_mask)[0]:
            # Get valid length (non-padding)
            # Assuming 0 padding. Check sum of features != 0
            valid_len = (X_full[i].sum(dim=1) != 0).sum().item()
            
            # Get importance and features for valid mitos
            imp = attn_full[i, :valid_len]
            feats = X_full[i, :valid_len]
            
            # Split into High/Low
            high_mask = imp >= attn_threshold
            low_mask = imp < attn_threshold
            
            if high_mask.any():
                high_attn_features.append(feats[high_mask])
            if low_mask.any():
                low_attn_features.append(feats[low_mask])
                
        if high_attn_features and low_attn_features:
            high_vals = torch.cat(high_attn_features).cpu().numpy()
            low_vals = torch.cat(low_attn_features).cpu().numpy()
            
            # Create DataFrame for plotting
            df_high = pd.DataFrame(high_vals, columns=feature_cols)
            df_high['Type'] = 'Alta Atención'
            
            df_low = pd.DataFrame(low_vals, columns=feature_cols)
            df_low['Type'] = 'Baja Atención'
            
            df_feats = pd.concat([df_high, df_low], ignore_index=True)
            
            # Plot boxplots for each feature
            # Use a selectbox to choose feature to visualize to avoid clutter
            selected_feat_viz = st.selectbox("Seleccionar Característica para Comparar", feature_cols)
            
            fig_box = px.box(
                df_feats, x='Type', y=selected_feat_viz, color='Type',
                title=f"Distribución de {selected_feat_viz} según Atención",
                points="outliers" # Show outliers
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Statistical Test
            from scipy.stats import ttest_ind
            t_stat, p_val = ttest_ind(df_high[selected_feat_viz], df_low[selected_feat_viz], equal_var=False)
            
            st.metric(
                label=f"Diferencia Significativa ({selected_feat_viz})", 
                value=f"p-value: {p_val:.2e}",
                delta="Significativo" if p_val < 0.05 else "No Significativo",
                delta_color="normal" if p_val < 0.05 else "off"
            )
            
        else:
            st.warning("No se encontraron suficientes mitocondrias para el análisis con este umbral.")

# --- Tab 4: Latent Space ---
with tab4:
    if 'model' in st.session_state:
        st.subheader("🌌 Espacio Latente")
        st.markdown("Visualización de los vectores latentes (antes del clasificador) usando t-SNE.")
        
        # Get latent vectors for ALL data
        # We need to run the full dataset through the model
        X_all, y_all, len_all = loader.get_sequences(feature_cols)
        model.eval()
        with torch.no_grad():
            _, _, latent_vecs = model(X_all, len_all)
            
        latent_np = latent_vecs.cpu().numpy()
        labels_np = y_all.cpu().numpy()
        
        # t-SNE
        n_samples = latent_np.shape[0]
        # Perplexity must be less than n_samples. Default is 30.
        # We set it to min(30, n_samples - 1) but ensure it's at least 1.
        perplexity = min(30, max(1, n_samples - 1))
        
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        projections = tsne.fit_transform(latent_np)
        
        df_latent = pd.DataFrame(projections, columns=['Dim1', 'Dim2'])
        df_latent['Label'] = ['ELA' if l==1 else 'CT' for l in labels_np]
        
        fig_tsne = px.scatter(
            df_latent, x='Dim1', y='Dim2', color='Label',
            title="t-SNE del Espacio Latente",
            template="plotly_white"
        )
        st.plotly_chart(fig_tsne, use_container_width=True)
        
        # Statistical Test on Latent Space (MANOVA equivalent or just t-test on dims)
        st.markdown("#### Test Estadístico en Espacio Latente")
        # Simple t-test on t-SNE dims for demonstration
        from scipy import stats
        
        g1 = df_latent[df_latent['Label']=='CT']
        g2 = df_latent[df_latent['Label']=='ELA']
        
        for dim in ['Dim1', 'Dim2']:
            t_stat, p_val = stats.ttest_ind(g1[dim], g2[dim])
            st.write(f"**{dim}**: p-value = {p_val:.4f} {'✅' if p_val<0.05 else '❌'}")
            
    else:
        st.info("Entrena el modelo para ver el espacio latente.")
