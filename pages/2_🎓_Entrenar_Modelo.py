"""
Page 2: Train VAE Models with Real-time Monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import subprocess
import time
import yaml
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MitochondriaDataLoader

# Page config
st.set_page_config(page_title="Entrenar Modelo", page_icon="🎓", layout="wide")

st.title("🎓 Entrenar Modelo VAE")

st.markdown("""
Esta página te permite entrenar modelos **Variational Autoencoder (VAE)** con clasificación integrada
para analizar morfología mitocondrial y distinguir entre grupos CT y ELA.
""")

# Load data for info
@st.cache_data
def load_data_info():
    loader = MitochondriaDataLoader()
    X_scaled, data = loader.prepare_features(standardize=True)
    return data, loader

data, loader = load_data_info()

# Show dataset info
with st.expander("📊 Información del Dataset", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Observaciones", len(data))
        st.metric("Participantes", data['Participant'].nunique())
    
    with col2:
        group_counts = data['Group'].value_counts()
        st.metric("Grupo CT", group_counts.get('CT', 0))
        st.metric("Grupo ELA", group_counts.get('ELA', 0))
    
    with col3:
        seq_lengths = data.groupby('Participant').size()
        st.metric("Min mediciones/participante", seq_lengths.min())
        st.metric("Max mediciones/participante", seq_lengths.max())
        st.metric("Promedio mediciones", f"{seq_lengths.mean():.1f}")

st.markdown("---")

# Training Configuration
st.markdown("## ⚙️ Configuración de Entrenamiento")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Tipo de Modelo")
    
    model_type = st.radio(
        "Selecciona el tipo de modelo a entrenar:",
        options=[
            "📊 VAE Estándar (mean pooling)",
            "🔬 LSTM-VAE (preserva secuencias)"
        ],
        help="""
        **VAE Estándar**: Agrega múltiples mediciones por participante usando mean pooling.
        Más simple y rápido, pero pierde información de variabilidad intra-participante.
        
        **LSTM-VAE**: Preserva todas las mediciones en secuencias. Captura la variabilidad 
        intra-participante que puede ser informativa para distinguir CT de ELA.
        """
    )
    
    use_lstm = "LSTM" in model_type
    
    st.markdown("### Arquitectura")
    
    if use_lstm:
        st.info("""
        **LSTM-VAE Architecture:**
        - Input: Secuencias de 4-36 mediciones × 8 features
        - Encoder: Bidirectional LSTM (2 capas, hidden=64)
        - Latent: 16D (μ, σ) desde hidden state final
        - Decoder: Unidirectional LSTM (2 capas, hidden=64)
        - Classifier: [32, 16] → 2 clases (CT/ELA)
        - Parámetros: ~205,850
        """)
    else:
        st.info("""
        **Standard VAE Architecture:**
        - Input: 8 features agregadas por participante
        - Encoder: [64, 32] → Latent 8D (μ, σ)
        - Decoder: [32, 64] → 8 features
        - Classifier: [16] → 2 clases (CT/ELA)
        - Parámetros: ~6,700
        """)

with col2:
    st.markdown("### Hiperparámetros")
    
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    max_epochs = st.number_input(
        "Max Epochs",
        min_value=10,
        max_value=500,
        value=config['autoencoder']['training']['max_epochs'],
        help="Máximo número de épocas. El entrenamiento puede parar antes con early stopping."
    )
    
    learning_rate = st.number_input(
        "Learning Rate",
        min_value=0.00001,
        max_value=0.01,
        value=config['autoencoder']['training']['learning_rate'],
        format="%.5f",
        help="Tasa de aprendizaje para el optimizador Adam"
    )
    
    batch_size = st.number_input(
        "Batch Size",
        min_value=2,
        max_value=32,
        value=4 if use_lstm else 16,
        help="Tamaño del batch. LSTM usa batches más pequeños debido a secuencias variables."
    )
    
    early_stopping_patience = st.number_input(
        "Early Stopping Patience",
        min_value=5,
        max_value=50,
        value=config['autoencoder']['training']['early_stopping_patience'],
        help="Épocas sin mejora antes de detener el entrenamiento"
    )

st.markdown("---")

# Training Section
st.markdown("## 🚀 Entrenar Modelo")

# Check existing models
existing_models = glob.glob("models/*.ckpt")
if use_lstm:
    existing_lstm_models = [m for m in existing_models if 'lstm' in os.path.basename(m).lower()]
    if existing_lstm_models:
        st.warning(f"⚠️ Ya existen {len(existing_lstm_models)} modelos LSTM-VAE entrenados. El nuevo modelo se guardará junto a ellos.")
else:
    existing_vae_models = [m for m in existing_models if 'lstm' not in os.path.basename(m).lower()]
    if existing_vae_models:
        st.warning(f"⚠️ Ya existen {len(existing_vae_models)} modelos VAE estándar entrenados. El nuevo modelo se guardará junto a ellos.")

col1, col2 = st.columns([3, 1])

with col1:
    train_button = st.button(
        "🚀 Iniciar Entrenamiento",
        type="primary",
        use_container_width=True
    )

with col2:
    if st.button("🧹 Limpiar Logs", help="Eliminar logs de TensorBoard anteriores"):
        log_dir = "logs/lstm_vae_classifier" if use_lstm else "logs/vae_classifier"
        if os.path.exists(log_dir):
            import shutil
            shutil.rmtree(log_dir)
            st.success(f"✓ Logs eliminados: {log_dir}")
        else:
            st.info("No hay logs para eliminar")

if train_button:
    # Create placeholders for dynamic updates
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    status_placeholder.info("🔧 Preparando entrenamiento...")
    
    try:
        # Create log directory
        log_dir = "logs/lstm_vae_classifier" if use_lstm else "logs/vae_classifier"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Kill any existing TensorBoard processes
        try:
            subprocess.run(["pkill", "-f", "tensorboard"], stderr=subprocess.DEVNULL)
            time.sleep(1)
        except:
            pass
        
        status_placeholder.info("📊 Iniciando TensorBoard...")
        
        # Start TensorBoard server
        tb_port = 6006
        tb_process = subprocess.Popen(
            ["tensorboard", "--logdir", "logs/", "--port", str(tb_port), "--bind_all"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for TensorBoard to start
        time.sleep(3)
        
        status_placeholder.success("✓ TensorBoard iniciado en puerto 6006")
        
        # Display TensorBoard
        st.markdown("### 📊 Monitoreo en Tiempo Real")
        st.markdown("Puedes ver el progreso del entrenamiento en tiempo real:")
        
        tb_container = st.container()
        with tb_container:
            st.components.v1.iframe(
                f"http://localhost:{tb_port}",
                height=600,
                scrolling=True
            )
        
        status_placeholder.info("🏋️ Entrenando modelo... (esto puede tomar varios minutos)")
        
        # Update progress
        progress_bar = progress_placeholder.progress(0)
        progress_text = st.empty()
        progress_text.text("Inicializando entrenamiento...")
        
        # Import training function
        from scripts.train_autoencoder import train_vae
        
        # Prepare training arguments
        kwargs = {
            'use_lstm': use_lstm,
            'max_epochs': max_epochs,
            'batch_size': batch_size,
        }
        
        # Train model
        progress_text.text("Entrenando... Consulta TensorBoard arriba para detalles")
        best_model_path = train_vae(**kwargs)
        
        # Training complete
        progress_bar.progress(100)
        progress_text.empty()
        status_placeholder.success("✅ ¡Entrenamiento completado exitosamente!")
        
        # Show results
        st.markdown("---")
        st.markdown("## 🎉 Resultados del Entrenamiento")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"✓ Modelo guardado en:")
            st.code(best_model_path, language="text")
            
            st.info("""
            **Siguiente paso**: Ve a la página **🤖 Autoencoder** para:
            - Visualizar el espacio latente
            - Evaluar métricas de clasificación
            - Comparar reconstrucciones
            """)
        
        with col2:
            st.markdown("**Archivos generados:**")
            st.markdown(f"- Modelo: `{os.path.basename(best_model_path)}`")
            st.markdown(f"- Logs: `{log_dir}/`")
            st.markdown(f"- TensorBoard: http://localhost:6006")
        
        st.balloons()
        
        # Keep TensorBoard running
        st.info("💡 TensorBoard seguirá corriendo. Puedes refrescar el iframe arriba para ver las métricas finales.")
        
    except Exception as e:
        status_placeholder.error(f"❌ Error durante el entrenamiento")
        st.exception(e)
        
        st.markdown("---")
        st.markdown("### 🛠️ Solución de Problemas")
        
        st.error(f"Error: {str(e)}")
        
        st.markdown("**Posibles soluciones:**")
        st.markdown("1. Verifica que todas las dependencias estén instaladas: `pip install -r requirements.txt`")
        st.markdown("2. Asegúrate de que el archivo `data/data.csv` existe y tiene el formato correcto")
        st.markdown("3. Intenta entrenar manualmente desde la terminal:")
        
        if use_lstm:
            st.code("python scripts/train_autoencoder.py --lstm", language="bash")
        else:
            st.code("python scripts/train_autoencoder.py", language="bash")
        
        st.markdown("4. Revisa los logs en la terminal para más detalles")

# View Previous Training Runs
st.markdown("---")
st.markdown("## 📈 Ver Entrenamientos Anteriores")

# Check if logs exist
log_dirs_info = []
if os.path.exists("logs/vae_classifier"):
    vae_runs = len([d for d in os.listdir("logs/vae_classifier") if os.path.isdir(os.path.join("logs/vae_classifier", d))])
    log_dirs_info.append(f"📊 VAE Estándar: {vae_runs} runs")

if os.path.exists("logs/lstm_vae_classifier"):
    lstm_runs = len([d for d in os.listdir("logs/lstm_vae_classifier") if os.path.isdir(os.path.join("logs/lstm_vae_classifier", d))])
    log_dirs_info.append(f"🔬 LSTM-VAE: {lstm_runs} runs")

if log_dirs_info:
    st.markdown("**Entrenamientos disponibles:**")
    for info in log_dirs_info:
        st.markdown(f"- {info}")
    
    view_logs = st.checkbox("📊 Ver TensorBoard de entrenamientos anteriores", value=False)
    
    if view_logs:
        st.markdown("### TensorBoard - Histórico de Entrenamientos")
        
        # Start TensorBoard if not running
        try:
            import requests
            tb_running = False
            try:
                response = requests.get("http://localhost:6006", timeout=1)
                tb_running = True
            except:
                pass
            
            if not tb_running:
                st.info("Iniciando TensorBoard...")
                subprocess.Popen(
                    ["tensorboard", "--logdir", "logs/", "--port", "6006", "--bind_all"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                time.sleep(3)
            
            # Display TensorBoard
            st.components.v1.iframe(
                "http://localhost:6006",
                height=800,
                scrolling=True
            )
            
            st.info("""
            **💡 Tip**: En TensorBoard puedes:
            - Comparar múltiples runs simultáneamente
            - Filtrar por nombre de modelo (vae_classifier vs lstm_vae_classifier)
            - Hacer zoom en regiones específicas de las curvas
            - Descargar datos como CSV para análisis adicional
            """)
            
        except Exception as e:
            st.error(f"Error al iniciar TensorBoard: {e}")
            st.info("Inicia TensorBoard manualmente: `tensorboard --logdir=logs/`")
else:
    st.info("📭 No hay entrenamientos anteriores. ¡Entrena tu primer modelo arriba!")

# Information section
st.markdown("---")
st.markdown("## 📚 Información Adicional")

with st.expander("❓ ¿Qué modelo elegir?", expanded=False):
    st.markdown("""
    ### VAE Estándar vs LSTM-VAE
    
    **Usa VAE Estándar si:**
    - Quieres un modelo más simple y rápido
    - La media de las mediciones es suficiente para tu análisis
    - Tienes limitaciones computacionales
    - Buscas interpretabilidad lineal
    
    **Usa LSTM-VAE si:**
    - La variabilidad intra-participante puede ser informativa
    - Quieres capturar patrones temporales/secuenciales
    - Tienes suficientes datos (múltiples mediciones por participante)
    - Buscas mayor capacidad representacional
    
    ### Recomendación
    Prueba ambos modelos y compara:
    - Accuracy en clasificación CT vs ELA
    - Separabilidad de clusters en espacio latente
    - Error de reconstrucción
    """)

with st.expander("🔬 Sobre el Espacio Latente", expanded=False):
    st.markdown("""
    El **espacio latente** es una representación comprimida de los datos originales.
    
    ### VAE vs Autoencoder Tradicional
    
    - **Autoencoder**: Aprende puntos fijos en el espacio latente
    - **VAE**: Aprende una distribución (μ, σ) por cada muestra
    
    ### Ventajas del VAE
    
    1. **Regularización**: KL divergence fuerza distribuciones a ser similares a N(0,1)
    2. **Generativo**: Puedes samplear nuevos puntos del espacio latente
    3. **Interpolación suave**: El espacio latente es más continuo y estructurado
    4. **Incertidumbre**: σ captura la incertidumbre de la representación
    
    ### Interpretación
    
    - Clusters bien separados → features discriminan bien los grupos
    - Overlap entre grupos → features no son suficientemente discriminativas
    - Alta varianza en cluster → grupo heterogéneo
    """)

with st.expander("⚙️ Hiperparámetros Explicados", expanded=False):
    st.markdown("""
    ### Learning Rate
    - **Muy bajo** (<0.0001): Entrenamiento lento, puede no converger
    - **Óptimo** (0.001): Balance entre velocidad y estabilidad
    - **Muy alto** (>0.01): Entrenamiento inestable, puede diverger
    
    ### Batch Size
    - **Pequeño** (2-8): Más ruido, mejor generalización, entrenamiento más lento
    - **Medio** (16-32): Balance entre velocidad y estabilidad
    - **Grande** (>32): Entrenamiento rápido, pero puede overfit
    
    ### Early Stopping Patience
    - **Bajo** (5-10): Detiene rápido, puede subentrenar
    - **Medio** (15-25): Balance recomendado
    - **Alto** (>30): Permite más exploración, riesgo de overfitting
    
    ### KL Weight
    - **Bajo** (<0.001): Reconstrucción precisa, espacio latente menos estructurado
    - **Alto** (>0.01): Espacio latente muy estructurado, reconstrucción peor
    - **Balance**: 0.001 para VAE, 0.0001 para LSTM-VAE (secuencias más complejas)
    """)

st.markdown("---")
st.markdown("### 🤝 ¿Necesitas ayuda?")
st.markdown("""
- **Documentación**: Revisa `LSTM_VAE_ARCHITECTURE.md` para detalles técnicos
- **Testing**: Ejecuta `python test_lstm_vae.py` para verificar la instalación
- **Issues**: Si encuentras bugs, reporta en el repositorio
""")
