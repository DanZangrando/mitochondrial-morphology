# LSTM-VAE: Preservando Variabilidad Intra-Participante

## 🎯 Problema Solucionado

### ❌ Problema con Mean Pooling

```python
# Mean pooling pierde información valiosa
Participante 1: [medida1, medida2, ..., medida_n]
                     ↓ mean()
                 valor_único  ← ¡Se pierde la variabilidad!
```

**¿Qué se pierde?**
- Variabilidad intra-participante (importante para discriminar CT vs ELA)
- Patrones temporales o espaciales en las medidas
- Información sobre heterogeneidad mitocondrial

### ✅ Solución: LSTM-VAE

```python
# LSTM procesa la secuencia completa
Participante 1: [medida1, medida2, ..., medida_n]
                     ↓ LSTM Encoder
                 hidden_state  ← ¡Captura toda la variabilidad!
                     ↓
                  μ, σ (latent)
```

**Ventajas:**
- ✅ Captura variabilidad intra-participante
- ✅ Sensible al orden/patrón de medidas
- ✅ Cada medida contribuye al embedding latente
- ✅ Reconstruye la secuencia completa (no solo un promedio)

## 🏗️ Arquitectura LSTM-VAE

### Flujo Completo

```
Input: Secuencia de medidas por participante
[batch, seq_len, 8 features]
         ↓
┌────────────────────────┐
│   LSTM Encoder         │
│   Bidirectional        │
│   2 layers, hidden=64  │
└────────────────────────┘
         ↓
   Final Hidden State
   [batch, 128]  (64*2 por bidireccional)
         ↓
    ┌────┴────┐
    ↓         ↓
   μ FC    logσ² FC
  [batch, 16]  [batch, 16]
         ↓
  Reparameterization
  z = μ + σ * ε
         ↓
    ┌────┴────┐
    ↓         ↓
┌──────────┐ ┌─────────────┐
│ Decoder  │ │ Classifier  │
│  LSTM    │ │   [32,16]   │
│2 layers  │ │  →2 classes │
│hidden=64 │ │             │
└──────────┘ └─────────────┘
    ↓             ↓
Reconstrucción   CT/ELA
de secuencia    prediction
```

### Componentes Detallados

#### 1. LSTM Encoder

```python
self.encoder_lstm = nn.LSTM(
    input_size=8,           # 8 features morfológicas
    hidden_size=64,         # Dimensión hidden
    num_layers=2,           # Stack de 2 LSTMs
    batch_first=True,       # Input: [batch, seq, features]
    dropout=0.3,            # Dropout entre layers
    bidirectional=True      # Lee secuencia en ambas direcciones
)
```

**¿Qué hace?**
- Procesa cada medida secuencialmente
- Mantiene "memoria" de medidas anteriores en hidden state
- Bidireccional: lee forward y backward
- Final hidden state = representación de toda la secuencia

#### 2. Latent Space Projection

```python
# Desde el último hidden state del LSTM
h_final = concat([h_forward, h_backward])  # [batch, 128]

μ = fc_mu(h_final)           # [batch, 16]
log σ² = fc_logvar(h_final)  # [batch, 16]
```

**Reparameterization Trick:**
```python
z = μ + exp(0.5 * log σ²) * ε,  donde ε ~ N(0,1)
```

Esto permite:
- Backpropagation a través de sampling
- Espacio latente probabilístico (distribución, no punto fijo)

#### 3. LSTM Decoder

```python
# Inicializar decoder hidden state desde z
h_0, c_0 = latent_to_hidden(z)  # [layers, batch, hidden]

# Expandir z para cada paso temporal
decoder_input = z.repeat(1, seq_len, 1)  # [batch, seq_len, latent_dim]

# Decodificar
lstm_out, _ = decoder_lstm(decoder_input, (h_0, c_0))

# Proyectar a features originales
recon = fc_decoder(lstm_out)  # [batch, seq_len, 8]
```

**Reconstruye:**
- Toda la secuencia de medidas
- No solo un valor agregado
- Captura la variabilidad original

#### 4. Classifier Head

```python
classifier(z) → [batch, 2]  # CT=0, ELA=1
```

Predice el grupo directamente desde el espacio latente.

## 📊 Manejo de Secuencias Variables

### Problema: Longitudes Diferentes

```
Participante 1: 25 medidas
Participante 2: 15 medidas  ← Diferente longitud
Participante 3: 30 medidas
```

### Solución: Padding + Pack/Unpack

```python
class ParticipantSequenceDataset:
    def __getitem__(self, idx):
        seq = self.sequences[idx]      # Variable length
        length = len(seq)
        return seq, length, label
```

```python
def collate_sequences(batch):
    # Encontrar max length en el batch
    max_len = max(lengths)
    
    # Pad todas las secuencias al max_len con zeros
    padded_seqs = pad_sequences(sequences, max_len)
    
    return padded_seqs, lengths, labels
```

```python
def encode(self, x, lengths):
    # Pack: ignora posiciones paddeadas
    packed = pack_padded_sequence(x, lengths, batch_first=True)
    
    # LSTM procesa solo posiciones reales
    _, (h_n, c_n) = self.encoder_lstm(packed)
    
    # h_n contiene el estado al final de cada secuencia real
    return h_n
```

**Ventajas:**
- ✅ Eficiente: no procesa padding
- ✅ Correcto: hidden state al final de secuencia real
- ✅ Flexible: soporta cualquier longitud

## 🔢 Función de Pérdida

### Loss Total

```python
L_total = L_recon + α * L_KL + β * L_class

donde:
  α = 0.0001  (más bajo que en VAE estándar)
  β = 1.0
```

### 1. Reconstruction Loss

```python
# Solo para posiciones reales (no padding)
L_recon = 0
for cada participante i:
    seq_len_i = lengths[i]
    L_recon += MSE(
        recon[i, :seq_len_i, :],  # Solo hasta seq_len_i
        original[i, :seq_len_i, :]
    )
L_recon /= batch_size
```

**Importante:** Ignora posiciones paddeadas en la pérdida.

### 2. KL Divergence

```python
L_KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```

**α más bajo (0.0001 vs 0.001)** porque:
- Secuencias tienen más información
- Evita "posterior collapse" (μ=0, σ=1 trivial)

### 3. Classification Loss

```python
L_class = CrossEntropy(classifier(z), true_label)
```

## 📈 Ventajas sobre VAE Estándar

| Aspecto | VAE Estándar | LSTM-VAE |
|---------|--------------|----------|
| **Información preservada** | Solo promedio | Secuencia completa |
| **Variabilidad** | ❌ Se pierde | ✅ Capturada |
| **Heterogeneidad** | ❌ Colapsada | ✅ Representada |
| **Reconstrucción** | 1 vector agregado | n medidas |
| **Parámetros** | ~6,700 | ~50,000 |
| **Complejidad** | Baja | Alta |
| **Training time** | Rápido | Más lento |
| **Interpretabilidad** | Alta | Media |

## 🚀 Uso

### Entrenar LSTM-VAE

```bash
# Con flag --lstm
python scripts/train_autoencoder.py --lstm

# O equivalentemente
python scripts/train_autoencoder.py -l
```

### Entrenar VAE Estándar (mean pooling)

```bash
# Sin flags
python scripts/train_autoencoder.py
```

### Comparar Ambos

```bash
# Entrenar VAE estándar
python scripts/train_autoencoder.py
# → models/vae-*.ckpt

# Entrenar LSTM-VAE
python scripts/train_autoencoder.py --lstm
# → models/lstm_vae-*.ckpt
```

Luego comparar en Streamlit:
- Accuracy de clasificación
- Espacio latente (clusterización)
- Calidad de reconstrucción

## 🔬 Interpretación

### ¿Cuándo LSTM-VAE es mejor?

Si **val_acc(LSTM-VAE) > val_acc(VAE)**:
- ✅ La variabilidad intra-participante es informativa
- ✅ Patrones en secuencias distinguen CT vs ELA
- ✅ Heterogeneidad mitocondrial es relevante

### ¿Cuándo VAE estándar es suficiente?

Si **val_acc(VAE) ≈ val_acc(LSTM-VAE)**:
- El promedio captura la información relevante
- Variabilidad es ruido, no señal
- VAE estándar es más simple y rápido

## 📊 Hiperparámetros LSTM-VAE

```python
LSTMVariationalAutoencoder(
    input_dim=8,              # Features por medida
    hidden_dim=64,            # LSTM hidden dimension
    num_lstm_layers=2,        # Profundidad del LSTM
    latent_dim=16,            # Espacio latente (más grande)
    classifier_layers=[32, 16],  # Clasificador más profundo
    num_classes=2,            # CT vs ELA
    learning_rate=0.0005,     # LR adaptativo
    kl_weight=0.0001,         # KL bajo (evitar collapse)
    classification_weight=1.0,
    dropout_rate=0.3,         # Dropout alto (secuencias)
    bidirectional=True        # Lee en ambas direcciones
)
```

### Justificación

- **hidden_dim=64**: Balance capacidad/overfitting
- **num_layers=2**: Captura patrones jerárquicos
- **latent_dim=16**: Más grande que VAE (8) para capturar variabilidad
- **bidirectional=True**: Contexto completo de secuencia
- **kl_weight=0.0001**: Evita colapso a N(0,1) trivial
- **dropout=0.3**: Regularización fuerte (secuencias overfit fácil)

## 💡 Análisis Esperados

### 1. Variabilidad Intra-Participante

```python
# En el espacio latente, ¿la "incertidumbre" (σ) difiere entre grupos?
σ_CT = latent_logvar[labels==CT].mean()
σ_ELA = latent_logvar[labels==ELA].mean()

if σ_ELA > σ_CT:
    print("ELA tiene mayor heterogeneidad mitocondrial")
```

### 2. Reconstrucción de Secuencias

```python
# ¿El modelo captura patrones en la secuencia?
plot_sequence_reconstruction(
    original=seq_original,
    reconstructed=seq_recon
)
# Buscar: ¿mantiene tendencias, picos, variaciones?
```

### 3. Importancia de Cada Medida

```python
# Atención implícita del LSTM
# ¿Qué medidas contribuyen más al hidden state?
attention_weights = analyze_lstm_attention(model, sequences)
```

## 🎓 Referencias

- **LSTM**: Hochreiter & Schmidhuber (1997)
- **VAE**: Kingma & Welling (2013)
- **VAE-RNN**: Chung et al. (2015) - "A Recurrent Latent Variable Model for Sequential Data"
- **Bidirectional LSTM**: Schuster & Paliwal (1997)

## 📝 Notas Técnicas

### Gradient Clipping

```python
trainer = pl.Trainer(
    gradient_clip_val=1.0  # Evita exploding gradients en LSTM
)
```

Las LSTMs pueden sufrir de gradientes explosivos, especialmente con secuencias largas.

### Batch Size

```python
batch_size = 4  # Más pequeño que VAE estándar (16)
```

Secuencias usan más memoria. Ajustar según GPU disponible.

### Sequence Length Distribution

```
Participante | Medidas
-------------|--------
1            | 10
2            | 25  ← Máximo en batch
3            | 15
4            | 20

→ Padding: todas a 25
→ LSTM ignora padding con pack_padded_sequence
```

## 🔮 Extensiones Futuras

1. **Attention Mechanism**: Ponderación explícita de medidas importantes
2. **Variational RNN**: Modelo más sofisticado (cada timestep tiene latent)
3. **Hierarchical VAE**: Latent por medida + latent global por participante
4. **Conditional LSTM-VAE**: Condicionar en Age, Sex, etc.

---

**Resumen:** LSTM-VAE preserva toda la riqueza de información de las medidas individuales, capturando variabilidad intra-participante que puede ser crucial para discriminar entre CT y ELA. Es más complejo que el VAE estándar, pero potencialmente más poderoso si esa variabilidad es informativa.
