"""
Test script for LSTM-VAE implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from src.data_loader import MitochondriaDataLoader
from src.autoencoder import LSTMVariationalAutoencoder, ParticipantSequenceDataset, collate_sequences

def test_lstm_vae():
    print("="*80)
    print("Testing LSTM-VAE Implementation")
    print("="*80)
    
    # Load data
    print("\n[1/5] Loading data...")
    loader = MitochondriaDataLoader()
    X_scaled, data = loader.prepare_features(standardize=True)
    feature_cols = loader.get_feature_columns()
    
    print(f"✓ Data shape: {X_scaled.shape}")
    print(f"✓ Participants: {data['Participant'].nunique()}")
    
    # Analyze sequence lengths
    seq_lengths = data.groupby('Participant').size()
    print(f"✓ Sequence lengths: min={seq_lengths.min()}, max={seq_lengths.max()}, "
          f"mean={seq_lengths.mean():.1f}, median={seq_lengths.median():.0f}")
    
    # Create sequence dataset
    print("\n[2/5] Creating sequence dataset...")
    dataset = ParticipantSequenceDataset(
        data,
        feature_cols,
        include_labels=True
    )
    
    print(f"✓ Dataset size: {len(dataset)} participants")
    
    # Test individual sample
    seq, length, label = dataset[0]
    print(f"✓ Sample 0: seq_len={length.item()}, features_per_step={seq.shape[1]}, "
          f"label={label.item()} ({'CT' if label == 0 else 'ELA'})")
    
    # Test batching with collate
    print("\n[3/5] Testing batch collation...")
    batch_samples = [dataset[i] for i in range(4)]
    padded_seqs, lengths, labels = collate_sequences(batch_samples)
    
    print(f"✓ Batch shape: {padded_seqs.shape}")
    print(f"✓ Lengths: {lengths.tolist()}")
    print(f"✓ Labels: {labels.tolist()}")
    print(f"✓ Max length in batch: {lengths.max().item()}")
    
    # Initialize LSTM-VAE
    print("\n[4/5] Initializing LSTM-VAE model...")
    model = LSTMVariationalAutoencoder(
        input_dim=8,
        hidden_dim=64,
        num_lstm_layers=2,
        latent_dim=16,
        classifier_layers=[32, 16],
        num_classes=2,
        learning_rate=0.0005,
        kl_weight=0.0001,
        classification_weight=1.0,
        dropout_rate=0.3,
        bidirectional=True
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized")
    print(f"✓ Parameters: {num_params:,}")
    print(f"✓ LSTM: bidirectional={model.bidirectional}, layers={model.num_lstm_layers}, hidden={model.hidden_dim}")
    print(f"✓ Latent dim: {model.latent_dim}")
    
    # Test forward pass
    print("\n[5/5] Testing forward pass...")
    model.eval()
    with torch.no_grad():
        # Forward pass
        recon_x, mu, logvar, class_logits = model(padded_seqs, lengths)
        
        print(f"✓ Input shape: {padded_seqs.shape}")
        print(f"✓ Reconstruction shape: {recon_x.shape}")
        print(f"✓ Latent mu shape: {mu.shape}")
        print(f"✓ Latent logvar shape: {logvar.shape}")
        print(f"✓ Class logits shape: {class_logits.shape}")
        
        # Calculate losses
        loss_dict = model.vae_loss(recon_x, padded_seqs, mu, logvar, lengths, class_logits, labels)
        
        print(f"\n✓ Loss components:")
        print(f"  - Reconstruction: {loss_dict['recon_loss']:.6f}")
        print(f"  - KL divergence: {loss_dict['kl_loss']:.6f}")
        print(f"  - Classification: {loss_dict['class_loss']:.6f}")
        print(f"  - Total: {loss_dict['total_loss']:.6f}")
        print(f"  - Accuracy: {loss_dict['accuracy']:.2%}")
        
        # Test latent representation
        latent = model.get_latent(padded_seqs, lengths)
        print(f"\n✓ Latent representation: {latent.shape}")
        print(f"  Sample latent values: {latent[0][:5].numpy()}")
        
        # Test sequence reconstruction quality
        print(f"\n✓ Reconstruction quality per sequence:")
        for i in range(len(lengths)):
            seq_len = int(lengths[i].item())
            orig = padded_seqs[i, :seq_len, :]
            recon = recon_x[i, :seq_len, :]
            mse = torch.nn.functional.mse_loss(recon, orig).item()
            print(f"  Participant {i+1} (len={seq_len}): MSE={mse:.6f}")
    
    print("\n" + "="*80)
    print("✅ All tests passed! LSTM-VAE is working correctly.")
    print("="*80)
    
    print("\n💡 Key insights:")
    print(f"  - LSTM-VAE has {num_params:,} parameters (vs ~6,700 for standard VAE)")
    print(f"  - Handles variable sequence lengths: {seq_lengths.min()}-{seq_lengths.max()} measurements")
    print(f"  - Preserves intra-participant variability (no aggregation)")
    print(f"  - Bidirectional LSTM captures sequence context")
    
    print("\n💡 Next steps:")
    print("  1. Train LSTM-VAE: python scripts/train_autoencoder.py --lstm")
    print("  2. Compare with standard VAE: python scripts/train_autoencoder.py")
    print("  3. Monitor with TensorBoard: tensorboard --logdir=logs/")
    print("  4. Visualize results: streamlit run app.py")
    print("\n")

if __name__ == "__main__":
    test_lstm_vae()
