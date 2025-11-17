"""
Quick test script to verify VAE implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from src.data_loader import MitochondriaDataLoader
from src.autoencoder import MitochondriaVAE, ParticipantDataset

def test_vae():
    print("="*80)
    print("Testing VAE Implementation")
    print("="*80)
    
    # Load data
    print("\n[1/4] Loading data...")
    loader = MitochondriaDataLoader()
    X_scaled, data = loader.prepare_features(standardize=True)
    feature_cols = loader.get_feature_columns()
    
    print(f"✓ Data shape: {X_scaled.shape}")
    print(f"✓ Participants: {data['Participant'].nunique()}")
    
    # Create participant dataset
    print("\n[2/4] Creating participant dataset...")
    dataset = ParticipantDataset(
        data,
        feature_cols,
        aggregation='mean',
        include_labels=True
    )
    
    print(f"✓ Dataset size: {len(dataset)} participants")
    
    # Test sample
    features, label = dataset[0]
    print(f"✓ Sample shape: {features.shape}")
    print(f"✓ Label: {label.item()} ({'CT' if label == 0 else 'ELA'})")
    
    # Initialize model
    print("\n[3/4] Initializing VAE model...")
    model = MitochondriaVAE(
        input_dim=8,
        encoder_layers=[64, 32],
        latent_dim=8,
        decoder_layers=[32, 64],
        classifier_layers=[16],
        num_classes=2,
        learning_rate=0.0005,
        kl_weight=0.001,
        classification_weight=1.0,
        dropout_rate=0.2
    )
    
    print(f"✓ Model initialized")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\n[4/4] Testing forward pass...")
    model.eval()
    with torch.no_grad():
        features_batch = features.unsqueeze(0)  # Add batch dimension
        label_batch = label.unsqueeze(0)
        
        # Forward pass
        recon_x, mu, logvar, class_logits = model(features_batch)
        
        print(f"✓ Input shape: {features_batch.shape}")
        print(f"✓ Reconstruction shape: {recon_x.shape}")
        print(f"✓ Latent mu shape: {mu.shape}")
        print(f"✓ Latent logvar shape: {logvar.shape}")
        print(f"✓ Class logits shape: {class_logits.shape}")
        
        # Calculate losses
        loss_dict = model.vae_loss(recon_x, features_batch, mu, logvar, class_logits, label_batch)
        
        print(f"\n✓ Loss components:")
        print(f"  - Reconstruction: {loss_dict['recon_loss']:.6f}")
        print(f"  - KL divergence: {loss_dict['kl_loss']:.6f}")
        print(f"  - Classification: {loss_dict['class_loss']:.6f}")
        print(f"  - Total: {loss_dict['total_loss']:.6f}")
        print(f"  - Accuracy: {loss_dict['accuracy']:.2%}")
        
        # Test latent representation
        latent = model.get_latent(features_batch)
        print(f"\n✓ Latent representation: {latent.shape}")
        print(f"  Sample latent values: {latent[0][:3].numpy()}")
    
    print("\n" + "="*80)
    print("✅ All tests passed! VAE is working correctly.")
    print("="*80)
    
    print("\n💡 Next steps:")
    print("  1. Train the model: python scripts/train_autoencoder.py")
    print("  2. Monitor with TensorBoard: tensorboard --logdir=logs/")
    print("  3. Visualize results: streamlit run app.py")
    print("\n")

if __name__ == "__main__":
    test_vae()
