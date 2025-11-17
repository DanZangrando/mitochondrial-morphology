"""
Training script for the Variational Autoencoder (VAE) with classification.
Run this script to train the VAE on mitochondrial data with group classification.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.data_loader import MitochondriaDataLoader
from src.autoencoder import MitochondriaVAE, prepare_dataloaders


def train_vae(config_path: str = "config/config.yaml"):
    """
    Train VAE model with classification head.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*80)
    print("VAE Training - Mitochondrial Morphology Analysis")
    print("="*80)
    
    print("\n[1/5] Loading data...")
    # Load and prepare data
    data_loader = MitochondriaDataLoader(config_path)
    X_scaled, data = data_loader.prepare_features(standardize=True)
    feature_cols = data_loader.get_feature_columns()
    
    print(f"✓ Data loaded: {X_scaled.shape}")
    print(f"✓ Features: {feature_cols}")
    print(f"✓ Participants: {data['Participant'].nunique()}")
    print(f"✓ Groups: {data['Group'].value_counts().to_dict()}")
    
    print("\n[2/5] Preparing dataloaders...")
    # Prepare dataloaders with participant-level aggregation
    batch_size = config['autoencoder']['training']['batch_size']
    
    train_loader, val_loader = prepare_dataloaders(
        data=data,
        feature_columns=feature_cols,
        batch_size=batch_size,
        train_split=0.8,
        use_participant_aggregation=True,  # Aggregate by participant
        aggregation='mean',  # Use mean of measurements
        random_state=42
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Batch size: {batch_size}")
    
    print("\n[3/5] Initializing VAE model...")
    # Initialize model
    ae_config = config['autoencoder']['architecture']
    training_config = config['autoencoder']['training']
    
    model = MitochondriaVAE(
        input_dim=ae_config['input_dim'],
        encoder_layers=[64, 32],  # Deeper encoder
        latent_dim=ae_config['latent_dim'],
        decoder_layers=[32, 64],  # Symmetric decoder
        classifier_layers=[16],  # Classification head
        num_classes=2,  # CT vs ELA
        learning_rate=training_config['learning_rate'],
        kl_weight=0.001,  # KL divergence weight
        classification_weight=1.0,  # Classification loss weight
        dropout_rate=0.2
    )
    
    print(f"✓ Architecture:")
    print(f"  Input: {ae_config['input_dim']} features")
    print(f"  Encoder: [64, 32] → Latent: {ae_config['latent_dim']}D (μ, σ)")
    print(f"  Decoder: [32, 64] → Output: {ae_config['input_dim']} features")
    print(f"  Classifier: [16] → 2 classes (CT/ELA)")
    print(f"✓ VAE parameters:")
    print(f"  KL weight: 0.001")
    print(f"  Classification weight: 1.0")
    print(f"  Dropout: 0.2")
    
    print("\n[4/5] Setting up training...")
    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=training_config['early_stopping_patience'],
        mode='min',
        verbose=True
    )
    
    checkpoint = ModelCheckpoint(
        dirpath='models/',
        filename='vae-{epoch:02d}-{val_loss:.4f}-{val_acc:.3f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,  # Keep top 3 models
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=config['autoencoder']['logging']['log_dir'],
        name='vae_classifier',
        version=None
    )
    
    print(f"✓ Early stopping patience: {training_config['early_stopping_patience']}")
    print(f"✓ Model checkpoints: models/")
    print(f"✓ TensorBoard logs: {config['autoencoder']['logging']['log_dir']}/vae_classifier")
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=training_config['max_epochs'],
        callbacks=[early_stopping, checkpoint, lr_monitor],
        logger=logger,
        accelerator='auto',
        devices=1,
        log_every_n_steps=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0  # Gradient clipping for stability
    )
    
    print(f"\n[5/5] Starting training...")
    print(f"✓ Max epochs: {training_config['max_epochs']}")
    print(f"✓ Learning rate: {training_config['learning_rate']}")
    print(f"\n💡 Monitor training in real-time:")
    print(f"   tensorboard --logdir={config['autoencoder']['logging']['log_dir']}")
    print("\n" + "="*80)
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    print("\n" + "="*80)
    print("✓ Training complete!")
    print("="*80)
    print(f"✓ Best model: {checkpoint.best_model_path}")
    print(f"✓ Best val_loss: {checkpoint.best_model_score:.4f}")
    print(f"\n💡 To view results:")
    print(f"   1. Launch Streamlit: streamlit run app.py")
    print(f"   2. Go to: 🤖 Autoencoder page")
    print(f"   3. Load model: {os.path.basename(checkpoint.best_model_path)}")
    print("="*80)
    
    return checkpoint.best_model_path


# Alias for backward compatibility
train_autoencoder = train_vae


if __name__ == "__main__":
    best_model_path = train_vae()
    print(f"\n🎉 Model ready for inference: {best_model_path}")
