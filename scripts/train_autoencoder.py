"""
Training script for the autoencoder model.
Run this script to train the autoencoder on mitochondrial data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

from src.data_loader import MitochondriaDataLoader
from src.autoencoder import MitochondriaAutoencoder, prepare_dataloaders


def train_autoencoder(config_path: str = "config/config.yaml"):
    """
    Train autoencoder model.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Loading data...")
    # Load and prepare data
    data_loader = MitochondriaDataLoader(config_path)
    X_scaled, data = data_loader.prepare_features(standardize=True)
    
    print(f"Data shape: {X_scaled.shape}")
    print(f"Features: {data_loader.get_feature_columns()}")
    
    # Split data
    X_train, X_val = train_test_split(
        X_scaled,
        test_size=0.2,
        random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Prepare dataloaders
    batch_size = config['autoencoder']['training']['batch_size']
    train_loader, val_loader = prepare_dataloaders(X_train, X_val, batch_size)
    
    # Initialize model
    ae_config = config['autoencoder']['architecture']
    training_config = config['autoencoder']['training']
    
    model = MitochondriaAutoencoder(
        input_dim=ae_config['input_dim'],
        encoder_layers=ae_config['encoder_layers'],
        latent_dim=ae_config['latent_dim'],
        decoder_layers=ae_config['decoder_layers'],
        learning_rate=training_config['learning_rate']
    )
    
    print(f"\nModel architecture:")
    print(f"Input: {ae_config['input_dim']} -> Encoder: {ae_config['encoder_layers']} -> "
          f"Latent: {ae_config['latent_dim']} -> Decoder: {ae_config['decoder_layers']} -> "
          f"Output: {ae_config['input_dim']}")
    
    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=training_config['early_stopping_patience'],
        mode='min',
        verbose=True
    )
    
    checkpoint = ModelCheckpoint(
        dirpath='models/',
        filename='autoencoder-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True
    )
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=config['autoencoder']['logging']['log_dir'],
        name=config['autoencoder']['logging']['experiment_name']
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=training_config['max_epochs'],
        callbacks=[early_stopping, checkpoint],
        logger=logger,
        accelerator='auto',
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    
    print(f"\nStarting training for max {training_config['max_epochs']} epochs...")
    print("Monitor training with: tensorboard --logdir=logs/")
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    print("\n✓ Training complete!")
    print(f"Best model saved to: {checkpoint.best_model_path}")
    print(f"Best validation loss: {checkpoint.best_model_score:.4f}")
    
    return checkpoint.best_model_path


if __name__ == "__main__":
    best_model_path = train_autoencoder()
    print(f"\nModel ready for inference: {best_model_path}")
