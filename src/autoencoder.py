"""
Autoencoder model for mitochondrial morphology analysis using PyTorch Lightning.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
import numpy as np


class MitochondriaAutoencoder(pl.LightningModule):
    """PyTorch Lightning Autoencoder for dimensionality reduction."""
    
    def __init__(
        self,
        input_dim: int = 8,
        encoder_layers: list = [16, 8],
        latent_dim: int = 3,
        decoder_layers: list = [8, 16],
        learning_rate: float = 0.001
    ):
        """
        Initialize autoencoder.
        
        Args:
            input_dim: Number of input features
            encoder_layers: List of hidden layer sizes for encoder
            latent_dim: Size of latent space
            decoder_layers: List of hidden layer sizes for decoder
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # Build encoder
        encoder_arch = [input_dim] + encoder_layers + [latent_dim]
        encoder_modules = []
        for i in range(len(encoder_arch) - 1):
            encoder_modules.append(nn.Linear(encoder_arch[i], encoder_arch[i+1]))
            if i < len(encoder_arch) - 2:  # No activation on latent layer
                encoder_modules.append(nn.ReLU())
                encoder_modules.append(nn.BatchNorm1d(encoder_arch[i+1]))
        
        self.encoder = nn.Sequential(*encoder_modules)
        
        # Build decoder
        decoder_arch = [latent_dim] + decoder_layers + [input_dim]
        decoder_modules = []
        for i in range(len(decoder_arch) - 1):
            decoder_modules.append(nn.Linear(decoder_arch[i], decoder_arch[i+1]))
            if i < len(decoder_arch) - 2:
                decoder_modules.append(nn.ReLU())
                decoder_modules.append(nn.BatchNorm1d(decoder_arch[i+1]))
        
        self.decoder = nn.Sequential(*decoder_modules)
        
    def forward(self, x):
        """Forward pass through autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Encode input to latent space."""
        return self.encoder(x)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x = batch[0]
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x = batch[0]
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


def prepare_dataloaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare PyTorch DataLoaders.
    
    Args:
        X_train: Training features
        X_val: Validation features
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader
