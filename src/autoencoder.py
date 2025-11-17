"""
Variational Autoencoder (VAE) with classification for mitochondrial morphology analysis.
Inspired by Nature Methods paper on quantized VAEs for biological data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd


class MitochondriaVAE(pl.LightningModule):
    """
    Variational Autoencoder with group classification head.
    Handles multiple measurements per participant through aggregation.
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        encoder_layers: list = [64, 32],
        latent_dim: int = 8,
        decoder_layers: list = [32, 64],
        classifier_layers: list = [16],
        num_classes: int = 2,
        learning_rate: float = 0.001,
        kl_weight: float = 0.001,
        classification_weight: float = 1.0,
        dropout_rate: float = 0.2
    ):
        """
        Initialize VAE with classifier.
        
        Args:
            input_dim: Number of input features (8 morphological metrics)
            encoder_layers: Hidden layers for encoder
            latent_dim: Dimensionality of latent space
            decoder_layers: Hidden layers for decoder
            classifier_layers: Hidden layers for classifier
            num_classes: Number of classes (2 for CT/ELA)
            learning_rate: Learning rate
            kl_weight: Weight for KL divergence loss
            classification_weight: Weight for classification loss
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.classification_weight = classification_weight
        
        # Encoder: input -> hidden -> mu & logvar
        encoder_modules = []
        prev_dim = input_dim
        for hidden_dim in encoder_layers:
            encoder_modules.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_modules)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder: latent -> hidden -> output
        decoder_modules = []
        prev_dim = latent_dim
        for hidden_dim in decoder_layers:
            decoder_modules.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        decoder_modules.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_modules)
        
        # Classifier head: latent -> classes
        classifier_modules = []
        prev_dim = latent_dim
        for hidden_dim in classifier_layers:
            classifier_modules.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        classifier_modules.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_modules)
        
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass: encode -> reparameterize -> decode + classify.
        
        Returns:
            recon_x: Reconstructed input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            class_logits: Classification logits
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        class_logits = self.classifier(z)
        return recon_x, mu, logvar, class_logits
    
    def get_latent(self, x):
        """Get latent representation (mu) for visualization."""
        mu, _ = self.encode(x)
        return mu
    
    def vae_loss(self, recon_x, x, mu, logvar, class_logits=None, labels=None):
        """
        VAE loss: reconstruction + KL divergence + optional classification.
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            class_logits: Classification logits (optional)
            labels: True labels (optional)
            
        Returns:
            Dictionary with total loss and components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL divergence loss
        # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # Total VAE loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        loss_dict = {
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'vae_loss': total_loss
        }
        
        # Add classification loss if labels provided
        if class_logits is not None and labels is not None:
            class_loss = F.cross_entropy(class_logits, labels)
            total_loss = total_loss + self.classification_weight * class_loss
            loss_dict['class_loss'] = class_loss
            
            # Calculate accuracy
            preds = torch.argmax(class_logits, dim=1)
            acc = (preds == labels).float().mean()
            loss_dict['accuracy'] = acc
        
        loss_dict['total_loss'] = total_loss
        return loss_dict
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        if len(batch) == 2:
            x, labels = batch
        else:
            x = batch[0]
            labels = None
            
        recon_x, mu, logvar, class_logits = self(x)
        loss_dict = self.vae_loss(recon_x, x, mu, logvar, class_logits, labels)
        
        # Log metrics
        self.log('train_loss', loss_dict['total_loss'], prog_bar=True)
        self.log('train_recon', loss_dict['recon_loss'])
        self.log('train_kl', loss_dict['kl_loss'])
        if 'class_loss' in loss_dict:
            self.log('train_class_loss', loss_dict['class_loss'])
            self.log('train_acc', loss_dict['accuracy'], prog_bar=True)
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if len(batch) == 2:
            x, labels = batch
        else:
            x = batch[0]
            labels = None
            
        recon_x, mu, logvar, class_logits = self(x)
        loss_dict = self.vae_loss(recon_x, x, mu, logvar, class_logits, labels)
        
        # Log metrics
        self.log('val_loss', loss_dict['total_loss'], prog_bar=True)
        self.log('val_recon', loss_dict['recon_loss'])
        self.log('val_kl', loss_dict['kl_loss'])
        if 'class_loss' in loss_dict:
            self.log('val_class_loss', loss_dict['class_loss'])
            self.log('val_acc', loss_dict['accuracy'], prog_bar=True)
        
        return loss_dict['total_loss']
    
    def configure_optimizers(self):
        """Configure optimizer with learning rate scheduling."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


class ParticipantDataset(Dataset):
    """
    Dataset that handles multiple measurements per participant.
    Each batch contains aggregated features per participant.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list,
        aggregation: str = 'mean',
        include_labels: bool = True
    ):
        """
        Initialize dataset with participant-level aggregation.
        
        Args:
            data: DataFrame with measurements
            feature_columns: List of feature column names
            aggregation: How to aggregate measurements ('mean', 'median', 'max')
            include_labels: Whether to include group labels
        """
        self.feature_columns = feature_columns
        self.aggregation = aggregation
        self.include_labels = include_labels
        
        # Group by participant and aggregate
        agg_dict = {col: aggregation for col in feature_columns}
        if include_labels:
            # Assume Group is consistent per participant
            agg_dict['Group'] = 'first'
        
        self.data_agg = data.groupby('Participant').agg(agg_dict).reset_index()
        
        # Prepare features
        self.features = self.data_agg[feature_columns].values.astype(np.float32)
        
        # Prepare labels if needed
        if include_labels:
            # Map CT=0, ELA=1
            self.labels = (self.data_agg['Group'] == 'ELA').astype(np.int64).values
        else:
            self.labels = None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        if self.include_labels:
            label = torch.LongTensor([self.labels[idx]])[0]
            return features, label
        return features,


class MeasurementDataset(Dataset):
    """
    Dataset that uses individual measurements (not aggregated).
    Useful for training on all data points.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list,
        include_labels: bool = True
    ):
        """
        Initialize dataset with individual measurements.
        
        Args:
            data: DataFrame with measurements
            feature_columns: List of feature column names
            include_labels: Whether to include group labels
        """
        self.features = data[feature_columns].values.astype(np.float32)
        
        if include_labels:
            # Map CT=0, ELA=1
            self.labels = (data['Group'] == 'ELA').astype(np.int64).values
        else:
            self.labels = None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        if self.labels is not None:
            label = torch.LongTensor([self.labels[idx]])[0]
            return features, label
        return features,


def prepare_dataloaders(
    data: pd.DataFrame,
    feature_columns: list,
    batch_size: int = 32,
    train_split: float = 0.8,
    use_participant_aggregation: bool = True,
    aggregation: str = 'mean',
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare PyTorch DataLoaders with train/val split.
    
    Args:
        data: Complete DataFrame
        feature_columns: List of feature column names
        batch_size: Batch size
        train_split: Proportion of data for training
        use_participant_aggregation: If True, aggregate by participant
        aggregation: Aggregation method ('mean', 'median', 'max')
        random_state: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Split by participants to avoid data leakage
    participants = data['Participant'].unique()
    np.random.seed(random_state)
    np.random.shuffle(participants)
    
    split_idx = int(len(participants) * train_split)
    train_participants = participants[:split_idx]
    val_participants = participants[split_idx:]
    
    train_data = data[data['Participant'].isin(train_participants)]
    val_data = data[data['Participant'].isin(val_participants)]
    
    # Create datasets
    DatasetClass = ParticipantDataset if use_participant_aggregation else MeasurementDataset
    
    if use_participant_aggregation:
        train_dataset = DatasetClass(
            train_data,
            feature_columns,
            aggregation=aggregation,
            include_labels=True
        )
        val_dataset = DatasetClass(
            val_data,
            feature_columns,
            aggregation=aggregation,
            include_labels=True
        )
    else:
        train_dataset = DatasetClass(
            train_data,
            feature_columns,
            include_labels=True
        )
        val_dataset = DatasetClass(
            val_data,
            feature_columns,
            include_labels=True
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader
