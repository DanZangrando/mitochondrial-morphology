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
            patience=10
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


class ParticipantSequenceDataset(Dataset):
    """
    Dataset for LSTM-VAE: preserves sequences of measurements per participant.
    
    Each sample is a sequence of variable length representing all measurements
    for one participant. This preserves intra-participant variability.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list,
        include_labels: bool = True
    ):
        """
        Initialize sequence dataset.
        
        Args:
            data: DataFrame with measurements
            feature_columns: List of feature column names
            include_labels: Whether to include group labels
        """
        self.feature_columns = feature_columns
        self.include_labels = include_labels
        
        # Group by participant
        self.sequences = []
        self.labels = []
        self.participants = []
        
        for participant_id, group_data in data.groupby('Participant'):
            # Extract feature sequence for this participant
            seq = group_data[feature_columns].values.astype(np.float32)
            self.sequences.append(seq)
            self.participants.append(participant_id)
            
            if include_labels:
                # Get label (should be same for all measurements)
                label = (group_data['Group'].iloc[0] == 'ELA')
                self.labels.append(int(label))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = torch.FloatTensor(self.sequences[idx])
        length = torch.LongTensor([len(seq)])
        
        if self.include_labels:
            label = torch.LongTensor([self.labels[idx]])[0]
            return seq, length, label
        
        return seq, length


def collate_sequences(batch):
    """
    Collate function for variable-length sequences.
    Pads sequences to max length in batch.
    
    Args:
        batch: List of (sequence, length, label) tuples
        
    Returns:
        padded_seqs: Padded sequences [batch, max_len, features]
        lengths: Original lengths [batch]
        labels: Labels [batch]
    """
    if len(batch[0]) == 3:
        sequences, lengths, labels = zip(*batch)
        has_labels = True
    else:
        sequences, lengths = zip(*batch)
        has_labels = False
    
    # Get max length in batch
    lengths = torch.cat(lengths)
    max_len = lengths.max().item()
    
    # Pad sequences
    batch_size = len(sequences)
    feature_dim = sequences[0].size(1)
    
    padded_seqs = torch.zeros(batch_size, max_len, feature_dim)
    for i, seq in enumerate(sequences):
        seq_len = seq.size(0)
        padded_seqs[i, :seq_len, :] = seq
    
    if has_labels:
        labels = torch.stack(labels)
        return padded_seqs, lengths, labels
    
    return padded_seqs, lengths
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
    use_sequences: bool = False,
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
        use_participant_aggregation: If True, aggregate by participant (for MitochondriaVAE)
        use_sequences: If True, use sequences (for LSTMVariationalAutoencoder)
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
    
    # Create datasets based on mode
    if use_sequences:
        # LSTM-VAE: use sequences
        train_dataset = ParticipantSequenceDataset(
            train_data,
            feature_columns,
            include_labels=True
        )
        val_dataset = ParticipantSequenceDataset(
            val_data,
            feature_columns,
            include_labels=True
        )
        
        # Use custom collate function for padding
        collate_fn = collate_sequences
    elif use_participant_aggregation:
        # Standard VAE with aggregation
        train_dataset = ParticipantDataset(
            train_data,
            feature_columns,
            aggregation=aggregation,
            include_labels=True
        )
        val_dataset = ParticipantDataset(
            val_data,
            feature_columns,
            aggregation=aggregation,
            include_labels=True
        )
        collate_fn = None
    else:
        # Individual measurements (original MeasurementDataset)
        train_dataset = MeasurementDataset(
            train_data,
            feature_columns,
            include_labels=True
        )
        val_dataset = MeasurementDataset(
            val_data,
            feature_columns,
            include_labels=True
        )
        collate_fn = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


class LSTMVariationalAutoencoder(pl.LightningModule):
    """
    LSTM-based Variational Autoencoder for sequential mitochondrial measurements.
    
    Captures intra-participant variability by processing sequences of measurements
    instead of aggregating them. Each participant has a variable number of measurements
    that are treated as a time series.
    
    Architecture:
    - LSTM Encoder: processes sequence of measurements → hidden states
    - Final hidden state → μ and log σ² (latent distribution parameters)
    - Reparameterization: z = μ + σ * ε
    - LSTM Decoder: expands latent z → reconstructs sequence
    - Classifier: z → group prediction (CT/ELA)
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        num_lstm_layers: int = 2,
        latent_dim: int = 16,
        classifier_layers: list = [32, 16],
        num_classes: int = 2,
        learning_rate: float = 0.0005,
        kl_weight: float = 0.0001,
        classification_weight: float = 1.0,
        dropout_rate: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM-VAE.
        
        Args:
            input_dim: Number of features per measurement (8 morphological metrics)
            hidden_dim: LSTM hidden dimension
            num_lstm_layers: Number of LSTM layers
            latent_dim: Dimensionality of latent space
            classifier_layers: Hidden layers for classifier
            num_classes: Number of classes (2 for CT/ELA)
            learning_rate: Learning rate
            kl_weight: Weight for KL divergence loss
            classification_weight: Weight for classification loss
            dropout_rate: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.classification_weight = classification_weight
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Latent space parameters from LSTM final hidden state
        lstm_output_dim = hidden_dim * self.num_directions
        self.fc_mu = nn.Linear(lstm_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(lstm_output_dim, latent_dim)
        
        # Decoder: latent → initial hidden state for LSTM
        self.latent_to_hidden = nn.Linear(
            latent_dim,
            num_lstm_layers * self.num_directions * hidden_dim
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,  # Each step gets the latent vector
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
            bidirectional=False  # Decoder is unidirectional
        )
        
        # Output projection
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Classifier head
        classifier_modules = []
        prev_dim = latent_dim
        for hidden_dim_cls in classifier_layers:
            classifier_modules.extend([
                nn.Linear(prev_dim, hidden_dim_cls),
                nn.BatchNorm1d(hidden_dim_cls),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim_cls
        classifier_modules.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_modules)
        
    def encode(self, x, lengths):
        """
        Encode sequence to latent distribution parameters.
        
        Args:
            x: Input sequences [batch, max_seq_len, input_dim]
            lengths: Actual sequence lengths [batch]
            
        Returns:
            mu: Mean of latent distribution [batch, latent_dim]
            logvar: Log variance of latent distribution [batch, latent_dim]
        """
        # Pack padded sequence for efficient processing
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Process through LSTM
        packed_output, (h_n, c_n) = self.encoder_lstm(packed_input)
        
        # Use final hidden state (from last layer)
        if self.bidirectional:
            # Concatenate forward and backward final states
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_final = h_n[-1]
        
        # Project to latent parameters
        mu = self.fc_mu(h_final)
        logvar = self.fc_logvar(h_final)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, seq_len):
        """
        Decode latent vector to sequence.
        
        Args:
            z: Latent vector [batch, latent_dim]
            seq_len: Target sequence length (max in batch)
            
        Returns:
            recon_x: Reconstructed sequence [batch, seq_len, input_dim]
        """
        batch_size = z.size(0)
        
        # Initialize decoder LSTM hidden state from latent z
        hidden_init = self.latent_to_hidden(z)
        hidden_init = hidden_init.view(
            batch_size,
            self.num_lstm_layers * self.num_directions,
            self.hidden_dim
        )
        
        # Separate h and c for LSTM (use same values)
        h_0 = hidden_init[:, :self.num_lstm_layers, :].contiguous().transpose(0, 1)
        c_0 = torch.zeros_like(h_0)
        
        # Create input: repeat latent vector for each time step
        decoder_input = z.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        lstm_out, _ = self.decoder_lstm(decoder_input, (h_0, c_0))
        
        # Project to output dimension
        recon_x = self.decoder_fc(lstm_out)
        
        return recon_x
    
    def forward(self, x, lengths):
        """
        Forward pass through LSTM-VAE.
        
        Args:
            x: Input sequences [batch, max_seq_len, input_dim]
            lengths: Actual sequence lengths [batch]
            
        Returns:
            recon_x: Reconstructed sequences
            mu: Latent mean
            logvar: Latent log variance
            class_logits: Classification logits
        """
        # Encode
        mu, logvar = self.encode(x, lengths)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        max_len = x.size(1)
        recon_x = self.decode(z, max_len)
        
        # Classify
        class_logits = self.classifier(z)
        
        return recon_x, mu, logvar, class_logits
    
    def get_latent(self, x, lengths):
        """Get latent representation (mu) for visualization."""
        mu, _ = self.encode(x, lengths)
        return mu
    
    def vae_loss(self, recon_x, x, mu, logvar, lengths, class_logits=None, labels=None):
        """
        VAE loss for sequences with variable lengths.
        
        Args:
            recon_x: Reconstructed sequences [batch, max_seq_len, input_dim]
            x: Original sequences [batch, max_seq_len, input_dim]
            mu: Latent mean [batch, latent_dim]
            logvar: Latent log variance [batch, latent_dim]
            lengths: Actual sequence lengths [batch]
            class_logits: Classification logits [batch, num_classes]
            labels: True labels [batch]
            
        Returns:
            Dictionary with losses
        """
        batch_size = x.size(0)
        
        # Reconstruction loss (MSE) - only for actual sequence positions
        recon_loss = 0
        for i in range(batch_size):
            seq_len = int(lengths[i].item())
            recon_loss += F.mse_loss(
                recon_x[i, :seq_len, :],
                x[i, :seq_len, :],
                reduction='sum'
            )
        recon_loss = recon_loss / batch_size
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # Total VAE loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        loss_dict = {
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'vae_loss': total_loss
        }
        
        # Classification loss
        if class_logits is not None and labels is not None:
            class_loss = F.cross_entropy(class_logits, labels)
            total_loss = total_loss + self.classification_weight * class_loss
            loss_dict['class_loss'] = class_loss
            
            # Accuracy
            preds = torch.argmax(class_logits, dim=1)
            acc = (preds == labels).float().mean()
            loss_dict['accuracy'] = acc
        
        loss_dict['total_loss'] = total_loss
        return loss_dict
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, lengths, labels = batch
        
        recon_x, mu, logvar, class_logits = self(x, lengths)
        loss_dict = self.vae_loss(recon_x, x, mu, logvar, lengths, class_logits, labels)
        
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
        x, lengths, labels = batch
        
        recon_x, mu, logvar, class_logits = self(x, lengths)
        loss_dict = self.vae_loss(recon_x, x, mu, logvar, lengths, class_logits, labels)
        
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
            patience=15
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
