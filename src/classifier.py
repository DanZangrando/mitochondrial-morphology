"""
Simple LSTM Classifier for CT vs ELA classification based on mitochondrial morphology.

Architecture:
- Bidirectional LSTM encoder processes variable-length sequences
- Fully connected layers for classification
- Binary classification (CT vs ELA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class SequenceDataset(Dataset):
    """Dataset for variable-length sequences with participant grouping"""
    
    def __init__(self, sequences: list, labels: list, participants: list):
        """
        Args:
            sequences: List of numpy arrays, each of shape (seq_len, n_features)
            labels: List of labels (0=CT, 1=ELA)
            participants: List of participant IDs
        """
        self.sequences = sequences
        self.labels = labels
        self.participants = participants
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': torch.FloatTensor(self.sequences[idx]),
            'label': torch.LongTensor([self.labels[idx]]),
            'participant': self.participants[idx]
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    sequences = [item['sequence'] for item in batch]
    labels = torch.cat([item['label'] for item in batch])
    participants = [item['participant'] for item in batch]
    
    # Get lengths
    lengths = torch.LongTensor([len(seq) for seq in sequences])
    
    # Pad sequences
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    
    return padded_sequences, labels, lengths, participants


class LSTMClassifier(pl.LightningModule):
    """
    Simple LSTM-based binary classifier for CT vs ELA.
    
    Architecture:
        Input: (batch, seq_len, n_features) - variable seq_len per sample
        ↓
        Bidirectional LSTM encoder
        ↓
        Take last hidden state
        ↓
        Fully connected layers
        ↓
        Output: (batch, 2) - logits for CT/ELA
    """
    
    def __init__(
        self,
        input_dim: int = 8,  # 8 morphometric features
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
    ):
        """
        Args:
            input_dim: Number of input features (morphometric variables)
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        
        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classifier head
        # Input: hidden_dim * 2 (bidirectional)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # CT vs ELA
        )
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input sequences (batch, max_seq_len, input_dim)
            lengths: Actual lengths of sequences (batch,)
        
        Returns:
            logits: Classification logits (batch, 2)
        """
        batch_size = x.size(0)
        
        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM encoding
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # hidden shape: (num_layers * 2, batch, hidden_dim) for bidirectional
        # Take last layer's forward and backward hidden states
        forward_hidden = hidden[-2, :, :]  # (batch, hidden_dim)
        backward_hidden = hidden[-1, :, :]  # (batch, hidden_dim)
        
        # Concatenate forward and backward
        encoding = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch, hidden_dim * 2)
        
        # Classification
        logits = self.classifier(encoding)  # (batch, 2)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        x, labels, lengths, _ = batch
        
        # Forward pass
        logits = self(x, lengths)
        
        # Loss
        loss = F.cross_entropy(logits, labels)
        
        # Accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, labels, lengths, _ = batch
        
        # Forward pass
        logits = self(x, lengths)
        
        # Loss
        loss = F.cross_entropy(logits, labels)
        
        # Accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


def prepare_dataloaders(
    X_scaled: np.ndarray,
    data: pd.DataFrame,
    batch_size: int = 16,
    val_split: float = 0.2,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare train and validation dataloaders
    
    Args:
        X_scaled: Scaled features (n_samples, n_features)
        data: Original dataframe with Participant and Group columns
        batch_size: Batch size
        val_split: Validation split ratio
        random_state: Random seed
    
    Returns:
        train_loader, val_loader
    """
    # Group by participant
    participants = data['Participant'].unique()
    
    # Split participants (not individual samples)
    train_participants, val_participants = train_test_split(
        participants, test_size=val_split, random_state=random_state,
        stratify=[data[data['Participant'] == p]['Group'].iloc[0] for p in participants]
    )
    
    # Prepare sequences
    def prepare_sequences(participant_list):
        sequences = []
        labels = []
        participant_ids = []
        
        for participant in participant_list:
            participant_data = data[data['Participant'] == participant]
            sequence = X_scaled[participant_data.index]
            label = 0 if participant_data['Group'].iloc[0] == 'CT' else 1
            
            sequences.append(sequence)
            labels.append(label)
            participant_ids.append(participant)
        
        return sequences, labels, participant_ids
    
    train_sequences, train_labels, train_ids = prepare_sequences(train_participants)
    val_sequences, val_labels, val_ids = prepare_sequences(val_participants)
    
    # Create datasets
    train_dataset = SequenceDataset(train_sequences, train_labels, train_ids)
    val_dataset = SequenceDataset(val_sequences, val_labels, val_ids)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    return train_loader, val_loader
