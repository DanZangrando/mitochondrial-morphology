"""
Training script for LSTM Classifier.

Simple supervised classification: LSTM encoder → Classifier → CT/ELA prediction
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
import json

from src.classifier import LSTMClassifier, prepare_dataloaders
from src.data_loader import MitochondriaDataLoader


def train_classifier(
    max_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    val_split: float = 0.2,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 0.001,
    checkpoint_monitor: str = 'val_acc',
    checkpoint_mode: str = 'max',
    checkpoint_save_top_k: int = 3,
    random_state: int = 42
) -> str:
    """
    Train LSTM Classifier for CT vs ELA classification.
    
    Args:
        max_epochs: Maximum number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        hidden_dim: Hidden dimension for LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        val_split: Validation split ratio
        early_stopping_patience: Early stopping patience (epochs)
        early_stopping_min_delta: Minimum change to qualify as improvement
        checkpoint_monitor: Metric to monitor for checkpointing
        checkpoint_mode: 'min' or 'max' for checkpoint metric
        checkpoint_save_top_k: Number of best models to save
        random_state: Random seed
    
    Returns:
        Path to best model checkpoint
    """
    
    print("=" * 80)
    print("🚀 LSTM CLASSIFIER TRAINING")
    print("=" * 80)
    
    # Set random seed
    pl.seed_everything(random_state)
    
    # Load data
    print("\n📊 Loading data...")
    loader = MitochondriaDataLoader()
    X_scaled, data = loader.prepare_features(standardize=True)
    
    print(f"✓ Loaded {len(data)} samples from {data['Participant'].nunique()} participants")
    print(f"  - CT: {(data['Group'] == 'CT').sum()} samples, {(data.groupby('Participant')['Group'].first() == 'CT').sum()} participants")
    print(f"  - ELA: {(data['Group'] == 'ELA').sum()} samples, {(data.groupby('Participant')['Group'].first() == 'ELA').sum()} participants")
    print(f"  - Features: {X_scaled.shape[1]}")
    
    # Prepare dataloaders
    print(f"\n🔀 Preparing dataloaders (val_split={val_split})...")
    train_loader, val_loader = prepare_dataloaders(
        X_scaled, data,
        batch_size=batch_size,
        val_split=val_split,
        random_state=random_state
    )
    
    # Get participant IDs for reference
    participants = data['Participant'].unique()
    train_participants, val_participants = train_test_split(
        participants,
        test_size=val_split,
        random_state=random_state,
        stratify=[data[data['Participant'] == p]['Group'].iloc[0] for p in participants]
    )
    
    print(f"✓ Train: {len(train_loader.dataset)} participants - IDs: {sorted(train_participants.tolist())}")
    print(f"✓ Val: {len(val_loader.dataset)} participants - IDs: {sorted(val_participants.tolist())}")
    
    # Create model
    print("\n🏗️  Creating LSTM Classifier...")
    model = LSTMClassifier(
        input_dim=X_scaled.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate
    )
    
    # Save val_split in model hparams for later evaluation
    model.save_hyperparameters({
        'val_split': val_split,
        'random_state': random_state,
        'training_mode': 'simple',
        'train_participants': train_participants.tolist(),
        'val_participants': val_participants.tolist()
    })
    
    print(f"✓ Model created")
    print(f"  - Input dim: {X_scaled.shape[1]}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Num layers: {num_layers}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Val split: {val_split}")
    
    # Setup callbacks
    print("\n⚙️  Setting up callbacks...")
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath='models',
        filename='lstm_classifier-{epoch:02d}-{val_acc:.4f}',
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
        save_top_k=checkpoint_save_top_k,
        verbose=True
    )
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=early_stopping_min_delta,
        patience=early_stopping_patience,
        verbose=True,
        mode='min'
    )
    
    callbacks = [checkpoint_callback, early_stop_callback]
    
    print(f"✓ ModelCheckpoint: monitor={checkpoint_monitor}, mode={checkpoint_mode}, save_top_k={checkpoint_save_top_k}")
    print(f"✓ EarlyStopping: monitor=val_loss, patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    
    # Setup logger
    logger = TensorBoardLogger('logs', name='lstm_classifier')
    print(f"✓ TensorBoard logger: logs/lstm_classifier")
    
    # Create trainer
    print(f"\n🎯 Creating trainer (max_epochs={max_epochs})...")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator='auto',
        devices=1,
        log_every_n_steps=5,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train
    print("\n" + "=" * 80)
    print("🏋️  STARTING TRAINING")
    print("=" * 80 + "\n")
    
    trainer.fit(model, train_loader, val_loader)
    
    # Results
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED")
    print("=" * 80)
    
    best_model_path = checkpoint_callback.best_model_path
    print(f"\n📁 Best model saved to: {best_model_path}")
    print(f"🎯 Best {checkpoint_monitor}: {checkpoint_callback.best_model_score:.4f}")
    
    # Save training metadata
    metadata = {
        'best_model_path': best_model_path,
        'best_score': float(checkpoint_callback.best_model_score),
        'val_split': val_split,
        'random_state': random_state,
        'train_participants': sorted(train_participants.tolist()),
        'val_participants': sorted(val_participants.tolist()),
        'hyperparameters': {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'max_epochs': max_epochs
        }
    }
    
    metadata_path = best_model_path.replace('.ckpt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Training metadata saved to: {metadata_path}")
    
    print("\n💡 To view training metrics:")
    print("   tensorboard --logdir logs/lstm_classifier")
    
    return best_model_path


if __name__ == "__main__":
    import argparse
    


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LSTM Classifier')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    
    args = parser.parse_args()
    
    best_model = train_classifier(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        val_split=args.val_split
    )
    print(f"\n✅ Training complete! Best model: {best_model}")
