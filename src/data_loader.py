"""
Data loader and preprocessing utilities for mitochondrial morphology data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import yaml


class MitochondriaDataLoader:
    """Load and preprocess mitochondrial morphology data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data loader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data = None
        self.scaler = StandardScaler()
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV."""
        try:
            data_path = self.config['data']['raw_data_path']
            self.data = pd.read_csv(data_path, encoding='utf-8')
        except Exception as e:
            # Try with explicit encoding if default fails
            try:
                self.data = pd.read_csv(data_path, encoding='latin-1')
            except:
                # Fallback to reading without config
                self.data = pd.read_csv('data/data.csv', encoding='utf-8')
        return self.data
    
    def get_feature_columns(self) -> List[str]:
        """Get list of numerical feature columns for analysis."""
        # Try to load from config file
        try:
            import json
            import os
            config_path = "config/selected_variables.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    selected_features = json.load(f)
                if selected_features and isinstance(selected_features, list):
                    return selected_features
        except Exception as e:
            print(f"Warning: Could not load selected variables config: {e}")

        # Default fallback
        features = [
            'PROM IsoVol', 'PROM Surface', 'PROM Length', 'PROM RoughSph',
            'SUMA IsoVol', 'SUMA Surface', 'SUMA Length', 'SUMA RoughSph'
        ]
        return features
    
    def prepare_features(self, standardize: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Prepare features for analysis.
        
        Args:
            standardize: Whether to standardize features
            
        Returns:
            Tuple of (scaled_features, original_dataframe)
        """
        if self.data is None:
            self.load_data()
        
        feature_cols = self.get_feature_columns()
        X = self.data[feature_cols].values
        
        if standardize:
            X_scaled = self.scaler.fit_transform(X)
            return X_scaled, self.data
        
        return X, self.data
    
    def get_groups(self) -> pd.Series:
        """Get group labels (CT/ELA)."""
        if self.data is None:
            self.load_data()
        return self.data['Group']
    
    def get_sex(self) -> pd.Series:
        """Get sex labels."""
        if self.data is None:
            self.load_data()
        return self.data['Sex']
    
    def get_summary_stats(self) -> pd.DataFrame:
        """Get summary statistics by group."""
        if self.data is None:
            self.load_data()
        
        feature_cols = self.get_feature_columns()
        summary = self.data.groupby('Group')[feature_cols].agg(['mean', 'std', 'count'])
        return summary
    
        ct_data = self.data[self.data['Group'] == 'CT']
        ela_data = self.data[self.data['Group'] == 'ELA']
        
        return ct_data, ela_data

    def get_sequences(self, feature_cols: List[str] = None) -> Tuple[any, any, any]:
        """
        Generate sequences for LSTM input.
        Returns:
            X: Padded sequences tensor (Batch, Max_Len, Features)
            y: Labels tensor (Batch)
            lengths: Sequence lengths tensor (Batch)
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence
        
        if self.data is None:
            self.load_data()
            
        if feature_cols is None:
            feature_cols = self.get_feature_columns()
            
        # Group by participant
        participants = self.data['Participant'].unique()
        sequences = []
        labels = []
        
        # Create label mapping
        label_map = {'CT': 0, 'ELA': 1}
        
        for p in participants:
            p_data = self.data[self.data['Participant'] == p]
            
            # Get features
            seq = p_data[feature_cols].values
            
            # Sort by the first feature (usually Volume) to give consistent order
            # This is important because mitochondria are a set, but LSTM expects sequence
            # Sorting makes the "sequence" invariant to permutation of the input file
            sort_idx = np.argsort(seq[:, 0])
            seq = seq[sort_idx]
            
            # Standardize (using global scaler if possible, but here we might need to be careful)
            # Ideally we standardize the whole dataset first. 
            # Let's assume the caller handles standardization or we do it here globally.
            # For now, let's just return raw/pre-standardized values if the user passed them.
            # Actually, let's standardize here to be safe using the class scaler.
            
            sequences.append(torch.tensor(seq, dtype=torch.float32))
            
            # Get label (assume all rows for participant have same group)
            group = p_data['Group'].iloc[0]
            labels.append(label_map.get(group, 0))
            
        # Pad sequences
        # batch_first=True -> (Batch, Max_Len, Features)
        X = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        y = torch.tensor(labels, dtype=torch.long)
        lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
        
        # Global standardization of the padded tensor (ignoring padding)
        # It's better to fit scaler on all data first.
        # Let's do a quick fit-transform on the flattened data to ensure valid scaling
        # But we need to mask padding.
        
        # Alternative: The user calls prepare_features first? 
        # Let's just fit the scaler on the concatenated sequences for now.
        all_data = torch.cat(sequences, dim=0).numpy()
        self.scaler.fit(all_data)
        
        # Transform
        X_numpy = X.numpy()
        B, L, F = X_numpy.shape
        # Reshape, transform, reshape back
        X_flat = X_numpy.reshape(-1, F)
        X_flat_scaled = self.scaler.transform(X_flat)
        X_scaled = torch.tensor(X_flat_scaled.reshape(B, L, F), dtype=torch.float32)
        
        # Re-apply padding (transform might have shifted 0 to something else)
        # We need to mask the padding positions again.
        mask = torch.arange(L)[None, :] < lengths[:, None]
        X_scaled[~mask] = 0.0
        
        return X_scaled, y, lengths
