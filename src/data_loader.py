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
    
    def split_by_group(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into CT and ELA groups."""
        if self.data is None:
            self.load_data()
        
        ct_data = self.data[self.data['Group'] == 'CT']
        ela_data = self.data[self.data['Group'] == 'ELA']
        
        return ct_data, ela_data
