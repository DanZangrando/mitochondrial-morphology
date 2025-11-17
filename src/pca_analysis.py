"""
PCA analysis module for dimensionality reduction.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Tuple, Dict
import plotly.express as px
import plotly.graph_objects as go


class MitochondriaPCA:
    """Perform PCA analysis on mitochondrial data."""
    
    def __init__(self, n_components: int = 3):
        """
        Initialize PCA analyzer.
        
        Args:
            n_components: Number of principal components to compute
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.components = None
        self.explained_variance = None
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data.
        
        Args:
            X: Feature matrix (scaled)
            
        Returns:
            Transformed data in PC space
        """
        self.components = self.pca.fit_transform(X)
        self.explained_variance = self.pca.explained_variance_ratio_
        return self.components
    
    def get_explained_variance(self) -> np.ndarray:
        """Get explained variance ratio for each component."""
        return self.explained_variance
    
    def get_loadings(self, feature_names: list) -> pd.DataFrame:
        """
        Get PCA loadings (feature contributions).
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with loadings for each PC
        """
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.n_components)],
            index=feature_names
        )
        return loadings
    
    def plot_explained_variance(self) -> go.Figure:
        """Create scree plot of explained variance."""
        fig = go.Figure()
        
        # Bar plot
        fig.add_trace(go.Bar(
            x=[f'PC{i+1}' for i in range(len(self.explained_variance))],
            y=self.explained_variance * 100,
            name='Individual',
            marker_color='steelblue'
        ))
        
        # Cumulative line
        cumsum = np.cumsum(self.explained_variance) * 100
        fig.add_trace(go.Scatter(
            x=[f'PC{i+1}' for i in range(len(cumsum))],
            y=cumsum,
            name='Cumulative',
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='PCA Explained Variance',
            xaxis_title='Principal Component',
            yaxis_title='Explained Variance (%)',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def plot_2d(self, labels: pd.Series, color_by: str = 'Group') -> go.Figure:
        """
        Create 2D PCA plot.
        
        Args:
            labels: Series with group/sex labels
            color_by: What to color points by
            
        Returns:
            Plotly figure
        """
        df_plot = pd.DataFrame({
            'PC1': self.components[:, 0],
            'PC2': self.components[:, 1],
            color_by: labels
        })
        
        fig = px.scatter(
            df_plot,
            x='PC1',
            y='PC2',
            color=color_by,
            title=f'PCA: PC1 vs PC2 (colored by {color_by})',
            template='plotly_white',
            opacity=0.7
        )
        
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
        
        return fig
    
    def plot_3d(self, labels: pd.Series, color_by: str = 'Group') -> go.Figure:
        """
        Create 3D PCA plot.
        
        Args:
            labels: Series with group/sex labels
            color_by: What to color points by
            
        Returns:
            Plotly figure
        """
        df_plot = pd.DataFrame({
            'PC1': self.components[:, 0],
            'PC2': self.components[:, 1],
            'PC3': self.components[:, 2],
            color_by: labels
        })
        
        fig = px.scatter_3d(
            df_plot,
            x='PC1',
            y='PC2',
            z='PC3',
            color=color_by,
            title=f'PCA: 3D Projection (colored by {color_by})',
            template='plotly_white',
            opacity=0.7
        )
        
        fig.update_traces(marker=dict(size=4, line=dict(width=0.3, color='white')))
        
        return fig
