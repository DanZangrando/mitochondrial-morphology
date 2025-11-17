"""
Utility functions for analysis and visualization.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from typing import Dict, List


def create_distribution_plot(
    data: pd.DataFrame,
    column: str,
    group_by: str = 'Group',
    plot_type: str = 'box'
) -> go.Figure:
    """
    Create distribution plot for a metric.
    
    Args:
        data: DataFrame with data
        column: Column to plot
        group_by: Column to group by
        plot_type: 'box', 'violin', or 'histogram'
        
    Returns:
        Plotly figure
    """
    if plot_type == 'box':
        fig = px.box(
            data,
            x=group_by,
            y=column,
            color=group_by,
            title=f'Distribution of {column} by {group_by}',
            template='plotly_white'
        )
    elif plot_type == 'violin':
        fig = px.violin(
            data,
            x=group_by,
            y=column,
            color=group_by,
            box=True,
            title=f'Distribution of {column} by {group_by}',
            template='plotly_white'
        )
    elif plot_type == 'histogram':
        fig = px.histogram(
            data,
            x=column,
            color=group_by,
            marginal='box',
            title=f'Distribution of {column} by {group_by}',
            template='plotly_white',
            opacity=0.7
        )
    
    fig.update_layout(height=500)
    return fig


def create_correlation_heatmap(data: pd.DataFrame, features: List[str]) -> go.Figure:
    """
    Create correlation heatmap.
    
    Args:
        data: DataFrame with data
        features: List of feature columns
        
    Returns:
        Plotly figure
    """
    corr_matrix = data[features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        template='plotly_white',
        height=600,
        width=700
    )
    
    return fig


def perform_statistical_tests(
    data: pd.DataFrame,
    feature: str,
    group_col: str = 'Group'
) -> Dict[str, float]:
    """
    Perform t-test or ANOVA between groups.
    
    Args:
        data: DataFrame with data
        feature: Feature to test
        group_col: Grouping column
        
    Returns:
        Dictionary with test results
    """
    groups = data[group_col].unique()
    
    if len(groups) == 2:
        # T-test
        group1 = data[data[group_col] == groups[0]][feature].dropna()
        group2 = data[data[group_col] == groups[1]][feature].dropna()
        
        statistic, p_value = stats.ttest_ind(group1, group2)
        
        return {
            'test': 't-test',
            'statistic': statistic,
            'p_value': p_value,
            'group1_mean': group1.mean(),
            'group2_mean': group2.mean(),
            'significant': p_value < 0.05
        }
    else:
        # ANOVA
        group_data = [data[data[group_col] == g][feature].dropna() for g in groups]
        statistic, p_value = stats.f_oneway(*group_data)
        
        return {
            'test': 'ANOVA',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }


def create_pairplot_data(data: pd.DataFrame, features: List[str], n_sample: int = 500):
    """
    Prepare data for pairplot (sample if needed).
    
    Args:
        data: DataFrame with data
        features: Features to include
        n_sample: Number of samples (to avoid overplotting)
        
    Returns:
        Sampled DataFrame
    """
    if len(data) > n_sample:
        return data.sample(n=n_sample, random_state=42)
    return data


def format_metrics_table(metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Format metrics for display.
    
    Args:
        metrics: Dictionary of metric names and values
        
    Returns:
        Formatted DataFrame
    """
    df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    return df


def calculate_reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> Dict:
    """
    Calculate reconstruction error metrics.
    
    Args:
        original: Original data
        reconstructed: Reconstructed data
        
    Returns:
        Dictionary with error metrics
    """
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    rmse = np.sqrt(mse)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse
    }
