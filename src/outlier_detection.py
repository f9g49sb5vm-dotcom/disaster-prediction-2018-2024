"""
Outlier Detection Module
Module phát hiện và xử lý outliers

This module contains functions for detecting and handling outliers using IQR and Z-score methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def detect_outliers_iqr(df, columns, multiplier=1.5):
    """
    Phát hiện outliers sử dụng phương pháp IQR (Interquartile Range)
    Detect outliers using IQR method
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list
        List of column names to check for outliers
    multiplier : float
        IQR multiplier (default 1.5)
        
    Returns:
    --------
    dict
        Dictionary with column names as keys and outlier indices as values
    """
    outliers_dict = {}
    
    for column in columns:
        if column not in df.columns:
            continue
            
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
        outliers_dict[column] = outliers
        
        print(f"\n{column}:")
        print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"  Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    
    return outliers_dict


def detect_outliers_zscore(df, columns, threshold=3):
    """
    Phát hiện outliers sử dụng phương pháp Z-score
    Detect outliers using Z-score method
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list
        List of column names to check for outliers
    threshold : float
        Z-score threshold (default 3)
        
    Returns:
    --------
    dict
        Dictionary with column names as keys and outlier indices as values
    """
    outliers_dict = {}
    
    for column in columns:
        if column not in df.columns:
            continue
            
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outliers = df[column].dropna().iloc[z_scores > threshold].index
        outliers_dict[column] = outliers
        
        print(f"\n{column}:")
        print(f"  Mean: {df[column].mean():.2f}, Std: {df[column].std():.2f}")
        print(f"  Z-score threshold: {threshold}")
        print(f"  Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    
    return outliers_dict


def visualize_outliers_boxplot(df, columns, figsize=(15, 10), save_path=None):
    """
    Trực quan hóa outliers bằng boxplot
    Visualize outliers using boxplots
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list
        List of column names to visualize
    figsize : tuple
        Figure size
    save_path : str
        Path to save the figure (optional)
    """
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, column in enumerate(columns):
        if column in df.columns:
            sns.boxplot(data=df, y=column, ax=axes[idx])
            axes[idx].set_title(f'Boxplot: {column}')
            axes[idx].set_ylabel(column)
    
    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Boxplot saved to {save_path}")
    
    plt.show()


def visualize_outliers_scatter(df, x_col, y_col, figsize=(10, 6), save_path=None):
    """
    Trực quan hóa outliers bằng scatter plot
    Visualize outliers using scatter plot
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    figsize : tuple
        Figure size
    save_path : str
        Path to save the figure (optional)
    """
    plt.figure(figsize=figsize)
    plt.scatter(df[x_col], df[y_col], alpha=0.5)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Scatter Plot: {x_col} vs {y_col}')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to {save_path}")
    
    plt.show()


def handle_outliers_cap(df, column, lower_percentile=1, upper_percentile=99):
    """
    Xử lý outliers bằng cách cap (giới hạn) giá trị
    Handle outliers by capping values at percentiles
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column name to process
    lower_percentile : float
        Lower percentile for capping
    upper_percentile : float
        Upper percentile for capping
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with capped values
    """
    df = df.copy()
    
    lower_bound = df[column].quantile(lower_percentile / 100)
    upper_bound = df[column].quantile(upper_percentile / 100)
    
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"Capped {column} to [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    return df


def apply_log_transformation(df, columns):
    """
    Áp dụng biến đổi logarithm để xử lý skewness
    Apply log transformation to handle skewness
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list
        List of column names to transform
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with log-transformed columns
    """
    df = df.copy()
    
    for column in columns:
        if column in df.columns:
            # Add 1 to avoid log(0)
            df[f'{column}_log'] = np.log1p(df[column])
            print(f"Created log-transformed feature: {column}_log")
    
    return df


def apply_sqrt_transformation(df, columns):
    """
    Áp dụng biến đổi căn bậc hai để xử lý skewness
    Apply square root transformation to handle skewness
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list
        List of column names to transform
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with sqrt-transformed columns
    """
    df = df.copy()
    
    for column in columns:
        if column in df.columns:
            df[f'{column}_sqrt'] = np.sqrt(df[column].clip(lower=0))
            print(f"Created sqrt-transformed feature: {column}_sqrt")
    
    return df


def analyze_outliers(df):
    """
    Phân tích toàn diện outliers cho dataset
    Comprehensive outlier analysis for the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    dict
        Dictionary containing outlier analysis results
    """
    print("=" * 80)
    print("OUTLIER DETECTION ANALYSIS")
    print("=" * 80)
    
    # Columns for IQR method (as specified in requirements)
    iqr_columns = ['casualties', 'economic_loss_usd', 'aid_amount_usd']
    
    # Columns for Z-score method (as specified in requirements)
    zscore_columns = ['response_time_hours', 'recovery_days']
    
    results = {}
    
    print("\n--- IQR Method ---")
    results['iqr_outliers'] = detect_outliers_iqr(df, iqr_columns)
    
    print("\n--- Z-Score Method ---")
    results['zscore_outliers'] = detect_outliers_zscore(df, zscore_columns, threshold=3)
    
    return results


if __name__ == "__main__":
    # Test the module
    print("Outlier Detection Module - Testing")
    print("=" * 50)
    
    # Create sample data with outliers
    np.random.seed(42)
    sample_data = {
        'casualties': np.concatenate([np.random.normal(100, 20, 95), [500, 600, 700, 800, 900]]),
        'economic_loss_usd': np.concatenate([np.random.normal(1000000, 200000, 95), 
                                             [10000000, 15000000, 20000000, 25000000, 30000000]]),
        'aid_amount_usd': np.concatenate([np.random.normal(500000, 100000, 95),
                                          [5000000, 6000000, 7000000, 8000000, 9000000]]),
        'response_time_hours': np.concatenate([np.random.normal(24, 5, 95), [100, 120, 150, 180, 200]]),
        'recovery_days': np.concatenate([np.random.normal(30, 8, 95), [150, 180, 200, 220, 250]])
    }
    
    df = pd.DataFrame(sample_data)
    print("\nSample DataFrame created with shape:", df.shape)
    
    # Analyze outliers
    results = analyze_outliers(df)
    
    print("\n\nOutlier detection complete!")
