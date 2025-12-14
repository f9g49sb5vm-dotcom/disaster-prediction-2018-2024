"""
Data Augmentation Module
Module tăng cường dữ liệu

This module contains functions for data augmentation including SMOTE for imbalanced data.
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter


def apply_smote(X, y, random_state=42, sampling_strategy='auto', k_neighbors=5):
    """
    Áp dụng SMOTE để cân bằng dữ liệu
    Apply SMOTE for balancing imbalanced data
    
    Parameters:
    -----------
    X : pd.DataFrame or np.array
        Feature matrix
    y : pd.Series or np.array
        Target variable
    random_state : int
        Random seed for reproducibility
    sampling_strategy : str or dict
        Sampling strategy for SMOTE
    k_neighbors : int
        Number of nearest neighbors for SMOTE
        
    Returns:
    --------
    X_resampled, y_resampled
        Resampled features and target
    """
    print("\n" + "=" * 60)
    print("APPLYING SMOTE FOR DATA AUGMENTATION")
    print("=" * 60)
    
    print(f"\nOriginal dataset shape: {X.shape}")
    print(f"Original class distribution:")
    print(Counter(y))
    
    # Apply SMOTE
    smote = SMOTE(
        random_state=random_state,
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors
    )
    
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"\nResampled dataset shape: {X_resampled.shape}")
    print(f"Resampled class distribution:")
    print(Counter(y_resampled))
    
    print(f"\nSMOTE applied successfully!")
    print(f"New samples generated: {len(X_resampled) - len(X)}")
    
    return X_resampled, y_resampled


def augment_disaster_data(df, target_column='disaster_type', random_state=42):
    """
    Tăng cường dữ liệu thảm họa sử dụng SMOTE
    Augment disaster data using SMOTE
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with features and target
    target_column : str
        Name of the target column
    random_state : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        Augmented DataFrame
    """
    # Separate features and target
    # Select only numeric columns for SMOTE
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target if it's in numeric columns
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    X = df[numeric_cols]
    y = df[target_column]
    
    # Apply SMOTE
    X_resampled, y_resampled = apply_smote(X, y, random_state=random_state)
    
    # Convert back to DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=numeric_cols)
    df_resampled[target_column] = y_resampled
    
    return df_resampled


def create_synthetic_samples(df, n_samples=100, random_state=42):
    """
    Tạo dữ liệu tổng hợp cho các lớp hiếm
    Create synthetic samples for rare classes
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    n_samples : int
        Number of synthetic samples to create
    random_state : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with synthetic samples
    """
    np.random.seed(random_state)
    
    print(f"\nCreating {n_samples} synthetic samples...")
    
    synthetic_samples = []
    
    for _ in range(n_samples):
        # Randomly select two samples
        idx1, idx2 = np.random.choice(len(df), 2, replace=False)
        sample1 = df.iloc[idx1]
        sample2 = df.iloc[idx2]
        
        # Create synthetic sample by interpolation
        alpha = np.random.random()
        
        synthetic_sample = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Interpolate numeric values
                synthetic_sample[col] = alpha * sample1[col] + (1 - alpha) * sample2[col]
            else:
                # Random choice for categorical
                synthetic_sample[col] = np.random.choice([sample1[col], sample2[col]])
        
        synthetic_samples.append(synthetic_sample)
    
    synthetic_df = pd.DataFrame(synthetic_samples)
    
    print(f"Synthetic samples created: {len(synthetic_df)}")
    
    return synthetic_df


def augment_by_class(df, target_column, min_samples=100, random_state=42):
    """
    Tăng cường dữ liệu theo từng lớp
    Augment data by class to ensure minimum samples
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_column : str
        Target column name
    min_samples : int
        Minimum number of samples per class
    random_state : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        Augmented DataFrame
    """
    print("\n" + "=" * 60)
    print("AUGMENTING DATA BY CLASS")
    print("=" * 60)
    
    augmented_dfs = []
    
    for class_value in df[target_column].unique():
        class_df = df[df[target_column] == class_value]
        current_count = len(class_df)
        
        print(f"\nClass: {class_value}")
        print(f"  Current samples: {current_count}")
        
        if current_count < min_samples:
            needed = min_samples - current_count
            print(f"  Generating {needed} synthetic samples...")
            
            synthetic = create_synthetic_samples(class_df, n_samples=needed, random_state=random_state)
            augmented_dfs.append(class_df)
            augmented_dfs.append(synthetic)
        else:
            print(f"  No augmentation needed")
            augmented_dfs.append(class_df)
    
    result_df = pd.concat(augmented_dfs, ignore_index=True)
    
    print(f"\n{'=' * 60}")
    print(f"Augmentation complete!")
    print(f"Original size: {len(df)}")
    print(f"Augmented size: {len(result_df)}")
    print(f"New samples added: {len(result_df) - len(df)}")
    
    return result_df


def balance_dataset(df, target_column='disaster_type', method='smote', random_state=42):
    """
    Cân bằng dataset sử dụng phương pháp được chỉ định
    Balance dataset using specified method
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_column : str
        Target column name
    method : str
        Balancing method ('smote' or 'synthetic')
    random_state : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        Balanced DataFrame
    """
    print(f"\nBalancing dataset using {method.upper()} method...")
    
    if method == 'smote':
        return augment_disaster_data(df, target_column=target_column, random_state=random_state)
    elif method == 'synthetic':
        return augment_by_class(df, target_column=target_column, random_state=random_state)
    else:
        print(f"Unknown method: {method}. Returning original dataset.")
        return df


if __name__ == "__main__":
    # Test the module
    print("Data Augmentation Module - Testing")
    print("=" * 60)
    
    # Create sample imbalanced data
    np.random.seed(42)
    
    sample_data = {
        'feature1': np.random.randn(150),
        'feature2': np.random.randn(150),
        'feature3': np.random.randn(150),
        'disaster_type': ['Earthquake'] * 50 + ['Flood'] * 70 + ['Wildfire'] * 30
    }
    
    df = pd.DataFrame(sample_data)
    
    print("\nOriginal DataFrame shape:", df.shape)
    print("\nOriginal class distribution:")
    print(df['disaster_type'].value_counts())
    
    # Test SMOTE augmentation
    df_balanced = balance_dataset(df, target_column='disaster_type', method='smote')
    
    print("\nBalanced DataFrame shape:", df_balanced.shape)
    print("\nBalanced class distribution:")
    print(df_balanced['disaster_type'].value_counts())
