"""
Data Split Module
Module chia dữ liệu train/test

This module contains functions for splitting data into train and test sets with stratification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os


def stratified_split(df, target_column, test_size=0.2, random_state=42):
    """
    Chia dữ liệu train/test với stratified sampling
    Split data into train/test with stratified sampling
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_column : str
        Target column name for stratification
    test_size : float
        Proportion of test set (default 0.2 = 20%)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    train_df, test_df
        Train and test DataFrames
    """
    print("\n" + "=" * 60)
    print("STRATIFIED TRAIN-TEST SPLIT")
    print("=" * 60)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Target column: {target_column}")
    print(f"Test size: {test_size * 100}%")
    
    # Check target distribution
    print(f"\nTarget distribution:")
    print(df[target_column].value_counts())
    print(f"\nTarget proportions:")
    print(df[target_column].value_counts(normalize=True))
    
    # Perform stratified split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_column],
        random_state=random_state
    )
    
    print(f"\nTrain set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    print(f"\nTrain set target distribution:")
    print(train_df[target_column].value_counts())
    print(f"\nTest set target distribution:")
    print(test_df[target_column].value_counts())
    
    # Verify stratification
    print(f"\n{'=' * 60}")
    print("STRATIFICATION VERIFICATION")
    print("=" * 60)
    
    train_proportions = train_df[target_column].value_counts(normalize=True).sort_index()
    test_proportions = test_df[target_column].value_counts(normalize=True).sort_index()
    
    comparison_df = pd.DataFrame({
        'Train': train_proportions,
        'Test': test_proportions,
        'Difference': abs(train_proportions - test_proportions)
    })
    
    print("\nProportions comparison:")
    print(comparison_df)
    print(f"\nMax difference: {comparison_df['Difference'].max():.4f}")
    
    return train_df, test_df


def save_split_data(train_df, test_df, encoders, output_dir='data/processed'):
    """
    Lưu dữ liệu train/test và encoders
    Save train/test data and encoders
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training DataFrame
    test_df : pd.DataFrame
        Test DataFrame
    encoders : dict
        Dictionary of encoders
    output_dir : str
        Output directory path
    """
    print("\n" + "=" * 60)
    print("SAVING SPLIT DATA")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train data
    train_path = os.path.join(output_dir, 'train.csv')
    train_df.to_csv(train_path, index=False)
    print(f"\nTrain data saved to: {train_path}")
    print(f"  Shape: {train_df.shape}")
    
    # Save test data
    test_path = os.path.join(output_dir, 'test.csv')
    test_df.to_csv(test_path, index=False)
    print(f"\nTest data saved to: {test_path}")
    print(f"  Shape: {test_df.shape}")
    
    # Save encoders
    encoders_path = os.path.join(output_dir, 'encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"\nEncoders saved to: {encoders_path}")
    print(f"  Encoders: {list(encoders.keys())}")
    
    print(f"\n{'=' * 60}")
    print("All data saved successfully!")
    print("=" * 60)


def load_split_data(input_dir='data/processed'):
    """
    Tải dữ liệu train/test và encoders
    Load train/test data and encoders
    
    Parameters:
    -----------
    input_dir : str
        Input directory path
        
    Returns:
    --------
    train_df, test_df, encoders
        Loaded train DataFrame, test DataFrame, and encoders
    """
    print("\n" + "=" * 60)
    print("LOADING SPLIT DATA")
    print("=" * 60)
    
    # Load train data
    train_path = os.path.join(input_dir, 'train.csv')
    train_df = pd.read_csv(train_path)
    print(f"\nTrain data loaded from: {train_path}")
    print(f"  Shape: {train_df.shape}")
    
    # Load test data
    test_path = os.path.join(input_dir, 'test.csv')
    test_df = pd.read_csv(test_path)
    print(f"\nTest data loaded from: {test_path}")
    print(f"  Shape: {test_df.shape}")
    
    # Load encoders
    encoders_path = os.path.join(input_dir, 'encoders.pkl')
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    print(f"\nEncoders loaded from: {encoders_path}")
    print(f"  Encoders: {list(encoders.keys())}")
    
    print(f"\n{'=' * 60}")
    print("All data loaded successfully!")
    print("=" * 60)
    
    return train_df, test_df, encoders


def split_features_target(df, target_columns, drop_columns=None):
    """
    Chia DataFrame thành features và target
    Split DataFrame into features and target
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_columns : str or list
        Target column name(s)
    drop_columns : list
        Columns to drop from features (optional)
        
    Returns:
    --------
    X, y
        Features and target
    """
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    
    # Create copy to avoid modifying original
    df = df.copy()
    
    # Extract target
    y = df[target_columns]
    if len(target_columns) == 1:
        y = y.iloc[:, 0]
    
    # Create features
    X = df.drop(columns=target_columns)
    
    # Drop additional columns if specified
    if drop_columns:
        existing_drop_cols = [col for col in drop_columns if col in X.columns]
        X = X.drop(columns=existing_drop_cols)
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape if hasattr(y, 'shape') else len(y)}")
    
    return X, y


def create_train_test_split(df, target_column='disaster_type', 
                            test_size=0.2, random_state=42,
                            output_dir='data/processed',
                            encoders=None, save=True):
    """
    Pipeline hoàn chỉnh để chia và lưu dữ liệu
    Complete pipeline for splitting and saving data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_column : str
        Target column for stratification
    test_size : float
        Test set proportion
    random_state : int
        Random seed
    output_dir : str
        Output directory
    encoders : dict
        Encoders dictionary (optional)
    save : bool
        Whether to save the data
        
    Returns:
    --------
    train_df, test_df, encoders
        Train/test DataFrames and encoders
    """
    # Perform stratified split
    train_df, test_df = stratified_split(
        df, 
        target_column=target_column,
        test_size=test_size,
        random_state=random_state
    )
    
    # Save if requested
    if save:
        if encoders is None:
            encoders = {}
        save_split_data(train_df, test_df, encoders, output_dir=output_dir)
    
    return train_df, test_df, encoders


if __name__ == "__main__":
    # Test the module
    print("Data Split Module - Testing")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    
    sample_data = {
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
        'feature4': np.random.randn(1000),
        'disaster_type': np.random.choice(
            ['Earthquake', 'Flood', 'Wildfire', 'Hurricane'], 
            size=1000,
            p=[0.3, 0.3, 0.2, 0.2]
        )
    }
    
    df = pd.DataFrame(sample_data)
    
    print("\nSample DataFrame created:")
    print(f"Shape: {df.shape}")
    print("\nTarget distribution:")
    print(df['disaster_type'].value_counts())
    
    # Test stratified split
    train_df, test_df = stratified_split(df, target_column='disaster_type')
    
    print("\n\nSplit completed successfully!")
    
    # Test feature-target split
    print("\n" + "=" * 60)
    print("Testing feature-target split...")
    X_train, y_train = split_features_target(train_df, target_columns='disaster_type')
    X_test, y_test = split_features_target(test_df, target_columns='disaster_type')
    
    print("\nFeature-target split completed!")
