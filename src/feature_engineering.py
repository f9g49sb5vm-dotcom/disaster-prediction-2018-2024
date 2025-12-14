"""
Feature Engineering Module
Module xử lý đặc trưng cho dữ liệu thảm họa

This module contains functions for creating temporal, geographic, and derived features.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def create_temporal_features(df):
    """
    Tạo các đặc trưng thời gian từ cột date
    Create temporal features from date column
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'date' column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added temporal features
    """
    df = df.copy()
    
    # Convert date to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Extract temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Season mapping
    def get_season(month):
        if month in [12, 1, 2]:
            return 1  # Winter
        elif month in [3, 4, 5]:
            return 2  # Spring
        elif month in [6, 7, 8]:
            return 3  # Summer
        else:
            return 4  # Autumn
    
    df['season'] = df['month'].apply(get_season)
    
    # Days since start
    start_date = df['date'].min()
    df['days_since_start'] = (df['date'] - start_date).dt.days
    
    return df


def create_geographic_features(df):
    """
    Tạo các đặc trưng địa lý (continent mapping)
    Create geographic features (continent mapping)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'country' column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added continent feature
    """
    df = df.copy()
    
    # Continent mapping as specified in requirements
    continent_map = {
        'United States': 'North America',
        'Canada': 'North America',
        'Mexico': 'North America',
        'Brazil': 'South America',
        'Chile': 'South America',
        'France': 'Europe',
        'Germany': 'Europe',
        'Greece': 'Europe',
        'Spain': 'Europe',
        'Italy': 'Europe',
        'Turkey': 'Europe',
        'India': 'Asia',
        'China': 'Asia',
        'Bangladesh': 'Asia',
        'Japan': 'Asia',
        'Indonesia': 'Southeast Asia',
        'Philippines': 'Southeast Asia',
        'Nigeria': 'Africa',
        'South Africa': 'Africa',
        'Australia': 'Oceania'
    }
    
    df['continent'] = df['country'].map(continent_map)
    
    return df


def create_derived_features(df):
    """
    Tạo các đặc trưng phái sinh
    Create derived features from existing columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with base features
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added derived features
    """
    df = df.copy()
    
    # Severity category
    def categorize_severity(severity):
        if severity <= 3:
            return 'Low'
        elif severity <= 6:
            return 'Medium'
        elif severity <= 8:
            return 'High'
        else:
            return 'Critical'
    
    df['severity_category'] = df['severity_index'].apply(categorize_severity)
    
    # Aid per casualty (handle division by zero)
    df['aid_per_casualty'] = np.where(
        df['casualties'] > 0,
        df['aid_amount_usd'] / df['casualties'],
        0
    )
    
    # Loss per casualty
    df['loss_per_casualty'] = np.where(
        df['casualties'] > 0,
        df['economic_loss_usd'] / df['casualties'],
        0
    )
    
    # Recovery efficiency
    df['recovery_efficiency'] = np.where(
        df['severity_index'] > 0,
        df['recovery_days'] / df['severity_index'],
        0
    )
    
    # Response effectiveness
    df['response_effectiveness'] = np.where(
        df['response_time_hours'] > 0,
        df['response_efficiency_score'] / df['response_time_hours'],
        0
    )
    
    return df


def encode_features(df, fit=True, encoders=None):
    """
    Mã hóa các đặc trưng phân loại
    Encode categorical features
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with categorical features
    fit : bool
        Whether to fit encoders (True for train, False for test)
    encoders : dict
        Dictionary of pre-fitted encoders (for test data)
        
    Returns:
    --------
    pd.DataFrame, dict
        Encoded DataFrame and dictionary of encoders
    """
    from sklearn.preprocessing import LabelEncoder
    
    df = df.copy()
    
    if fit:
        encoders = {}
        
        # Label encoding for disaster_type
        if 'disaster_type' in df.columns:
            encoders['disaster_type'] = LabelEncoder()
            df['disaster_type_encoded'] = encoders['disaster_type'].fit_transform(df['disaster_type'])
        
        # One-hot encoding for country
        if 'country' in df.columns:
            country_dummies = pd.get_dummies(df['country'], prefix='country')
            df = pd.concat([df, country_dummies], axis=1)
            encoders['country_columns'] = country_dummies.columns.tolist()
        
        # One-hot encoding for continent
        if 'continent' in df.columns:
            continent_dummies = pd.get_dummies(df['continent'], prefix='continent')
            df = pd.concat([df, continent_dummies], axis=1)
            encoders['continent_columns'] = continent_dummies.columns.tolist()
            
    else:
        # Use pre-fitted encoders
        if 'disaster_type' in df.columns and 'disaster_type' in encoders:
            df['disaster_type_encoded'] = encoders['disaster_type'].transform(df['disaster_type'])
        
        # One-hot encoding with same columns as training
        if 'country' in df.columns and 'country_columns' in encoders:
            country_dummies = pd.get_dummies(df['country'], prefix='country')
            # Ensure same columns as training
            for col in encoders['country_columns']:
                if col not in country_dummies.columns:
                    country_dummies[col] = 0
            country_dummies = country_dummies[encoders['country_columns']]
            df = pd.concat([df, country_dummies], axis=1)
        
        if 'continent' in df.columns and 'continent_columns' in encoders:
            continent_dummies = pd.get_dummies(df['continent'], prefix='continent')
            for col in encoders['continent_columns']:
                if col not in continent_dummies.columns:
                    continent_dummies[col] = 0
            continent_dummies = continent_dummies[encoders['continent_columns']]
            df = pd.concat([df, continent_dummies], axis=1)
    
    return df, encoders


def engineer_all_features(df, fit=True, encoders=None):
    """
    Áp dụng tất cả các bước feature engineering
    Apply all feature engineering steps
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw DataFrame
    fit : bool
        Whether to fit encoders
    encoders : dict
        Pre-fitted encoders (for test data)
        
    Returns:
    --------
    pd.DataFrame, dict
        Fully engineered DataFrame and encoders
    """
    print("Creating temporal features...")
    df = create_temporal_features(df)
    
    print("Creating geographic features...")
    df = create_geographic_features(df)
    
    print("Creating derived features...")
    df = create_derived_features(df)
    
    print("Encoding categorical features...")
    df, encoders = encode_features(df, fit=fit, encoders=encoders)
    
    print(f"Feature engineering complete! Total features: {len(df.columns)}")
    
    return df, encoders


if __name__ == "__main__":
    # Test the module
    print("Feature Engineering Module - Testing")
    print("=" * 50)
    
    # Create sample data
    sample_data = {
        'date': ['2018-01-01', '2019-06-15', '2020-12-25'],
        'country': ['United States', 'India', 'Brazil'],
        'disaster_type': ['Earthquake', 'Flood', 'Wildfire'],
        'severity_index': [5, 8, 3],
        'casualties': [100, 500, 50],
        'economic_loss_usd': [1000000, 5000000, 500000],
        'response_time_hours': [24, 48, 12],
        'aid_amount_usd': [500000, 2000000, 200000],
        'response_efficiency_score': [75, 60, 90],
        'recovery_days': [30, 60, 15]
    }
    
    df = pd.DataFrame(sample_data)
    print("\nOriginal DataFrame:")
    print(df)
    
    df_engineered, encoders = engineer_all_features(df)
    print("\nEngineered DataFrame shape:", df_engineered.shape)
    print("\nNew features created:")
    print(df_engineered.columns.tolist())
