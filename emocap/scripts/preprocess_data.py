"""
Preprocessing script for IEMOCAP train dataset.
Applies feature removal, transformation, and scaling.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_CSV = os.path.join(BASE_DIR, 'data', 'iemocap', 'train.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'iemocap')
TRAIN_PROCESSED_CSV = os.path.join(OUTPUT_DIR, 'train_processed.csv')
SCALER_PATH = os.path.join(OUTPUT_DIR, 'scaler.pkl')

# Features to remove (highly correlated)
FEATURES_TO_REMOVE = ['zcr_mean', 'spectral_rolloff_mean']

# Features for log transformation (right-skewed, skewness > 2)
LOG_TRANSFORM_FEATURES = [
    'delta chroma_1_mean', 'delta chroma_8_mean', 'delta spectral_flux_mean',
    'delta mfcc_1_mean', 'energy_mean', 'chroma_2_mean', 'chroma_8_mean',
    'chroma_4_mean', 'chroma_6_mean', 'chroma_9_mean', 'chroma_10_mean',
    'chroma_12_mean', 'delta mfcc_8_mean', 'delta chroma_3_mean',
    'delta chroma_4_mean', 'delta chroma_6_mean', 'delta chroma_7_mean'
]

# Features for sqrt transformation (left-skewed, skewness < -2)
SQRT_TRANSFORM_FEATURES = [
    'delta energy_entropy_mean', 'delta mfcc_4_mean', 'delta chroma_5_mean',
    'delta chroma_9_mean', 'delta chroma_12_mean', 'delta zcr_mean',
    'delta energy_mean', 'delta spectral_rolloff_mean', 'delta mfcc_5_mean',
    'delta chroma_11_mean'
]

def apply_log_transform(df, features):
    """Apply log(x + 1) transformation to handle zeros and negative values."""
    df_transformed = df.copy()
    for feat in features:
        if feat in df_transformed.columns:
            min_val = df_transformed[feat].min()
            if min_val <= 0:
                shift = abs(min_val) + 1
                df_transformed[feat] = np.log1p(df_transformed[feat] + shift)
            else:
                df_transformed[feat] = np.log1p(df_transformed[feat])
    return df_transformed

def apply_sqrt_transform(df, features):
    """Apply sqrt(|x|) * sign(x) transformation."""
    df_transformed = df.copy()
    for feat in features:
        if feat in df_transformed.columns:
            signs = np.sign(df_transformed[feat])
            df_transformed[feat] = signs * np.sqrt(np.abs(df_transformed[feat]))
    return df_transformed

def main():
    print("Loading train data...")
    df = pd.read_csv(TRAIN_CSV)
    print(f"Original shape: {df.shape}")
    
    # Separate metadata and features
    metadata_cols = ['session', 'method', 'gender', 'emotion', 'n_annotators', 'agreement']
    metadata_df = df[metadata_cols].copy()
    feature_df = df.drop(columns=metadata_cols)
    
    # Step 1: Remove redundant features
    print("\nStep 1: Removing redundant features...")
    features_to_remove = [f for f in FEATURES_TO_REMOVE if f in feature_df.columns]
    feature_df = feature_df.drop(columns=features_to_remove)
    print(f"Removed {len(features_to_remove)} features: {features_to_remove}")
    
    # Step 2: Transform skewed features
    print("\nStep 2: Transforming skewed features...")
    
    # Log transform (right-skewed)
    log_features = [f for f in LOG_TRANSFORM_FEATURES if f in feature_df.columns]
    if log_features:
        print(f"Applying log transform to {len(log_features)} features...")
        feature_df = apply_log_transform(feature_df, log_features)
    
    # Sqrt transform (left-skewed)
    sqrt_features = [f for f in SQRT_TRANSFORM_FEATURES if f in feature_df.columns]
    if sqrt_features:
        print(f"Applying sqrt transform to {len(sqrt_features)} features...")
        feature_df = apply_sqrt_transform(feature_df, sqrt_features)
    
    # Step 3: Scale features
    print("\nStep 3: Scaling features with StandardScaler...")
    scaler = StandardScaler()
    feature_array = scaler.fit_transform(feature_df)
    feature_df_scaled = pd.DataFrame(
        feature_array,
        columns=feature_df.columns,
        index=feature_df.index
    )
    
    # Combine metadata and scaled features
    df_processed = pd.concat([metadata_df, feature_df_scaled], axis=1)
    
    print(f"\nProcessed shape: {df_processed.shape}")
    print(f"Features: {len(feature_df_scaled.columns)}")
    
    # Save processed data
    print("\nSaving processed data...")
    df_processed.to_csv(TRAIN_PROCESSED_CSV, index=False)
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"✅ Saved: {TRAIN_PROCESSED_CSV}")
    print(f"✅ Saved scaler: {SCALER_PATH}")
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()
