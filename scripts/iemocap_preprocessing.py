"""
Data Preprocessing Script for IEMOCAP Features
Προετοιμάζει τα extracted features για machine learning

Usage:
    python scripts/iemocap_preprocessing.py --input datasets/iemocap_features.pkl --output datasets/iemocap_features_processed.pkl
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pickle
from pathlib import Path
import argparse
import warnings

warnings.filterwarnings('ignore')


def load_features(input_path):
    """
    Φορτώνει τα features από pickle file
    
    Parameters:
    -----------
    input_path : str
        Διαδρομή προς το .pkl file με τα features
    
    Returns:
    --------
    pd.DataFrame : DataFrame με τα features
    """
    print(f"Loading features from {input_path}...")
    df = pd.read_pickle(input_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    return df


def handle_missing_values(df, strategy='mean'):
    """
    Χειρίζεται missing values (NaN) στα features
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame με τα features
    strategy : str
        Strategy για imputation: 'mean', 'median', 'most_frequent', 'constant'
    
    Returns:
    --------
    pd.DataFrame : DataFrame με imputed values
    """
    print(f"\nHandling missing values (strategy: {strategy})...")
    
    # Χωρίζουμε features από metadata
    feature_cols = [c for c in df.columns 
                   if c not in ['session', 'method', 'gender', 'emotion', 
                               'n_annotators', 'agreement', 'audio_path']]
    metadata_cols = [c for c in df.columns if c in ['session', 'method', 'gender', 
                                                     'emotion', 'n_annotators', 'agreement', 'audio_path']]
    
    # Ελέγχουμε πόσα missing values υπάρχουν
    missing_count = df[feature_cols].isna().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values")
        
        # Impute missing values
        imputer = SimpleImputer(strategy=strategy)
        df_features = pd.DataFrame(
            imputer.fit_transform(df[feature_cols]),
            columns=feature_cols,
            index=df.index
        )
        
        # Συνδυάζουμε με metadata
        df_processed = pd.concat([df_features, df[metadata_cols]], axis=1)
        print(f"Imputed {missing_count} missing values")
    else:
        print("No missing values found")
        df_processed = df.copy()
    
    return df_processed


def remove_inf_values(df):
    """
    Αφαιρεί Inf values (infinity) από τα features
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame με τα features
    
    Returns:
    --------
    pd.DataFrame : DataFrame χωρίς Inf values
    """
    print("\nRemoving infinite values...")
    
    feature_cols = [c for c in df.columns 
                   if c not in ['session', 'method', 'gender', 'emotion', 
                               'n_annotators', 'agreement', 'audio_path']]
    
    # Αντικαθιστούμε Inf με NaN
    df_processed = df.copy()
    inf_count = np.isinf(df_processed[feature_cols]).sum().sum()
    
    if inf_count > 0:
        print(f"Found {inf_count} infinite values")
        df_processed[feature_cols] = df_processed[feature_cols].replace([np.inf, -np.inf], np.nan)
        # Impute με mean
        imputer = SimpleImputer(strategy='mean')
        df_processed[feature_cols] = pd.DataFrame(
            imputer.fit_transform(df_processed[feature_cols]),
            columns=feature_cols,
            index=df_processed.index
        )
        print(f"Replaced {inf_count} infinite values with mean")
    else:
        print("No infinite values found")
    
    return df_processed


def scale_features(df, scaler_type='standard', exclude_cols=None):
    """
    Κάνει scaling/normalization των features
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame με τα features
    scaler_type : str
        Τύπος scaler: 'standard', 'minmax', 'robust'
    exclude_cols : list
        Columns που δεν θα κλιμακωθούν (metadata)
    
    Returns:
    --------
    pd.DataFrame : DataFrame με scaled features
    tuple : (scaler, feature_columns)
    """
    print(f"\nScaling features (scaler: {scaler_type})...")
    
    if exclude_cols is None:
        exclude_cols = ['session', 'method', 'gender', 'emotion', 
                       'n_annotators', 'agreement', 'audio_path']
    
    # Χωρίζουμε features από metadata
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    metadata_cols = [c for c in df.columns if c in exclude_cols]
    
    # Επιλέγουμε scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()  # Mean=0, Std=1
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()  # Range [0, 1]
    elif scaler_type == 'robust':
        scaler = RobustScaler()  # Robust to outliers
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Fit και transform
    df_features_scaled = pd.DataFrame(
        scaler.fit_transform(df[feature_cols]),
        columns=feature_cols,
        index=df.index
    )
    
    # Συνδυάζουμε με metadata
    df_processed = pd.concat([df_features_scaled, df[metadata_cols]], axis=1)
    
    print(f"Scaled {len(feature_cols)} features using {scaler_type} scaler")
    
    return df_processed, scaler, feature_cols


def split_data(df, test_size=0.2, random_state=42, stratify_col='emotion'):
    """
    Χωρίζει τα δεδομένα σε train/test sets (χωρίς validation)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame με τα features
    test_size : float
        Ποσοστό για test set (default: 0.2 = 20%)
    random_state : int
        Random seed για reproducibility
    stratify_col : str
        Column για stratified splitting (default: 'emotion')
    
    Returns:
    --------
    dict : Dictionary με train, test DataFrames
    """
    print(f"\nSplitting data (test: {test_size*100}%)...")
    
    # Χωρίζουμε features από labels
    feature_cols = [c for c in df.columns 
                   if c not in ['session', 'method', 'gender', 'emotion', 
                               'n_annotators', 'agreement', 'audio_path']]
    X = df[feature_cols]
    y = df[stratify_col] if stratify_col in df.columns else None
    
    # Train/Test split (μόνο)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y if stratify_col in df.columns else None
    )
    
    # Συνδυάζουμε με metadata
    train_indices = X_train.index
    test_indices = X_test.index
    
    splits = {
        'train': df.loc[train_indices],
        'test': df.loc[test_indices]
    }
    
    print(f"Train set: {len(splits['train'])} samples")
    print(f"Test set: {len(splits['test'])} samples")
    
    # Emotion distribution
    if stratify_col in df.columns:
        print("\nEmotion distribution:")
        for split_name, split_df in splits.items():
            print(f"  {split_name}:")
            emotion_counts = split_df[stratify_col].value_counts()
            for emotion, count in emotion_counts.items():
                print(f"    {emotion}: {count}")
    
    return splits


def preprocess_features(input_path, output_path=None, scaler_type='standard', 
                       test_size=0.2, save_splits=True):
    """
    Κύριο preprocessing pipeline
    
    Parameters:
    -----------
    input_path : str
        Διαδρομή προς το input .pkl file
    output_path : str
        Διαδρομή για αποθήκευση (default: input_path με _processed suffix)
    scaler_type : str
        Τύπος scaler: 'standard', 'minmax', 'robust'
    test_size : float
        Ποσοστό για test set
    save_splits : bool
        Αν να αποθηκευτούν τα train/test splits
    """
    # Load features
    df = load_features(input_path)
    
    # 1. Handle missing values
    df = handle_missing_values(df)
    
    # 2. Remove infinite values
    df = remove_inf_values(df)
    
    # 3. Scale features
    df_scaled, scaler, feature_cols = scale_features(df, scaler_type=scaler_type)
    
    # 4. Split data (μόνο train/test, χωρίς validation)
    splits = split_data(df_scaled, test_size=test_size)
    
    # Set output path
    if output_path is None:
        output_path = input_path.replace('.pkl', '_processed.pkl')
    
    # Save processed features
    print(f"\nSaving processed features to {output_path}...")
    df_scaled.to_pickle(output_path)
    df_scaled.to_csv(output_path.replace('.pkl', '.csv'), index=False)
    print(f"Also saved as CSV: {output_path.replace('.pkl', '.csv')}")
    
    # Save scaler
    scaler_path = output_path.replace('.pkl', '_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")
    
    # Save splits if requested
    if save_splits:
        splits_path = output_path.replace('.pkl', '_splits.pkl')
        with open(splits_path, 'wb') as f:
            pickle.dump(splits, f)
        print(f"Saved train/test splits to {splits_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Preprocessing complete!")
    print(f"{'='*60}")
    print(f"Original features: {len(df.columns)}")
    print(f"Processed features: {len(df_scaled.columns)}")
    print(f"Scaler used: {scaler_type}")
    print(f"Train samples: {len(splits['train'])}")
    print(f"Test samples: {len(splits['test'])}")
    print(f"{'='*60}")
    
    return df_scaled, scaler, splits


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Preprocess IEMOCAP features for machine learning'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input .pkl file with extracted features'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output .pkl file (default: input_processed.pkl)'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default='standard',
        choices=['standard', 'minmax', 'robust'],
        help='Type of scaler (default: standard)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size (default: 0.2)'
    )
    parser.add_argument(
        '--no-splits',
        action='store_true',
        help='Do not save train/test splits'
    )
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Run preprocessing
    preprocess_features(
        input_path=args.input,
        output_path=args.output,
        scaler_type=args.scaler,
        test_size=args.test_size,
        save_splits=not args.no_splits
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

