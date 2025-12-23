"""
Preprocessing script for IEMOCAP with Fold Support.

Λειτουργίες:
1. Επεξεργασία ανά Fold (1-5).
2. Emotion Alignment: Κρατάει μόνο τα συναισθήματα που υπάρχουν σε Train, Val και Test.
3. Feature Transformations (Log/Sqrt).
4. Feature Selection (CFS) προσαρμοσμένο ανά fold.
5. Resampling (SMOTE/Random) προσαρμοσμένο ανά fold.
6. Scaling (StandardScaler) fitted μόνο στο Training set.

Usage:
    python 04_preprocess_data.py --fold 1 --cfs --resample smote
    python 04_preprocess_data.py --all --cfs --resample smote
"""

import os
import pandas as pd
import numpy as np
import argparse
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE, RandomOverSampler

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'iemocap', 'dataset')

# Config
FEATURES_TO_REMOVE = ['zcr_mean', 'spectral_rolloff_mean']
LOG_TRANSFORM_FEATURES = [
    'delta chroma_1_mean', 'delta chroma_8_mean', 'delta spectral_flux_mean',
    'delta mfcc_1_mean', 'energy_mean', 'chroma_2_mean', 'chroma_8_mean',
    'chroma_4_mean', 'chroma_6_mean', 'chroma_9_mean', 'chroma_10_mean',
    'chroma_12_mean', 'delta mfcc_8_mean', 'delta chroma_3_mean',
    'delta chroma_4_mean', 'delta chroma_6_mean', 'delta chroma_7_mean'
]
SQRT_TRANSFORM_FEATURES = [
    'delta energy_entropy_mean', 'delta mfcc_4_mean', 'delta chroma_5_mean',
    'delta chroma_9_mean', 'delta chroma_12_mean', 'delta zcr_mean',
    'delta energy_mean', 'delta spectral_rolloff_mean', 'delta mfcc_5_mean',
    'delta chroma_11_mean'
]

def align_emotions(train_df, val_df, test_df):
    """
    Κρατάει μόνο τα συναισθήματα που υπάρχουν και στα 3 datasets
    ΚΑΙ έχουν αρκετά δείγματα για SMOTE (>= 6 στο train).
    """
    train_emotions = set(train_df['emotion'].unique())
    val_emotions = set(val_df['emotion'].unique())
    test_emotions = set(test_df['emotion'].unique())
    common = train_emotions & val_emotions & test_emotions
    
    counts = train_df['emotion'].value_counts()
    valid_emotions = [e for e in common if counts[e] >= 6]
    
    removed = common - set(valid_emotions)
    if removed:
        print(f"  ⚠️ Removing rare emotions (n < 6 in train): {removed}")
    
    common_emotions = sorted(valid_emotions)
    print(f"  ✅ Final emotions: {common_emotions}")
    
    train_df = train_df[train_df['emotion'].isin(common_emotions)].copy()
    val_df = val_df[val_df['emotion'].isin(common_emotions)].copy()
    test_df = test_df[test_df['emotion'].isin(common_emotions)].copy()
    
    return train_df, val_df, test_df

def apply_log_transform(df, features):
    for feat in features:
        if feat in df.columns:
            min_val = df[feat].min()
            shift = abs(min_val) + 1 if min_val <= 0 else 0
            df[feat] = np.log1p(df[feat] + shift)
    return df

def apply_sqrt_transform(df, features):
    for feat in features:
        if feat in df.columns:
            df[feat] = np.sign(df[feat]) * np.sqrt(np.abs(df[feat]))
    return df

def correlation_based_feature_selection(X, y, max_features=None):
    """CFS Implementation"""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    feature_names = X.columns.tolist()
    
    correlations_with_class = [abs(np.corrcoef(X[feat].fillna(0), y_encoded)[0,1]) for feat in feature_names]
    corr_matrix = X.fillna(0).corr().abs()
    
    selected = [feature_names[np.argmax(correlations_with_class)]]
    remaining = [f for f in feature_names if f not in selected]
    
    if max_features is None:
        max_features = 100  # Increased from sqrt(N) to 100 for better representation

    while len(selected) < max_features and remaining:
        best_score = -1
        best_feat = None
        for feat in remaining:
            curr_selection = selected + [feat]
            k = len(curr_selection)
            rcf = sum(correlations_with_class[feature_names.index(f)] for f in curr_selection)
            rff = sum(corr_matrix.loc[f1, f2] for i, f1 in enumerate(curr_selection) for j, f2 in enumerate(curr_selection) if i < j)
            score = rcf / np.sqrt(k + 2 * rff) if (k + 2 * rff) > 0 else 0
            if score > best_score:
                best_score, best_feat = score, feat
        if best_feat:
            selected.append(best_feat)
            remaining.remove(best_feat)
        else: break
    return selected

def preprocess_fold(fold_num, args):
    print(f"\n--- PROCESSING FOLD {fold_num} ---")
    
    fold_in_dir = os.path.join(DATA_DIR, f'fold_{fold_num}', 'unprocessed')
    train_path = os.path.join(fold_in_dir, 'train.csv')
    val_path = os.path.join(fold_in_dir, 'val.csv')
    test_path = os.path.join(fold_in_dir, 'test.csv')
    
    if not os.path.exists(train_path):
        print(f"  ❌ Fold {fold_num} unprocessed files not found!")
        return

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    train_df, val_df, test_df = align_emotions(train_df, val_df, test_df)
    
    meta_cols = ['session', 'method', 'gender', 'emotion', 'n_annotators', 'agreement']
    
    def get_features_meta(df):
        return df.drop(columns=meta_cols), df[meta_cols]

    X_train, m_train = get_features_meta(train_df)
    X_val, m_val = get_features_meta(val_df)
    X_test, m_test = get_features_meta(test_df)
    
    X_train = X_train.drop(columns=[f for f in FEATURES_TO_REMOVE if f in X_train.columns])
    X_val = X_val.drop(columns=[f for f in FEATURES_TO_REMOVE if f in X_val.columns])
    X_test = X_test.drop(columns=[f for f in FEATURES_TO_REMOVE if f in X_test.columns])
    
    X_train = apply_log_transform(X_train, LOG_TRANSFORM_FEATURES)
    X_train = apply_sqrt_transform(X_train, SQRT_TRANSFORM_FEATURES)
    X_val = apply_log_transform(X_val, LOG_TRANSFORM_FEATURES)
    X_val = apply_sqrt_transform(X_val, SQRT_TRANSFORM_FEATURES)
    X_test = apply_log_transform(X_test, LOG_TRANSFORM_FEATURES)
    X_test = apply_sqrt_transform(X_test, SQRT_TRANSFORM_FEATURES)
    
    selected_features = X_train.columns.tolist()
    if args.mi:
        k = args.k if args.k else 60
        print(f"  Running Mutual Information feature selection (k={k})...")
        selector = SelectKBest(mutual_info_classif, k=k)
        X_train_arr = X_train.fillna(0)
        selector.fit(X_train_arr, m_train['emotion'])
        selected_features = X_train.columns[selector.get_support()].tolist()
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]
        print(f"  MI selected top {len(selected_features)} features")
    elif args.anova:
        k = args.k if args.k else 40
        selector = SelectKBest(f_classif, k=k)
        X_train_arr = X_train.fillna(0)
        selector.fit(X_train_arr, m_train['emotion'])
        selected_features = X_train.columns[selector.get_support()].tolist()
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]
        print(f"  ANOVA selected top {len(selected_features)} features (k={k})")
    elif args.cfs:
        selected_features = correlation_based_feature_selection(X_train, m_train['emotion'])
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]
        print(f"  CFS selected {len(selected_features)} features")

    # RESAMPLING
    if args.resample:
        if args.resample == 'smote':
            sampler = SMOTE(random_state=42)
        else:
            sampler = RandomOverSampler(random_state=42)
        
        X_res, y_res = sampler.fit_resample(X_train.fillna(0), m_train['emotion'])
        
        # Create new metadata for resampled data
        # Keep original metadata for original samples, 
        # and first available metadata of that emotion for synthetic ones
        n_original = len(X_train)
        new_rows = []
        for i in range(len(y_res)):
            if i < n_original:
                # Original sample: keep original metadata
                sample_meta = m_train.iloc[i].to_dict()
            else:
                # Synthetic sample: use first available metadata for this emotion
                emotion = y_res[i]
                sample_meta = m_train[m_train['emotion'] == emotion].iloc[0].to_dict()
                # Mark as synthetic (optional, but good for tracking)
                sample_meta['method'] = 'synthetic_' + (args.resample if args.resample else 'unknown')
            
            new_rows.append(sample_meta)
        
        m_train = pd.DataFrame(new_rows)
        X_train = pd.DataFrame(X_res, columns=selected_features)
        print(f"  Resampled Train size: {len(X_train)} (Balanced)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.fillna(0))
    X_val_scaled = scaler.transform(X_val.fillna(0))
    X_test_scaled = scaler.transform(X_test.fillna(0))
    
    fold_out_dir = os.path.join(DATA_DIR, f'fold_{fold_num}', 'processed')
    os.makedirs(fold_out_dir, exist_ok=True)
    
    def save_set(X, m, name):
        # Ensure alignment
        df_feat = pd.DataFrame(X, columns=selected_features)
        df_meta = m.reset_index(drop=True)
        df_final = pd.concat([df_meta, df_feat], axis=1)
        df_final.to_csv(os.path.join(fold_out_dir, f'{name}.csv'), index=False)
    
    save_set(X_train_scaled, m_train, 'train')
    save_set(X_val_scaled, m_val, 'val')
    save_set(X_test_scaled, m_test, 'test')
    
    joblib.dump(scaler, os.path.join(fold_out_dir, 'scaler.pkl'))
    joblib.dump(selected_features, os.path.join(fold_out_dir, 'selected_features.pkl'))
    print(f"  ✅ Fold {fold_num} saved to {fold_out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=[1,2,3,4,5], help='Specific fold')
    parser.add_argument('--all', action='store_true', help='Process all 5 folds')
    parser.add_argument('--cfs', action='store_true', help='Enable CFS')
    parser.add_argument('--anova', action='store_true', help='Enable ANOVA feature selection')
    parser.add_argument('--mi', action='store_true', help='Enable Mutual Information feature selection')
    parser.add_argument('--k', type=int, default=40, help='Number of features for Selection')
    parser.add_argument('--resample', type=str, choices=['smote', 'random'], help='Resampling method')
    args = parser.parse_args()
    
    folds = [args.fold] if args.fold else ([1,2,3,4,5] if args.all else [])
    
    if not folds:
        print("Please specify --fold X or --all")
        return

    for f in folds:
        preprocess_fold(f, args)

if __name__ == "__main__":
    main()
