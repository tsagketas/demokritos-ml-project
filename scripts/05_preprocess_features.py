import sys
import argparse
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, str(Path(__file__).parent.parent))


def remove_correlated_features(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > threshold)]
    df_cleaned = df.drop(columns=to_drop)
    return df_cleaned, to_drop


def clip_outliers(X, n_std=3):
    X_clipped = X.copy()
    clipping_stats = {}
    
    for col in X.columns:
        mean = X[col].mean()
        std = X[col].std()
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        
        outliers_below = (X[col] < lower_bound).sum()
        outliers_above = (X[col] > upper_bound).sum()
        
        X_clipped[col] = X_clipped[col].clip(lower=lower_bound, upper=upper_bound)
        
        clipping_stats[col] = {
            'outliers_below': outliers_below,
            'outliers_above': outliers_above,
            'total_outliers': outliers_below + outliers_above
        }
    
    return X_clipped, clipping_stats


def add_feature_interactions(X, top_n=5):
    X_with_interactions = X.copy()
    interaction_features = []
    
    feature_variances = X.var().sort_values(ascending=False)
    top_features = feature_variances.head(top_n).index.tolist()
    
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i+1:]:
            interaction_name = f"{feat1}_x_{feat2}"
            X_with_interactions[interaction_name] = X[feat1] * X[feat2]
            interaction_features.append(interaction_name)
    
    return X_with_interactions, interaction_features


def main():
    parser = argparse.ArgumentParser(description='Preprocess features for model training')
    parser.add_argument('--scaler', type=str, default='standard', choices=['standard', 'robust'],
                       help='Scaler to use: standard (mean/std) or robust (median/IQR)')
    parser.add_argument('--clip-outliers', action='store_true',
                       help='Clip outliers beyond N standard deviations')
    parser.add_argument('--clip-n-std', type=float, default=3.0,
                       help='Number of standard deviations for outlier clipping')
    parser.add_argument('--add-interactions', action='store_true',
                       help='Add feature interactions for top features')
    parser.add_argument('--interaction-top-n', type=int, default=5,
                       help='Number of top features to create interactions for')
    parser.add_argument('--mi-percentile', type=float, default=25.0,
                       help='Percentile threshold for Mutual Information')
    parser.add_argument('--variance-threshold', type=float, default=0.001,
                       help='Variance threshold for removing low variance features')
    parser.add_argument('--correlation-threshold', type=float, default=0.9,
                       help='Correlation threshold for removing highly correlated features')
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    features_path = project_root / "data" / "cremad_features.csv"
    output_path = project_root / "data" / "cremad_features_preprocessed.csv"
    scaler_path = project_root / "data" / "scaler.pkl"
    reports_dir = project_root / "reports"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    if not features_path.exists():
        print(f"ERROR: Features file not found: {features_path}")
        sys.exit(1)
    
    print("Loading features...")
    features_df = pd.read_csv(features_path)
    
    metadata_cols = ['filename', 'filepath', 'actor_id', 'sentence_code', 'emotion', 'intensity']
    if 'label' in features_df.columns:
        metadata_cols.append('label')
    if 'duration' in features_df.columns:
        metadata_cols.append('duration')
    
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    
    metadata_df = features_df[metadata_cols].copy()
    X = features_df[feature_cols].copy()
    
    original_feature_count = len(feature_cols)
    
    important_features = []
    mi_threshold = None
    if 'emotion' in features_df.columns:
        print("Calculating feature importance...")
        y = features_df['emotion']
        X_filled = X.fillna(0)
        
        mi_scores = mutual_info_classif(X_filled, y, random_state=42)
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        mi_threshold = np.percentile(mi_scores, args.mi_percentile)
        important_features = feature_importance[
            feature_importance['importance'] >= mi_threshold
        ]['feature'].tolist()
    
    print("Removing low variance features...")
    feature_variances = X.var()
    low_variance_threshold = args.variance_threshold
    
    selected_features = []
    removed_variance = []
    
    for feat in feature_cols:
        if feat in important_features:
            selected_features.append(feat)
        elif feature_variances[feat] >= low_variance_threshold:
            selected_features.append(feat)
        else:
            removed_variance.append(feat)
    
    X_variance_df = X[selected_features].copy()
    
    print("Removing highly correlated features...")
    X_corr, removed_corr = remove_correlated_features(X_variance_df, threshold=args.correlation_threshold)
    
    final_feature_cols = X_corr.columns.tolist()
    
    clipping_stats = None
    if args.clip_outliers:
        print("Clipping outliers...")
        X_corr, clipping_stats = clip_outliers(X_corr, n_std=args.clip_n_std)
    
    interaction_features = []
    if args.add_interactions:
        print("Adding feature interactions...")
        X_corr, interaction_features = add_feature_interactions(X_corr, top_n=args.interaction_top_n)
        final_feature_cols = X_corr.columns.tolist()
    
    print("Scaling features...")
    if args.scaler == 'robust':
        scaler = RobustScaler()
        scaler_name = "RobustScaler"
    else:
        scaler = StandardScaler()
        scaler_name = "StandardScaler"
    
    X_scaled = scaler.fit_transform(X_corr)
    X_scaled_df = pd.DataFrame(X_scaled, columns=final_feature_cols, index=X_corr.index)
    
    preprocessed_df = pd.concat([metadata_df, X_scaled_df], axis=1)
    
    metadata_cols_final = [col for col in metadata_cols if col in preprocessed_df.columns]
    feature_cols_final = [col for col in final_feature_cols if col in preprocessed_df.columns]
    preprocessed_df = preprocessed_df[metadata_cols_final + feature_cols_final]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessed_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved: {scaler_path}")
    
    reduction_pct = ((original_feature_count - len(final_feature_cols)) / original_feature_count * 100)
    
    report_path = reports_dir / "preprocessing_report.md"
    
    report = f"""# Feature Preprocessing Report

## Summary

- **Scaler**: {scaler_name}
- **Outlier Clipping**: {'Yes' if args.clip_outliers else 'No'}
- **Feature Interactions**: {'Yes' if args.add_interactions else 'No'}
- **MI Percentile Threshold**: {args.mi_percentile}%
- **Variance Threshold**: {args.variance_threshold}
- **Correlation Threshold**: {args.correlation_threshold}

## Preprocessing Steps

### Feature Selection

- **MI Threshold**: {f'{mi_threshold:.6f}' if mi_threshold is not None else 'N/A'}
- **Important features kept**: {len(important_features)}
- **Removed low variance**: {len(removed_variance)}
- **Removed correlated**: {len(removed_corr)}
- **Added interactions**: {len(interaction_features) if args.add_interactions else 0}

### Feature Scaling

- **Method**: {scaler_name}
- **Features scaled**: {len(final_feature_cols)}

## Final Dataset

- **Total samples**: {len(preprocessed_df)}
- **Total features**: {len(final_feature_cols)}
- **Feature reduction**: {original_feature_count} -> {len(final_feature_cols)} ({reduction_pct:.1f}%)
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nOriginal features: {original_feature_count}")
    print(f"Final features: {len(final_feature_cols)}")
    print(f"Reduction: {reduction_pct:.1f}%")


if __name__ == "__main__":
    main()
