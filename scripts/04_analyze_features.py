import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    project_root = Path(__file__).parent.parent
    features_path = project_root / "data" / "cremad_features.csv"
    reports_dir = project_root / "reports"
    plots_dir = project_root / "plots"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if not features_path.exists():
        print(f"ERROR: Features file not found: {features_path}")
        sys.exit(1)
    
    print("Loading features...")
    features_df = pd.read_csv(features_path)
    
    metadata_cols = ['filename', 'filepath', 'actor_id', 'sentence_code', 'emotion', 'intensity']
    if 'label' in features_df.columns:
        metadata_cols.append('label')
    
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    
    if 'emotion' in features_df.columns:
        emotion_counts = features_df['emotion'].value_counts()
        emotion_percentages = features_df['emotion'].value_counts(normalize=True) * 100
        
        print("\nEmotion Distribution:")
        for emotion, count in emotion_counts.items():
            pct = emotion_percentages[emotion]
            print(f"  {emotion}: {count} ({pct:.1f}%)")
        
        max_pct = emotion_percentages.max()
        min_pct = emotion_percentages.min()
        imbalance_ratio = max_pct / min_pct
        
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")
    
    numeric_feature_cols = features_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    missing = features_df[feature_cols].isnull().sum()
    missing_total = missing.sum()
    print(f"\nMissing values: {missing_total}")
    
    inf_count = np.isinf(features_df[feature_cols].select_dtypes(include=[np.number])).sum().sum()
    print(f"Infinite values: {inf_count}")
    
    print("\nAnalyzing variance...")
    numeric_cols = features_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    feature_variances = features_df[numeric_cols].var()
    variance_threshold = 0.01
    low_variance_features = feature_variances[feature_variances < variance_threshold]
    
    if len(low_variance_features) > 0:
        print(f"Found {len(low_variance_features)} low variance features")
    
    feature_ranges = features_df[numeric_cols].max() - features_df[numeric_cols].min()
    max_range = feature_ranges.max()
    min_range = feature_ranges.min()
    scale_ratio = max_range / min_range if min_range > 0 else float('inf')
    
    print(f"\nScale ratio: {scale_ratio:.2f}")
    
    print("\nAnalyzing correlations...")
    if len(numeric_feature_cols) > 100:
        sample_cols = numeric_feature_cols[:50]
    else:
        sample_cols = numeric_feature_cols
    
    corr_matrix = features_df[sample_cols].corr()
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.9:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))
    
    if high_corr_pairs:
        print(f"Found {len(high_corr_pairs)} highly correlated pairs")
    
    if 'emotion' in features_df.columns:
        print("\nCalculating feature importance...")
        X = features_df[numeric_cols].fillna(0)
        y = features_df['emotion']
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        feature_importance = pd.DataFrame({
            'feature': numeric_cols,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Features:")
        for idx, row in feature_importance.head(15).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        top_features = feature_importance.head(15)
    else:
        feature_importance = None
        top_features = None
    
    print("\nGenerating visualizations...")
    
    if 'emotion' in features_df.columns:
        sample_features = [col for col in numeric_feature_cols if 'mean' in col][:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feat in enumerate(sample_features):
            if i < len(axes):
                features_df.boxplot(column=feat, by='emotion', ax=axes[i])
                axes[i].set_title(f'{feat}')
                axes[i].set_xlabel('Emotion')
                axes[i].set_ylabel('Value')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Feature Distributions by Emotion', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_distribution_by_emotion.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={'label': 'Correlation'}
    )
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    if 'emotion' in features_df.columns:
        numeric_cols = features_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        X = features_df[numeric_cols].fillna(0)
        y = features_df['emotion']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 8))
        for emotion in y.unique():
            mask = y == emotion
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=emotion, alpha=0.6)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        plt.title('PCA Visualization by Emotion', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "pca_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    if top_features is not None and len(top_features) > 0:
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Mutual Information Score', fontsize=12)
        plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    if 'emotion' in features_df.columns:
        plt.figure(figsize=(10, 6))
        emotion_counts = features_df['emotion'].value_counts()
        plt.bar(emotion_counts.index, emotion_counts.values)
        plt.xlabel('Emotion', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title('Class Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / "class_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    report_path = reports_dir / "feature_analysis_report.md"
    
    report = f"""# Feature Analysis Report

## Dataset Overview

- **Total Samples**: {len(features_df)}
- **Total Features**: {len(feature_cols)}

## Feature Quality

- **Missing Values**: {missing_total}
- **Infinite Values**: {inf_count}
- **Low Variance Features**: {len(low_variance_features)}

## Class Imbalance

"""
    if 'emotion' in features_df.columns:
        emotion_counts = features_df['emotion'].value_counts()
        emotion_percentages = features_df['emotion'].value_counts(normalize=True) * 100
        max_pct = emotion_percentages.max()
        min_pct = emotion_percentages.min()
        imbalance_ratio = max_pct / min_pct
        
        report += f"- **Imbalance Ratio**: {imbalance_ratio:.2f}\n"
        report += f"- **Most Common**: {emotion_counts.index[0]} ({emotion_percentages.iloc[0]:.1f}%)\n"
        report += f"- **Least Common**: {emotion_counts.index[-1]} ({emotion_percentages.iloc[-1]:.1f}%)\n"
    
    report += f"""
## Feature Scaling

- **Scale Ratio**: {scale_ratio:.2f}

## Feature Importance

"""
    if feature_importance is not None:
        report += "### Top 15 Features:\n\n"
        for idx, row in feature_importance.head(15).iterrows():
            report += f"1. `{row['feature']}`: {row['importance']:.4f}\n"
    
    report += f"""
## Correlation Analysis

- **Highly Correlated Pairs**: {len(high_corr_pairs)}
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
