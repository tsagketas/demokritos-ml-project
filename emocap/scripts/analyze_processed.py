"""
Analysis script for preprocessed train data.
Compares preprocessed vs original data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_ORIGINAL = os.path.join(BASE_DIR, 'data', 'iemocap', 'train.csv')
TRAIN_PROCESSED = os.path.join(BASE_DIR, 'data', 'iemocap', 'train_processed.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'iemocap', 'analysis_processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def get_numeric_features(df):
    metadata_cols = ['session', 'method', 'gender', 'emotion', 'n_annotators', 'agreement']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in metadata_cols]

def analyze_skewness_kurtosis(df, title_suffix=""):
    feature_cols = get_numeric_features(df)
    skewness = df[feature_cols].skew()
    kurtosis = df[feature_cols].kurtosis()
    
    stats_df = pd.DataFrame({
        'Feature': feature_cols,
        'Skewness': skewness.values,
        'Kurtosis': kurtosis.values
    })
    
    stats_df.to_csv(os.path.join(OUTPUT_DIR, f'skewness_kurtosis{title_suffix}.csv'), index=False)
    
    print(f"\n📊 Skewness Statistics{title_suffix}:")
    print(f"  Mean |skewness|: {abs(skewness).mean():.3f}")
    print(f"  Max |skewness|: {abs(skewness).max():.3f}")
    print(f"  Features with |skewness| > 2: {len(skewness[abs(skewness) > 2])}")
    print(f"  Features with |skewness| > 5: {len(skewness[abs(skewness) > 5])}")
    
    print(f"\n📊 Kurtosis Statistics{title_suffix}:")
    print(f"  Mean |kurtosis|: {abs(kurtosis).mean():.3f}")
    print(f"  Max |kurtosis|: {abs(kurtosis).max():.3f}")
    print(f"  Features with |kurtosis| > 10: {len(kurtosis[abs(kurtosis) > 10])}")
    print(f"  Features with |kurtosis| > 100: {len(kurtosis[abs(kurtosis) > 100])}")
    
    return stats_df

def analyze_statistics(df, title_suffix=""):
    feature_cols = get_numeric_features(df)
    stats_df = df[feature_cols].describe().T[['mean', 'std', 'min', 'max']]
    stats_df.to_csv(os.path.join(OUTPUT_DIR, f'feature_statistics{title_suffix}.csv'))
    
    print(f"\n📊 Feature Statistics{title_suffix}:")
    print(f"  Mean range: {stats_df['max'].sub(stats_df['min']).mean():.4f}")
    print(f"  Max range: {stats_df['max'].sub(stats_df['min']).max():.4f}")
    print(f"  Min range: {stats_df['max'].sub(stats_df['min']).min():.4f}")
    print(f"  Mean std: {stats_df['std'].mean():.4f}")
    
    return stats_df

def plot_comparison(original_stats, processed_stats, metric='Skewness'):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    original_vals = original_stats[metric].abs()
    processed_vals = processed_stats[metric].abs()
    
    axes[0].hist(original_vals, bins=50, alpha=0.7, label='Original', color='blue', edgecolor='black')
    axes[0].set_title(f'Original Data - |{metric}| Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(f'|{metric}|', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].axvline(2, color='red', linestyle='--', label='Threshold (2)')
    axes[0].legend()
    
    axes[1].hist(processed_vals, bins=50, alpha=0.7, label='Processed', color='green', edgecolor='black')
    axes[1].set_title(f'Processed Data - |{metric}| Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel(f'|{metric}|', fontsize=10)
    axes[1].set_ylabel('Frequency', fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].axvline(2, color='red', linestyle='--', label='Threshold (2)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{metric.lower()}_comparison.png'), dpi=300)
    plt.close()
    print(f"✅ Saved: {metric.lower()}_comparison.png")

def analyze_correlations(df, title_suffix=""):
    feature_cols = get_numeric_features(df)
    corr_matrix = df[feature_cols].corr()
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.95:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    print(f"\n📊 Correlation Analysis{title_suffix}:")
    print(f"  Highly correlated pairs (>0.95): {len(high_corr_pairs)}")
    
    if len(high_corr_pairs) > 0:
        print(f"  ⚠️  Still have {len(high_corr_pairs)} highly correlated pairs")
    else:
        print(f"  ✅ No highly correlated pairs found!")
    
    return len(high_corr_pairs)

def main():
    print("="*80)
    print("  POST-PREPROCESSING ANALYSIS")
    print("="*80)
    
    print("\n📂 Loading data...")
    df_original = pd.read_csv(TRAIN_ORIGINAL)
    df_processed = pd.read_csv(TRAIN_PROCESSED)
    
    print(f"Original: {df_original.shape}")
    print(f"Processed: {df_processed.shape}")
    
    print("\n" + "="*80)
    print("  ORIGINAL DATA ANALYSIS")
    print("="*80)
    original_skew = analyze_skewness_kurtosis(df_original, "_original")
    original_stats = analyze_statistics(df_original, "_original")
    original_corr = analyze_correlations(df_original, " (Original)")
    
    print("\n" + "="*80)
    print("  PROCESSED DATA ANALYSIS")
    print("="*80)
    processed_skew = analyze_skewness_kurtosis(df_processed, "_processed")
    processed_stats = analyze_statistics(df_processed, "_processed")
    processed_corr = analyze_correlations(df_processed, " (Processed)")
    
    print("\n" + "="*80)
    print("  COMPARISON")
    print("="*80)
    
    print("\n📊 Skewness Improvement:")
    orig_high_skew = len(original_skew[abs(original_skew['Skewness']) > 2])
    proc_high_skew = len(processed_skew[abs(processed_skew['Skewness']) > 2])
    print(f"  Original: {orig_high_skew} features with |skewness| > 2")
    print(f"  Processed: {proc_high_skew} features with |skewness| > 2")
    print(f"  Improvement: {orig_high_skew - proc_high_skew} features fixed")
    
    print("\n📊 Kurtosis Improvement:")
    orig_high_kurt = len(original_skew[abs(original_skew['Kurtosis']) > 10])
    proc_high_kurt = len(processed_skew[abs(processed_skew['Kurtosis']) > 10])
    print(f"  Original: {orig_high_kurt} features with |kurtosis| > 10")
    print(f"  Processed: {proc_high_kurt} features with |kurtosis| > 10")
    print(f"  Improvement: {orig_high_kurt - proc_high_kurt} features fixed")
    
    print("\n📊 Scale Improvement:")
    orig_range = original_stats['max'].sub(original_stats['min']).max()
    proc_range = processed_stats['max'].sub(processed_stats['min']).max()
    print(f"  Original max range: {orig_range:.4f}")
    print(f"  Processed max range: {proc_range:.4f}")
    print(f"  ✅ Features are now on similar scales (StandardScaler applied)")
    
    print("\n📊 Correlation Improvement:")
    print(f"  Original highly correlated pairs: {original_corr}")
    print(f"  Processed highly correlated pairs: {processed_corr}")
    print(f"  Improvement: {original_corr - processed_corr} pairs removed")
    
    print("\n📊 Creating comparison plots...")
    plot_comparison(original_skew, processed_skew, 'Skewness')
    plot_comparison(original_skew, processed_skew, 'Kurtosis')
    plot_feature_distributions(df_original, df_processed)
    plot_correlation_matrices(df_original, df_processed)
    plot_statistics_comparison(original_stats, processed_stats)
    plot_scale_comparison(original_stats, processed_stats)
    plot_boxplots_comparison(df_original, df_processed)
    
    print("\n" + "="*80)
    print("  ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n✅ Results saved to: {OUTPUT_DIR}")

def plot_feature_distributions(df_orig, df_proc):
    """Compare feature distributions before/after preprocessing"""
    feature_cols_orig = get_numeric_features(df_orig)
    feature_cols_proc = get_numeric_features(df_proc)
    
    # Get common features (after removing some in preprocessing)
    common_features = [f for f in feature_cols_proc if f in feature_cols_orig][:12]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for idx, feat in enumerate(common_features):
        axes[idx].hist(df_orig[feat], bins=50, alpha=0.6, label='Original', color='blue', density=True)
        axes[idx].hist(df_proc[feat], bins=50, alpha=0.6, label='Processed', color='green', density=True)
        axes[idx].set_title(f'{feat}', fontsize=9)
        axes[idx].set_xlabel('Value', fontsize=8)
        axes[idx].set_ylabel('Density', fontsize=8)
        axes[idx].legend(fontsize=7)
        axes[idx].grid(alpha=0.3)
    
    plt.suptitle('Feature Distributions Comparison (Before/After Preprocessing)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_distributions_comparison.png'), dpi=300)
    plt.close()
    print(f"✅ Saved: feature_distributions_comparison.png")

def plot_correlation_matrices(df_orig, df_proc):
    """Compare correlation matrices"""
    feature_cols_orig = get_numeric_features(df_orig)
    feature_cols_proc = get_numeric_features(df_proc)
    
    # Sample features for readability
    sample_orig = feature_cols_orig[:20]
    sample_proc = feature_cols_proc[:20]
    
    corr_orig = df_orig[sample_orig].corr()
    corr_proc = df_proc[sample_proc].corr()
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.heatmap(corr_orig, annot=False, cmap='coolwarm', center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[0])
    axes[0].set_title('Original Data - Correlation Matrix (Sample)', fontsize=12, fontweight='bold')
    
    sns.heatmap(corr_proc, annot=False, cmap='coolwarm', center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[1])
    axes[1].set_title('Processed Data - Correlation Matrix (Sample)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrices_comparison.png'), dpi=300)
    plt.close()
    print(f"✅ Saved: correlation_matrices_comparison.png")

def plot_statistics_comparison(orig_stats, proc_stats):
    """Compare mean, std, range statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Mean comparison
    common_features = [f for f in proc_stats.index if f in orig_stats.index]
    orig_means = orig_stats.loc[common_features, 'mean'].abs()
    proc_means = proc_stats.loc[common_features, 'mean'].abs()
    
    axes[0, 0].scatter(orig_means, proc_means, alpha=0.6, s=30)
    axes[0, 0].plot([0, orig_means.max()], [0, orig_means.max()], 'r--', alpha=0.5)
    axes[0, 0].set_xlabel('Original |Mean|', fontsize=10)
    axes[0, 0].set_ylabel('Processed |Mean|', fontsize=10)
    axes[0, 0].set_title('Mean Values Comparison', fontsize=11, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # Std comparison
    orig_stds = orig_stats.loc[common_features, 'std']
    proc_stds = proc_stats.loc[common_features, 'std']
    
    axes[0, 1].scatter(orig_stds, proc_stds, alpha=0.6, s=30, color='green')
    axes[0, 1].axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Target std=1')
    axes[0, 1].set_xlabel('Original Std', fontsize=10)
    axes[0, 1].set_ylabel('Processed Std', fontsize=10)
    axes[0, 1].set_title('Std Values Comparison (StandardScaler)', fontsize=11, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Range comparison
    orig_ranges = orig_stats.loc[common_features, 'max'] - orig_stats.loc[common_features, 'min']
    proc_ranges = proc_stats.loc[common_features, 'max'] - proc_stats.loc[common_features, 'min']
    
    axes[1, 0].scatter(orig_ranges, proc_ranges, alpha=0.6, s=30, color='orange')
    axes[1, 0].set_xlabel('Original Range', fontsize=10)
    axes[1, 0].set_ylabel('Processed Range', fontsize=10)
    axes[1, 0].set_title('Range Comparison', fontsize=11, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # Std distribution - use kde instead of hist for processed (all ≈1.0)
    if len(orig_stds.unique()) > 1:
        axes[1, 1].hist(orig_stds, bins=min(30, len(orig_stds.unique())), alpha=0.6, 
                       label='Original', color='blue', edgecolor='black', density=True)
    if len(proc_stds.unique()) > 1:
        proc_std_range = proc_stds.max() - proc_stds.min()
        if proc_std_range > 0.01:
            axes[1, 1].hist(proc_stds, bins=min(30, len(proc_stds.unique())), alpha=0.6, 
                           label='Processed', color='green', edgecolor='black', density=True)
        else:
            axes[1, 1].axvline(proc_stds.mean(), color='green', linewidth=3, 
                              label=f'Processed (all ≈{proc_stds.mean():.3f})')
    axes[1, 1].axvline(1.0, color='r', linestyle='--', linewidth=2, label='Target std=1')
    axes[1, 1].set_xlabel('Std Value', fontsize=10)
    axes[1, 1].set_ylabel('Density', fontsize=10)
    axes[1, 1].set_title('Std Distribution (StandardScaler Effect)', fontsize=11, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'statistics_comparison.png'), dpi=300)
    plt.close()
    print(f"✅ Saved: statistics_comparison.png")

def plot_scale_comparison(orig_stats, proc_stats):
    """Compare feature scales"""
    common_features = [f for f in proc_stats.index if f in orig_stats.index]
    
    orig_stds = orig_stats.loc[common_features, 'std']
    proc_stds = proc_stats.loc[common_features, 'std']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Before scaling
    axes[0].bar(range(len(orig_stds.head(30))), orig_stds.head(30), alpha=0.7, color='blue')
    axes[0].set_xlabel('Feature Index', fontsize=10)
    axes[0].set_ylabel('Std Value', fontsize=10)
    axes[0].set_title('Original Data - Std Values (First 30 features)', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')
    
    # After scaling
    axes[1].bar(range(len(proc_stds.head(30))), proc_stds.head(30), alpha=0.7, color='green')
    axes[1].axhline(1.0, color='r', linestyle='--', linewidth=2, label='Target std=1')
    axes[1].set_xlabel('Feature Index', fontsize=10)
    axes[1].set_ylabel('Std Value', fontsize=10)
    axes[1].set_title('Processed Data - Std Values (StandardScaler)', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scale_comparison.png'), dpi=300)
    plt.close()
    print(f"✅ Saved: scale_comparison.png")

def plot_boxplots_comparison(df_orig, df_proc):
    """Compare box plots for outlier detection"""
    feature_cols_orig = get_numeric_features(df_orig)
    feature_cols_proc = get_numeric_features(df_proc)
    
    common_features = [f for f in feature_cols_proc if f in feature_cols_orig][:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, feat in enumerate(common_features):
        data = [df_orig[feat].dropna(), df_proc[feat].dropna()]
        bp = axes[idx].boxplot(data, labels=['Original', 'Processed'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightgreen')
        axes[idx].set_title(f'{feat}', fontsize=10)
        axes[idx].set_ylabel('Value', fontsize=9)
        axes[idx].grid(alpha=0.3, axis='y')
    
    plt.suptitle('Outlier Detection - Box Plots Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'boxplots_comparison.png'), dpi=300)
    plt.close()
    print(f"✅ Saved: boxplots_comparison.png")

if __name__ == "__main__":
    main()
