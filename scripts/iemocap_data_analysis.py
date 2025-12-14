"""
Data Analysis and Visualization Script for IEMOCAP Features
Δημιουργεί comprehensive report με metrics, statistics και visualizations

Usage:
    python scripts/iemocap_data_analysis.py --input datasets/iemocap_features.pkl --output reports/
    
    # Με processed features (για before/after scaling comparison)
    python scripts/iemocap_data_analysis.py --input datasets/iemocap_features_processed.pkl --output reports/ --processed
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def get_feature_columns(df):
    """Επιστρέφει τη λίστα των feature columns (όχι metadata) - μόνο numeric"""
    metadata_cols = ['session', 'method', 'gender', 'emotion', 
                     'n_annotators', 'agreement', 'audio_path']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    # Filter only numeric columns (exclude object/array types)
    numeric_cols = []
    for col in feature_cols:
        try:
            # Try to convert to numeric - if it works, it's numeric
            pd.to_numeric(df[col], errors='raise')
            numeric_cols.append(col)
        except (ValueError, TypeError):
            # Skip non-numeric columns
            continue
    return numeric_cols


def analyze_class_distribution(df, output_dir):
    """
    1. DATA DISTRIBUTION ANALYSIS
    - Class distribution (emotion labels)
    - Train/test split distribution
    - Session/gender/method distribution
    """
    print("\n" + "="*60)
    print("1. DATA DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # 1.1 Emotion Class Distribution
    if 'emotion' in df.columns:
        print("\n1.1 Emotion Class Distribution")
        emotion_counts = df['emotion'].value_counts().sort_index()
        emotion_percentages = (emotion_counts / len(df) * 100).round(2)
        
        print(f"Total samples: {len(df)}")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count} ({emotion_percentages[emotion]}%)")
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        emotion_counts.plot(kind='bar', ax=ax1, color='steelblue')
        ax1.set_title('Emotion Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Emotion', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Pie chart
        emotion_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Emotion Class Distribution (%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1_class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {os.path.join(output_dir, '1_class_distribution.png')}")
        
        # Save to CSV
        emotion_stats = pd.DataFrame({
            'emotion': emotion_counts.index,
            'count': emotion_counts.values,
            'percentage': emotion_percentages.values
        })
        emotion_stats.to_csv(os.path.join(output_dir, '1_class_distribution.csv'), index=False)
    
    # 1.2 Train/Test Split Distribution (αν υπάρχει split)
    if 'split' in df.columns or any('train' in str(c).lower() or 'test' in str(c).lower() for c in df.columns):
        print("\n1.2 Train/Test Split Distribution")
        # This will be handled if splits are loaded separately
        pass
    
    # 1.3 Session Distribution
    if 'session' in df.columns:
        print("\n1.3 Session Distribution")
        session_counts = df['session'].value_counts().sort_index()
        print(session_counts.to_string())
        
        plt.figure(figsize=(10, 6))
        session_counts.plot(kind='bar', color='coral')
        plt.title('Session Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Session', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1_session_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {os.path.join(output_dir, '1_session_distribution.png')}")
    
    # 1.4 Gender Distribution
    if 'gender' in df.columns:
        print("\n1.4 Gender Distribution")
        gender_counts = df['gender'].value_counts()
        print(gender_counts.to_string())
        
        plt.figure(figsize=(8, 6))
        gender_counts.plot(kind='bar', color=['pink', 'lightblue'])
        plt.title('Gender Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Gender', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1_gender_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {os.path.join(output_dir, '1_gender_distribution.png')}")
    
    # 1.5 Method Distribution
    if 'method' in df.columns:
        print("\n1.5 Method Distribution")
        method_counts = df['method'].value_counts()
        print(method_counts.to_string())
        
        plt.figure(figsize=(8, 6))
        method_counts.plot(kind='bar', color='lightgreen')
        plt.title('Method Distribution (Script vs Improvisation)', fontsize=14, fontweight='bold')
        plt.xlabel('Method', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1_method_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {os.path.join(output_dir, "1_method_distribution.png")}")


def analyze_feature_statistics(df, output_dir):
    """
    2. FEATURE STATISTICS
    - Mean, std, min, max per feature
    - Missing values count/percentage
    - Infinite values count
    - Zero variance features
    """
    print("\n" + "="*60)
    print("2. FEATURE STATISTICS")
    print("="*60)
    
    feature_cols = get_feature_columns(df)
    df_features = df[feature_cols]
    
    # 2.1 Basic Statistics
    print("\n2.1 Basic Feature Statistics")
    stats_df = df_features.describe().T
    stats_df['variance'] = df_features.var()
    stats_df['skewness'] = df_features.skew()
    stats_df['kurtosis'] = df_features.kurtosis()
    
    # Save statistics
    stats_df.to_csv(os.path.join(output_dir, '2_feature_statistics.csv'))
    print(f"  [OK] Saved: {os.path.join(output_dir, "2_feature_statistics.csv")}")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Mean statistics range: [{stats_df['mean'].min():.4f}, {stats_df['mean'].max():.4f}]")
    print(f"  Std statistics range: [{stats_df['std'].min():.4f}, {stats_df['std'].max():.4f}]")
    
    # 2.2 Missing Values
    print("\n2.2 Missing Values Analysis")
    missing_data = df_features.isnull().sum()
    missing_percent = (missing_data / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({
        'feature': missing_data.index,
        'missing_count': missing_data.values,
        'missing_percentage': missing_percent.values
    })
    missing_summary = missing_summary[missing_summary['missing_count'] > 0].sort_values('missing_count', ascending=False)
    
    if len(missing_summary) > 0:
        print(f"  Features with missing values: {len(missing_summary)}")
        print(f"  Total missing values: {missing_data.sum()}")
        missing_summary.to_csv(os.path.join(output_dir, '2_missing_values.csv'), index=False)
        print(f"  [OK] Saved: {os.path.join(output_dir, "2_missing_values.csv")}")
        
        # Visualization
        if len(missing_summary) > 0:
            top_missing = missing_summary.head(20)
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_missing)), top_missing['missing_count'].values)
            plt.yticks(range(len(top_missing)), top_missing['feature'].values)
            plt.xlabel('Missing Count', fontsize=12)
            plt.title('Top 20 Features with Missing Values', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '2_missing_values.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  [OK] Saved: {os.path.join(output_dir, "2_missing_values.png")}")
    else:
        print("  [OK] No missing values found")
    
    # 2.3 Infinite Values
    print("\n2.3 Infinite Values Analysis")
    inf_data = np.isinf(df_features).sum()
    inf_summary = pd.DataFrame({
        'feature': inf_data.index,
        'infinite_count': inf_data.values
    })
    inf_summary = inf_summary[inf_summary['infinite_count'] > 0].sort_values('infinite_count', ascending=False)
    
    if len(inf_summary) > 0:
        print(f"  Features with infinite values: {len(inf_summary)}")
        print(f"  Total infinite values: {inf_data.sum()}")
        inf_summary.to_csv(os.path.join(output_dir, '2_infinite_values.csv'), index=False)
        print(f"  [OK] Saved: {os.path.join(output_dir, "2_infinite_values.csv")}")
    else:
        print("  [OK] No infinite values found")
    
    # 2.4 Zero Variance Features
    print("\n2.4 Zero Variance Features")
    variances = df_features.var()
    zero_var_features = variances[variances == 0].index.tolist()
    
    if len(zero_var_features) > 0:
        print(f"  Zero variance features: {len(zero_var_features)}")
        zero_var_df = pd.DataFrame({'feature': zero_var_features})
        zero_var_df.to_csv(os.path.join(output_dir, '2_zero_variance_features.csv'), index=False)
        print(f"  [OK] Saved: {os.path.join(output_dir, "2_zero_variance_features.csv")}")
        print(f"  Features: {', '.join(zero_var_features[:10])}{'...' if len(zero_var_features) > 10 else ''}")
    else:
        print("  [OK] No zero variance features found")


def create_visualizations(df, output_dir):
    """
    3. VISUALIZATIONS
    - Correlation matrix (feature correlations)
    - Feature distributions (histograms/box plots)
    - Outlier detection (box plots, IQR)
    - Class imbalance (bar charts)
    """
    print("\n" + "="*60)
    print("3. VISUALIZATIONS")
    print("="*60)
    
    feature_cols = get_feature_columns(df)
    df_features = df[feature_cols]
    
    # 3.1 Correlation Matrix
    print("\n3.1 Correlation Matrix")
    # Sample features για correlation (για performance)
    if len(feature_cols) > 50:
        # Select diverse features
        sample_size = 50
        # Select first, middle, and last features
        step = len(feature_cols) // sample_size
        sample_features = feature_cols[::step][:sample_size]
        print(f"  Computing correlation for {len(sample_features)} features (sampled from {len(feature_cols)})")
    else:
        sample_features = feature_cols
        print(f"  Computing correlation for {len(sample_features)} features")
    
    corr_matrix = df[sample_features].corr()
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.1, cbar_kws={"shrink": 0.8},
                xticklabels=False, yticklabels=False)
    plt.title('Feature Correlation Matrix (Sample)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {os.path.join(output_dir, "3_correlation_matrix.png")}")
    
    # Save correlation matrix
    corr_matrix.to_csv(os.path.join(output_dir, '3_correlation_matrix.csv'))
    
    # 3.2 Feature Distributions (Histograms)
    print("\n3.2 Feature Distributions (Histograms)")
    # Select diverse features for visualization
    num_features_to_plot = min(12, len(feature_cols))
    step = len(feature_cols) // num_features_to_plot
    features_to_plot = feature_cols[::step][:num_features_to_plot]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, feat in enumerate(features_to_plot):
        if i < len(axes):
            df[feat].hist(bins=50, ax=axes[i], color='steelblue', alpha=0.7)
            axes[i].set_title(f'{feat[:40]}...' if len(feat) > 40 else feat, fontsize=10)
            axes[i].set_xlabel('Value', fontsize=9)
            axes[i].set_ylabel('Frequency', fontsize=9)
            axes[i].grid(alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(features_to_plot), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Feature Distributions (Sample)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {os.path.join(output_dir, "3_feature_distributions.png")}")
    
    # 3.3 Box Plots for Outlier Detection
    print("\n3.3 Outlier Detection (Box Plots)")
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, feat in enumerate(features_to_plot):
        if i < len(axes):
            df.boxplot(column=feat, ax=axes[i])
            axes[i].set_title(f'{feat[:40]}...' if len(feat) > 40 else feat, fontsize=10)
            axes[i].tick_params(axis='x', rotation=45, labelsize=8)
            axes[i].grid(alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(features_to_plot), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Outlier Detection - Box Plots (Sample)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_outlier_detection.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {os.path.join(output_dir, "3_outlier_detection.png")}")
    
    # 3.4 IQR-based Outlier Analysis
    print("\n3.4 IQR-based Outlier Analysis")
    outlier_summary = []
    for feat in feature_cols[:100]:  # Analyze first 100 features
        Q1 = df[feat].quantile(0.25)
        Q3 = df[feat].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[feat] < lower_bound) | (df[feat] > upper_bound)).sum()
        outlier_summary.append({
            'feature': feat,
            'outliers_count': outliers,
            'outliers_percentage': (outliers / len(df) * 100).round(2),
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    outlier_df = outlier_df[outlier_df['outliers_count'] > 0].sort_values('outliers_count', ascending=False)
    outlier_df.to_csv(os.path.join(output_dir, '3_outlier_analysis.csv'), index=False)
    print(f"  [OK] Saved: {os.path.join(output_dir, "3_outlier_analysis.csv")}")
    if len(outlier_df) > 0:
        print(f"  Features with outliers: {len(outlier_df)}")
        print(f"  Top 5 features with most outliers:")
        for _, row in outlier_df.head(5).iterrows():
            print(f"    {row['feature']}: {row['outliers_count']} ({row['outliers_percentage']}%)")


def analyze_data_quality(df, output_dir):
    """
    4. DATA QUALITY
    - Missing values heatmap
    - Feature variance analysis
    - Skewness/kurtosis
    """
    print("\n" + "="*60)
    print("4. DATA QUALITY ANALYSIS")
    print("="*60)
    
    feature_cols = get_feature_columns(df)
    df_features = df[feature_cols]
    
    # 4.1 Missing Values Heatmap
    print("\n4.1 Missing Values Heatmap")
    missing_data = df_features.isnull()
    if missing_data.sum().sum() > 0:
        # Sample features with missing values
        features_with_missing = missing_data.columns[missing_data.sum() > 0]
        if len(features_with_missing) > 0:
            sample_size = min(50, len(features_with_missing))
            features_to_plot = features_with_missing[:sample_size]
            
            plt.figure(figsize=(12, max(8, len(features_to_plot) * 0.3)))
            sns.heatmap(missing_data[features_to_plot].T, 
                       cbar=True, yticklabels=True, xticklabels=False,
                       cmap='YlOrRd', cbar_kws={'label': 'Missing'})
            plt.title('Missing Values Heatmap (Sample Features)', fontsize=14, fontweight='bold')
            plt.ylabel('Features', fontsize=12)
            plt.xlabel('Samples', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '4_missing_values_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  [OK] Saved: {os.path.join(output_dir, "4_missing_values_heatmap.png")}")
    else:
        print("  [OK] No missing values to visualize")
    
    # 4.2 Feature Variance Analysis
    print("\n4.2 Feature Variance Analysis")
    variances = df_features.var().sort_values(ascending=False)
    
    plt.figure(figsize=(14, 8))
    top_variances = variances.head(30)
    plt.barh(range(len(top_variances)), top_variances.values)
    plt.yticks(range(len(top_variances)), top_variances.index)
    plt.xlabel('Variance', fontsize=12)
    plt.title('Top 30 Features by Variance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_variance_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {os.path.join(output_dir, "4_variance_analysis.png")}")
    
    # Variance statistics
    variance_stats = pd.DataFrame({
        'feature': variances.index,
        'variance': variances.values
    })
    variance_stats.to_csv(os.path.join(output_dir, '4_variance_analysis.csv'), index=False)
    print(f"  Mean variance: {variances.mean():.4f}")
    print(f"  Median variance: {variances.median():.4f}")
    print(f"  Min variance: {variances.min():.4f}")
    print(f"  Max variance: {variances.max():.4f}")
    
    # 4.3 Skewness and Kurtosis Analysis
    print("\n4.3 Skewness and Kurtosis Analysis")
    skewness = df_features.skew()
    kurtosis = df_features.kurtosis()
    
    quality_df = pd.DataFrame({
        'feature': feature_cols,
        'skewness': skewness.values,
        'kurtosis': kurtosis.values,
        'variance': variances.values
    })
    quality_df.to_csv(os.path.join(output_dir, '4_skewness_kurtosis.csv'), index=False)
    print(f"  [OK] Saved: {os.path.join(output_dir, "4_skewness_kurtosis.csv")}")
    
    # Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Skewness distribution
    ax1.hist(skewness.values, bins=50, color='steelblue', alpha=0.7)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Normal (0)')
    ax1.set_xlabel('Skewness', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Skewness Values', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Kurtosis distribution
    ax2.hist(kurtosis.values, bins=50, color='coral', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Normal (0)')
    ax2.set_xlabel('Kurtosis', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Kurtosis Values', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_skewness_kurtosis_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {os.path.join(output_dir, "4_skewness_kurtosis_distribution.png")}")
    
    print(f"\n  Skewness statistics:")
    print(f"    Mean: {skewness.mean():.4f}")
    print(f"    Std: {skewness.std():.4f}")
    print(f"    Min: {skewness.min():.4f}")
    print(f"    Max: {skewness.max():.4f}")
    print(f"    Features with |skewness| > 2: {(abs(skewness) > 2).sum()}")
    
    print(f"\n  Kurtosis statistics:")
    print(f"    Mean: {kurtosis.mean():.4f}")
    print(f"    Std: {kurtosis.std():.4f}")
    print(f"    Min: {kurtosis.min():.4f}")
    print(f"    Max: {kurtosis.max():.4f}")
    print(f"    Features with |kurtosis| > 2: {(abs(kurtosis) > 2).sum()}")


def compare_before_after(original_path, processed_path, output_dir):
    """
    Before/After Scaling Comparison
    """
    print("\n" + "="*60)
    print("BEFORE/AFTER SCALING COMPARISON")
    print("="*60)
    
    df_original = pd.read_pickle(original_path)
    df_processed = pd.read_pickle(processed_path)
    
    original_features = get_feature_columns(df_original)
    processed_features = get_feature_columns(df_processed)
    
    # Find common features
    common_features = [f for f in original_features if f in processed_features][:6]
    
    if len(common_features) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feat in enumerate(common_features):
            if i < len(axes):
                axes[i].hist(df_original[feat].values, bins=50, alpha=0.5, 
                           label='Before Scaling', color='blue')
                axes[i].hist(df_processed[feat].values, bins=50, alpha=0.5, 
                           label='After Scaling', color='red')
                axes[i].set_title(f'{feat[:40]}...' if len(feat) > 40 else feat, fontsize=10)
                axes[i].set_xlabel('Value', fontsize=9)
                axes[i].set_ylabel('Frequency', fontsize=9)
                axes[i].legend()
                axes[i].grid(alpha=0.3)
        
        plt.suptitle('Before/After Scaling Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'before_after_scaling.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {os.path.join(output_dir, "before_after_scaling.png")}")


def generate_summary_report(df, output_dir):
    """Δημιουργεί text summary report"""
    feature_cols = get_feature_columns(df)
    
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("IEMOCAP DATA ANALYSIS SUMMARY REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Dataset Overview:\n")
        f.write(f"  Total Samples: {len(df)}\n")
        f.write(f"  Total Features: {len(feature_cols)}\n")
        f.write(f"  Metadata Columns: {len(df.columns) - len(feature_cols)}\n\n")
        
        if 'emotion' in df.columns:
            f.write(f"Class Distribution:\n")
            emotion_counts = df['emotion'].value_counts()
            for emotion, count in emotion_counts.items():
                percentage = (count / len(df) * 100)
                f.write(f"  {emotion}: {count} ({percentage:.2f}%)\n")
            f.write("\n")
        
        # Missing values
        missing_count = df[feature_cols].isnull().sum().sum()
        f.write(f"Data Quality:\n")
        f.write(f"  Missing Values: {missing_count}\n")
        f.write(f"  Infinite Values: {np.isinf(df[feature_cols]).sum().sum()}\n")
        f.write(f"  Zero Variance Features: {(df[feature_cols].var() == 0).sum()}\n\n")
        
        # Statistics
        f.write(f"Feature Statistics:\n")
        f.write(f"  Mean range: [{df[feature_cols].mean().min():.4f}, {df[feature_cols].mean().max():.4f}]\n")
        f.write(f"  Std range: [{df[feature_cols].std().min():.4f}, {df[feature_cols].std().max():.4f}]\n")
        f.write(f"  Variance range: [{df[feature_cols].var().min():.4f}, {df[feature_cols].var().max():.4f}]\n")
        f.write(f"  Skewness range: [{df[feature_cols].skew().min():.4f}, {df[feature_cols].skew().max():.4f}]\n")
        f.write(f"  Kurtosis range: [{df[feature_cols].kurtosis().min():.4f}, {df[feature_cols].kurtosis().max():.4f}]\n")
    
    print(f"\n  [OK] Saved: {os.path.join(output_dir, "summary_report.txt")}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive data analysis report for IEMOCAP features'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input .pkl file with features'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports',
        help='Output directory for reports (default: reports/)'
    )
    parser.add_argument(
        '--original',
        type=str,
        default=None,
        help='Original (unprocessed) .pkl file for before/after comparison'
    )
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("IEMOCAP DATA ANALYSIS")
    print("="*60)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_pickle(args.input)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Run all analyses
    analyze_class_distribution(df, args.output)
    analyze_feature_statistics(df, args.output)
    create_visualizations(df, args.output)
    analyze_data_quality(df, args.output)
    generate_summary_report(df, args.output)
    
    # Before/after comparison if original file provided
    if args.original and os.path.exists(args.original):
        compare_before_after(args.original, args.input, args.output)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"All reports saved to: {args.output}/")
    print("\nGenerated files:")
    print("  - Class distribution plots and CSV")
    print("  - Feature statistics CSV")
    print("  - Missing/infinite values analysis")
    print("  - Correlation matrix")
    print("  - Feature distributions")
    print("  - Outlier detection")
    print("  - Data quality metrics")
    print("  - Summary report")
    print("="*60)


if __name__ == "__main__":
    main()

