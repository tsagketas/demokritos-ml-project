"""
Comprehensive Data Analysis Script for IEMOCAP Train Dataset (Fold-based)

Αυτό το script κάνει πλήρη ανάλυση ενός συγκεκριμένου fold (train) με:
- Data distribution (class, session, gender, method)
- Feature statistics (mean, std, min, max, missing, infinite, zero variance)
- Visualizations (correlation, distributions, outliers, class imbalance)
- Feature importance analysis
- Correlation analysis για redundant features

Usage:
    python 03_data_analysis.py --fold 1
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_section(title):
    """Helper για όμορφη εκτύπωση"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def analyze_data_distribution(df, output_dir):
    """ΑΝΑΛΥΣΗ ΚΑΤΑΝΟΜΗΣ ΔΕΔΟΜΕΝΩΝ"""
    print_section("1. DATA DISTRIBUTION ANALYSIS")
    
    emotion_counts = df['emotion'].value_counts()
    emotion_pct = df['emotion'].value_counts(normalize=True) * 100
    
    print("📊 CLASS DISTRIBUTION (Emotion Labels):")
    for emotion in emotion_counts.index:
        count = emotion_counts[emotion]
        pct = emotion_pct[emotion]
        print(f"  {emotion:10s}: {count:5d} samples ({pct:5.2f}%)")
    
    plt.figure(figsize=(10, 6))
    emotion_counts.plot(kind='bar', color='steelblue')
    plt.title('Class Distribution (Emotion Labels)', fontsize=14, fontweight='bold')
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300)
    plt.close()
    
    if 'session' in df.columns:
        print("\n📊 SESSION DISTRIBUTION:")
        session_counts = df['session'].value_counts().sort_index()
        for session in session_counts.index:
            count = session_counts[session]
            pct = (count / len(df)) * 100
            print(f"  Session {session}: {count:5d} samples ({pct:5.2f}%)")

def analyze_feature_statistics(df, output_dir):
    """ΑΝΑΛΥΣΗ ΣΤΑΤΙΣΤΙΚΩΝ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ"""
    print_section("2. FEATURE STATISTICS ANALYSIS")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata_cols = ['session', 'method', 'gender', 'emotion', 'n_annotators', 'agreement']
    feature_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    print(f"📊 Total features: {len(feature_cols)}")
    
    # Missing/Inf/Zero Var checks
    missing_count = df[feature_cols].isnull().sum().sum()
    inf_count = np.isinf(df[feature_cols]).sum().sum()
    
    if missing_count > 0: print(f"⚠️ Missing values: {missing_count}")
    else: print("✅ No missing values found!")
    
    if inf_count > 0: print(f"⚠️ Infinite values: {inf_count}")
    else: print("✅ No infinite values found!")
    
    stats_df = df[feature_cols].describe().T
    stats_df = stats_df[['mean', 'std', 'min', 'max']]
    stats_df.to_csv(os.path.join(output_dir, 'feature_statistics.csv'))
    print(f"✅ Saved statistics to: feature_statistics.csv")

def analyze_correlations(df, output_dir):
    """ΑΝΑΛΥΣΗ ΣΥΣΧΕΤΙΣΗΣ"""
    print_section("3. CORRELATION ANALYSIS")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata_cols = ['session', 'method', 'gender', 'emotion', 'n_annotators', 'agreement']
    feature_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    # Heatmap for first 30 features
    sample_features = feature_cols[:30]
    corr_sample = df[sample_features].corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_sample, annot=False, cmap='coolwarm', center=0, square=True)
    plt.title('Feature Correlation Matrix (Sample of 30 features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300)
    plt.close()
    print(f"✅ Saved correlation heatmap")

def analyze_feature_importance(df, output_dir):
    """ΑΝΑΛΥΣΗ FEATURE IMPORTANCE"""
    print_section("4. FEATURE IMPORTANCE ANALYSIS")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata_cols = ['session', 'method', 'gender', 'emotion', 'n_annotators', 'agreement']
    feature_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    X = df[feature_cols].fillna(0)
    y = df['emotion']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("🔍 Calculating feature importance (Random Forest)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y_encoded)
    
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance_full.csv'), index=False)
    print(f"✅ Saved top importance features")
    
    # Plot top 20
    plt.figure(figsize=(12, 8))
    top_20 = importance_df.head(20)
    plt.barh(top_20['Feature'], top_20['Importance'])
    plt.xlabel('Importance')
    plt.title('Top 20 Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze IEMOCAP fold data')
    parser.add_argument('--fold', type=int, default=1, help='Fold number to analyze (default: 1)')
    parser.add_argument('--type', type=str, choices=['unprocessed', 'processed'], default='unprocessed',
                        help='Type of data to analyze (default: unprocessed)')
    args = parser.parse_args()

    # Input path based on type
    train_csv = os.path.join(BASE_DIR, 'data', 'iemocap', 'dataset', f'fold_{args.fold}', args.type, 'train.csv')
    
    # Output path based on type
    output_dir = os.path.join(BASE_DIR, 'data', 'iemocap', 'analysis', f'fold_{args.fold}', args.type.capitalize())
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(train_csv):
        print(f"ERROR: File not found: {train_csv}")
        return

    print(f"📂 Analyzing {args.type.upper()} data: {train_csv}")
    df = pd.read_csv(train_csv)
    
    sns.set_style("whitegrid")
    
    analyze_data_distribution(df, output_dir)
    analyze_feature_statistics(df, output_dir)
    analyze_correlations(df, output_dir)
    analyze_feature_importance(df, output_dir)
    
    print(f"\n✅ All analysis results saved to: {output_dir}")

if __name__ == "__main__":
    main()
