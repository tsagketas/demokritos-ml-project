"""
Comprehensive Data Analysis Script for IEMOCAP Train Dataset

Αυτό το script κάνει πλήρη ανάλυση του train.csv με:
- Data distribution (class, session, gender, method)
- Feature statistics (mean, std, min, max, missing, infinite, zero variance)
- Visualizations (correlation, distributions, outliers, class imbalance)
- Feature importance analysis
- Correlation analysis για redundant features
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_CSV = os.path.join(BASE_DIR, 'data', 'iemocap', 'train.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'iemocap', 'analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def print_section(title):
    """Helper για όμορφη εκτύπωση"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def analyze_data_distribution(df):
    """
    ΑΝΑΛΥΣΗ ΚΑΤΑΝΟΜΗΣ ΔΕΔΟΜΕΝΩΝ
    
    ΓΙΑΤΙ ΕΙΝΑΙ ΣΗΜΑΝΤΙΚΟ:
    - Class distribution: Αν υπάρχει imbalance, το μοντέλο θα bias προς τις πιο συχνές κλάσεις
    - Session/gender/method: Δείχνει αν υπάρχει bias στα δεδομένα
    """
    print_section("1. DATA DISTRIBUTION ANALYSIS")
    
    # Class distribution (emotion labels) - ΠΟΛΥ ΣΗΜΑΝΤΙΚΟ!
    print("📊 CLASS DISTRIBUTION (Emotion Labels):")
    print("-" * 60)
    emotion_counts = df['emotion'].value_counts()
    emotion_pct = df['emotion'].value_counts(normalize=True) * 100
    
    for emotion in emotion_counts.index:
        count = emotion_counts[emotion]
        pct = emotion_pct[emotion]
        print(f"  {emotion:10s}: {count:5d} samples ({pct:5.2f}%)")
    
    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    emotion_counts.plot(kind='bar', color='steelblue')
    plt.title('Class Distribution (Emotion Labels)', fontsize=14, fontweight='bold')
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'), dpi=300)
    plt.close()
    print(f"\n✅ Saved: class_distribution.png")
    
    # Session distribution
    if 'session' in df.columns:
        print("\n📊 SESSION DISTRIBUTION:")
        print("-" * 60)
        session_counts = df['session'].value_counts().sort_index()
        for session in session_counts.index:
            count = session_counts[session]
            pct = (count / len(df)) * 100
            print(f"  Session {session}: {count:5d} samples ({pct:5.2f}%)")
    
    # Gender distribution
    if 'gender' in df.columns:
        print("\n📊 GENDER DISTRIBUTION:")
        print("-" * 60)
        gender_counts = df['gender'].value_counts()
        for gender in gender_counts.index:
            count = gender_counts[gender]
            pct = (count / len(df)) * 100
            print(f"  {gender}: {count:5d} samples ({pct:5.2f}%)")
    
    # Method distribution
    if 'method' in df.columns:
        print("\n📊 METHOD DISTRIBUTION:")
        print("-" * 60)
        method_counts = df['method'].value_counts()
        for method in method_counts.index:
            count = method_counts[method]
            pct = (count / len(df)) * 100
            print(f"  {method}: {count:5d} samples ({pct:5.2f}%)")
    
    # Class imbalance ratio
    print("\n⚠️  CLASS IMBALANCE ANALYSIS:")
    print("-" * 60)
    max_class = emotion_counts.max()
    min_class = emotion_counts.min()
    imbalance_ratio = max_class / min_class
    print(f"  Max class count: {max_class}")
    print(f"  Min class count: {min_class}")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 2:
        print("  ⚠️  WARNING: Significant class imbalance detected!")
        print("     Consider: SMOTE, class weights, or undersampling")
    else:
        print("  ✅ Classes are relatively balanced")

def analyze_feature_statistics(df):
    """
    ΑΝΑΛΥΣΗ ΣΤΑΤΙΣΤΙΚΩΝ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ
    
    ΓΙΑΤΙ ΕΙΝΑΙ ΣΗΜΑΝΤΙΚΟ:
    - Mean/std/min/max: Δείχνει αν χρειάζεται scaling/normalization
    - Missing values: Πρέπει να τα handle (imputation ή removal)
    - Infinite values: Bug που πρέπει να διορθωθεί
    - Zero variance: Features που δεν προσφέρουν πληροφορία
    """
    print_section("2. FEATURE STATISTICS ANALYSIS")
    
    # Get numeric features only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata_cols = ['session', 'method', 'gender', 'emotion', 'n_annotators', 'agreement']
    feature_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    print(f"📊 Total features: {len(feature_cols)}")
    print(f"📊 Total samples: {len(df)}")
    
    # Missing values - ΠΟΛΥ ΣΗΜΑΝΤΙΚΟ!
    print("\n🔍 MISSING VALUES ANALYSIS:")
    print("-" * 60)
    missing_count = df[feature_cols].isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        'Feature': missing_count.index,
        'Missing Count': missing_count.values,
        'Missing %': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        print("  ⚠️  Features with missing values:")
        for _, row in missing_df.iterrows():
            print(f"    {row['Feature']:30s}: {row['Missing Count']:5d} ({row['Missing %']:5.2f}%)")
    else:
        print("  ✅ No missing values found!")
    
    # Infinite values
    print("\n🔍 INFINITE VALUES ANALYSIS:")
    print("-" * 60)
    inf_count = np.isinf(df[feature_cols]).sum()
    inf_features = inf_count[inf_count > 0]
    if len(inf_features) > 0:
        print("  ⚠️  Features with infinite values:")
        for feat in inf_features.index:
            print(f"    {feat:30s}: {inf_count[feat]} infinite values")
    else:
        print("  ✅ No infinite values found!")
    
    # Zero variance features - ΣΗΜΑΝΤΙΚΟ!
    print("\n🔍 ZERO VARIANCE FEATURES:")
    print("-" * 60)
    variances = df[feature_cols].var()
    zero_var_features = variances[variances == 0].index.tolist()
    if len(zero_var_features) > 0:
        print(f"  ⚠️  Found {len(zero_var_features)} zero variance features:")
        for feat in zero_var_features:
            print(f"    - {feat}")
        print("  💡 These features should be removed (they provide no information)")
    else:
        print("  ✅ No zero variance features found!")
    
    # Basic statistics (mean, std, min, max)
    print("\n📊 BASIC STATISTICS (Sample of 10 features):")
    print("-" * 60)
    stats_df = df[feature_cols].describe().T
    stats_df = stats_df[['mean', 'std', 'min', 'max']]
    print(stats_df.head(10).to_string())
    print(f"\n  ... (showing first 10 of {len(feature_cols)} features)")
    
    # Save full statistics to CSV
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_statistics.csv'))
    print(f"\n✅ Saved full statistics to: feature_statistics.csv")
    
    # Scale analysis - για να δούμε αν χρειάζεται normalization
    print("\n📊 SCALE ANALYSIS:")
    print("-" * 60)
    max_vals = df[feature_cols].max()
    min_vals = df[feature_cols].min()
    ranges = max_vals - min_vals
    print(f"  Feature value ranges:")
    print(f"    Min range: {ranges.min():.4f}")
    print(f"    Max range: {ranges.max():.4f}")
    print(f"    Ratio: {ranges.max() / ranges.min():.2f}:1")
    if ranges.max() / ranges.min() > 100:
        print("  ⚠️  Large scale differences detected!")
        print("     Consider: StandardScaler or MinMaxScaler")
    else:
        print("  ✅ Scales are relatively similar")

def analyze_correlations(df):
    """
    ΑΝΑΛΥΣΗ ΣΥΣΧΕΤΙΣΗΣ
    
    ΓΙΑΤΙ ΕΙΝΑΙ ΠΟΛΥ ΣΗΜΑΝΤΙΚΟ:
    - Αν features είναι πολύ συσχετισμένα (>0.95), είναι redundant
    - Μπορείς να αφαιρέσεις μερικά για να μειώσεις dimensionality
    - Βοηθάει να καταλάβεις ποια features είναι παρόμοια
    """
    print_section("3. CORRELATION ANALYSIS")
    
    # Get numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata_cols = ['session', 'method', 'gender', 'emotion', 'n_annotators', 'agreement']
    feature_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr()
    
    # Find highly correlated pairs
    print("🔍 HIGHLY CORRELATED FEATURE PAIRS (>0.95):")
    print("-" * 60)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.95:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))
    
    if len(high_corr_pairs) > 0:
        print(f"  ⚠️  Found {len(high_corr_pairs)} highly correlated pairs:")
        for feat1, feat2, corr in high_corr_pairs[:20]:  # Show first 20
            print(f"    {feat1:30s} <-> {feat2:30s}: {corr:.4f}")
        if len(high_corr_pairs) > 20:
            print(f"    ... and {len(high_corr_pairs) - 20} more pairs")
        print("\n  💡 Consider removing one feature from each highly correlated pair")
    else:
        print("  ✅ No highly correlated pairs found (>0.95)")
    
    # Visualize correlation matrix (sample of features for readability)
    print("\n📊 Creating correlation matrix visualization...")
    sample_features = feature_cols[:30]  # First 30 features for readability
    corr_sample = df[sample_features].corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_sample, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix (Sample of 30 features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), dpi=300)
    plt.close()
    print(f"✅ Saved: correlation_matrix.png")
    
    # Save full correlation matrix
    corr_matrix.to_csv(os.path.join(OUTPUT_DIR, 'correlation_matrix_full.csv'))
    print(f"✅ Saved full correlation matrix to: correlation_matrix_full.csv")

def analyze_outliers(df):
    """
    ΑΝΑΛΥΣΗ OUTLIERS
    
    ΓΙΑΤΙ ΕΙΝΑΙ ΣΗΜΑΝΤΙΚΟ:
    - Outliers μπορεί να προκαλούν προβλήματα στο training
    - Μπορεί να είναι errors ή legitimate extreme values
    """
    print_section("4. OUTLIER DETECTION")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata_cols = ['session', 'method', 'gender', 'emotion', 'n_annotators', 'agreement']
    feature_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    # IQR method for outlier detection
    print("🔍 OUTLIER DETECTION (IQR Method):")
    print("-" * 60)
    
    outlier_counts = {}
    for col in feature_cols[:20]:  # Check first 20 features
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
    
    outlier_df = pd.DataFrame(list(outlier_counts.items()), columns=['Feature', 'Outlier Count'])
    outlier_df = outlier_df[outlier_df['Outlier Count'] > 0].sort_values('Outlier Count', ascending=False)
    
    if len(outlier_df) > 0:
        print(f"  Features with outliers (showing top 10):")
        for _, row in outlier_df.head(10).iterrows():
            pct = (row['Outlier Count'] / len(df)) * 100
            print(f"    {row['Feature']:30s}: {row['Outlier Count']:5d} outliers ({pct:5.2f}%)")
    else:
        print("  ✅ No outliers detected in sample features")
    
    # Box plots for sample features
    print("\n📊 Creating box plots for outlier visualization...")
    sample_features = feature_cols[:6]  # First 6 features
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(sample_features):
        df.boxplot(column=col, ax=axes[idx])
        axes[idx].set_title(f'{col}', fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Outlier Detection - Box Plots (Sample Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'outliers_boxplots.png'), dpi=300)
    plt.close()
    print(f"✅ Saved: outliers_boxplots.png")

def analyze_skewness_kurtosis(df):
    """
    ΑΝΑΛΥΣΗ SKEWNESS & KURTOSIS
    
    ΓΙΑΤΙ ΕΙΝΑΙ ΣΗΜΑΝΤΙΚΟ:
    - Skewed features μπορεί να χρειάζονται transformation (log, sqrt)
    - Δείχνει αν η κατανομή είναι normal ή όχι
    """
    print_section("5. SKEWNESS & KURTOSIS ANALYSIS")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata_cols = ['session', 'method', 'gender', 'emotion', 'n_annotators', 'agreement']
    feature_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    skewness = df[feature_cols].skew()
    kurtosis = df[feature_cols].kurtosis()
    
    # Highly skewed features
    highly_skewed = skewness[abs(skewness) > 2].sort_values(key=abs, ascending=False)
    
    print("🔍 HIGHLY SKEWED FEATURES (|skewness| > 2):")
    print("-" * 60)
    if len(highly_skewed) > 0:
        print(f"  Found {len(highly_skewed)} highly skewed features:")
        for feat, skew_val in highly_skewed.head(10).items():
            print(f"    {feat:30s}: skewness = {skew_val:7.3f}")
        print("\n  💡 Consider: log transformation, sqrt transformation, or Box-Cox")
    else:
        print("  ✅ No highly skewed features found")
    
    # Save full statistics
    skew_kurt_df = pd.DataFrame({
        'Feature': feature_cols,
        'Skewness': skewness.values,
        'Kurtosis': kurtosis.values
    })
    skew_kurt_df.to_csv(os.path.join(OUTPUT_DIR, 'skewness_kurtosis.csv'), index=False)
    print(f"\n✅ Saved full skewness/kurtosis to: skewness_kurtosis.csv")

def analyze_feature_importance(df):
    """
    ΑΝΑΛΥΣΗ FEATURE IMPORTANCE
    
    ΓΙΑΤΙ ΕΙΝΑΙ ΠΟΛΥ ΣΗΜΑΝΤΙΚΟ:
    - Δείχνει ποια features είναι πιο σημαντικά για το prediction
    - Μπορείς να αφαιρέσεις χαμηλής σημασίας features
    """
    print_section("6. FEATURE IMPORTANCE ANALYSIS")
    
    # Prepare data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata_cols = ['session', 'method', 'gender', 'emotion', 'n_annotators', 'agreement']
    feature_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    X = df[feature_cols].fillna(0)  # Fill missing with 0 for importance calculation
    y = df['emotion']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("🔍 Calculating feature importance using Random Forest...")
    print("   (This may take a few minutes...)")
    
    # Use Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y_encoded)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n📊 TOP 20 MOST IMPORTANT FEATURES:")
    print("-" * 60)
    for idx, row in importance_df.head(20).iterrows():
        print(f"  {row['Feature']:30s}: {row['Importance']:.6f}")
    
    # Features with very low importance
    low_importance = importance_df[importance_df['Importance'] < 0.001]
    print(f"\n⚠️  FEATURES WITH VERY LOW IMPORTANCE (<0.001): {len(low_importance)}")
    if len(low_importance) > 0:
        print("  Consider removing these features:")
        for _, row in low_importance.head(10).iterrows():
            print(f"    - {row['Feature']}")
    
    # Visualize top features
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(30)
    plt.barh(range(len(top_features)), top_features['Importance'].values)
    plt.yticks(range(len(top_features)), top_features['Feature'].values)
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 30 Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300)
    plt.close()
    print(f"\n✅ Saved: feature_importance.png")
    
    # Save full importance
    importance_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance_full.csv'), index=False)
    print(f"✅ Saved full feature importance to: feature_importance_full.csv")

def create_missing_values_heatmap(df):
    """
    HEATMAP για Missing Values
    
    ΓΙΑΤΙ ΕΙΝΑΙ ΣΗΜΑΝΤΙΚΟ:
    - Δείχνει visually ποια features έχουν missing values
    - Βοηθάει να δεις patterns στα missing data
    """
    print_section("7. MISSING VALUES HEATMAP")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata_cols = ['session', 'method', 'gender', 'emotion', 'n_annotators', 'agreement']
    feature_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    missing_data = df[feature_cols].isnull()
    
    if missing_data.sum().sum() > 0:
        # Get features with missing values
        features_with_missing = missing_data.columns[missing_data.any()].tolist()
        
        if len(features_with_missing) > 0:
            plt.figure(figsize=(14, max(8, len(features_with_missing) * 0.3)))
            sns.heatmap(missing_data[features_with_missing].T, 
                       cbar=True, yticklabels=True, xticklabels=False, cmap='viridis')
            plt.title('Missing Values Heatmap', fontsize=14, fontweight='bold')
            plt.ylabel('Features', fontsize=12)
            plt.xlabel('Samples', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'missing_values_heatmap.png'), dpi=300)
            plt.close()
            print(f"✅ Saved: missing_values_heatmap.png")
        else:
            print("✅ No missing values to visualize")
    else:
        print("✅ No missing values found in dataset")

def create_feature_distributions(df):
    """
    DISTRIBUTIONS για Sample Features
    
    ΓΙΑΤΙ ΕΙΝΑΙ ΣΗΜΑΝΤΙΚΟ:
    - Δείχνει την κατανομή των features
    - Βοηθάει να δεις αν είναι normal, skewed, κλπ
    """
    print_section("8. FEATURE DISTRIBUTIONS")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata_cols = ['session', 'method', 'gender', 'emotion', 'n_annotators', 'agreement']
    feature_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    # Sample features for visualization
    sample_features = feature_cols[:12]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for idx, col in enumerate(sample_features):
        df[col].hist(bins=50, ax=axes[idx], edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col}', fontsize=10)
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(alpha=0.3)
    
    plt.suptitle('Feature Distributions (Sample of 12 features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_distributions.png'), dpi=300)
    plt.close()
    print(f"✅ Saved: feature_distributions.png")

def main():
    """Main function"""
    print("\n" + "="*80)
    print("  COMPREHENSIVE DATA ANALYSIS FOR IEMOCAP TRAIN DATASET")
    print("="*80)
    
    # Load data
    print(f"\n📂 Loading data from: {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)
    print(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Run all analyses
    analyze_data_distribution(df)
    analyze_feature_statistics(df)
    analyze_correlations(df)
    analyze_outliers(df)
    analyze_skewness_kurtosis(df)
    analyze_feature_importance(df)
    create_missing_values_heatmap(df)
    create_feature_distributions(df)
    
    # Summary
    print_section("ANALYSIS COMPLETE")
    print(f"✅ All analysis results saved to: {OUTPUT_DIR}")
    print("\n📋 Generated files:")
    print("  - class_distribution.png")
    print("  - feature_statistics.csv")
    print("  - correlation_matrix.png")
    print("  - correlation_matrix_full.csv")
    print("  - outliers_boxplots.png")
    print("  - skewness_kurtosis.csv")
    print("  - feature_importance.png")
    print("  - feature_importance_full.csv")
    print("  - missing_values_heatmap.png")
    print("  - feature_distributions.png")
    
    print("\n💡 NEXT STEPS:")
    print("  1. Check class imbalance - apply balancing if needed")
    print("  2. Handle missing values (if any)")
    print("  3. Remove zero variance features")
    print("  4. Remove highly correlated features (>0.95)")
    print("  5. Remove low importance features (<0.001)")
    print("  6. Apply scaling/normalization if needed")
    print("  7. Handle outliers if necessary")

if __name__ == "__main__":
    main()
