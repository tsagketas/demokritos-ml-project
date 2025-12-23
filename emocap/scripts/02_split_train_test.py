"""
Leave-One-Session-Out (LOSO) Cross-Validation for IEMOCAP Dataset.

IEMOCAP has 5 sessions. LOSO creates 5 folds where:
- Each session becomes the test set exactly once
- Each session becomes the validation set exactly once (rotating)
- Remaining 3 sessions form the training set

Fold Structure:
  Fold 1: Test=Session1, Val=Session2, Train=Sessions[3,4,5]
  Fold 2: Test=Session2, Val=Session3, Train=Sessions[1,4,5]
  Fold 3: Test=Session3, Val=Session4, Train=Sessions[1,2,5]
  Fold 4: Test=Session4, Val=Session5, Train=Sessions[1,2,3]
  Fold 5: Test=Session5, Val=Session1, Train=Sessions[2,3,4]

Usage:
    python 02_split_train_test.py
"""

import os
import pandas as pd
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = os.path.join(BASE_DIR, 'data', 'iemocap', 'iemocap_features.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'iemocap', 'dataset')

def create_loso_folds(df):
    """
    Create 5 LOSO folds.
    
    Each fold:
    - Test set: One session (held out)
    - Validation set: Next session (rotating)
    - Train set: Remaining 3 sessions
    
    Args:
        df: DataFrame with 'session' column
    
    Returns:
        List of 5 tuples: [(train_df, val_df, test_df), ...]
    """
    sessions = sorted(df['session'].unique().tolist())
    
    if len(sessions) != 5:
        raise ValueError(f"Expected 5 sessions, found {len(sessions)}: {sessions}")
    
    folds = []
    
    # Create 5 folds
    for i in range(5):
        test_session = sessions[i]
        val_session = sessions[(i + 1) % 5]  # Next session (wraps around)
        
        # Remaining sessions for training
        train_sessions = [s for s in sessions if s != test_session and s != val_session]
        
        # Create splits
        test_df = df[df['session'] == test_session].copy()
        val_df = df[df['session'] == val_session].copy()
        train_df = df[df['session'].isin(train_sessions)].copy()
        
        folds.append((train_df, val_df, test_df))
    
    return folds

def save_fold(fold_num, train_df, val_df, test_df, output_dir):
    """
    Save fold data to disk in unprocessed folder.
    """
    fold_dir = os.path.join(output_dir, f'fold_{fold_num}', 'unprocessed')
    os.makedirs(fold_dir, exist_ok=True)
    
    train_path = os.path.join(fold_dir, 'train.csv')
    val_path = os.path.join(fold_dir, 'val.csv')
    test_path = os.path.join(fold_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    return train_path, val_path, test_path

def print_fold_stats(fold_num, train_df, val_df, test_df, total_samples):
    """
    Print statistics for a fold.
    
    Args:
        fold_num: Fold number (1-5)
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        total_samples: Total number of samples in dataset
    """
    # Get session numbers
    test_session = test_df['session'].iloc[0]
    val_session = val_df['session'].iloc[0]
    train_sessions = sorted(train_df['session'].unique().tolist())
    
    print(f"\n=== FOLD {fold_num} ===")
    print(f"Test: Session {test_session} ({len(test_df)} samples, {100*len(test_df)/total_samples:.1f}%)")
    print(f"Validation: Session {val_session} ({len(val_df)} samples, {100*len(val_df)/total_samples:.1f}%)")
    print(f"Train: Sessions {train_sessions} ({len(train_df)} samples, {100*len(train_df)/total_samples:.1f}%)")
    
    # Emotion distribution
    train_emotions = Counter(train_df['emotion'])
    val_emotions = Counter(val_df['emotion'])
    test_emotions = Counter(test_df['emotion'])
    
    all_emotions = sorted(set(train_emotions.keys()) | set(val_emotions.keys()) | set(test_emotions.keys()))
    
    print("\nEmotion distribution:")
    for emotion in all_emotions:
        train_count = train_emotions.get(emotion, 0)
        val_count = val_emotions.get(emotion, 0)
        test_count = test_emotions.get(emotion, 0)
        print(f"  {emotion}: Train={train_count}, Val={val_count}, Test={test_count}")

def print_summary(folds):
    """
    Print summary statistics across all folds.
    
    Args:
        folds: List of (train_df, val_df, test_df) tuples
    """
    print("\n" + "="*60)
    print("=== LOSO SUMMARY ===")
    print("="*60)
    
    total_folds = len(folds)
    train_sizes = [len(train_df) for train_df, _, _ in folds]
    val_sizes = [len(val_df) for _, val_df, _ in folds]
    test_sizes = [len(test_df) for _, _, test_df in folds]
    
    avg_train = sum(train_sizes) / total_folds
    avg_val = sum(val_sizes) / total_folds
    avg_test = sum(test_sizes) / total_folds
    
    print(f"Total folds: {total_folds}")
    print(f"Average train size: {avg_train:.1f} samples")
    print(f"Average val size: {avg_val:.1f} samples")
    print(f"Average test size: {avg_test:.1f} samples")
    
    # Validation: Check that all samples appear exactly once in test across all folds
    all_test_samples = set()
    all_val_samples = set()
    all_train_samples = set()
    
    for fold_num, (train_df, val_df, test_df) in enumerate(folds, start=1):
        # Check for overlaps within fold
        train_indices = set(train_df.index)
        val_indices = set(val_df.index)
        test_indices = set(test_df.index)
        
        if train_indices & val_indices:
            print(f"\n⚠️  WARNING: Fold {fold_num} has overlap between train and validation!")
        if train_indices & test_indices:
            print(f"\n⚠️  WARNING: Fold {fold_num} has overlap between train and test!")
        if val_indices & test_indices:
            print(f"\n⚠️  WARNING: Fold {fold_num} has overlap between validation and test!")
        
        all_test_samples.update(test_indices)
        all_val_samples.update(val_indices)
        all_train_samples.update(train_indices)
    
    # Count occurrences
    test_counts = {}
    val_counts = {}
    train_counts = {}
    
    for train_df, val_df, test_df in folds:
        for idx in test_df.index:
            test_counts[idx] = test_counts.get(idx, 0) + 1
        for idx in val_df.index:
            val_counts[idx] = val_counts.get(idx, 0) + 1
        for idx in train_df.index:
            train_counts[idx] = train_counts.get(idx, 0) + 1
    
    # Verify each sample appears exactly once in test, once in val, 3 times in train
    total_samples = len(all_test_samples | all_val_samples | all_train_samples)
    expected_test_count = 1
    expected_val_count = 1
    expected_train_count = 3
    
    test_errors = [idx for idx, count in test_counts.items() if count != expected_test_count]
    val_errors = [idx for idx, count in val_counts.items() if count != expected_val_count]
    train_errors = [idx for idx, count in train_counts.items() if count != expected_train_count]
    
    if test_errors:
        print(f"\n⚠️  WARNING: {len(test_errors)} samples appear in test set {expected_test_count} times (expected once)")
    if val_errors:
        print(f"\n⚠️  WARNING: {len(val_errors)} samples appear in validation set {expected_val_count} times (expected once)")
    if train_errors:
        print(f"\n⚠️  WARNING: {len(train_errors)} samples appear in train set {expected_train_count} times (expected 3 times)")
    
    if not (test_errors or val_errors or train_errors):
        print(f"\n✅ Validation passed: All {total_samples} samples correctly distributed across folds")

def main():
    """Main function to create LOSO folds."""
    print("="*60)
    print("Leave-One-Session-Out (LOSO) Cross-Validation")
    print("="*60)
    
    # Load data
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: Input CSV not found: {INPUT_CSV}")
        return
    
    df = pd.read_csv(INPUT_CSV)
    print(f"\nTotal samples: {len(df)}")
    
    # Check for session column
    if 'session' not in df.columns:
        print("\nERROR: 'session' column not found in dataset!")
        print("LOSO requires session information. Please check your input CSV.")
        exit(1)
    
    # Check available sessions
    sessions = sorted(df['session'].unique().tolist())
    print(f"Sessions found: {sessions}")
    
    if len(sessions) != 5:
        print(f"\nERROR: Expected 5 sessions, found {len(sessions)}: {sessions}")
        print("LOSO requires exactly 5 sessions.")
        exit(1)
    
    # Create folds
    print("\nCreating LOSO folds...")
    folds = create_loso_folds(df)
    
    # Process each fold
    for fold_num, (train_df, val_df, test_df) in enumerate(folds, start=1):
        print_fold_stats(fold_num, train_df, val_df, test_df, len(df))
        
        train_path, val_path, test_path = save_fold(
            fold_num, train_df, val_df, test_df, OUTPUT_DIR
        )
        
        print(f"\nSaved to:")
        print(f"  {train_path}")
        print(f"  {val_path}")
        print(f"  {test_path}")
    
    # Print summary
    print_summary(folds)
    
    print("\n" + "="*60)
    print("✅ LOSO cross-validation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
