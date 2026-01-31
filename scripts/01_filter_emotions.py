"""
Filter Emotions Script
======================

This script removes 'disgust' and 'fear' samples from feature datasets,
keeping only the 4 main emotions: angry, happy, sad, neutral.

Author: ML Team
Date: 2026-01-31
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================
# Auto-detect if running locally or in Docker
PROJECT_ROOT = Path(__file__).parent.parent
if os.path.exists("/workspace"):
    BASE_PATH = Path("/workspace")
else:
    BASE_PATH = PROJECT_ROOT

# Emotions to keep
KEEP_EMOTIONS = ['angry', 'happy', 'sad', 'neutral']

# Datasets to process
DATASETS = {
    "iemocap": {
        "input_csv": str(BASE_PATH / "processed/features/iemocap/iemocap_features.csv"),
        "output_csv": str(BASE_PATH / "processed/features/iemocap/iemocap_features_filtered.csv"),
        "output_report": str(BASE_PATH / "processed/features/iemocap/iemocap_filter_report.txt")
    },
    "cremad": {
        "input_csv": str(BASE_PATH / "processed/features/cremad/cremad_features.csv"),
        "output_csv": str(BASE_PATH / "processed/features/cremad/cremad_features_filtered.csv"),
        "output_report": str(BASE_PATH / "processed/features/cremad/cremad_filter_report.txt")
    }
}


# ==============================
# FILTERING FUNCTION
# ==============================

def filter_emotions(df, keep_emotions):
    """
    Filter dataframe to keep only specified emotions.
    
    Args:
        df: Input dataframe with 'label' column
        keep_emotions: List of emotion labels to keep
    
    Returns:
        Filtered dataframe and statistics dict
    """
    initial_count = len(df)
    initial_labels = df['label'].value_counts().to_dict()
    
    # Filter to keep only specified emotions
    df_filtered = df[df['label'].isin(keep_emotions)].copy()
    
    final_count = len(df_filtered)
    final_labels = df_filtered['label'].value_counts().to_dict()
    removed_count = initial_count - final_count
    
    # Calculate removed samples per label
    removed_labels = {}
    for label in initial_labels:
        if label not in keep_emotions:
            removed_labels[label] = initial_labels[label]
    
    stats = {
        'initial_count': initial_count,
        'final_count': final_count,
        'removed_count': removed_count,
        'initial_labels': initial_labels,
        'final_labels': final_labels,
        'removed_labels': removed_labels
    }
    
    return df_filtered, stats


# ==============================
# PROCESS DATASET
# ==============================

def process_dataset(dataset_name, config):
    """Process a single dataset."""
    
    print("\n" + "="*70)
    print(f"Processing: {dataset_name.upper()}")
    print("="*70)
    
    # Load features
    input_csv = config['input_csv']
    print(f"Loading features from: {input_csv}")
    
    if not os.path.exists(input_csv):
        print(f"ERROR: File not found: {input_csv}")
        return None
    
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} samples")
    
    # Show initial distribution
    print(f"\nInitial label distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"   {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # Filter emotions
    print(f"\nFiltering to keep only: {', '.join(KEEP_EMOTIONS)}")
    df_filtered, stats = filter_emotions(df, KEEP_EMOTIONS)
    
    # Show results
    print(f"\nFiltering results:")
    print(f"   Initial samples: {stats['initial_count']}")
    print(f"   Final samples:   {stats['final_count']}")
    print(f"   Removed samples: {stats['removed_count']} ({stats['removed_count']/stats['initial_count']*100:.1f}%)")
    
    if stats['removed_labels']:
        print(f"\nRemoved emotions:")
        for label, count in stats['removed_labels'].items():
            print(f"   {label}: {count} samples")
    
    print(f"\nFinal label distribution:")
    for label, count in stats['final_labels'].items():
        print(f"   {label}: {count} ({count/stats['final_count']*100:.1f}%)")
    
    # Save filtered dataset
    output_csv = config['output_csv']
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_filtered.to_csv(output_csv, index=False)
    print(f"\nSaved filtered features to: {output_csv}")
    
    # Generate report
    output_report = config['output_report']
    generate_report(dataset_name, stats, output_report)
    print(f"Saved report to: {output_report}")
    
    return stats


def generate_report(dataset_name, stats, output_path):
    """Generate filtering report."""
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"EMOTION FILTERING REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Dataset: {dataset_name.upper()}\n")
        f.write(f"Date: 2026-01-31\n")
        f.write(f"Kept emotions: {', '.join(KEEP_EMOTIONS)}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("FILTERING STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Initial samples: {stats['initial_count']}\n")
        f.write(f"Final samples:   {stats['final_count']}\n")
        f.write(f"Removed samples: {stats['removed_count']} ({stats['removed_count']/stats['initial_count']*100:.1f}%)\n\n")
        
        f.write("-"*70 + "\n")
        f.write("INITIAL LABEL DISTRIBUTION\n")
        f.write("-"*70 + "\n")
        for label, count in sorted(stats['initial_labels'].items()):
            f.write(f"{label}: {count} ({count/stats['initial_count']*100:.1f}%)\n")
        
        f.write("\n")
        f.write("-"*70 + "\n")
        f.write("REMOVED LABELS\n")
        f.write("-"*70 + "\n")
        if stats['removed_labels']:
            for label, count in sorted(stats['removed_labels'].items()):
                f.write(f"{label}: {count} samples\n")
        else:
            f.write("None\n")
        
        f.write("\n")
        f.write("-"*70 + "\n")
        f.write("FINAL LABEL DISTRIBUTION\n")
        f.write("-"*70 + "\n")
        for label, count in sorted(stats['final_labels'].items()):
            f.write(f"{label}: {count} ({count/stats['final_count']*100:.1f}%)\n")
        
        f.write("\n")
        f.write("-"*70 + "\n")
        f.write("NOTES\n")
        f.write("-"*70 + "\n")
        f.write("- This filtered dataset contains only 4 main emotions\n")
        f.write("- Removed emotions: disgust, fear\n")
        f.write("- Use this dataset for training models on balanced classes\n")


# ==============================
# MAIN EXECUTION
# ==============================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("EMOTION FILTERING - REMOVE DISGUST & FEAR")
    print("="*70)
    
    print(f"\nKeeping only: {', '.join(KEEP_EMOTIONS)}")
    print(f"Removing: disgust, fear")
    
    all_results = {}
    
    # Process each dataset
    for dataset_name, config in DATASETS.items():
        result = process_dataset(dataset_name, config)
        if result:
            all_results[dataset_name] = result
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for dataset_name, stats in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  {stats['initial_count']} samples -> {stats['final_count']} samples "
              f"(removed {stats['removed_count']})")
    
    print("\nEmotion filtering completed successfully!")
    print("="*70 + "\n")
