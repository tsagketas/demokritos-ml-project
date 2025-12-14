"""
Script to extract audio features from IEMOCAP dataset using pyAudioAnalysis.
Reads the CSV file, extracts features for each audio file, and creates a new CSV
with features and all original columns (except path).

Uses MidTermFeatures for segment-level feature extraction as recommended.
"""

import os
import pandas as pd
import numpy as np
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import MidTermFeatures
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = os.path.join(BASE_DIR, 'datasets', 'iemocap_full_dataset.csv')
AUDIO_BASE_DIR = os.path.join(BASE_DIR, 'datasets', 'iemocap')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'iemocap')
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'iemocap_features.csv')

def extract_audio_features(audio_path):
    """
    Extract segment-level features from an audio file using pyAudioAnalysis MidTermFeatures.
    
    MidTermFeatures extracts statistics (mean, std, etc.) over short-term features
    within mid-term windows (e.g., 1 second). We then aggregate these mid-term features
    across the entire segment using mean to get a single feature vector per audio file.
    Returns 68 features (34 mean + 34 std from short-term features, aggregated with mean).
    
    Args:
        audio_path: Full path to the audio file
        
    Returns:
        numpy array of features (flattened) and feature names, or None if extraction fails
    """
    try:
        [Fs, x] = audioBasicIO.read_audio_file(audio_path)
        
        if len(x) == 0:
            return None, None
        
        mid_window = 1.0
        mid_step = 1.0
        short_window = 0.050
        short_step = 0.025
        
        mid_window_samples = int(mid_window * Fs)
        mid_step_samples = int(mid_step * Fs)
        short_window_samples = int(short_window * Fs)
        short_step_samples = int(short_step * Fs)
        
        if mid_window_samples < 1 or short_window_samples < 1:
            return None, None
        
        mt_features, _, mt_feature_names = MidTermFeatures.mid_feature_extraction(
            x, Fs, mid_window_samples, mid_step_samples, 
            short_window_samples, short_step_samples
        )
        
        if mt_features is None or mt_features.size == 0:
            return None, None
        
        feature_vector = []
        feature_names = []
        
        for i, feat_name in enumerate(mt_feature_names):
            feature_values = mt_features[i, :]
            if np.isnan(feature_values).all() or len(feature_values) == 0:
                continue
            feature_vector.append(np.mean(feature_values))
            feature_names.append(feat_name)
        
        if len(feature_vector) == 0:
            return None, None
            
        return np.array(feature_vector), feature_names
        
    except Exception:
        return None, None

def main():
    """Main function to process the CSV and extract features."""
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: Input CSV not found: {INPUT_CSV}")
        return
    
    print(f"Reading CSV from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total rows: {len(df)}")
    
    # Remove rows with invalid emotion labels (xxx) - these have no annotations
    initial_count = len(df)
    df = df[df['emotion'] != 'xxx']
    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"Removed {removed_count} rows with invalid emotion labels (xxx)")
    print(f"Valid rows after filtering: {len(df)}")
    
    feature_names = None
    for idx, row in df.iterrows():
        first_audio_path = os.path.join(AUDIO_BASE_DIR, row['path'])
        if os.path.exists(first_audio_path):
            _, feature_names = extract_audio_features(first_audio_path)
            if feature_names is not None:
                break
    
    if feature_names is None:
        print("ERROR: Failed to extract features. Check audio file paths.")
        return
    
    print(f"Extracting {len(feature_names)} features per audio file...")
    
    # Initialize feature columns
    for feat_name in feature_names:
        df[feat_name] = None
    
    failed_files = []
    successful = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        audio_path = os.path.join(AUDIO_BASE_DIR, row['path'])
        
        if not os.path.exists(audio_path):
            failed_files.append(audio_path)
            continue
        
        feature_vector, _ = extract_audio_features(audio_path)
        
        if feature_vector is not None and len(feature_vector) == len(feature_names):
            for i, feat_name in enumerate(feature_names):
                df.at[idx, feat_name] = feature_vector[i]
            successful += 1
        else:
            failed_files.append(audio_path)
    
    if failed_files:
        print(f"Failed: {len(failed_files)} files")
        df = df.dropna(subset=[feature_names[0]])
    
    print(f"Successful: {successful}/{len(df)}")
    
    # Remove the 'path' column
    df = df.drop(columns=['path'])
    
    # Save the final CSV
    print(f"\nSaving features CSV to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully saved {len(df)} rows with {len(df.columns)} columns.")
    print(f"Features saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
