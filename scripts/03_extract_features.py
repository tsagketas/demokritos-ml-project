import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import CREMADLoader
from src.feature_extractor import FeatureExtractor


def validate_audio_files(filepaths):
    existing_files = []
    missing_files = []
    
    for filepath in filepaths:
        if Path(filepath).exists():
            existing_files.append(filepath)
        else:
            missing_files.append(filepath)
    
    return existing_files, missing_files


def validate_features(features_dict):
    if not features_dict or len(features_dict) == 0:
        return False, "Empty features"
    
    if 'filepath' not in features_dict or 'filename' not in features_dict:
        return False, "Missing keys"
    
    metadata_keys = {'filepath', 'filename', 'label', 'duration'}
    feature_keys = [k for k in features_dict.keys() if k not in metadata_keys]
    
    if len(feature_keys) == 0:
        return False, "No features"
    
    for key in feature_keys:
        value = features_dict[key]
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                return False, f"Invalid value: {key}"
    
    return True, "OK"


def main():
    parser = argparse.ArgumentParser(description='Extract features from CREMA-D dataset')
    parser.add_argument('--sr', type=int, default=22050,
                       help='Sample rate for audio loading')
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "datasets" / "cremad"
    metadata_path = project_root / "data" / "cremad_metadata.csv"
    output_path = project_root / "data" / "cremad_features.csv"
    
    if not metadata_path.exists():
        print(f"ERROR: Metadata file not found: {metadata_path}")
        sys.exit(1)
    
    print("Loading metadata...")
    metadata_df = pd.read_csv(metadata_path)
    
    metadata_df['filepath'] = metadata_df['filepath'].apply(
        lambda x: str(project_root / x) if not Path(x).is_absolute() else x
    )
    
    print("Validating audio files...")
    existing_files, missing_files = validate_audio_files(metadata_df['filepath'].tolist())
    
    print(f"Found: {len(existing_files)} files")
    if missing_files:
        print(f"Missing: {len(missing_files)} files")
    
    if len(existing_files) == 0:
        print("ERROR: No valid audio files found!")
        sys.exit(1)
    
    metadata_df = metadata_df[metadata_df['filepath'].isin(existing_files)].copy()
    
    print(f"Extracting features (sample rate: {args.sr} Hz)...")
    extractor = FeatureExtractor(sr=args.sr)
    
    features_df = extractor.extract_batch(
        filepaths=metadata_df['filepath'].tolist(),
        labels=metadata_df['emotion'].tolist(),
        progress=True
    )
    
    print("Validating features...")
    valid_indices = []
    failed_files = []
    
    for idx, row in features_df.iterrows():
        features_dict = row.to_dict()
        is_valid, reason = validate_features(features_dict)
        
        if is_valid:
            valid_indices.append(idx)
        else:
            failed_files.append({
                'filename': features_dict.get('filename', 'unknown'),
                'reason': reason
            })
    
    print(f"Valid: {len(valid_indices)}, Failed: {len(failed_files)}")
    
    if len(valid_indices) < len(features_df):
        features_df = features_df.loc[valid_indices].copy()
    
    if len(features_df) == 0:
        print("ERROR: No valid features extracted!")
        sys.exit(1)
    
    print("Merging with metadata...")
    features_df = features_df.merge(
        metadata_df[['filename', 'actor_id', 'sentence_code', 'emotion', 'intensity']],
        on='filename',
        how='left'
    )
    
    metadata_cols = ['filename', 'filepath', 'actor_id', 'sentence_code', 'emotion', 'intensity']
    if 'label' in features_df.columns:
        metadata_cols.insert(2, 'label')
    
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    features_df = features_df[metadata_cols + feature_cols]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    
    print(f"Saved: {output_path}")
    print(f"Samples: {len(features_df)}, Features: {len(feature_cols)}")


if __name__ == "__main__":
    main()
