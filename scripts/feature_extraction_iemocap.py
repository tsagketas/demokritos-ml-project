import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm

from src.preprocessing.audio_preprocess import preprocess_audio
from src.features.label_mapping import map_label
from pyAudioAnalysis import ShortTermFeatures

# ==============================
# CONFIGURATION
# ==============================
DATASET_NAME = "iemocap"
METADATA_CSV = "/workspace/datasets/iemocap/iemocap_full_dataset.csv"
OUTPUT_DIR = "/workspace/processed/features"

FEATURES_CSV = OUTPUT_DIR + "/iemocap/iemocap_features.csv"
REPORT_TXT = OUTPUT_DIR + "/iemocap/iemocap_feature_report.txt"

DEBUG_LIMIT = None  # set to None for full dataset
ST_WIN = 0.050  # 50 ms
ST_STEP = 0.025  # 25 ms

AGGREGATION_STATS = ["mean", "min", "max", "std"]

# ==============================
# LOAD METADATA
# ==============================
df = pd.read_csv(METADATA_CSV)
print(f"Loaded metadata with {len(df)} samples")

# ==============================
# STORAGE
# ==============================
X = []
y = []
file_paths = []

label_counter = {}
dropped_other = 0
processed = 0

# ==============================
# MAIN LOOP
# ==============================
for idx, row in tqdm(df.iterrows(), total=len(df)):
    if DEBUG_LIMIT is not None and processed >= DEBUG_LIMIT:
        break

    wav_path = "/workspace/datasets/iemocap/IEMOCAP_full_release/" + row["path"]
    print(f"\nProcessing file: {wav_path}")

    # Label mapping
    original_label = row["emotion"]
    label = map_label(DATASET_NAME, original_label)

    if label is None or label == "other":
        dropped_other += 1
        continue

    label_counter[label] = label_counter.get(label, 0) + 1

    # ==============================
    # PREPROCESS AUDIO
    # ==============================
    frames, sr = preprocess_audio(
        wav_path,
        target_sr=16000,
        frame_ms=25,
        save_path=None
    )

    # ==============================
    # FEATURE EXTRACTION
    # ==============================
    audio_signal = frames.flatten()

    win_samp = int(ST_WIN * sr)
    step_samp = int(ST_STEP * sr)

    F, f_names = ShortTermFeatures.feature_extraction(
        audio_signal, sr, win_samp, step_samp
    )

    # ==============================
    # AGGREGATE FEATURES
    # ==============================
    features_mean = np.mean(F, axis=1)
    features_min = np.min(F, axis=1)
    features_max = np.max(F, axis=1)
    features_std = np.std(F, axis=1)

    features_agg = np.concatenate([
        features_mean,
        features_min,
        features_max,
        features_std
    ])

    X.append(features_agg)
    y.append(label)
    file_paths.append(wav_path)

    print(
        f"Frames: {frames.shape}, "
        f"Raw features: {F.shape}, "
        f"Aggregated features: {features_agg.shape}"
    )

    processed += 1

# ==============================
# FINAL ARRAYS
# ==============================
X = np.array(X)
y = np.array(y)

# ==============================
# FEATURE NAMES
# ==============================
f_names_agg = (
    [f"{name}_mean" for name in f_names] +
    [f"{name}_min"  for name in f_names] +
    [f"{name}_max"  for name in f_names] +
    [f"{name}_std"  for name in f_names]
)

# ==============================
# SAVE FEATURES (CSV)
# ==============================
df_features = pd.DataFrame(X, columns=f_names_agg)
df_features["label"] = y
df_features["file_path"] = file_paths
df_features["dataset"] = DATASET_NAME

df_features.to_csv(FEATURES_CSV, index=False)
print(f"Saved features to: {FEATURES_CSV}")

# ==============================
# FEATURE EXTRACTION REPORT
# ==============================
with open(REPORT_TXT, "w") as f:
    f.write("FEATURE EXTRACTION REPORT\n")
    f.write("=========================\n\n")

    f.write(f"Dataset: {DATASET_NAME}\n")
    f.write(f"Total metadata samples: {len(df)}\n")
    f.write(f"Processed samples: {processed}\n")
    f.write(f"Dropped samples (other/invalid): {dropped_other}\n\n")

    f.write("Label distribution:\n")
    for lbl, cnt in label_counter.items():
        f.write(f"  {lbl}: {cnt}\n")

    f.write("\nFeature extraction details:\n")
    f.write("  Short-term features: pyAudioAnalysis\n")
    f.write(f"  Window size: {ST_WIN * 1000:.1f} ms\n")
    f.write(f"  Step size: {ST_STEP * 1000:.1f} ms\n")
    f.write(f"  Aggregation statistics: {', '.join(AGGREGATION_STATS)}\n")

    f.write("\nFeature statistics:\n")
    f.write(f"  Base feature count: {len(f_names)}\n")
    f.write(f"  Aggregated feature vector size: {X.shape[1]}\n")
    f.write(f"  Total samples: {X.shape[0]}\n")

    f.write("\nFeature names:\n")
    for name in f_names_agg:
        f.write(f"  {name}\n")

print(f"Saved report to: {REPORT_TXT}")
