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

FEATURES_PKL = OUTPUT_DIR + "/iemocap_features_debug.pkl"
REPORT_TXT = OUTPUT_DIR + "/iemocap_feature_report_debug.txt"

DEBUG_LIMIT = 10  # set to None for full dataset
ST_WIN = 0.050  # 50 ms
ST_STEP = 0.025  # 25 ms

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
    frames, sr = preprocess_audio(wav_path, target_sr=16000, frame_ms=25,save_path=None)

    # ==============================
    # FEATURE EXTRACTION FROM FRAMES
    # ==============================
    # Convert frames to 1D audio signal for pyAudioAnalysis
    audio_signal = frames.flatten()

    # Compute window/step in samples
    win_samp = int(ST_WIN * sr)
    step_samp = int(ST_STEP * sr)

    # Extract features using pyAudioAnalysis's ShortTermFeatures
    F, f_names = ShortTermFeatures.feature_extraction(audio_signal, sr, win_samp, step_samp)

    # Aggregate features across frames (mean)
    features_mean = np.mean(F, axis=1)

    X.append(features_mean)
    y.append(label)  # now works correctly
    file_paths.append(wav_path)

    print(f"Frames shape: {frames.shape}, Features shape: {features_mean.shape}")
    processed += 1

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"\nProcessed samples: {processed}")
print(f"Dropped 'other'/invalid labels: {dropped_other}")

# ==============================
# SAVE FEATURES (PKL)
# ==============================
output = {
    "X": X,
    "y": y,
    "feature_names": f_names,
    "file_paths": file_paths,
    "dataset": DATASET_NAME
}

with open(FEATURES_PKL, "wb") as f:
    pickle.dump(output, f)

print(f"Saved features to: {FEATURES_PKL}")

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

    f.write("\nFeature statistics:\n")
    f.write(f"  Feature vector size: {X.shape[1]}\n")
    f.write(f"  Total samples: {X.shape[0]}\n")

    f.write("\nFeature names:\n")
    for name in f_names:
        f.write(f"  {name}\n")

print(f"Saved report to: {REPORT_TXT}")
