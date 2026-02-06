"""
01 — Preprocess data: load audio, preprocess, extract features, save CSV + report.
Supports CREMA-D (--cremad) or IEMOCAP (--iemocap). All logic in this script.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

from pyAudioAnalysis import ShortTermFeatures

# ---------------------------------------------------------------------------
# Label mapping (inlined from src.features.label_mapping)
# Labels we skip are mapped to DROP so we filter them before loading audio.
# ---------------------------------------------------------------------------
DROP = "drop"
IEMOCAP_LABEL_MAP = {
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",
    "sad": "sad",
    "neu": "neutral",
    "fea": DROP,
    "dis": DROP,
    "fru": DROP,
    "sur": DROP,
    "oth": DROP,
    "xxx": DROP,
}
CREMAD_LABEL_MAP = {
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "NEU": "neutral",
    "FEA": DROP,
    "DIS": DROP,
}


def map_label(dataset, original_label):
    dataset = dataset.lower()
    if dataset == "iemocap":
        return IEMOCAP_LABEL_MAP.get(original_label, DROP)
    if dataset == "cremad":
        return CREMAD_LABEL_MAP.get(original_label, DROP)
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Audio preprocessing (inlined from src.preprocessing.audio_preprocess)
# ---------------------------------------------------------------------------
def preprocess_audio(file_path, target_sr=16000, frame_ms=30, hop_ms=10,
                     trim_silence=True, save_path=None):
    """Load, resample, normalize, trim silence, frame. Optionally save processed wav."""
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=20)
    frame_length = int(sr * frame_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sf.write(save_path, frames.flatten(), sr)
    return frames, sr

# ==============================
# CONFIGURATION
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEBUG_LIMIT = None  # set to int for limiting samples
ST_WIN = 0.050   # 50 ms
ST_STEP = 0.025  # 25 ms
AGGREGATION_STATS = ["mean", "min", "max", "std"]

DATASETS_ROOT = PROJECT_ROOT / "datasets"
CONFIG = {
    "cremad": {
        "name": "cremad",
        "audio_dir": DATASETS_ROOT / "cremad" / "AudioWAV",
        "path_prefix": None,
        "metadata_csv": None,
        "get_items": None,
        "get_path_and_label": None,
    },
    "iemocap": {
        "name": "iemocap",
        "audio_dir": None,
        "path_prefix": DATASETS_ROOT / "iemocap",  # CSV path = Session1/sentences/wav/...
        "metadata_csv": DATASETS_ROOT / "iemocap" / "iemocap_full_dataset.csv",
        "get_items": None,
        "get_path_and_label": None,
    },
}


def get_cremad_items(cfg):
    """Return (items, total_meta). items = [(wav_path, label), ...] with drop already filtered out."""
    wav_files = sorted(Path(cfg["audio_dir"]).glob("*.wav"))
    dataset_name = cfg["name"]
    out = []
    for p in wav_files:
        wav_path = str(p)
        filename = os.path.basename(wav_path)
        try:
            emotion_code = filename.split("_")[2]
        except IndexError:
            continue
        label = map_label(dataset_name, emotion_code)
        if label is None or label == DROP:
            continue
        out.append((wav_path, label))
    return out, len(wav_files)


def get_iemocap_items(cfg):
    """Return (items, total_meta). items = [(wav_path, label), ...] with drop already filtered out."""
    df = pd.read_csv(cfg["metadata_csv"])
    path_prefix = Path(cfg["path_prefix"])
    df["_label"] = df["emotion"].map(lambda e: map_label(cfg["name"], e))
    df_keep = df[df["_label"] != DROP].dropna(subset=["_label"])
    out = [(str(path_prefix / row["path"]), row["_label"]) for _, row in df_keep.iterrows()]
    return out, len(df)


def extract_features_for_dataset(dataset_key, output_base, flat=False):
    cfg = CONFIG[dataset_key].copy()
    dataset_name = cfg["name"]
    output_base = Path(output_base)
    output_dir = output_base if flat else (output_base / dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_csv = output_dir / f"{dataset_name}_features.csv"
    report_txt = output_dir / f"{dataset_name}_feature_report.txt"

    if dataset_key == "cremad":
        items, total_meta = get_cremad_items(cfg)
    else:
        items, total_meta = get_iemocap_items(cfg)
    dropped = total_meta - len(items)

    X, y, file_paths = [], [], []
    label_counter = {}
    processed = 0

    for (wav_path, label) in tqdm(items, total=len(items)):
        if DEBUG_LIMIT is not None and processed >= DEBUG_LIMIT:
            break

        label_counter[label] = label_counter.get(label, 0) + 1

        frames, sr = preprocess_audio(
            wav_path,
            target_sr=16000,
            frame_ms=25,
            save_path=None,
        )
        audio_signal = frames.flatten()
        win_samp = int(ST_WIN * sr)
        step_samp = int(ST_STEP * sr)
        F, f_names = ShortTermFeatures.feature_extraction(
            audio_signal, sr, win_samp, step_samp
        )

        features_mean = np.mean(F, axis=1)
        features_min = np.min(F, axis=1)
        features_max = np.max(F, axis=1)
        features_std = np.std(F, axis=1)
        features_agg = np.concatenate([
            features_mean, features_min, features_max, features_std
        ])

        X.append(features_agg)
        y.append(label)
        file_paths.append(wav_path)
        processed += 1

    if not X:
        print(f"No samples processed for {dataset_name}. Check paths and labels.")
        return

    X = np.array(X)
    y = np.array(y)
    f_names_agg = (
        [f"{n}_mean" for n in f_names]
        + [f"{n}_min" for n in f_names]
        + [f"{n}_max" for n in f_names]
        + [f"{n}_std" for n in f_names]
    )

    df_features = pd.DataFrame(X, columns=f_names_agg)
    df_features["label"] = y
    df_features["file_path"] = file_paths
    df_features["dataset"] = dataset_name
    df_features.to_csv(features_csv, index=False)
    print(f"Saved features to {features_csv}")

    with open(report_txt, "w") as f:
        f.write("FEATURE EXTRACTION REPORT\n")
        f.write("=========================\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total metadata samples: {total_meta}\n")
        f.write(f"Processed samples: {processed}\n")
        f.write(f"Dropped (label=drop or invalid): {dropped}\n\n")
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
    print(f"Saved report to {report_txt}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess data: audio load, preprocessing, feature extraction (CREMA-D or IEMOCAP)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cremad", action="store_true", help="Run for CREMA-D dataset")
    group.add_argument("--iemocap", action="store_true", help="Run for IEMOCAP dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write outputs directly into this directory (no extra subfolders). "
             "Example: --output-dir datasets/cremad_zero_shot_dataset",
    )
    parser.add_argument(
        "--workflow-dir",
        type=Path,
        default=None,
        help="Workflow output root; writes to <workflow-dir>/features/ (no dataset subfolder). Default: project root features/<dataset>/.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        output_base = Path(args.output_dir)
        flat_features = True  # write directly into output_base
    elif args.workflow_dir is not None:
        output_base = Path(args.workflow_dir) / "features"
        flat_features = True  # workflow: features/* directly, no dataset subfolder
    else:
        output_base = PROJECT_ROOT / "features"
    output_base = output_base.resolve()
    if args.cremad:
        extract_features_for_dataset("cremad", output_base, flat=flat_features)
    else:
        extract_features_for_dataset("iemocap", output_base, flat=flat_features)


if __name__ == "__main__":
    main()
