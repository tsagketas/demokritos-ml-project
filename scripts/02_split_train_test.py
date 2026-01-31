import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "processed" / "features" / "iemocap" / "iemocap_features.csv"
SPLITS_DIR = PROJECT_ROOT / "processed" / "features" / "iemocap" / "splits"

NON_FEATURE_COLS = ["label", "file_path", "dataset"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=Path, default=FEATURES_CSV)
    parser.add_argument("--out-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--split", type=str, default="stratified_80_20", choices=["stratified_80_20", "loso"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true", help="Fit StandardScaler on train and normalize train/test")
    args = parser.parse_args()

    if args.split == "loso":
        raise NotImplementedError("LOSO will be implemented later")

    features_path = Path(args.features_csv)
    if not features_path.is_file():
        raise FileNotFoundError(f"Features CSV not found: {features_path}")

    split_folder = "80_20" if args.split == "stratified_80_20" else "loso"
    out_dir = Path(args.out_dir) / split_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain column 'label' for stratified split.")

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df["label"],
        random_state=args.seed,
    )

    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    if args.normalize:
        scaler = StandardScaler()
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
        joblib.dump(scaler, out_dir / "scaler.pkl")

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    report_path = out_dir / "split_report.txt"
    with open(report_path, "w") as f:
        f.write("Split report (stratified 80-20)\n")
        f.write("==============================\n\n")
        f.write(f"Total: {len(df)}  Train: {len(train_df)}  Test: {len(test_df)}\n\n")
        f.write("Label distribution (overall):\n")
        for lbl, cnt in df["label"].value_counts().sort_index().items():
            pct = 100.0 * cnt / len(df)
            f.write(f"  {lbl}: {cnt} ({pct:.1f}%)\n")
        f.write("\nTrain:\n")
        for lbl, cnt in train_df["label"].value_counts().sort_index().items():
            pct = 100.0 * cnt / len(train_df)
            f.write(f"  {lbl}: {cnt} ({pct:.1f}%)\n")
        f.write("\nTest:\n")
        for lbl, cnt in test_df["label"].value_counts().sort_index().items():
            pct = 100.0 * cnt / len(test_df)
            f.write(f"  {lbl}: {cnt} ({pct:.1f}%)\n")
        if args.normalize:
            f.write("\nTrain/test features normalized (StandardScaler fit on train). scaler.pkl saved.\n")


if __name__ == "__main__":
    main()
