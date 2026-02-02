import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT
FEATURES_CSV = DEFAULT_OUTPUT / "features" / "iemocap" / "iemocap_features.csv"  # project root: keep dataset subfolder
SPLITS_DIR = DEFAULT_OUTPUT / "features" / "iemocap" / "splits"

NON_FEATURE_COLS = ["label", "file_path", "dataset"]


def _run_stratified_80_20(df, out_dir, test_size, seed, normalize, non_feature_cols):
    split_folder = "80_20"
    out_dir = out_dir / split_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )

    feature_cols = [c for c in train_df.columns if c not in non_feature_cols]
    if normalize:
        scaler = StandardScaler()
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
        joblib.dump(scaler, out_dir / "scaler.pkl")

    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    with open(out_dir / "split_report.txt", "w") as f:
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
        if normalize:
            f.write("\nTrain/test features normalized (StandardScaler fit on train). scaler.pkl saved.\n")


def _run_loso(df, out_dir, normalize, non_feature_cols):
    """Leave-One-Subject-Out: subject derived from file_path (e.g. Session1, Session2)."""
    # Derive subject from path: .../Session1/... or .../Session2/...
    df = df.copy()
    df["subject"] = df["file_path"].astype(str).str.extract(r"(Session\d+)", expand=False)
    missing = df["subject"].isna()
    if missing.any():
        df = df[~missing].copy()
    subjects = sorted(df["subject"].unique().tolist())
    if not subjects:
        raise ValueError("Could not derive any subject from file_path (expected Session1, Session2, ...).")

    split_folder = "loso"
    out_dir = out_dir / split_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c not in non_feature_cols and c != "subject"]

    report_lines = [
        "Split report (LOSO - Leave-One-Subject-Out)",
        "===========================================",
        "",
        f"Subjects: {len(subjects)}  {', '.join(subjects)}",
        f"Total samples: {len(df)}",
        "",
    ]

    for fold_i, test_subject in enumerate(subjects):
        fold_dir = out_dir / f"fold_{fold_i}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df = df[df["subject"] != test_subject].drop(columns=["subject"])
        test_df = df[df["subject"] == test_subject].drop(columns=["subject"])

        if normalize:
            scaler = StandardScaler()
            train_df = train_df.copy()
            test_df = test_df.copy()
            train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
            test_df[feature_cols] = scaler.transform(test_df[feature_cols])
            joblib.dump(scaler, fold_dir / "scaler.pkl")

        train_df.to_csv(fold_dir / "train.csv", index=False)
        test_df.to_csv(fold_dir / "test.csv", index=False)

        with open(fold_dir / "split_report.txt", "w") as f:
            f.write(f"LOSO fold {fold_i} (test subject: {test_subject})\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Train: {len(train_df)}  Test: {len(test_df)}\n\n")
            f.write("Train label distribution:\n")
            for lbl, cnt in train_df["label"].value_counts().sort_index().items():
                pct = 100.0 * cnt / len(train_df)
                f.write(f"  {lbl}: {cnt} ({pct:.1f}%)\n")
            f.write("\nTest label distribution:\n")
            for lbl, cnt in test_df["label"].value_counts().sort_index().items():
                pct = 100.0 * cnt / len(test_df)
                f.write(f"  {lbl}: {cnt} ({pct:.1f}%)\n")
            if normalize:
                f.write("\nNormalized (StandardScaler fit on train). scaler.pkl saved.\n")

        report_lines.append(f"fold_{fold_i} (test={test_subject}): train={len(train_df)}, test={len(test_df)}")

    report_lines.append("")
    if normalize:
        report_lines.append("Each fold: StandardScaler fit on train, transform train+test. scaler.pkl per fold.")
    with open(out_dir / "split_report.txt", "w") as f:
        f.write("\n".join(report_lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-dir", type=Path, default=None, help="Workflow output root; uses <workflow-dir>/features/...")
    parser.add_argument("--features-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="stratified_80_20", choices=["stratified_80_20", "loso"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true", help="Fit StandardScaler on train and normalize train/test")
    args = parser.parse_args()

    if args.workflow_dir is not None:
        base = Path(args.workflow_dir).resolve()
        features_csv = args.features_csv or (base / "features" / "iemocap_features.csv")
        out_dir = args.out_dir or (base / "features" / "splits")
    else:
        features_csv = args.features_csv or FEATURES_CSV
        out_dir = args.out_dir or SPLITS_DIR
    features_csv = Path(features_csv)
    out_dir = Path(out_dir)

    if not features_csv.is_file():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")

    df = pd.read_csv(features_csv)
    if "label" not in df.columns:
        raise ValueError("CSV must contain column 'label' for stratified split.")
    if "file_path" not in df.columns and args.split == "loso":
        raise ValueError("CSV must contain column 'file_path' for LOSO (to derive subject).")

    if args.split == "loso":
        _run_loso(df, out_dir, args.normalize, NON_FEATURE_COLS)
    else:
        _run_stratified_80_20(df, out_dir, args.test_size, args.seed, args.normalize, NON_FEATURE_COLS)


if __name__ == "__main__":
    main()
