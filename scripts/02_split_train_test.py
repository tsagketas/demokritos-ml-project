import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def _run_pca(
    workflow_dir,
    n_components=None,
    seed=42,
    variance_threshold=0.95,
    train_csv=None,
    test_csv=None,
):
    workflow_dir = Path(workflow_dir)

    non_feature_cols = ["label", "file_path", "dataset"]

    # Default: operate on existing split train/test
    splits_80_20 = workflow_dir / "features" / "splits" / "80_20"
    train_csv = Path(train_csv) if train_csv is not None else (splits_80_20 / "train.csv")
    test_csv = Path(test_csv) if test_csv is not None else (splits_80_20 / "test.csv")
    if not train_csv.is_file():
        raise FileNotFoundError(f"Train CSV not found for PCA: {train_csv}")
    if not test_csv.is_file():
        raise FileNotFoundError(f"Test CSV not found for PCA: {test_csv}")

    df_train_raw = pd.read_csv(train_csv)
    df_test_raw = pd.read_csv(test_csv)

    feature_cols = [c for c in df_train_raw.columns if c not in non_feature_cols]
    if not feature_cols:
        raise ValueError("No feature columns found in train.csv (expected columns besides label/file_path/dataset).")
    missing_in_test = [c for c in feature_cols if c not in df_test_raw.columns]
    if missing_in_test:
        raise ValueError(f"Test CSV missing {len(missing_in_test)} feature columns (examples: {missing_in_test[:10]})")

    # --- Fit scaler + PCA on TRAIN only ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train_raw[feature_cols])
    X_test_scaled = scaler.transform(df_test_raw[feature_cols])

    if n_components is None:
        n_components = variance_threshold
    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    pca_cols = [f"pca_{i}" for i in range(X_train_pca.shape[1])]

    df_train_pca = pd.DataFrame(X_train_pca, columns=pca_cols)
    df_train_pca["label"] = df_train_raw["label"].values
    df_train_pca["file_path"] = df_train_raw["file_path"].values
    df_train_pca["dataset"] = df_train_raw["dataset"].values

    df_test_pca = pd.DataFrame(X_test_pca, columns=pca_cols)
    df_test_pca["label"] = df_test_raw["label"].values
    df_test_pca["file_path"] = df_test_raw["file_path"].values
    df_test_pca["dataset"] = df_test_raw["dataset"].values

    # Save PCA artifacts + plots/info
    pca_info_dir = workflow_dir / "pca_info"
    pca_info_dir.mkdir(parents=True, exist_ok=True)

    # scaler/pca are fit on IEMOCAP train only
    joblib.dump(scaler, pca_info_dir / "scaler.pkl")
    joblib.dump(pca, pca_info_dir / "pca.pkl")
    joblib.dump(feature_cols, pca_info_dir / "pca_feature_cols.pkl")

    # Save PCA explained variance report + plot
    evr = np.array(getattr(pca, "explained_variance_ratio_", []), dtype=float)
    if evr.size:
        cumulative = np.cumsum(evr)
        df_var = pd.DataFrame(
            {
                "component": np.arange(1, evr.size + 1),
                "explained_variance_ratio": evr,
                "cumulative_explained_variance": cumulative,
            }
        )
        df_var.to_csv(pca_info_dir / "explained_variance_ratio.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_var["component"], df_var["cumulative_explained_variance"], marker="o", linewidth=2)
        ax.set_xlabel("Number of components")
        ax.set_ylabel("Cumulative explained variance")
        ax.set_title("PCA cumulative explained variance (fit on IEMOCAP train)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.01)
        plt.tight_layout()
        fig.savefig(str(pca_info_dir / "cumulative_explained_variance.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        with open(pca_info_dir / "pca_report.txt", "w") as f:
            f.write("PCA report\n")
            f.write("==========\n\n")
            f.write(f"Fit on: IEMOCAP train split only (seed={seed})\n")
            f.write(f"Original feature dims: {len(feature_cols)}\n")
            f.write(f"Requested n_components: {n_components}\n")
            f.write(f"Selected n_components_: {getattr(pca, 'n_components_', 'unknown')}\n")
            f.write(f"Cumulative explained variance: {float(cumulative[-1]):.6f}\n")
            f.write("\nArtifacts:\n")
            f.write(f"- pca_info/scaler.pkl\n- pca_info/pca.pkl\n- pca_info/pca_feature_cols.pkl\n")
            f.write(f"- pca_info/explained_variance_ratio.csv\n- pca_info/cumulative_explained_variance.png\n")
    
    # Backup the old split files (no PCA) and overwrite with PCA versions
    out_dir = Path(train_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_old = out_dir / "train_old.csv"
    test_old = out_dir / "test_old.csv"
    # Overwrite backups each run to keep it simple
    df_train_raw.to_csv(train_old, index=False)
    df_test_raw.to_csv(test_old, index=False)

    df_train_pca.to_csv(out_dir / "train.csv", index=False)
    df_test_pca.to_csv(out_dir / "test.csv", index=False)

    with open(out_dir / "split_report.txt", "w") as f:
        f.write("Split report (existing 80-20) + PCA\n")
        f.write("==================================\n\n")
        f.write(f"Train: {len(df_train_pca)}  Test: {len(df_test_pca)}\n")
        f.write(f"PCA components: {len(pca_cols)}  (requested: {n_components})\n")
        if evr.size:
            f.write(f"Cumulative explained variance: {float(np.cumsum(evr)[-1]):.6f}\n")
        f.write("\nTrain:\n")
        for lbl, cnt in pd.Series(df_train_pca["label"]).value_counts().sort_index().items():
            pct = 100.0 * cnt / len(df_train_pca)
            f.write(f"  {lbl}: {cnt} ({pct:.1f}%)\n")
        f.write("\nTest:\n")
        for lbl, cnt in pd.Series(df_test_pca["label"]).value_counts().sort_index().items():
            pct = 100.0 * cnt / len(df_test_pca)
            f.write(f"  {lbl}: {cnt} ({pct:.1f}%)\n")


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
