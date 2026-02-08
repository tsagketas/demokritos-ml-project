import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_pca(
    workflow_dir,
    n_components=None,
    seed=42,
    variance_threshold=0.95,
    train_csv=None,
    test_csv=None,
):
    workflow_dir = Path(workflow_dir)
    
    # Define paths
    splits_80_20 = workflow_dir / "features" / "splits" / "80_20"
    train_csv = Path(train_csv) if train_csv is not None else (splits_80_20 / "train.csv")
    test_csv = Path(test_csv) if test_csv is not None else (splits_80_20 / "test.csv")
    
    # 1. Load Data
    if not train_csv.is_file():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not test_csv.is_file():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    df_train_raw = pd.read_csv(train_csv)
    df_test_raw = pd.read_csv(test_csv)

    # 2. Safety Check: Is it already PCA-transformed?
    if any(c.startswith("pca_") for c in df_train_raw.columns):
        print(f"WARNING: '{train_csv.name}' already contains PCA features.")
        # Try to find the backup
        train_old = train_csv.parent / "train_old.csv"
        test_old = test_csv.parent / "test_old.csv"
        if train_old.is_file() and test_old.is_file():
            print(f"-> Found backup 'train_old.csv'. Using that as source instead.")
            df_train_raw = pd.read_csv(train_old)
            df_test_raw = pd.read_csv(test_old)
        else:
            raise ValueError(
                "Cannot run PCA: Input data is already PCA-transformed and no 'train_old.csv' backup found.\n"
                "Please re-run '02_split_train_test.py' to regenerate raw split files."
            )

    # 3. Identify Feature Columns
    non_feature_cols = ["label", "file_path", "dataset", "subject"]
    feature_cols = [c for c in df_train_raw.columns if c not in non_feature_cols]
    
    if not feature_cols:
        raise ValueError("No feature columns found in train.csv.")

    print(f"Source features: {len(feature_cols)} columns.")
    
    # 4. Standardize (Fit on Train, Transform Train & Test)
    # Critical: PCA requires unit variance to work as expected on diverse features.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train_raw[feature_cols])
    X_test_scaled = scaler.transform(df_test_raw[feature_cols])

    # 5. Fit PCA
    if n_components is None:
        n_components = variance_threshold  # float (0.0 to 1.0) means "keep enough for this variance"
    
    print(f"Fitting PCA on train set (target variance/components: {n_components})...")
    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    n_selected = pca.n_components_
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"-> Selected {n_selected} components.")
    print(f"-> Cumulative explained variance: {explained_var:.4f}")

    # 6. Save PCA Artifacts
    pca_info_dir = workflow_dir / "pca_info"
    pca_info_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, pca_info_dir / "scaler.pkl")
    joblib.dump(pca, pca_info_dir / "pca.pkl")
    joblib.dump(feature_cols, pca_info_dir / "pca_feature_cols.pkl")

    # 7. Generate Detailed Report
    # 7a. Variance Report
    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)
    df_var = pd.DataFrame({
        "component": range(1, len(evr) + 1),
        "explained_variance_ratio": evr,
        "cumulative_explained_variance": cumulative
    })
    df_var.to_csv(pca_info_dir / "explained_variance_ratio.csv", index=False)

    # 7b. Variance Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_var["component"], df_var["cumulative_explained_variance"], marker=".", linewidth=2)
    ax.axhline(y=explained_var, color='r', linestyle='--', alpha=0.5, label=f'Total: {explained_var:.4f}')
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Explained Variance (IEMOCAP Train)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(str(pca_info_dir / "cumulative_explained_variance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 7c. Loadings (Feature Contributions)
    # Which original features contribute most to the top 5 components?
    loadings = pca.components_.T  # Shape: (n_features, n_components)
    top_components = min(5, n_selected)
    
    report_lines = [
        "PCA Detailed Report",
        "===================",
        f"Input features: {len(feature_cols)}",
        f"Selected components: {n_selected}",
        f"Total Variance Explained: {explained_var:.6f}",
        "",
        "Top Components Analysis (Loadings):",
        "-----------------------------------"
    ]
    
    for i in range(top_components):
        # Get indices of top 5 absolute loadings for this component
        loading_vec = loadings[:, i]
        # Sort by absolute value descending
        top_indices = np.argsort(np.abs(loading_vec))[::-1][:5]
        
        report_lines.append(f"\nComponent {i+1} (Explains {evr[i]*100:.2f}% var):")
        for idx in top_indices:
            feat_name = feature_cols[idx]
            weight = loading_vec[idx]
            report_lines.append(f"  {weight:+.4f} * {feat_name}")

    with open(pca_info_dir / "pca_report.txt", "w") as f:
        f.write("\n".join(report_lines))
        f.write("\n\nArtifacts:\n- pca_info/scaler.pkl\n- pca_info/pca.pkl\n- pca_info/pca_feature_cols.pkl\n")
        f.write("- pca_info/explained_variance_ratio.csv\n- pca_info/cumulative_explained_variance.png")

    print(f"Report saved to: {pca_info_dir / 'pca_report.txt'}")

    # 8. Save Transformed Data
    # Construct new DataFrames
    pca_col_names = [f"pca_{i}" for i in range(n_selected)]
    
    # Helper to attach meta columns
    def create_out_df(pca_data, source_df):
        out = pd.DataFrame(pca_data, columns=pca_col_names)
        for col in non_feature_cols:
            if col in source_df.columns:
                out[col] = source_df[col].values
        return out

    df_train_pca = create_out_df(X_train_pca, df_train_raw)
    df_test_pca = create_out_df(X_test_pca, df_test_raw)

    # Backup only if we loaded from 'train.csv' (not if we already loaded from backup)
    # If we loaded from train_old, train.csv is currently 'bad' (PCA'd), so we can overwrite it safely.
    # If we loaded from train.csv (raw), we must backup.
    
    out_dir = train_csv.parent
    train_old = out_dir / "train_old.csv"
    test_old = out_dir / "test_old.csv"

    # Save backups if they don't exist (preserve the ORIGINAL raw data)
    if not train_old.exists():
        df_train_raw.to_csv(train_old, index=False)
    if not test_old.exists():
        df_test_raw.to_csv(test_old, index=False)

    df_train_pca.to_csv(out_dir / "train.csv", index=False)
    df_test_pca.to_csv(out_dir / "test.csv", index=False)
    
    print(f"Saved transformed data to: {out_dir}")
    print(f"(Original data backed up to *_old.csv)")

def main():
    parser = argparse.ArgumentParser(description="Run PCA on features and update train/test CSVs.")
    parser.add_argument("--workflow-dir", type=Path, required=True, help="Workflow directory")
    parser.add_argument("--n-components", type=int, default=None, help="Force fixed number of components")
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.99,
        help="Keep variance ratio (default 0.99)",
    )
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    run_pca(
        args.workflow_dir,
        args.n_components,
        args.seed,
        args.variance_threshold,
        args.train_csv,
        args.test_csv,
    )

if __name__ == "__main__":
    main()
