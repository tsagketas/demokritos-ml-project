"""
One-shot prediction + evaluation runner.

Usage (from project root):
  python scripts/08_one_shot_predict_eval.py --workflow-dir workflows/iemocap_80_20 --one-shot-csv path/to/one_shot.csv

Expected one-shot CSV format:
  - Must contain the same feature columns as the workflow's trained models (loaded from models/feature_cols.pkl).
  - Can optionally include:
      - label: to enable evaluation (will run scripts/04_evaluate_models.py)
      - file_path / dataset: kept in the saved prediction outputs

Outputs:
  <workflow-dir>/one_shot_results/   (REPLACED each run unless --out-dir is set)
    - one_shot_input_raw.csv (copy of input)
    - one_shot_features_used.csv (features actually fed to models; scaled unless --one-shot-already-normalized)
    - predictions_<model>.csv
    - evaluation/<model>/* (report + confusion matrix)
"""

import argparse
from pathlib import Path
import shutil

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAMES = ["rf", "xgb", "svm", "knn", "dtr", "logistic", "nb"]


def main():
    parser = argparse.ArgumentParser(description="Run one-shot predictions (and optional evaluation) using workflow models.")
    parser.add_argument("--workflow-dir", type=Path, required=True, help="Workflow directory (contains models/, features/, etc.)")
    parser.add_argument("--one-shot-csv", type=Path, required=True, help="One-shot CSV (features + optional label)")
    parser.add_argument(
        "--one-shot-already-normalized",
        action="store_true",
        help="If set, do NOT apply the workflow scaler to the one-shot features before predicting/evaluating.",
    )
    parser.add_argument("--models-dir", type=Path, default=None, help="Override models dir (default: <workflow-dir>/models)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override output dir (default: <workflow-dir>/one_shot_results)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation step even if labels exist.")
    args = parser.parse_args()

    workflow_dir = Path(args.workflow_dir).resolve()
    one_shot_csv = Path(args.one_shot_csv).resolve()
    if not one_shot_csv.is_file():
        raise FileNotFoundError(f"One-shot CSV not found: {one_shot_csv}")

    models_dir = Path(args.models_dir or (workflow_dir / "models")).resolve()
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models dir not found: {models_dir}")

    feature_cols_path = models_dir / "feature_cols.pkl"
    scaler_path = models_dir / "scaler.pkl"
    le_path = models_dir / "label_encoder.pkl"
    if not feature_cols_path.is_file():
        raise FileNotFoundError(f"Missing model artifact: {feature_cols_path}")
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Missing model artifact: {scaler_path}")
    if not le_path.is_file():
        raise FileNotFoundError(f"Missing model artifact: {le_path}")

    feature_cols = joblib.load(feature_cols_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)

    df_one = pd.read_csv(one_shot_csv)
    missing = [c for c in feature_cols if c not in df_one.columns]
    if missing:
        # PCA workflow support:
        # If models were trained on PCA features (pca_*) but the one-shot CSV contains the ORIGINAL features,
        # try to transform using workflow_dir/pca_info/{scaler.pkl,pca.pkl,pca_feature_cols.pkl}.
        pca_info_dir = workflow_dir / "pca_info"
        pca_scaler_path = pca_info_dir / "scaler.pkl"
        pca_path = pca_info_dir / "pca.pkl"
        pca_feature_cols_path = pca_info_dir / "pca_feature_cols.pkl"

        can_try_pca = (
            pca_info_dir.is_dir()
            and pca_scaler_path.is_file()
            and pca_path.is_file()
            and pca_feature_cols_path.is_file()
            and all(str(c).startswith("pca_") for c in feature_cols)
        )
        if not can_try_pca:
            raise ValueError(f"One-shot CSV missing {len(missing)} required feature columns (examples: {missing[:10]})")

        pca_scaler = joblib.load(pca_scaler_path)
        pca = joblib.load(pca_path)
        pca_feature_cols = joblib.load(pca_feature_cols_path)

        # Repair older PCA workflows that accidentally saved pca_* instead of raw feature cols
        if pca_feature_cols and all(str(c).startswith("pca_") for c in pca_feature_cols):
            train_old = workflow_dir / "features" / "splits" / "80_20" / "train_old.csv"
            if train_old.is_file():
                df_train_old = pd.read_csv(train_old, nrows=1)
                recovered = [c for c in df_train_old.columns if c not in ["label", "file_path", "dataset"]]
                if recovered:
                    pca_feature_cols = recovered
                    joblib.dump(pca_feature_cols, pca_feature_cols_path)
                    print("Repaired pca_feature_cols.pkl using train_old.csv raw feature columns.")

        missing_raw = [c for c in pca_feature_cols if c not in df_one.columns]
        if missing_raw:
            raise ValueError(
                "One-shot CSV does not match required model features, and PCA fallback failed. "
                f"Missing raw PCA feature columns: {len(missing_raw)} (examples: {missing_raw[:10]})."
            )

        # raw -> pca_info scaler -> pca transform => pca_* DataFrame
        X_raw = df_one[pca_feature_cols]
        X_raw_scaled = pca_scaler.transform(X_raw)
        X_pca = pca.transform(X_raw_scaled)
        X_pca_df = pd.DataFrame(X_pca, columns=feature_cols)

        # Replace df_one with a view that contains the PCA columns needed for the model
        df_one = df_one.copy()
        for c in feature_cols:
            df_one[c] = X_pca_df[c]
        missing = []  # satisfied

    # Prepare output directory
    out_dir = Path(args.out_dir or (workflow_dir / "one_shot_results")).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep a copy of the raw input
    df_one.to_csv(out_dir / "one_shot_input_raw.csv", index=False)

    X_one_df = df_one[feature_cols]
    if args.one_shot_already_normalized:
        X_one_scaled = X_one_df
    else:
        X_one_scaled = scaler.transform(X_one_df)

    # Ensure feature names are preserved for downstream predict() to avoid sklearn warnings
    if not isinstance(X_one_scaled, pd.DataFrame):
        X_one_scaled = pd.DataFrame(X_one_scaled, columns=feature_cols)

    # Save the exact features used for predictions (helps reproducibility/debugging)
    df_used = df_one.copy()
    df_used.loc[:, feature_cols] = X_one_scaled.values
    df_used.to_csv(out_dir / "one_shot_features_used.csv", index=False)

    # Predict with each available model and save per-model prediction files
    meta_cols = [c for c in ["file_path", "dataset", "label"] if c in df_one.columns]
    has_labels = "label" in df_one.columns
    if has_labels:
        y_true = df_one["label"]
        y_true_enc = le.transform(y_true)
        class_names = list(le.classes_)
        labels_enc = list(range(len(class_names)))

    eval_summary_rows = []
    for name in MODEL_NAMES:
        model_path = models_dir / f"{name}.pkl"
        if not model_path.is_file():
            continue
        model = joblib.load(model_path)

        y_pred_enc = model.predict(X_one_scaled)
        y_pred = le.inverse_transform(y_pred_enc)

        out = pd.DataFrame()
        for c in meta_cols:
            out[c] = df_one[c]
        out["pred_label"] = y_pred
        out["pred_label_enc"] = y_pred_enc

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_one_scaled)
                # Align columns to label encoder class order
                class_names = list(le.classes_)
                for i, cls in enumerate(class_names):
                    out[f"proba_{cls}"] = proba[:, i]
            except Exception:
                # Some models might not support predict_proba depending on training settings
                pass

        out.to_csv(out_dir / f"predictions_{name}.csv", index=False)

        # Optional evaluation (only if ground-truth labels exist)
        if args.skip_eval or not has_labels:
            continue

        eval_out = out_dir / "evaluation" / name
        eval_out.mkdir(parents=True, exist_ok=True)

        accuracy = accuracy_score(y_true_enc, y_pred_enc)
        f1 = f1_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)
        precision = precision_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)
        recall = recall_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true_enc, y_pred_enc, labels=labels_enc)

        with open(eval_out / "report.txt", "w") as f:
            f.write(f"One-shot evaluation report: {name}\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1-score (weighted): {f1:.4f}\n")
            f.write(f"Precision (weighted): {precision:.4f}\n")
            f.write(f"Recall (weighted): {recall:.4f}\n\n")
            f.write("Confusion matrix:\n")
            f.write(f"Labels: {class_names}\n\n")
            for i, row in enumerate(cm):
                f.write("  " + " ".join(f"{v:5d}" for v in row) + f"  | {class_names[i]}\n")
            f.write("  " + "-" * (6 * len(class_names)) + "\n")
            f.write("  " + " ".join(f"{c:>5s}" for c in class_names) + "\n")

        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(eval_out / "confusion_matrix.csv")

        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            xticklabels=class_names,
            yticklabels=class_names,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax_cm,
            cbar_kws={"label": "Count"},
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        ax_cm.set_title(f"One-shot confusion matrix: {name}")
        plt.tight_layout()
        fig_cm.savefig(str(eval_out / "confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close(fig_cm)

        eval_summary_rows.append(
            {
                "model": name,
                "accuracy": accuracy,
                "f1_weighted": f1,
                "precision_weighted": precision,
                "recall_weighted": recall,
            }
        )

    if (not args.skip_eval) and has_labels and eval_summary_rows:
        pd.DataFrame(eval_summary_rows).sort_values("model").to_csv(out_dir / "evaluation_summary.csv", index=False)


if __name__ == "__main__":
    main()

