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
  <workflow-dir>/one_shot_results/   (REPLACED each run)
    - one_shot_input_raw.csv (copy of input)
    - train.csv / test.csv (scaled to match model input; used for evaluation)
    - scaler.pkl (marker so evaluation script treats train/test as already-normalized)
    - predictions_<model>.csv
    - evaluation/<model>/* (reports, confusion matrix, learning curves)
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil

import joblib
import pandas as pd

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

    # Workflow train split (needed to evaluate)
    workflow_train_csv = workflow_dir / "features" / "splits" / "80_20" / "train.csv"
    if not workflow_train_csv.is_file():
        raise FileNotFoundError(f"Workflow train.csv not found: {workflow_train_csv}")

    # Determine whether the workflow's train.csv is already normalized (split script with --normalize)
    workflow_split_scaler = workflow_train_csv.parent / "scaler.pkl"
    workflow_train_already_normalized = workflow_split_scaler.is_file()

    df_one = pd.read_csv(one_shot_csv)
    missing = [c for c in feature_cols if c not in df_one.columns]
    if missing:
        raise ValueError(f"One-shot CSV missing {len(missing)} required feature columns (examples: {missing[:10]})")

    # Prepare output directory
    out_dir = workflow_dir / "one_shot_results"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep a copy of the raw input
    df_one.to_csv(out_dir / "one_shot_input_raw.csv", index=False)

    # Load workflow train and build scaled copies to avoid double-scaling logic inside 04_evaluate_models.py
    df_train = pd.read_csv(workflow_train_csv)
    if any(c not in df_train.columns for c in feature_cols):
        raise ValueError("Workflow train.csv does not contain the expected feature columns from models/feature_cols.pkl")

    X_train_df = df_train[feature_cols]
    if workflow_train_already_normalized:
        X_train_scaled = X_train_df
    else:
        X_train_scaled = scaler.transform(X_train_df)

    X_one_df = df_one[feature_cols]
    if args.one_shot_already_normalized:
        X_one_scaled = X_one_df
    else:
        X_one_scaled = scaler.transform(X_one_df)

    # Ensure feature names are preserved for downstream predict() to avoid sklearn warnings
    if not isinstance(X_train_scaled, pd.DataFrame):
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    if not isinstance(X_one_scaled, pd.DataFrame):
        X_one_scaled = pd.DataFrame(X_one_scaled, columns=feature_cols)

    # Write scaled train/test CSVs into out_dir and place a scaler.pkl marker next to them
    # so scripts/04_evaluate_models.py treats them as already-normalized (and won't re-transform).
    df_train_scaled = df_train.copy()
    df_train_scaled.loc[:, feature_cols] = X_train_scaled.values
    df_train_scaled.to_csv(out_dir / "train.csv", index=False)

    df_one_scaled = df_one.copy()
    df_one_scaled.loc[:, feature_cols] = X_one_scaled.values
    df_one_scaled.to_csv(out_dir / "test.csv", index=False)

    # Marker file (contents not used by 04_evaluate_models.py when already_normalized=True)
    joblib.dump(scaler, out_dir / "scaler.pkl")

    # Predict with each available model and save per-model prediction files
    meta_cols = [c for c in ["file_path", "dataset", "label"] if c in df_one.columns]
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
    if args.skip_eval:
        return
    if "label" not in df_one.columns:
        return

    eval_out = out_dir / "evaluation"
    eval_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "04_evaluate_models.py"),
        "--train-csv",
        str(out_dir / "train.csv"),
        "--test-csv",
        str(out_dir / "test.csv"),
        "--models-dir",
        str(models_dir),
        "--results-dir",
        str(eval_out),
    ]
    r = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()

