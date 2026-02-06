"""
IEMOCAP PCA workflow: preprocess both datasets → PCA + split → train → evaluate → hyper-tune → evaluate.
Run from project root: python workflows/iemocap_pca/work_iemocap_pca.py
"""
import subprocess
import sys
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKFLOW_DIR.parent.parent  # workflows/ -> project root
SCRIPTS = PROJECT_ROOT / "scripts"
WDIR = str(WORKFLOW_DIR)


def run(cmd, description):
    print(f"\n--- {description} ---")
    print(" ".join(cmd))
    r = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        print(f"Failed: {description}", file=sys.stderr)
        sys.exit(r.returncode)


def main():
    py = sys.executable
    run(
        [py, str(SCRIPTS / "01_preprocess_data.py"), "--iemocap", "--workflow-dir", WDIR],
        "1. Preprocess IEMOCAP",
    )
    run(
        [py, str(SCRIPTS / "02_split_train_test.py"), "--workflow-dir", WDIR],
        "3. Split 80-20 (raw features, no normalize)",
    )
    run(
        [
            py,
            str(SCRIPTS / "07_run_pca.py"),
            "--workflow-dir",
            WDIR,
            "--variance-threshold",
            "0.99",
        ],
        "4. PCA on TRAIN only (overwrite train/test; backups to *_old.csv; pca_info/)",
    )
    run(
        [py, str(SCRIPTS / "03_train_models.py"), "--workflow-dir", WDIR],
        "5. Train models",
    )
    run(
        [py, str(SCRIPTS / "04_evaluate_models.py"), "--workflow-dir", WDIR],
        "6. Evaluate models",
    )
    run(
        [py, str(SCRIPTS / "05_hyperparam_tuning.py"), "--workflow-dir", WDIR],
        "7. Hyperparameter tuning (saves to models/)",
    )
    run(
        [py, str(SCRIPTS / "04_evaluate_models.py"), "--workflow-dir", WDIR],
        "8. Evaluate (tuned models in models/)",
    )
    run(
        [
            py,
            str(SCRIPTS / "08_one_shot_predict_eval.py"),
            "--workflow-dir",
            WDIR,
            "--one-shot-csv",
            str(PROJECT_ROOT / "cremad_zero_shot_dataset" / "cremad_features.csv"),
        ],
        "9. Zero-shot (CREMAD as one-shot test)",
    )


if __name__ == "__main__":
    main()