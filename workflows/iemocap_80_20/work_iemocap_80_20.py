"""
IEMOCAP 80-20 workflow: preprocess → split → train → evaluate → hyper-tune → evaluate.
Tuning saves to models/; final evaluate uses those models.
Run from project root: python workflows/iemocap_80_20/work_iemocap_80_20.py
"""
import argparse
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
    parser = argparse.ArgumentParser(description="IEMOCAP 80-20 workflow runner")
    parser.add_argument(
        "--one-shot-csv",
        type=str,
        default=str(PROJECT_ROOT / "cremad_zero_shot_dataset" / "cremad_features.csv"),
        help="One-shot CSV to evaluate at the end (default: CREMAD features).",
    )
    parser.add_argument(
        "--one-shot-already-normalized",
        action="store_true",
        help="Pass through to one-shot step (skips scaling of one-shot features).",
    )
    args = parser.parse_args()

    py = sys.executable
    # run(
    #     [py, str(SCRIPTS / "01_preprocess_data.py"), "--iemocap", "--workflow-dir", WDIR],
    #     "1. Preprocess IEMOCAP",
    # )
    # run(
    #     [py, str(SCRIPTS / "02_split_train_test.py"), "--workflow-dir", WDIR, "--normalize"],
    #     "2. Split 80-20 (stratified, normalized)",
    # )
    # run(
    #     [py, str(SCRIPTS / "03_train_models.py"), "--workflow-dir", WDIR],
    #     "3. Train models",
    # )
    # run(
    #     [py, str(SCRIPTS / "04_evaluate_models.py"), "--workflow-dir", WDIR],
    #     "4. Evaluate models",
    # )
    run(
        [py, str(SCRIPTS / "05_hyperparam_tuning.py"), "--workflow-dir", WDIR],
        "5. Hyperparameter tuning (saves to models/)",
    )
    run(
        [py, str(SCRIPTS / "04_evaluate_models.py"), "--workflow-dir", WDIR],
        "6. Evaluate (tuned models in models/)",
    )

    cmd = [
        py,
        str(SCRIPTS / "08_one_shot_predict_eval.py"),
        "--workflow-dir",
        WDIR,
        "--one-shot-csv",
        args.one_shot_csv,
    ]
    if args.one_shot_already_normalized:
        cmd.append("--one-shot-already-normalized")
    run(cmd, "7. One-shot predict + evaluate (saved to one_shot_results/)")
    print("\n--- Workflow done ---")
    print(f"Outputs: {WORKFLOW_DIR} (features/, models/, results/, tuning/)")


if __name__ == "__main__":
    main()
