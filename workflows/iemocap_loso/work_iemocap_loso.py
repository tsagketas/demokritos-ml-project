"""
IEMOCAP LOSO workflow: preprocess → split (LOSO) → per-fold train [or tune] → per-fold evaluate → aggregate.
Run from project root: python workflows/iemocap_loso/work_iemocap_loso.py
Use --tune to run hyperparameter tuning per fold instead of fixed-param training (slower).
"""
import argparse
import subprocess
import sys
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKFLOW_DIR.parent.parent
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
    parser = argparse.ArgumentParser(description="IEMOCAP LOSO workflow (optionally with per-fold hyperparameter tuning)")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning per fold instead of fixed-param training")
    parser.add_argument("--n-iter", type=int, default=20, help="RandomizedSearchCV iterations per model when --tune (default: 20)")
    parser.add_argument("--cv", type=int, default=5, help="CV folds for tuning when --tune (default: 5)")
    args = parser.parse_args()

    py = sys.executable

    run(
        [py, str(SCRIPTS / "01_preprocess_data.py"), "--iemocap", "--workflow-dir", WDIR],
        "1. Preprocess IEMOCAP",
    )
    run(
        [py, str(SCRIPTS / "02_split_train_test.py"), "--workflow-dir", WDIR, "--split", "loso", "--normalize"],
        "2. Split LOSO (Leave-One-Subject-Out, normalized)",
    )

    splits_loso = WORKFLOW_DIR / "features" / "splits" / "loso"
    if not splits_loso.is_dir():
        print("LOSO splits not found.", file=sys.stderr)
        sys.exit(1)
    fold_dirs = sorted(d for d in splits_loso.iterdir() if d.is_dir() and d.name.startswith("fold_"))
    if not fold_dirs:
        print("No fold_* directories under features/splits/loso.", file=sys.stderr)
        sys.exit(1)

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        train_csv = str(fold_dir / "train.csv")
        test_csv = str(fold_dir / "test.csv")
        models_dir = str(WORKFLOW_DIR / "models" / fold_name)
        results_dir = str(WORKFLOW_DIR / "results" / fold_name)

        if args.tune:
            run(
                [
                    py, str(SCRIPTS / "05_hyperparam_tuning.py"),
                    "--workflow-dir", WDIR,
                    "--train-csv", train_csv,
                    "--out-dir", models_dir,
                    "--n-iter", str(args.n_iter),
                    "--cv", str(args.cv),
                ],
                f"3. Hyperparameter tuning ({fold_name})",
            )
        else:
            run(
                [
                    py, str(SCRIPTS / "03_train_models.py"),
                    "--workflow-dir", WDIR,
                    "--train-csv", train_csv,
                    "--out-dir", models_dir,
                ],
                f"3. Train models ({fold_name})",
            )
        run(
            [
                py, str(SCRIPTS / "04_evaluate_models.py"),
                "--workflow-dir", WDIR,
                "--train-csv", train_csv,
                "--test-csv", test_csv,
                "--models-dir", models_dir,
                "--results-dir", results_dir,
            ],
            f"4. Evaluate models ({fold_name})",
        )

    run(
        [py, str(SCRIPTS / "06_aggregate_loso_results.py"), "--workflow-dir", WDIR],
        "5. Aggregate LOSO results (mean ± std)",
    )

    print("\n--- LOSO workflow done ---")
    print(f"Outputs: {WORKFLOW_DIR}")
    print(f"  features/splits/loso/fold_*/  (train.csv, test.csv, scaler.pkl)")
    print(f"  models/fold_*/  (trained models per fold)")
    print(f"  results/fold_*/  (per-fold reports)")
    print(f"  results/loso_summary.txt  (aggregated metrics)")


if __name__ == "__main__":
    main()
