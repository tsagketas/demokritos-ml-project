"""
Hyperparameter tuning script.
Uses RandomizedSearchCV with weighted F1 (aligned with evaluation metrics).
Best params and optional best model are saved. Docker-friendly.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT
SPLITS_DIR = DEFAULT_OUTPUT / "features" / "iemocap" / "splits"
MODELS_DIR = DEFAULT_OUTPUT / "models"

NON_FEATURE_COLS = ["label", "file_path", "dataset"]

# Param grids for RandomizedSearchCV (subset for speed; extend as needed)
PARAM_GRIDS = {
    "rf": {
        "n_estimators": [50, 100, 200],
        "max_depth": [8, 12, 15, 20],
        "min_samples_leaf": [2, 5, 10],
    },
    "xgb": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.05, 0.1, 0.2],
    },
    "svm": {
        "C": [0.5, 1.0, 2.0],
        "kernel": ["rbf"],
        "gamma": ["scale", "auto"],
    },
    "knn": {
        "n_neighbors": [5, 10, 15, 25],
        "weights": ["uniform", "distance"],
    },
    "dtr": {
        "max_depth": [8, 12, 16],
        "min_samples_leaf": [2, 5, 10],
    },
    "logistic": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "saga"],
        "max_iter": [500, 1000],
    },
    "nb": {
        "var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
    },
}

# Base estimators (same as train script for consistency)
def get_estimators(seed: int):
    return {
        "rf": RandomForestClassifier(random_state=seed),
        "xgb": XGBClassifier(random_state=seed),
        "svm": SVC(probability=True, random_state=seed),
        "knn": KNeighborsClassifier(),
        "dtr": DecisionTreeClassifier(random_state=seed),
        "logistic": LogisticRegression(random_state=seed, max_iter=1000),
        "nb": GaussianNB(),
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning (RandomizedSearchCV, F1-weighted)")
    parser.add_argument("--workflow-dir", type=Path, default=None, help="Workflow output root; uses <workflow-dir>/models/..., features/...")
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--models", type=str, nargs="+", default=None, help="e.g. rf xgb svm (default: all)")
    parser.add_argument("--n-iter", type=int, default=20, help="RandomizedSearchCV iterations per model")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-scale", action="store_true", help="Skip StandardScaler")
    parser.add_argument("--verbose", type=int, default=1, help="RandomizedSearchCV verbosity (0=silent)")
    parser.add_argument("--save-best-model", action="store_true", help="(Deprecated) Best models are always saved to out_dir as <name>.pkl")
    args = parser.parse_args()

    if args.workflow_dir is not None:
        base = Path(args.workflow_dir).resolve()
        train_path = args.train_csv or (base / "features" / "splits" / "80_20" / "train.csv")
        out_dir = args.out_dir or (base / "models")
    else:
        train_path = args.train_csv or SPLITS_DIR / "80_20" / "train.csv"
        out_dir = args.out_dir or MODELS_DIR
    train_path = Path(train_path)
    out_dir = Path(out_dir)

    if not train_path.is_file():
        raise FileNotFoundError(f"Train CSV not found: {train_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df["label"]

    split_dir = train_path.parent
    scaler_path = split_dir / "scaler.pkl"
    
    if args.no_scale:
        print("Skipping scaling as requested (--no-scale).")
        X_scaled = X
        scaler = None
    elif scaler_path.is_file():
        print(f"Loading existing scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
    else:
        print("Fitting new StandardScaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    estimators = get_estimators(args.seed)
    model_names = list(args.models) if args.models else list(estimators.keys())
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    total_models = len([m for m in model_names if m in estimators])
    print(f"Starting tuning: models={total_models}, cv={args.cv}, n_iter={args.n_iter}")

    for name in model_names:
        if name not in estimators:
            print(f"Unknown model '{name}', skipping.")
            continue
        param_grid = PARAM_GRIDS.get(name, {})
        if not param_grid:
            print(f"No param grid for '{name}', skipping.")
            continue

        grid_size = 1
        for values in param_grid.values():
            grid_size *= len(values)
        actual_iters = min(args.n_iter, grid_size) if grid_size > 0 else args.n_iter
        total_fits = actual_iters * args.cv
        print(f"\n[{name}] grid_size={grid_size}, iters={actual_iters}, total_fits={total_fits}")
        start_ts = time.perf_counter()

        base = estimators[name]
        search = RandomizedSearchCV(
            base,
            param_distributions=param_grid,
            n_iter=args.n_iter,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
            random_state=args.seed,
            refit=True,
            verbose=args.verbose,
        )
        search.fit(X_scaled, y_enc)

        best_params = search.best_params_
        best_score = search.best_score_
        elapsed = time.perf_counter() - start_ts

        with open(out_dir / f"{name}_best_params.json", "w") as f:
            json.dump({"best_params": best_params, "best_cv_f1_weighted": best_score}, f, indent=2)

        print(f"{name}: best F1 (weighted) = {best_score:.4f}, params = {best_params}")
        print(f"[{name}] done in {elapsed:.1f}s")

        joblib.dump(search.best_estimator_, out_dir / f"{name}.pkl")

    # Save artifacts so train script can use same feature setup
    joblib.dump(scaler, out_dir / "scaler.pkl")
    joblib.dump(le, out_dir / "label_encoder.pkl")
    joblib.dump(feature_cols, out_dir / "feature_cols.pkl")
    print(f"Tuning results and artifacts saved under {out_dir}")


if __name__ == "__main__":
    main()
