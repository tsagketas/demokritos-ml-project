import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "processed" / "features" / "iemocap" / "splits"
MODELS_DIR = PROJECT_ROOT / "processed" / "models"

NON_FEATURE_COLS = ["label", "file_path", "dataset"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=SPLITS_DIR / "80_20" / "train.csv")
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = Path(args.train_csv)
    if not train_path.is_file():
        raise FileNotFoundError(f"Train CSV not found: {train_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_path)
    if len(df) < 500:
        raise ValueError(f"Train set too small: {len(df)} rows. Check that the correct train.csv is loaded.")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df["label"]

    split_dir = train_path.parent
    scaler_path = split_dir / "scaler.pkl"
    if scaler_path.is_file():
        scaler = joblib.load(scaler_path)
        X_scaled = X
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    models = {
        "rf": RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=5, random_state=args.seed),
        "xgb": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=args.seed),
        "svm": SVC(C=0.5, kernel="rbf", gamma="scale", random_state=args.seed),
        "knn": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "dtr": DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=args.seed),
    }

    for name, model in models.items():
        model.fit(X_scaled, y_enc)
        joblib.dump(model, out_dir / f"{name}.pkl")

    joblib.dump(scaler, out_dir / "scaler.pkl")
    joblib.dump(le, out_dir / "label_encoder.pkl")
    joblib.dump(feature_cols, out_dir / "feature_cols.pkl")


if __name__ == "__main__":
    main()
