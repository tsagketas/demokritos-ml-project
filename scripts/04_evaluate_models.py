import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import learning_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "processed" / "features" / "iemocap" / "splits"
MODELS_DIR = PROJECT_ROOT / "processed" / "models"
TUNING_DIR = PROJECT_ROOT / "processed" / "tuning"
RESULTS_DIR = PROJECT_ROOT / "processed" / "results"

MODEL_NAMES = ["rf", "xgb", "svm", "knn", "dtr", "logistic", "nb"]


def safe_mape(y_true, y_pred):
    denom = np.where(y_true != 0, np.abs(y_true), 1)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=SPLITS_DIR / "80_20" / "train.csv")
    parser.add_argument("--test-csv", type=Path, default=SPLITS_DIR / "80_20" / "test.csv")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--n-train-sizes", type=int, default=10)
    parser.add_argument("--best", action="store_true", help="Use best models from tuning (processed/tuning/<model>/best_model.pkl)")
    args = parser.parse_args()

    train_path = Path(args.train_csv)
    test_path = Path(args.test_csv)
    if not train_path.is_file():
        raise FileNotFoundError(f"Train CSV not found: {train_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test CSV not found: {test_path}")

    if args.best:
        models_dir = Path(TUNING_DIR)
    else:
        models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    le = joblib.load(models_dir / "label_encoder.pkl")
    feature_cols = joblib.load(models_dir / "feature_cols.pkl")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    split_dir = Path(args.test_csv).parent
    already_normalized = (split_dir / "scaler.pkl").is_file()
    if already_normalized:
        X_train = df_train[feature_cols]
        X_test = df_test[feature_cols]
    else:
        scaler = joblib.load(models_dir / "scaler.pkl")
        X_train = scaler.transform(df_train[feature_cols])
        X_test = scaler.transform(df_test[feature_cols])
    y_train_enc = le.transform(df_train["label"])
    y_true = df_test["label"]
    y_true_enc = le.transform(y_true)
    class_names = list(le.classes_)
    labels_enc = list(range(len(class_names)))

    for name in MODEL_NAMES:
        if args.best:
            pkl_path = models_dir / name / "best_model.pkl"
        else:
            pkl_path = models_dir / f"{name}.pkl"
        if not pkl_path.is_file():
            continue
        model = joblib.load(pkl_path)
        y_pred_enc = model.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)

        accuracy = accuracy_score(y_true_enc, y_pred_enc)
        f1 = f1_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)
        precision = precision_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)
        recall = recall_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)
        r2 = r2_score(y_true_enc, y_pred_enc)
        mse = mean_squared_error(y_true_enc, y_pred_enc)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_enc, y_pred_enc)
        mape = safe_mape(y_true_enc, y_pred_enc)
        cm = confusion_matrix(y_true_enc, y_pred_enc, labels=labels_enc)

        out_dir = results_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "report.txt", "w") as f:
            f.write(f"Evaluation report: {name}\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1-score (weighted): {f1:.4f}\n")
            f.write(f"Precision (weighted): {precision:.4f}\n")
            f.write(f"Recall (weighted): {recall:.4f}\n")
            f.write(f"R²: {r2:.4f}\n")
            f.write(f"MSE: {mse:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"MAPE: {mape:.4f}%\n\n")
            f.write("Confusion matrix:\n")
            f.write(f"Labels: {class_names}\n\n")
            for i, row in enumerate(cm):
                f.write("  " + " ".join(f"{v:5d}" for v in row) + f"  | {class_names[i]}\n")
            f.write("  " + "-" * (6 * len(class_names)) + "\n")
            f.write("  " + " ".join(f"{c:>5s}" for c in class_names) + "\n")

        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / "confusion_matrix.csv")

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
        ax_cm.set_title(f"Confusion matrix: {name}")
        plt.tight_layout()
        fig_cm.savefig(str(out_dir / "confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close(fig_cm)

        cv_folds = 5
        train_sizes = np.linspace(0.1, 1.0, args.n_train_sizes)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            clone(model), X_train, y_train_enc, train_sizes=train_sizes, cv=cv_folds, n_jobs=-1
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        # Loss = 1 - score (0-1 loss) for all models, so we always have both score and loss
        train_loss_mean = 1.0 - train_mean
        train_loss_std = train_std
        val_loss_mean = 1.0 - val_mean
        val_loss_std = val_std

        fig_lc, (ax_score, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

        # Score plot
        ax_score.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2, color="C0")
        ax_score.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.2, color="C1")
        ax_score.plot(train_sizes_abs, train_mean, "o-", color="C0", label="Train score")
        ax_score.plot(train_sizes_abs, val_mean, "s-", color="C1", label="Validation score")
        ax_score.set_xlabel("Training set size")
        ax_score.set_ylabel("Score")
        ax_score.set_title(f"Learning curve (score): {name}")
        ax_score.legend()

        # Loss plot (0-1 loss) – always for all models
        ax_loss.fill_between(train_sizes_abs, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.2, color="C0")
        ax_loss.fill_between(train_sizes_abs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.2, color="C1")
        ax_loss.plot(train_sizes_abs, train_loss_mean, "o-", color="C0", label="Train loss")
        ax_loss.plot(train_sizes_abs, val_loss_mean, "s-", color="C1", label="Validation loss")
        ax_loss.set_xlabel("Training set size")
        ax_loss.set_ylabel("Loss (1 - score)")
        ax_loss.set_title(f"Learning curve (loss): {name}")
        ax_loss.legend()

        plt.tight_layout()
        fig_lc.savefig(str(out_dir / "learning_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig_lc)


if __name__ == "__main__":
    main()
