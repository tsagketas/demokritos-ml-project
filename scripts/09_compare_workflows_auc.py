import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WORKFLOW_DIRS = {
    "80_20": PROJECT_ROOT / "workflows" / "iemocap_80_20",
    "loso": PROJECT_ROOT / "workflows" / "iemocap_loso",
    "pca": PROJECT_ROOT / "workflows" / "iemocap_pca",
}

MODEL_NAMES = ["rf", "xgb", "svm", "knn", "dtr", "logistic", "nb"]


def load_artifacts(models_dir: Path):
    le_path = models_dir / "label_encoder.pkl"
    feature_cols_path = models_dir / "feature_cols.pkl"
    scaler_path = models_dir / "scaler.pkl"
    if not le_path.is_file():
        raise FileNotFoundError(f"Missing label encoder: {le_path}")
    if not feature_cols_path.is_file():
        raise FileNotFoundError(f"Missing feature cols: {feature_cols_path}")
    le = joblib.load(le_path)
    feature_cols = joblib.load(feature_cols_path)
    scaler = joblib.load(scaler_path) if scaler_path.is_file() else None
    return le, feature_cols, scaler


def load_test_data(test_csv: Path, feature_cols, scaler, label_encoder):
    df_test = pd.read_csv(test_csv)
    X = df_test[feature_cols]
    if scaler is not None:
        X = scaler.transform(X)
    y_true_enc = label_encoder.transform(df_test["label"])
    return X, y_true_enc


def align_proba(model, proba, n_classes):
    proba = np.asarray(proba)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    if hasattr(model, "classes_"):
        full = np.zeros((proba.shape[0], n_classes))
        for idx, cls in enumerate(model.classes_):
            cls_idx = int(cls)
            if 0 <= cls_idx < n_classes:
                full[:, cls_idx] = proba[:, idx]
        return full
    return proba


def macro_roc_curve(y_true_enc, y_proba, n_classes):
    y_bin = label_binarize(y_true_enc, classes=list(range(n_classes)))
    fpr = {}
    tpr = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    return all_fpr, mean_tpr


def evaluate_single_testset(workflow_dir: Path, test_csv: Path, models_dir: Path, make_plots: bool, plot_path: Path):
    le, feature_cols, scaler = load_artifacts(models_dir)
    X_test, y_true_enc = load_test_data(test_csv, feature_cols, scaler, le)
    n_classes = len(le.classes_)

    auc_by_model = {}
    plot_data = {}

    for name in MODEL_NAMES:
        model_path = models_dir / f"{name}.pkl"
        if not model_path.is_file():
            auc_by_model[name] = np.nan
            continue
        model = joblib.load(model_path)
        if not hasattr(model, "predict_proba"):
            auc_by_model[name] = np.nan
            continue
        try:
            proba = model.predict_proba(X_test)
            proba = align_proba(model, proba, n_classes)
            auc_val = roc_auc_score(
                y_true_enc,
                proba,
                multi_class="ovr",
                average="macro",
            )
            auc_by_model[name] = float(auc_val)
            if make_plots:
                fpr, tpr = macro_roc_curve(y_true_enc, proba, n_classes)
                plot_data[name] = (fpr, tpr, auc_val)
        except ValueError:
            auc_by_model[name] = np.nan

    if make_plots and plot_data:
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, (fpr, tpr, auc_val) in plot_data.items():
            ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Macro ROC (OVR): {workflow_dir.name}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return auc_by_model


def evaluate_loso(workflow_dir: Path):
    splits_dir = workflow_dir / "features" / "splits" / "loso"
    fold_dirs = sorted(d for d in splits_dir.iterdir() if d.is_dir() and d.name.startswith("fold_"))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found under {splits_dir}")

    auc_by_model = {name: [] for name in MODEL_NAMES}

    for fold_dir in fold_dirs:
        test_csv = fold_dir / "test.csv"
        models_dir = workflow_dir / "models" / fold_dir.name
        if not test_csv.is_file() or not models_dir.is_dir():
            continue
        try:
            le, feature_cols, scaler = load_artifacts(models_dir)
        except FileNotFoundError:
            continue
        if scaler is None:
            scaler_path = fold_dir / "scaler.pkl"
            if scaler_path.is_file():
                scaler = joblib.load(scaler_path)

        X_test, y_true_enc = load_test_data(test_csv, feature_cols, scaler, le)
        n_classes = len(le.classes_)

        for name in MODEL_NAMES:
            model_path = models_dir / f"{name}.pkl"
            if not model_path.is_file():
                auc_by_model[name].append(np.nan)
                continue
            model = joblib.load(model_path)
            if not hasattr(model, "predict_proba"):
                auc_by_model[name].append(np.nan)
                continue
            try:
                proba = model.predict_proba(X_test)
                proba = align_proba(model, proba, n_classes)
                auc_val = roc_auc_score(
                    y_true_enc,
                    proba,
                    multi_class="ovr",
                    average="macro",
                )
                auc_by_model[name].append(float(auc_val))
            except ValueError:
                auc_by_model[name].append(np.nan)

    mean_auc = {}
    for name, vals in auc_by_model.items():
        vals = np.array(vals, dtype=float)
        mean_auc[name] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
    return mean_auc


def main():
    parser = argparse.ArgumentParser(description="Compare workflows by macro AUC (OVR).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "workflows_comparison",
        help="Output directory for comparison CSV and plots.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate ROC plots for 80_20 and PCA workflows.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {name: {} for name in MODEL_NAMES}

    wf_80 = WORKFLOW_DIRS["80_20"]
    test_80 = wf_80 / "features" / "splits" / "80_20" / "test.csv"
    models_80 = wf_80 / "models"
    auc_80 = evaluate_single_testset(
        wf_80,
        test_80,
        models_80,
        args.plots,
        out_dir / "roc_80_20.png",
    )
    for name in MODEL_NAMES:
        results[name]["80_20"] = auc_80.get(name, np.nan)

    wf_pca = WORKFLOW_DIRS["pca"]
    test_pca = wf_pca / "features" / "splits" / "80_20" / "test.csv"
    models_pca = wf_pca / "models"
    auc_pca = evaluate_single_testset(
        wf_pca,
        test_pca,
        models_pca,
        args.plots,
        out_dir / "roc_pca.png",
    )
    for name in MODEL_NAMES:
        results[name]["pca"] = auc_pca.get(name, np.nan)

    wf_loso = WORKFLOW_DIRS["loso"]
    auc_loso = evaluate_loso(wf_loso)
    for name in MODEL_NAMES:
        results[name]["loso"] = auc_loso.get(name, np.nan)

    df_out = pd.DataFrame.from_dict(results, orient="index")
    df_out.index.name = "model"
    df_out = df_out[["80_20", "loso", "pca"]]
    out_csv = out_dir / "auc_comparison.csv"
    df_out.to_csv(out_csv)
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
