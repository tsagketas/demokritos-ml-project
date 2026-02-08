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


def _average_roc_curves(curves_per_fold, common_fpr=None):
    """Interpolate each (fpr, tpr) to common_fpr and return (common_fpr, mean_tpr)."""
    if common_fpr is None:
        common_fpr = np.linspace(0, 1, 101)
    tprs = []
    for fpr, tpr in curves_per_fold:
        tpr_interp = np.interp(common_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        tprs.append(tpr_interp)
    mean_tpr = np.mean(tprs, axis=0)
    return common_fpr, mean_tpr


def evaluate_loso(workflow_dir: Path, make_plots: bool = False, plot_path: Path = None):
    splits_dir = workflow_dir / "features" / "splits" / "loso"
    fold_dirs = sorted(d for d in splits_dir.iterdir() if d.is_dir() and d.name.startswith("fold_"))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found under {splits_dir}")

    auc_by_model = {name: [] for name in MODEL_NAMES}
    plot_curves_by_model = {name: [] for name in MODEL_NAMES} if make_plots else None

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
                if make_plots and plot_curves_by_model is not None:
                    fpr, tpr = macro_roc_curve(y_true_enc, proba, n_classes)
                    plot_curves_by_model[name].append((fpr, tpr))
            except ValueError:
                auc_by_model[name].append(np.nan)

    mean_auc = {}
    for name, vals in auc_by_model.items():
        vals = np.array(vals, dtype=float)
        mean_auc[name] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan

    if make_plots and plot_path is not None and plot_curves_by_model is not None:
        common_fpr = np.linspace(0, 1, 101)
        fig, ax = plt.subplots(figsize=(8, 6))
        for name in MODEL_NAMES:
            curves = plot_curves_by_model.get(name, [])
            if not curves:
                continue
            fpr_avg, tpr_avg = _average_roc_curves(curves, common_fpr)
            auc_mean = mean_auc.get(name, np.nan)
            if np.isfinite(auc_mean):
                ax.plot(fpr_avg, tpr_avg, linewidth=2, label=f"{name} (AUC={auc_mean:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Macro ROC (OVR): iemocap_loso (mean over folds)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return mean_auc


def write_summary_report(out_dir: Path, df: pd.DataFrame) -> None:
    """Write a text summary report explaining the AUC comparison."""
    out_path = out_dir / "summary_report.txt"
    lines = [
        "Workflow comparison — Macro AUC (one-vs-rest)",
        "==============================================",
        "",
        "What you are looking at",
        "------------------------",
        "This comparison evaluates how well each model discriminates between emotion",
        "classes (macro AUC) in three different evaluation setups:",
        "",
        "  • 80_20: Single 80/20 train/test split (IEMOCAP). One test set, one AUC per model.",
        "  • loso:  Leave-One-Subject-Out. One fold per left-out subject; AUC is computed",
        "           per fold and then averaged across folds (mean AUC per model).",
        "  • pca:   Same 80/20 split as 80_20 but features are PCA-transformed (reduced",
        "           dimensionality). One test set, one AUC per model.",
        "",
        "The table (auc_comparison.csv) has one row per model and one column per workflow.",
        "Higher AUC = better class separation. Empty/NaN means the model does not support",
        "probability outputs (e.g. SVM without probability=True), so AUC was not computed.",
        "",
        "Summary by workflow",
        "--------------------",
    ]

    for col in ["80_20", "loso", "pca"]:
        if col not in df.columns:
            continue
        s = df[col].replace("", np.nan).astype(float)
        valid = s.dropna()
        if len(valid) == 0:
            lines.append(f"  {col}: no AUC values (all NaN).")
        else:
            best = valid.idxmax()
            best_val = valid[best]
            mean_auc = valid.mean()
            lines.append(f"  {col}: best model = {best} (AUC = {best_val:.4f}); mean AUC over models = {mean_auc:.4f}.")
        lines.append("")

    lines.append("Summary by model")
    lines.append("----------------")
    for model in df.index:
        row = df.loc[model]
        vals = []
        for c in ["80_20", "loso", "pca"]:
            v = row.get(c)
            if pd.isna(v) or v == "":
                vals.append(f"{c}=—")
            else:
                try:
                    vals.append(f"{c}={float(v):.3f}")
                except (TypeError, ValueError):
                    vals.append(f"{c}=—")
        best_wf = None
        best_auc = -1.0
        for c in ["80_20", "loso", "pca"]:
            v = row.get(c)
            if v != "" and not pd.isna(v):
                try:
                    a = float(v)
                    if a > best_auc:
                        best_auc = a
                        best_wf = c
                except (TypeError, ValueError):
                    pass
        if best_wf is not None:
            lines.append(f"  {model}: {', '.join(vals)}  → best in {best_wf} ({best_auc:.3f})")
        else:
            lines.append(f"  {model}: {', '.join(vals)}  (no AUC)")
    lines.append("")

    lines.extend([
        "Outputs",
        "--------",
        "  • auc_comparison.csv — table model × workflow with macro AUC values.",
        "  • roc_80_20.png, roc_pca.png, roc_loso.png — macro-averaged ROC curves per workflow (LOSO: mean over folds).",
        "  • summary_report.txt — this file.",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare workflows by macro AUC (OVR).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "workflows_comparison",
        help="Output directory for comparison CSV and plots.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate ROC plots (only CSV and summary report).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    make_plots = not args.no_plots

    results = {name: {} for name in MODEL_NAMES}

    wf_80 = WORKFLOW_DIRS["80_20"]
    test_80 = wf_80 / "features" / "splits" / "80_20" / "test.csv"
    models_80 = wf_80 / "models"
    auc_80 = evaluate_single_testset(
        wf_80,
        test_80,
        models_80,
        make_plots,
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
        make_plots,
        out_dir / "roc_pca.png",
    )
    for name in MODEL_NAMES:
        results[name]["pca"] = auc_pca.get(name, np.nan)

    wf_loso = WORKFLOW_DIRS["loso"]
    auc_loso = evaluate_loso(
        wf_loso,
        make_plots=make_plots,
        plot_path=out_dir / "roc_loso.png",
    )
    for name in MODEL_NAMES:
        results[name]["loso"] = auc_loso.get(name, np.nan)

    df_out = pd.DataFrame.from_dict(results, orient="index")
    df_out.index.name = "model"
    df_out = df_out[["80_20", "loso", "pca"]]
    out_csv = out_dir / "auc_comparison.csv"
    df_out.to_csv(out_csv)
    print(f"Wrote: {out_csv}")

    write_summary_report(out_dir, df_out)


if __name__ == "__main__":
    main()
