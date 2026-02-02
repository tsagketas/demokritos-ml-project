"""
06 — Aggregate LOSO (Leave-One-Subject-Out) fold results.
Reads report.txt from each results/fold_<i>/<model>/ and writes
loso_summary.txt (and optional CSV) with mean ± std per model.
Reusable: call from any workflow that produces per-fold reports in this format.
"""
import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Metrics we parse from each report.txt (key: line prefix, value: display name)
METRIC_KEYS = [
    ("Accuracy:", "Accuracy"),
    ("F1-score (weighted):", "F1_weighted"),
    ("Precision (weighted):", "Precision_weighted"),
    ("Recall (weighted):", "Recall_weighted"),
]


def parse_report(path: Path) -> dict:
    """Parse a single report.txt; return dict of metric_name -> float."""
    out = {}
    text = path.read_text(encoding="utf-8")
    for line_prefix, key in METRIC_KEYS:
        # e.g. "Accuracy: 0.5881" or "MAPE: 40.2138%"
        m = re.search(re.escape(line_prefix) + r"\s*([\d.]+)", text)
        if m:
            out[key] = float(m.group(1))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate LOSO fold results (mean ± std per model)."
    )
    parser.add_argument(
        "--workflow-dir",
        type=Path,
        default=None,
        help="Workflow root; results under <workflow-dir>/results/",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override: directory containing fold_0, fold_1, ... (default: <workflow-dir>/results or PROJECT_ROOT/results)",
    )
    parser.add_argument(
        "--out-summary-txt",
        type=Path,
        default=None,
        help="Output summary .txt path (default: <results-dir>/loso_summary.txt)",
    )
    parser.add_argument(
        "--out-summary-csv",
        type=Path,
        default=None,
        help="Optional: output summary CSV path (default: none)",
    )
    args = parser.parse_args()

    if args.workflow_dir is not None:
        base = Path(args.workflow_dir).resolve()
        results_dir = args.results_dir or (base / "results")
    else:
        results_dir = Path(args.results_dir or RESULTS_DIR).resolve()

    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Discover folds: results/fold_0, results/fold_1, ...
    fold_dirs = sorted(
        d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")
    )
    if not fold_dirs:
        raise FileNotFoundError(
            f"No fold_* directories found in {results_dir}. Run LOSO evaluate first."
        )

    # Discover models from first fold
    first_fold = fold_dirs[0]
    model_names = sorted(
        d.name for d in first_fold.iterdir() if d.is_dir() and (d / "report.txt").is_file()
    )
    if not model_names:
        raise FileNotFoundError(
            f"No model report.txt found under {first_fold}. Run 04_evaluate_models per fold."
        )

    # Collect metrics per (model, fold)
    # data[model_name][metric_key] = [value_fold0, value_fold1, ...]
    data = {m: {k: [] for _, k in METRIC_KEYS} for m in model_names}

    for fold_dir in fold_dirs:
        for model_name in model_names:
            report_path = fold_dir / model_name / "report.txt"
            if not report_path.is_file():
                continue
            parsed = parse_report(report_path)
            for k in data[model_name]:
                if k in parsed:
                    data[model_name][k].append(parsed[k])

    # Compute mean ± std per model (only where we have values for all folds)
    import statistics
    summary_rows = []
    for model_name in model_names:
        row = {"model": model_name}
        for _, key in METRIC_KEYS:
            vals = data[model_name].get(key, [])
            if len(vals) == len(fold_dirs):
                mean = statistics.mean(vals)
                stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
                row[f"{key}_mean"] = mean
                row[f"{key}_std"] = stdev
                row[key] = f"{mean:.4f} ± {stdev:.4f}"
            else:
                row[key] = "N/A"
        summary_rows.append(row)

    out_txt = args.out_summary_txt or (results_dir / "loso_summary.txt")
    out_txt = Path(out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("LOSO aggregated results (mean ± std over folds)\n")
        f.write("================================================\n\n")
        f.write(f"Folds: {len(fold_dirs)} ({', '.join(d.name for d in fold_dirs)})\n")
        f.write(f"Models: {', '.join(model_names)}\n\n")
        for _, key in METRIC_KEYS:
            f.write(f"{key}\n")
            f.write("-" * 50 + "\n")
            for row in summary_rows:
                f.write(f"  {row['model']}: {row[key]}\n")
            f.write("\n")
    print(f"Wrote {out_txt}")

    if args.out_summary_csv is not None:
        import csv
        out_csv = Path(args.out_summary_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["model"] + [key for _, key in METRIC_KEYS]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in summary_rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
