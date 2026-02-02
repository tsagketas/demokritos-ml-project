# IEMOCAP LOSO workflow

**LOSO** = Leave-One-Subject-Out. Subject is derived from `file_path` (e.g. `Session1`, `Session2`, …). For each fold, one subject is held out for test and the rest are used for train, so evaluation is speaker-independent. IEMOCAP has 5 sessions → 5 folds.

Preprocess IEMOCAP, split with LOSO (normalized per fold), train and evaluate per fold, then aggregate metrics (mean ± std) with `scripts/06_aggregate_loso_results.py`. All outputs are under this workflow folder.

**Run from project root (local):**
```bash
python workflows/iemocap_loso/work_iemocap_loso.py
```

**Run with per-fold hyperparameter tuning** (RandomizedSearchCV per fold; slower but can improve metrics):
```bash
python workflows/iemocap_loso/work_iemocap_loso.py --tune
python workflows/iemocap_loso/work_iemocap_loso.py --tune --n-iter 30 --cv 5   # optional: more iterations, CV folds
```

**Run full workflow in Docker:**
```bash
docker exec mlproject-container python /workspace/workflows/iemocap_loso/work_iemocap_loso.py
```

**Steps:**
1. Preprocess IEMOCAP (features CSV + report)
2. Split LOSO (one fold per session: `features/splits/loso/fold_0/`, …)
3. For each fold: train models (or run hyperparameter tuning if `--tune`) → evaluate
4. Aggregate results → `results/loso_summary.txt` (mean ± std per model)

**Outputs:**
- `features/splits/loso/fold_*/` — train.csv, test.csv, scaler.pkl per fold
- `models/fold_*/` — trained models per fold
- `results/fold_*/` — per-fold reports (confusion matrix, learning curve, report.txt)
- `results/loso_summary.txt` — aggregated metrics (Accuracy, F1, Precision, Recall) mean ± std

**Re-run only aggregation** (if you already have `results/fold_*/`):
```bash
python scripts/06_aggregate_loso_results.py --workflow-dir workflows/iemocap_loso
```
