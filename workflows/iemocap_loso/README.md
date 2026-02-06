# IEMOCAP LOSO workflow

**LOSO** = Leave-One-Subject-Out. 5 sessions → 5 folds. Ένα session για test, τα άλλα για train.

Preprocess → LOSO split (normalized ανά fold) → train/evaluate ανά fold → aggregation.

**Run από project root (local):**
```bash
python workflows/iemocap_loso/work_iemocap_loso.py
```

**Run με tuning ανά fold:**
```bash
python workflows/iemocap_loso/work_iemocap_loso.py --tune
python workflows/iemocap_loso/work_iemocap_loso.py --tune --n-iter 30 --cv 5   # optional: more iterations, CV folds
```

**Run σε Docker:**
```bash
docker exec mlproject-container python /workspace/workflows/iemocap_loso/work_iemocap_loso.py
```

**Steps:**
1. Preprocess
2. Split LOSO
3. Train/Eval ανά fold (ή tuning)
4. Aggregate → `results/loso_summary.txt`

**Outputs:**
- `features/splits/loso/fold_*/`
- `models/fold_*/`
- `results/fold_*/`
- `results/loso_summary.txt`

**Μόνο aggregation:**
```bash
python scripts/06_aggregate_loso_results.py --workflow-dir workflows/iemocap_loso
```
