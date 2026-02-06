# IEMOCAP PCA workflow

Preprocess → split 80-20 (χωρίς normalize) → PCA μόνο στο train → train/eval → tuning → eval → one-shot (CREMAD).

**Run από project root (local):**
```bash
python workflows/iemocap_pca/work_iemocap_pca.py
```

**Run σε Docker:**
```bash
docker exec mlproject-container python /workspace/workflows/iemocap_pca/work_iemocap_pca.py
```

**Steps:**
1. Preprocess
2. Split 80-20 (raw)
3. PCA στο train (backup `*_old.csv`)
4. Train
5. Evaluate
6. Tuning
7. Evaluate (tuned)
8. One-shot (CREMAD)

**Outputs:**
- `features/`
- `pca_info/`
- `models/`
- `results/`, `tuning/`
- `one_shot_results/`
