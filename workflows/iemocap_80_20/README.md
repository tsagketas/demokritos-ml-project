# IEMOCAP 80-20 workflow

Preprocess IEMOCAP, split 80-20 (stratified, normalized), train models, evaluate, hyper-tune, then evaluate best models. All data, models, and results are written under this workflow folder (`features/`, `models/`, `results/`, `tuning/`) so they do not overwrite other workflows or the project root.

**Run from project root (local):**
```bash
python workflows/iemocap_80_20/work_iemocap_80_20.py
```

**Run full workflow in Docker:**
```bash
docker exec mlproject-container python /workspace/workflows/iemocap_80_20/work_iemocap_80_20.py
```

**Run single steps in Docker** (use `--workflow-dir /workspace/workflows/iemocap_80_20` so outputs go to this workflow):
```bash
# Evaluate models (from models/)
docker exec mlproject-container python /workspace/scripts/04_evaluate_models.py --workflow-dir /workspace/workflows/iemocap_80_20
```

**Steps:**
1. Preprocess IEMOCAP (features CSV + report)
2. Split train/test 80-20 with StandardScaler
3. Train baseline models
4. Evaluate baseline models
5. Hyperparameter tuning (best models to `models/` or `tuning/`)
6. Evaluate (tuned models with `--best`)

Outputs: `features/`, `models/`, `results/`, `tuning/`.
