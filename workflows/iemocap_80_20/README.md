# IEMOCAP 80-20 workflow

Προεπεξεργασία → split 80-20 (stratified, normalized) → εκπαίδευση → αξιολόγηση → tuning → αξιολόγηση. Όλα τα outputs μένουν στον φάκελο workflow.

**Run από project root (local):**
```bash
python workflows/iemocap_80_20/work_iemocap_80_20.py
```

**Run σε Docker:**
```bash
docker exec mlproject-container python /workspace/workflows/iemocap_80_20/work_iemocap_80_20.py
```

**Steps:**
1. Preprocess
2. Split 80-20 + normalize
3. Train
4. Evaluate
5. Tuning
6. Evaluate (tuned)

Outputs: `features/`, `models/`, `results/`, `tuning/`, `one_shot_results/`.
