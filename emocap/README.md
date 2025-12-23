# IEMOCAP Speech Emotion Recognition (LOSO Pipeline)

Οδηγός για την εκτέλεση των πειραμάτων Machine Learning στο IEMOCAP χρησιμοποιώντας Leave-One-Session-Out (LOSO).

## 🐳 Docker Command Prefix
Όλες οι εντολές πρέπει να εκτελούνται μέσω του container:
`docker exec mlproject-container python ...`

---

## 🚀 1. Optimized Pipeline (Best Results)
Αυτό το workflow χρησιμοποιεί τις βέλτιστες τεχνικές: **4-class mapping**, **Agreement Filter (>=2)**, **ANOVA/MI Feature Selection** και **Class Weights** (όχι SMOTE).

### Εκτέλεση για SVM (Προτεινόμενο):
```bash
docker exec mlproject-container python emocap/workflows/emocap_loso_optimized_pipeline.py --model svm --mi --k 150
```

### Εκτέλεση για Random Forest:
```bash
docker exec mlproject-container python emocap/workflows/emocap_loso_optimized_pipeline.py --model random_forest --k 60
```

*Flags:*
- `--model`: `svm`, `random_forest`, `xgboost`
- `--mi`: Χρήση Mutual Information (για μη-γραμμικά features)
- `--anova`: Χρήση ANOVA F-test (πιο γρήγορο)
- `--k`: Αριθμός features που θα κρατηθούν (π.χ. 60, 100, 150)

---

## ⚖️ 2. SMOTE Pipeline (Baseline)
Το κλασικό workflow που χρησιμοποιεί SMOTE για εξισορρόπηση των κλάσεων.

```bash
docker exec mlproject-container python emocap/workflows/emocap_loso_smote_pipeline.py --model svm
```

---

## 🧠 3. Ensemble (Συνδυασμός Μοντέλων)
Για να τρέξει το Ensemble, πρέπει πρώτα να έχουν εκπαιδευτεί τα μοντέλα πάνω στα **ίδια ακριβώς features** (ίδιο k και ίδιο selection method).

### Soft Voting (Πιθανότητες - Ίσα Βάρη):
```bash
docker exec mlproject-container python emocap/workflows/ensemble_soft/run_ensemble.py --models svm,random_forest
```

### Weighted Ensemble (Αυτόματος υπολογισμός βαρών βάσει UA):
Αυτό το workflow διαβάζει το UA κάθε μοντέλου από τα αποτελέσματα και δίνει μεγαλύτερη βαρύτητα στο καλύτερο μοντέλο.
```bash
docker exec mlproject-container python emocap/workflows/ensemble_weighted/run_ensemble.py --models svm,random_forest
```

---

## 📊 Αποτελέσματα
Τα αποτελέσματα αποθηκεύονται στους αντίστοιχους φακέλους:
- **Metrics/Summary**: `emocap/results/[model_name]/loso_summary.csv`
- **Confusion Matrix**: `emocap/results/[model_name]/loso_confusion_matrix.png`
- **Reports**: `emocap/results/[model_name]/loso_classification_report.csv`

---
*Σημείωση: Για το IEMOCAP, το SVM με RBF kernel και k=120-150 features συνήθως δίνει το καλύτερο UA (>50%).*
