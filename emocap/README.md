# IEMOCAP Emotion Recognition - Master Model (Random Forest)

This project implements a high-performance machine learning pipeline for **Speech Emotion Recognition (SER)** on the IEMOCAP dataset.

## 📊 Methodology
The current approach focuses on training a **Master Model** using the entire IEMOCAP dataset (all 5 sessions) after balancing the classes to ensure maximum generalization across all available speakers.

### Data Strategy:
- **Resampling:** **SMOTE** (Synthetic Minority Over-sampling Technique) is applied to all sessions to achieve 100% class balance.
- **Features:** A full set of **136 acoustic features** (MFCC, Spectral, Chroma, etc.) is used, providing a comprehensive representation of audio data.
- **Cross-Validation:** Hyperparameter tuning is performed using a stratified hold-out set from the combined data.

## 🛠 Pipeline Architecture

### 1. Data Preprocessing (`04_preprocess_data.py`)
- Standardizes all 5 folds.
- Applies SMOTE to balance emotion classes.
- Ensures all features (136) are preserved across all folds for consistency.

### 2. Master Model Training (`05_model_train.py`)
- Combines data from all 5 folds into a single master dataset.
- Uses **Optuna** for Bayesian Hyperparameter Optimization.
- Trains the final **Random Forest** classifier on 100% of the balanced data.
- Saves the model as `emocap/models/random_forest.pkl`.

### 3. Model Evaluation (`06_model_evaluation.py`)
- Evaluates the master model on the entire dataset.
- Generates:
    - **Final Validation Metrics** (WA, UA, F1).
    - **Confusion Matrix** (Heatmap).
    - **Classification Report** per emotion.

## 🚀 How to Run
To run the full pipeline (Preprocess -> Train -> Evaluate):
```bash
python emocap/scripts/emocap_loso_pipeline.py
```

## 📈 Final Results (Random Forest Master Model)
- **WA (Accuracy):** 82.9%
- **UA (Unweighted Accuracy):** 82.8%
- **F1-Score (Weighted):** 83.0%

---
*Results are stored in the `emocap/results/` directory.*
