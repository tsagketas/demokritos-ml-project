import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'svm', 'xgboost', 'decision_tree'],
                        help='Model to evaluate')
    args = parser.parse_args()

    model_name = args.model
    models_dir = os.path.join("emocap/models", model_name)
    base_data_path = "data/iemocap/dataset"
    results_dir = os.path.join("emocap/results", model_name)
    os.makedirs(results_dir, exist_ok=True)

    all_preds = []
    all_y_true = []
    
    meta_cols = ['session', 'method', 'gender', 'emotion', 'n_annotators', 'agreement']
    
    logger.info(f"--- Starting LOSO Evaluation for {model_name} ---")

    for fold in [1, 2, 3, 4, 5]:
        model_path = os.path.join(models_dir, f"fold_{fold}.pkl")
        if not os.path.exists(model_path):
            logger.warning(f"Fold {fold} model not found at {model_path}, skipping.")
            continue
            
        logger.info(f"Evaluating Fold {fold}...")
        model = joblib.load(model_path)
        
        # Load TEST data for this fold (Unseen session)
        fold_path = os.path.join(base_data_path, f"fold_{fold}", "processed")
        test_df = pd.read_csv(os.path.join(fold_path, 'test.csv'))
        
        features = [c for c in test_df.columns if c not in meta_cols]
        X_test = test_df[features]
        y_test = test_df['emotion']
        
        # Predict
        if model_name == 'xgboost' and hasattr(model, '_label_encoder'):
            le = model._label_encoder
            y_pred_enc = model.predict(X_test)
            y_pred = le.inverse_transform(y_pred_enc)
        else:
            y_pred = model.predict(X_test)
            
        all_preds.extend(y_pred)
        all_y_true.extend(y_test)
        
        # Individual fold metrics (optional but useful)
        fold_wa = accuracy_score(y_test, y_pred)
        logger.info(f"  Fold {fold} WA: {fold_wa:.4f}")

    if not all_preds:
        logger.error("No predictions were made. Check model and data paths.")
        return

    # Aggregate Metrics
    wa = accuracy_score(all_y_true, all_preds)
    ua = recall_score(all_y_true, all_preds, average='macro')
    f1_w = f1_score(all_y_true, all_preds, average='weighted')
    
    labels = sorted(list(set(all_y_true)))
    
    logger.info(f"\n--- Final LOSO Results for {model_name} ---")
    logger.info(f"Weighted Accuracy (WA): {wa:.4f}")
    logger.info(f"Unweighted Accuracy (UA): {ua:.4f}")
    logger.info(f"F1-Score (Weighted): {f1_w:.4f}")

    # Save summary
    summary_results = {
        'Model': model_name,
        'WA': wa,
        'UA': ua,
        'F1_Weighted': f1_w
    }
    pd.DataFrame([summary_results]).to_csv(os.path.join(results_dir, "loso_summary.csv"), index=False)
    
    # Classification Report
    report = classification_report(all_y_true, all_preds, target_names=labels, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(results_dir, "loso_classification_report.csv"))
    
    # Confusion Matrix
    cm = confusion_matrix(all_y_true, all_preds, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix: {model_name} (LOSO Aggregate)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "loso_confusion_matrix.png"))
    plt.close()

    print("\n" + "="*60)
    print(f"LOSO EVALUATION COMPLETE FOR {model_name}")
    print(f"WA: {wa:.1%}, UA: {ua:.1%}")
    print("="*60)

if __name__ == "__main__":
    main()
