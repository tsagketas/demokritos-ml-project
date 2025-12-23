import os
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
from scipy.stats import mode
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Ensemble Evaluation for Emotion Classification')
    parser.add_argument('--method', type=str, choices=['hard', 'soft'], required=True, 
                        help='Voting method: hard (majority) or soft (average probabilities)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results. If None, uses default results path.')
    args = parser.parse_args()

    base_data_path = "data/iemocap/dataset"
    models_root = "emocap/models"
    results_root = "emocap/results"
    
    if args.output_dir:
        ensemble_results_dir = args.output_dir
    else:
        ensemble_results_dir = os.path.join(results_root, f"ensemble_{args.method}")
    
    os.makedirs(ensemble_results_dir, exist_ok=True)
    
    # Models to include in ensemble
    model_types = ['random_forest']
    folds = [1, 2, 3, 4, 5]
    
    all_ensemble_preds = []
    all_y_true = []
    
    meta_cols = ['session', 'method', 'gender', 'emotion', 'n_annotators', 'agreement']
    
    logger.info(f"--- Starting Ensemble LOSO Evaluation ({args.method.capitalize()} Voting) ---")

    for fold in folds:
        logger.info(f"Evaluating Fold {fold}...")
        
        # Load test data for this fold (Unseen session)
        fold_path = os.path.join(base_data_path, f"fold_{fold}", "processed")
        test_file = os.path.join(fold_path, 'test.csv')
        
        if not os.path.exists(test_file):
            logger.error(f"Test file not found: {test_file}")
            continue
            
        test_df = pd.read_csv(test_file)
        
        features = [c for c in test_df.columns if c not in meta_cols]
        X_test = test_df[features]
        y_test = test_df['emotion']
        
        fold_outputs = []

        for m_type in model_types:
            model_path = os.path.join(models_root, m_type, f"fold_{fold}.pkl")
            if not os.path.exists(model_path):
                logger.error(f"Model not found: {model_path}. Please train all models first.")
                return
            
            model = joblib.load(model_path)
            
            if args.method == 'hard':
                # Predict classes
                if m_type == 'xgboost' and hasattr(model, '_label_encoder'):
                    le = model._label_encoder
                    y_pred_enc = model.predict(X_test)
                    y_pred = le.inverse_transform(y_pred_enc)
                else:
                    y_pred = model.predict(X_test)
                fold_outputs.append(y_pred)
            else:
                # Predict probabilities (Soft Voting)
                if not hasattr(model, 'predict_proba'):
                    logger.error(f"Model {m_type} does not support predict_proba. For SVM, ensure probability=True during training.")
                    return
                
                # Predict probas
                y_proba = model.predict_proba(X_test)
                
                # XGBoost might have different class order if label encoder was used
                # But here we assume they were trained on the same classes
                # Most sklearn-compatible models keep classes_ attribute
                fold_outputs.append(y_proba)

        if args.method == 'hard':
            # Majority Vote logic that works robustly with strings
            n_samples = len(fold_outputs[0])
            ensemble_pred = []
            for i in range(n_samples):
                # Get predictions from all models for this sample
                sample_preds = [m_preds[i] for m_preds in fold_outputs]
                # Find the most frequent prediction
                values, counts = np.unique(sample_preds, return_counts=True)
                ensemble_pred.append(values[np.argmax(counts)])
            ensemble_pred = np.array(ensemble_pred)
        else:
            # Average Probabilities
            avg_proba = np.mean(fold_outputs, axis=0)
            best_idx = np.argmax(avg_proba, axis=1)
            
            # Get class labels from one of the models (e.g., Random Forest)
            ref_model = joblib.load(os.path.join(models_root, 'random_forest', f"fold_{fold}.pkl"))
            classes = ref_model.classes_
            ensemble_pred = classes[best_idx]
            
        all_ensemble_preds.extend(ensemble_pred)
        all_y_true.extend(y_test)
        
        fold_wa = accuracy_score(y_test, ensemble_pred)
        logger.info(f"  Fold {fold} Ensemble WA: {fold_wa:.4f}")

    if not all_ensemble_preds:
        logger.error("No predictions were made.")
        return

    # Aggregate Metrics
    wa = accuracy_score(all_y_true, all_ensemble_preds)
    ua = recall_score(all_y_true, all_ensemble_preds, average='macro')
    f1_w = f1_score(all_y_true, all_ensemble_preds, average='weighted')
    
    labels = sorted(list(set(all_y_true)))
    
    logger.info(f"\n--- Final LOSO Results for Ensemble ({args.method}) ---")
    logger.info(f"Weighted Accuracy (WA): {wa:.4f}")
    logger.info(f"Unweighted Accuracy (UA): {ua:.4f}")
    logger.info(f"F1-Score (Weighted): {f1_w:.4f}")

    # Save summary
    summary_results = {
        'Method': f'ensemble_{args.method}',
        'WA': wa,
        'UA': ua,
        'F1_Weighted': f1_w
    }
    pd.DataFrame([summary_results]).to_csv(os.path.join(ensemble_results_dir, "loso_summary.csv"), index=False)
    
    # Classification Report
    # Ensure labels includes everything present in both true and pred
    all_possible_labels = sorted(list(set(all_y_true) | set(all_ensemble_preds)))
    report = classification_report(all_y_true, all_ensemble_preds, labels=all_possible_labels, target_names=all_possible_labels, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(ensemble_results_dir, "loso_classification_report.csv"))
    
    # Confusion Matrix
    cm = confusion_matrix(all_y_true, all_ensemble_preds, labels=all_possible_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if args.method == 'hard' else 'Oranges', xticklabels=all_possible_labels, yticklabels=all_possible_labels)
    plt.title(f"Confusion Matrix: Ensemble {args.method.capitalize()} Voting (LOSO Aggregate)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(ensemble_results_dir, "loso_confusion_matrix.png"))
    plt.close()

    print("\n" + "="*60)
    print(f"ENSEMBLE ({args.method.upper()}) EVALUATION COMPLETE")
    print(f"WA: {wa:.1%}, UA: {ua:.1%}")
    print(f"Results saved in: {ensemble_results_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

