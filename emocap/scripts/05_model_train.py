import os
import pandas as pd
import numpy as np
import pickle
import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
import json
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_fold_data(base_path, fold):
    """Loads train and validation data for a specific fold."""
    fold_path = os.path.join(base_path, f"fold_{fold}", "processed")
    
    train_df = pd.read_csv(os.path.join(fold_path, 'train.csv'))
    val_df = pd.read_csv(os.path.join(fold_path, 'val.csv'))
    
    meta_cols = ['session', 'method', 'gender', 'emotion', 'n_annotators', 'agreement']
    features = [c for c in train_df.columns if c not in meta_cols]
    
    X_train = train_df[features]
    y_train = train_df['emotion']
    X_val = val_df[features]
    y_val = val_df['emotion']
    
    return X_train, y_train, X_val, y_val

def get_tuning_function(model_name):
    """Returns the objective function for Optuna based on model name."""
    
    def rf_objective(trial, X_train, y_train, X_val, y_val):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 10, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        return recall_score(y_val, model.predict(X_val), average='macro')

    def xgb_objective(trial, X_train, y_train, X_val, y_val):
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_val_enc = le.transform(y_val)
        # Calculate scale_pos_weight estimate for binary or use a sample weight approach
        # For multi-class, XGBoost doesn't have a single scale_pos_weight
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42,
            'verbosity': 0
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train_enc)
        return recall_score(y_val_enc, model.predict(X_val), average='macro')

    def svm_objective(trial, X_train, y_train, X_val, y_val):
        params = {
            'C': trial.suggest_float('C', 0.1, 100, log=True),
            'kernel': 'rbf',
            'gamma': trial.suggest_float('gamma', 1e-4, 1e-1, log=True),
            'class_weight': 'balanced',
            'random_state': 42
        }
        model = SVC(**params)
        model.fit(X_train, y_train)
        return recall_score(y_val, model.predict(X_val), average='macro')

    def dt_objective(trial, X_train, y_train, X_val, y_val):
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'random_state': 42
        }
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        return recall_score(y_val, model.predict(X_val), average='macro')

    objectives = {
        'random_forest': rf_objective,
        'xgboost': xgb_objective,
        'svm': svm_objective,
        'decision_tree': dt_objective
    }
    return objectives.get(model_name)

def train_fold_model(model_name, fold, X_train, y_train, X_val, y_val, models_dir, results_dir):
    logger.info(f"--- Training {model_name} for Fold {fold} ---")
    
    tuning_func = get_tuning_function(model_name)
    study = optuna.create_study(direction='maximize')
    n_trials = 30 if model_name == 'svm' else 15
    study.optimize(lambda trial: tuning_func(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
    
    best_params = study.best_params
    logger.info(f"Best params for {model_name} Fold {fold}: {best_params}")
    
    # Train final model for this fold
    if model_name == 'xgboost':
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train_enc)
        model._label_encoder = le
    elif model_name == 'svm':
        # Ensure balanced weights and probability for soft voting
        final_params = best_params.copy()
        final_params['class_weight'] = 'balanced'
        final_params['probability'] = True
        model = SVC(**final_params)
        model.fit(X_train, y_train)
    elif model_name == 'random_forest':
        # Ensure balanced weights
        final_params = best_params.copy()
        final_params['class_weight'] = 'balanced'
        model = RandomForestClassifier(**final_params)
        model.fit(X_train, y_train)
    else: # decision_tree
        model = DecisionTreeClassifier(**best_params)
        model.fit(X_train, y_train)
        
    # Save model in its dedicated folder
    fold_model_dir = os.path.join(models_dir, model_name)
    os.makedirs(fold_model_dir, exist_ok=True)
    model_path = os.path.join(fold_model_dir, f"fold_{fold}.pkl")
    joblib.dump(model, model_path)
    
    # Save params
    fold_results_dir = os.path.join(results_dir, model_name)
    os.makedirs(fold_results_dir, exist_ok=True)
    with open(os.path.join(fold_results_dir, f"best_params_fold_{fold}.json"), 'w') as f:
        json.dump(best_params, f)
    
    logger.info(f"Model saved to {model_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='random_forest', 
                        choices=['random_forest', 'svm', 'xgboost', 'decision_tree'],
                        help='Model to train')
    args = parser.parse_args()

    base_data_path = "data/iemocap/dataset"
    models_root = "emocap/models"
    results_root = "emocap/results"
    
    model_to_train = args.model
    
    fold_model_dir = os.path.join(models_root, model_to_train)
    if os.path.exists(fold_model_dir):
        shutil.rmtree(fold_model_dir)
    os.makedirs(fold_model_dir, exist_ok=True)

    for fold in [1, 2, 3, 4, 5]:
        X_train, y_train, X_val, y_val = load_fold_data(base_data_path, fold)
        train_fold_model(model_to_train, fold, X_train, y_train, X_val, y_val, models_root, results_root)

    logger.info(f"\n--- Training for all folds complete for {model_to_train} ---")

if __name__ == "__main__":
    main()
