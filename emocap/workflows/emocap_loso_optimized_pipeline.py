import subprocess
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 Starting Stage: {description}")
    logger.info(f"Command: {command}")
    logger.info(f"{'='*60}")
    try:
        subprocess.run(command, shell=True, check=True)
        logger.info(f"✅ Stage Completed: {description}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error during Stage: {description}")
        logger.error(f"Error details: {e}")
        sys.exit(1)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='svm', 
                        choices=['random_forest', 'svm', 'xgboost', 'decision_tree'],
                        help='Model to train and evaluate')
    parser.add_argument('--k', type=int, default=150, help='Number of features for Selection')
    parser.add_argument('--mi', action='store_true', help='Use Mutual Information instead of ANOVA')
    args = parser.parse_args()
    
    model_to_run = args.model
    feature_flag = "--mi" if args.mi else "--anova"
    
    logger.info(f"Starting Optimized IEMOCAP LOSO Pipeline for {model_to_run} ({feature_flag} k={args.k})")
    
    # 0. Data Cleanup & 4-Class Mapping + Agreement Filter
    run_command(
        "python emocap/scripts/00_data_cleanup_and_class_merges.py",
        "Data Cleanup & 4-Class Mapping (Step 00)"
    )

    # 2. Create LOSO Folds (Split)
    run_command(
        "python emocap/scripts/02_split_train_test.py",
        "LOSO Data Splitting (Step 02)"
    )

    # 4. Preprocessing (Scaling & Selection, NO SMOTE)
    run_command(
        f"python emocap/scripts/04_preprocess_data.py --all {feature_flag} --k {args.k}",
        f"Data Preprocessing & Feature Selection (Step 04)"
    )

    # 5. Training (with Class Weights)
    run_command(
        f"python emocap/scripts/05_model_train.py --model {model_to_run}",
        f"Model Training & Tuning for {model_to_run} with Class Weights (Step 05)"
    )

    # 6. Evaluation (Aggregate LOSO Metrics)
    run_command(
        f"python emocap/scripts/06_model_evaluation.py --model {model_to_run}",
        f"Aggregate LOSO Evaluation for {model_to_run} (Step 06)"
    )

    logger.info("\n" + "="*60)
    logger.info(f"🎯 OPTIMIZED LOSO PIPELINE COMPLETE FOR {model_to_run}!")
    logger.info(f"Check results in 'emocap/results/{model_to_run}/'")
    logger.info("="*60)

if __name__ == "__main__":
    main()

