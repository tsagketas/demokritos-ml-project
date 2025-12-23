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
    logger.info("Starting IEMOCAP LOSO Pipeline (Steps 02-06)")
    
    # 2. Create LOSO Folds (Split)
    run_command(
        "python emocap/scripts/02_split_train_test.py",
        "LOSO Data Splitting (Step 02)"
    )

    # 3. Data Analysis (Optional/Descriptive)
    run_command(
        "python emocap/scripts/03_data_analysis.py",
        "Data Analysis & Visualization (Step 03)"
    )

    # 4. Preprocessing (Scaling & SMOTE per fold)
    run_command(
        "python emocap/scripts/04_preprocess_data.py --all --resample smote",
        "Data Preprocessing & Balancing (Step 04)"
    )

    # 5. Training (LOSO Per-Fold Training with Optuna)
    run_command(
        "python emocap/scripts/05_model_train.py",
        "LOSO Model Training & Tuning (Step 05)"
    )

    # 6. Evaluation (Aggregate LOSO Metrics)
    run_command(
        "python emocap/scripts/06_model_evaluation.py",
        "Aggregate LOSO Evaluation & Reports (Step 06)"
    )

    logger.info("\n" + "="*60)
    logger.info("🎯 LOSO PIPELINE EXECUTION COMPLETE!")
    logger.info("Results available in 'emocap/results/random_forest/'")
    logger.info("="*60)

if __name__ == "__main__":
    main()