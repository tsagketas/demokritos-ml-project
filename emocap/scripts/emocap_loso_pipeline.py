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
        sys.exit(1)

def main():
    logger.info("Starting IEMOCAP LOSO Cross-Validation Pipeline (Random Forest)")
    
    # 1. Preprocessing (SMOTE on all folds, all 136 features)
    run_command(
        "python emocap/scripts/04_preprocess_data.py --all --resample smote",
        "Data Preprocessing (LOSO Folds + SMOTE)"
    )

    # 2. Training (LOSO Per-Fold Training)
    run_command(
        "python emocap/scripts/05_model_train.py",
        "LOSO Per-Fold Training & Tuning"
    )

    # 3. Evaluation (Aggregate LOSO Metrics)
    run_command(
        "python emocap/scripts/06_model_evaluation.py",
        "Aggregate LOSO Evaluation"
    )

    logger.info("\n" + "="*60)
    logger.info("🎯 LOSO PIPELINE EXECUTION COMPLETE!")
    logger.info("Results available in 'emocap/results/random_forest/'")
    logger.info("="*60)

if __name__ == "__main__":
    main()
