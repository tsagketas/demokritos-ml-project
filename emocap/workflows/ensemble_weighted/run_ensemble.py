import subprocess
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 Starting Weighted Ensemble Stage: {description}")
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
    parser.add_argument('--models', type=str, default='svm,random_forest,xgboost',
                        help='Comma-separated list of models to include')
    args = parser.parse_args()

    logger.info("Starting Weighted Ensemble Evaluation (Auto-weights based on UA)")
    
    # Define results directory
    results_dir = "emocap/workflows/ensemble_weighted/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run Ensemble Evaluation Script with Soft Voting and Auto Weights
    # The --weights auto flag will look at emocap/results/[model]/loso_summary.csv
    # and calculate weights proportional to their UA score.
    run_command(
        f"python emocap/scripts/07_ensemble_evaluation.py --method soft --models {args.models} --weights auto --output_dir {results_dir}",
        "Weighted Ensemble Aggregate Evaluation (Soft Voting + Auto Weights)"
    )

    logger.info("\n" + "="*60)
    logger.info("🎯 WEIGHTED ENSEMBLE WORKFLOW COMPLETE!")
    logger.info(f"Results available in '{results_dir}/'")
    logger.info("="*60)

if __name__ == "__main__":
    main()

