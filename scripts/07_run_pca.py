import argparse
import importlib.util
from pathlib import Path

# Import the _run_pca function from 02_split_train_test.py
spec = importlib.util.spec_from_file_location("split_module", Path(__file__).parent / "02_split_train_test.py")
split_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split_module)
_run_pca = split_module._run_pca


def main():
    parser = argparse.ArgumentParser(description="Run PCA on features and split IEMOCAP into train/test")
    parser.add_argument("--workflow-dir", type=Path, required=True, help="Workflow directory")
    parser.add_argument("--n-components", type=int, default=None, help="Number of PCA components")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    _run_pca(args.workflow_dir, args.n_components, args.seed)


if __name__ == "__main__":
    main()