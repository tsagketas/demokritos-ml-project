"""
Download and extract Dusha dataset from Kaggle
"""
import sys
import zipfile
import tarfile
from pathlib import Path

def extract_archive(archive_path, extract_to):
    """Extract archive with progress"""
    archive_path = Path(archive_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix in ['.tar', '.gz'] or '.tar.' in archive_path.name:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        return False
    return True

def main():
    try:
        import kagglehub
        
        target_dir = Path("datasets/dusha")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print("Downloading...")
        dataset_path = kagglehub.dataset_download("sigireddybalasai/dusha-datasetcrowd")
        
        # Find archives
        archives = []
        for ext in ['*.zip', '*.tar', '*.tar.gz', '*.tgz']:
            archives.extend(Path(dataset_path).rglob(ext))
        
        if archives:
            print("Extracting...")
            for archive in archives:
                extract_archive(archive, target_dir)
                archive.unlink()
            print("Done!")
        else:
            print("Done!")
        
    except ImportError:
        print("Error: kagglehub not installed")
        print("Run: pip install kagglehub")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
