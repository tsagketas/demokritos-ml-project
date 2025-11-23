import sys
import zipfile
import tarfile
import shutil
from pathlib import Path

def extract_archive(archive_path, extract_to):
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

def copy_directory(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    
    files = list(src.rglob('*'))
    total_files = sum(1 for f in files if f.is_file())
    copied = 0
    
    for item in files:
        if item.is_file():
            relative_path = item.relative_to(src)
            target_path = dst / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target_path)
            copied += 1
            if copied % 100 == 0 or copied == total_files:
                print(f"Copied {copied}/{total_files} files...")

def main():
    try:
        import kagglehub
        
        target_dir = Path("datasets/cremad")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print("Downloading...")
        dataset_path = kagglehub.dataset_download("ejlok1/cremad")
        dataset_path = Path(dataset_path)
        
        archives = []
        for ext in ['*.zip', '*.tar', '*.tar.gz', '*.tgz']:
            archives.extend(dataset_path.rglob(ext))
        
        if archives:
            print("Extracting...")
            for archive in archives:
                extract_archive(archive, target_dir)
                archive.unlink()
                print(f"Deleted: {archive.name}")
            print("Done!")
        else:
            print("Copying files...")
            copy_directory(dataset_path, target_dir)
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
