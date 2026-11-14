import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DIST_DIR = PROJECT_ROOT / 'dist'
ZIP_NAME = 'Customer-Segmentation-AI.zip'
ZIP_PATH = DIST_DIR / ZIP_NAME

# Ensure dist directory exists
DIST_DIR.mkdir(exist_ok=True)

# Define what to include
include_paths = [
    'app.py',
    'ui_app.py',
    'train.py',
    'predict.py',
    'requirements.txt',
    'README.md',
    'DESIGN_SYSTEM.md',
    'run.bat',
    'run_nicegui.bat',
    '.gitignore',
    'src',
    'data',
    'models',
    'reports',
    'tests'
]

def copy_for_zip(temp_dir: Path):
    ignore_patterns = shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo', '.DS_Store')
    for rel in include_paths:
        src = PROJECT_ROOT / rel
        dst = temp_dir / rel
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore_patterns)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} does not exist")

def create_zip():
    # Create a temporary staging area
    temp_dir = PROJECT_ROOT / 'tmp_zip_staging'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    copy_for_zip(temp_dir)
    # Create zip archive
    shutil.make_archive(base_name=str(ZIP_PATH.with_suffix('')), format='zip', root_dir=temp_dir)
    # Clean up staging area
    shutil.rmtree(temp_dir)
    print(f"Created zip at {ZIP_PATH}")

if __name__ == '__main__':
    create_zip()

