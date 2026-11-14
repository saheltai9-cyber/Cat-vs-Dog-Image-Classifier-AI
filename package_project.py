"""
Packaging and Distribution Verification Script
Builds a clean, portable distribution ZIP and verifies extraction integrity.
"""

import os
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import DIST_DIR, BEST_MODEL_PATH

ZIP_OUTPUT_PATH = DIST_DIR / "Cat-vs-Dog-Image-Classifier-AI.zip"

# Exclusion patterns for portable release
EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
    ".git",
    ".idea",
    ".vscode",
    "dist",
    "cats_and_dogs_filtered"
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".tmp",
    ".log",
    ".DS_Store"
}


def should_exclude(rel_path: Path) -> bool:
    """Check if file or directory should be excluded from package."""
    parts = rel_path.parts
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
        if part.startswith("."):
            return True

    if rel_path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return True

    # Exclude raw dataset zip or raw dataset files (to keep package portable)
    if "data" in parts and "raw" in parts:
        if rel_path.name.endswith(".zip") or "cats_and_dogs_filtered" in parts:
            return True

    return False


def build_package() -> Path:
    """Builds the final distribution ZIP archive."""
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_OUTPUT_PATH.exists():
        ZIP_OUTPUT_PATH.unlink()

    print(f"[*] Packaging project to: {ZIP_OUTPUT_PATH}")
    file_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(ZIP_OUTPUT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Ensure data and report directories exist with placeholder .gitkeep
        for keep_dir in ["data/raw", "data/processed", "models", "reports"]:
            zf.writestr(f"05-Cat-vs-Dog-Image-Classifier-AI/{keep_dir}/.gitkeep", b"")

        # Walk project directory
        for root, dirs, files in os.walk(PROJECT_DIR):
            root_path = Path(root)
            rel_dir = root_path.relative_to(PROJECT_DIR)

            # Skip excluded directories
            if should_exclude(rel_dir):
                dirs[:] = []
                continue

            for file in files:
                file_path = root_path / file
                rel_file = file_path.relative_to(PROJECT_DIR)

                if should_exclude(rel_file):
                    continue

                archive_arcname = Path("05-Cat-vs-Dog-Image-Classifier-AI") / rel_file
                zf.write(file_path, arcname=str(archive_arcname))
                file_count += 1
                total_uncompressed_bytes += file_path.stat().st_size

    zip_size_mb = ZIP_OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"[+] Package created: {file_count} files, {zip_size_mb:.2f} MB (Uncompressed: {total_uncompressed_bytes/(1024*1024):.2f} MB)")
    return ZIP_OUTPUT_PATH


def verify_package(zip_path: Path) -> bool:
    """Extracts zip into a temporary directory and verifies integrity and essential files."""
    print("\n[*] Verifying package integrity...")
    if not zipfile.is_zipfile(zip_path):
        print("[-] Verification failed: Not a valid ZIP file.")
        return False

    with zipfile.ZipFile(zip_path, "r") as zf:
        corrupted = zf.testzip()
        if corrupted:
            print(f"[-] Verification failed: Corrupted file detected in ZIP: {corrupted}")
            return False

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"[*] Extracting package to test sandbox: {tmp_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_path)

        extracted_root = tmp_path / "05-Cat-vs-Dog-Image-Classifier-AI"
        if not extracted_root.exists():
            print("[-] Verification failed: Root directory missing in ZIP.")
            return False

        # Check required files
        required_files = [
            "app.py",
            "train.py",
            "predict.py",
            "download_data.py",
            "requirements.txt",
            "README.md",
            "run.bat",
            "train_model.bat",
            "src/config.py",
            "src/model.py",
            "src/predictor.py",
            "src/image_preprocessing.py",
            "src/data_loader.py",
            "src/evaluation.py",
            "src/ui_components.py",
            "tests/test_image_preprocessing.py",
            "tests/test_prediction.py"
        ]

        missing = []
        for rf in required_files:
            if not (extracted_root / rf).exists():
                missing.append(rf)

        if missing:
            print(f"[-] Verification failed: Missing required files in ZIP:\n    {missing}")
            return False

        print("[+] All required application and source files present.")

        # Check model artifact if present in original
        if BEST_MODEL_PATH.exists():
            extracted_model = extracted_root / "models" / BEST_MODEL_PATH.name
            if extracted_model.exists():
                print(f"[+] Trained model artifact verified in ZIP ({extracted_model.stat().st_size / (1024*1024):.1f} MB).")
            else:
                print("[-] Warning: Model artifact was expected in ZIP but not found.")

    print("[+] ZIP Package verification PASSED with 100% integrity!")
    return True


if __name__ == "__main__":
    pkg = build_package()
    ok = verify_package(pkg)
    if not ok:
        sys.exit(1)
