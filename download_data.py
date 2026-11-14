"""
Automatic Dataset Downloader and Integrity Validator
Downloads and extracts the official TensorFlow Cats vs Dogs filtered dataset.
"""

import os
import sys
import zipfile
import urllib.request
import shutil
from pathlib import Path
from PIL import Image

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import (
    DATASET_URL,
    DATASET_ZIP_PATH,
    DATASET_EXTRACT_DIR,
    RAW_TRAIN_DIR,
    RAW_VAL_DIR,
    RAW_DATA_DIR
)


def is_dataset_ready() -> tuple[bool, dict]:
    """Check if the dataset is already downloaded, extracted, and verified."""
    if not (RAW_TRAIN_DIR.exists() and RAW_VAL_DIR.exists()):
        return False, {}

    train_cats = list((RAW_TRAIN_DIR / "cats").glob("*.jpg"))
    train_dogs = list((RAW_TRAIN_DIR / "dogs").glob("*.jpg"))
    val_cats = list((RAW_VAL_DIR / "cats").glob("*.jpg"))
    val_dogs = list((RAW_VAL_DIR / "dogs").glob("*.jpg"))

    # Expected ~1000 cats, ~1000 dogs in train; ~500 cats, ~500 dogs in val
    if len(train_cats) >= 950 and len(train_dogs) >= 950 and len(val_cats) >= 450 and len(val_dogs) >= 450:
        counts = {
            "train_cats": len(train_cats),
            "train_dogs": len(train_dogs),
            "train_total": len(train_cats) + len(train_dogs),
            "val_cats": len(val_cats),
            "val_dogs": len(val_dogs),
            "val_total": len(val_cats) + len(val_dogs),
            "total_images": len(train_cats) + len(train_dogs) + len(val_cats) + len(val_dogs)
        }
        return True, counts

    return False, {}


def download_progress_hook(block_num, block_size, total_size):
    """Console progress reporter for urllib download."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  -> Downloading: {percent:5.1f}% [{mb_down:6.1f} MB / {mb_total:6.1f} MB]")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\r  -> Downloading: {downloaded / (1024*1024):6.1f} MB")
        sys.stdout.flush()


def download_dataset(force: bool = False) -> Path:
    """Download the Cats vs Dogs filtered dataset archive with integrity checks."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_ZIP_PATH.exists() and not force:
        print(f"[*] Found existing archive: {DATASET_ZIP_PATH}")
        if zipfile.is_zipfile(DATASET_ZIP_PATH):
            try:
                with zipfile.ZipFile(DATASET_ZIP_PATH, 'r') as zf:
                    if zf.testzip() is None:
                        print("[+] Archive integrity verified.")
                        return DATASET_ZIP_PATH
                    else:
                        print("[-] Existing archive is corrupted. Re-downloading...")
            except Exception as e:
                print(f"[-] Error testing existing archive: {e}. Re-downloading...")
        else:
            print("[-] Existing file is not a valid zip archive. Re-downloading...")

    # Download to temporary file first to prevent partial/corrupted files
    temp_zip = RAW_DATA_DIR / "cats_and_dogs_filtered.tmp"
    if temp_zip.exists():
        temp_zip.unlink()

    print(f"[*] Downloading dataset from:\n    {DATASET_URL}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PawVision/1.0"}
    req = urllib.request.Request(DATASET_URL, headers=headers)

    with urllib.request.urlopen(req) as response, open(temp_zip, 'wb') as out_file:
        total_size = int(response.info().get('Content-Length', -1))
        downloaded = 0
        block_size = 65536
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            downloaded += len(buffer)
            out_file.write(buffer)
            if total_size > 0:
                percent = min(100.0, downloaded * 100 / total_size)
                sys.stdout.write(f"\r  -> Progress: {percent:5.1f}% [{downloaded/(1024*1024):6.1f} MB / {total_size/(1024*1024):6.1f} MB]")
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\r  -> Progress: {downloaded/(1024*1024):6.1f} MB")
                sys.stdout.flush()
    print("\n[+] Download completed successfully.")

    # Verify temp file
    if not zipfile.is_zipfile(temp_zip):
        temp_zip.unlink(missing_ok=True)
        raise RuntimeError("Downloaded file failed zipfile validation.")

    with zipfile.ZipFile(temp_zip, 'r') as zf:
        corrupt_file = zf.testzip()
        if corrupt_file:
            temp_zip.unlink(missing_ok=True)
            raise RuntimeError(f"Corrupted file inside downloaded zip: {corrupt_file}")

    if DATASET_ZIP_PATH.exists():
        DATASET_ZIP_PATH.unlink()
    temp_zip.rename(DATASET_ZIP_PATH)
    print(f"[+] Saved verified archive to: {DATASET_ZIP_PATH}")
    return DATASET_ZIP_PATH


def extract_and_validate(force_extract: bool = False) -> dict:
    """Extract archive and clean/validate corrupted image files."""
    ready, counts = is_dataset_ready()
    if ready and not force_extract:
        print("[+] Valid dataset already extracted and verified.")
        print(f"    Train: {counts['train_cats']} cats, {counts['train_dogs']} dogs ({counts['train_total']} total)")
        print(f"    Validation: {counts['val_cats']} cats, {counts['val_dogs']} dogs ({counts['val_total']} total)")
        print(f"    Total: {counts['total_images']} images")
        return counts

    zip_path = download_dataset(force=force_extract)

    print(f"[*] Extracting {zip_path.name} to {RAW_DATA_DIR}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(RAW_DATA_DIR)
    print("[+] Extraction complete.")

    # Validate all images and prune corrupted ones if any
    print("[*] Validating all extracted image files with Pillow...")
    valid_count = 0
    corrupt_count = 0

    for split in ["train", "validation"]:
        for cls in ["cats", "dogs"]:
            folder = DATASET_EXTRACT_DIR / split / cls
            if not folder.exists():
                raise FileNotFoundError(f"Missing expected folder: {folder}")
            for img_path in list(folder.glob("*.*")):
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    # Reopen to test actual decompression
                    with Image.open(img_path) as img:
                        img.load()
                    valid_count += 1
                except Exception as e:
                    print(f"[-] Corrupt image removed: {img_path.name} ({e})")
                    img_path.unlink(missing_ok=True)
                    corrupt_count += 1

    print(f"[+] Image validation done: {valid_count} valid images, {corrupt_count} corrupt removed.")

    ready, counts = is_dataset_ready()
    if not ready:
        raise RuntimeError(f"Dataset extraction incomplete or insufficient valid images: {counts}")

    print(f"[+] Dataset is ready for training and evaluation.")
    return counts


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    try:
        extract_and_validate(force_extract=force_flag)
    except Exception as e:
        print(f"\n[!] Dataset download/extraction failed: {e}")
        sys.exit(1)

