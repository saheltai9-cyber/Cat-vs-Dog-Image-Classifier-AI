"""
Root Training CLI Entrypoint
Usage:
  python train.py
  python train.py --epochs 8 --fine-tune-epochs 3
"""

import sys
import argparse
from pathlib import Path

# Ensure project root in sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from download_data import extract_and_validate
from src.model_training import train_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train PawVision Cat vs Dog Classifier")
    parser.add_argument("--epochs", type=int, default=10, help="Initial training epochs (frozen base)")
    parser.add_argument("--fine-tune-epochs", type=int, default=4, help="Fine-tuning epochs")
    parser.add_argument("--no-fine-tune", action="store_true", help="Skip fine-tuning")
    parser.add_argument("--force-download", action="store_true", help="Force dataset download")

    args = parser.parse_args()

    print("[*] Ensuring dataset is available...")
    extract_and_validate(force_extract=args.force_download)

    print("\n[*] Launching training pipeline...")
    train_pipeline(
        initial_epochs=args.epochs,
        fine_tune_epochs=0 if args.no_fine_tune else args.fine_tune_epochs,
        do_fine_tuning=not args.no_fine_tune
    )


if __name__ == "__main__":
    main()

