"""
Standalone Prediction CLI Module
Usage:
  python predict.py path/to/cat_or_dog.jpg
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure project root in sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.predictor import get_predictor


def main():
    parser = argparse.ArgumentParser(description="Predict Cat or Dog for an input image.")
    parser.add_argument("image_path", type=str, help="Path to input image file (JPG, PNG, WEBP)")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")

    args = parser.parse_args()
    image_path = Path(args.image_path)

    if not image_path.exists():
        print(f"[!] Error: Image not found at {image_path}")
        sys.exit(1)

    predictor = get_predictor()
    result = predictor.predict(image_path)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=" * 50)
    print("   PawVision AI — Prediction Result")
    print("=" * 50)
    if not result["success"]:
        print(f"[!] Prediction failed: {result['error']}")
        sys.exit(1)

    print(f"File: {image_path.name}")
    print(f"Predicted Class : {result['predicted_class']}")
    print(f"Cat Score       : {result['cat_percentage']}% ({result['cat_probability']:.4f})")
    print(f"Dog Score       : {result['dog_percentage']}% ({result['dog_probability']:.4f})")
    print(f"Model Confidence: {result['confidence_percentage']}%")

    if result["is_uncertain"]:
        print(f"[!] Warning: {result['uncertainty_warning']}")

    print("=" * 50)


if __name__ == "__main__":
    main()

