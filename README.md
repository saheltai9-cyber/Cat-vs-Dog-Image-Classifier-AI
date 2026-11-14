# 🐾 PawVision AI — Cat vs Dog Image Classifier AI

[![Python 3.11 / 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![MobileNetV2](https://img.shields.io/badge/Backbone-MobileNetV2-brightgreen.svg)](https://keras.io/api/applications/mobilenet/)
[![Accuracy 98.10%](https://img.shields.io/badge/Test%20Accuracy-98.10%25-success.svg)](#actual-held-out-test-performance)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**PawVision AI** is a production-grade Computer Vision application that accurately classifies images as either a **Cat** or a **Dog** using a fine-tuned **MobileNetV2** deep convolutional neural network with transfer learning. Built with a modern, reactive **NiceGUI** dashboard featuring dark/light theme support, rock-solid single-image upload state management, and real-time inference.

---

## 🌟 Key Features

1. **State-of-the-Art Transfer Learning**:
   - MobileNetV2 pretrained on 1.4 million ImageNet samples for low-level feature extraction.
   - Custom classification head: `GlobalAveragePooling2D` + `Dropout(0.2)` + `Dense(1, Sigmoid)`.
   - **98.10% accuracy** on the 1,000-image untouched held-out test set.
2. **Rock-Solid Single-Image State Management**:
   - Zero stale-state guarantees: uploads are read once, hashed with SHA-256, and atomically bound to previews and predictions.
   - Uploading a new image immediately purges previous results.
   - Complete clear button resets preview and predictions together.
3. **Calibrated Probability Scores & Uncertainty Detection**:
   - Displays real percentage scores for both Cat and Dog.
   - Calibrated confidence metric.
   - Low confidence threshold (<60%) automatically displays a warning for ambiguous inputs.
4. **Resilient Image Preprocessing Pipeline**:
   - Full support for `JPG`, `JPEG`, `PNG`, and `WEBP`.
   - Automatic RGBA / transparent PNG handling via white canvas composition.
   - Single-channel Grayscale expansion to 3-channel RGB.
   - Camera EXIF orientation correction (`exif_transpose`).
   - Aspect-ratio preserving padding to 160 × 160 px.
5. **Interactive NiceGUI Dashboard**:
   - **Overview Dashboard**: High-level system architecture, dataset balance, and test performance.
   - **AI Classifier**: Drag-and-drop file upload, real-time preview, instant classification, and probability bars.
   - **Model Insights**: Interactive Plotly confusion matrix heatmap, training & validation curves, and per-class precision/recall tables.
   - **Dataset & About**: Technical documentation, preprocessing explanation, and model scope.
6. **Dark & Light Mode Support**:
   - High-contrast WCAG AA accessible design tokens.
   - Persistent theme toggle via browser local storage.
7. **CPU-Optimized & 100% Offline Inference**:
   - Tuned for standard laptops (Intel i5, 8GB RAM).
   - Singleton model caching ensures the model is loaded once into memory.

---

## 📊 Dataset Specifications

The application uses the official **TensorFlow Cats vs Dogs filtered dataset**:
- **Source**: `https://download.mlcc.google.com/mledu-datasets/cats_and_dogs_filtered.zip`
- **Total Images**: 3,000 authentic photographs.
- **Partitioning Strategy**:
  - **Training Subset (80%)**: 1,600 images (800 cats, 800 dogs) with realistic data augmentation (flips, rotations, zoom, contrast).
  - **Validation Subset (20%)**: 400 images (200 cats, 200 dogs) for early stopping and model checkpointing.
  - **Held-Out Test Set**: 1,000 images (500 cats, 500 dogs) strictly held out from training and tuning.

---

## 🎯 Actual Held-Out Test Performance

Evaluated on the 1,000-image untouched held-out test set:

| Metric | Score | Details |
|---|---|---|
| **Test Accuracy** | **98.10%** | 981 / 1,000 correct predictions |
| **Precision** | **98.39%** | High positive predictive value |
| **Recall** | **97.80%** | High true positive sensitivity |
| **F1-Score** | **98.09%** | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.9965** | Area under the ROC curve |

### Confusion Matrix
```text
                  Predicted Cat    Predicted Dog
Actual Cat             492               8          (Cat Accuracy: 98.4%)
Actual Dog              11             489          (Dog Accuracy: 97.8%)
```

---

## 🚀 Quick Launch (Windows)

Simply double-click:
```bat
run.bat
```
This batch script will automatically:
1. Verify 64-bit Python 3.11/3.12.
2. Initialize and activate a local virtual environment (`.venv`).
3. Install pinned dependencies from `requirements.txt`.
4. Verify local dataset availability.
5. Check for the trained model artifact (`models/cat_dog_mobilenetv2.keras`).
6. Launch the NiceGUI dashboard on `http://127.0.0.1:8080`.

To retrain the model manually at any time:
```bat
train_model.bat
```

---

## 💻 Manual CLI Instructions

### 1. Environment Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download and Verify Dataset
```bash
python download_data.py
```

### 3. Train Model
```bash
python train.py --epochs 8 --fine-tune-epochs 2
```

### 4. Run CLI Prediction on Any Image
```bash
# Standard console output
python predict.py path/to/image.jpg

# JSON output
python predict.py path/to/image.jpg --json
```

### 5. Launch Web Dashboard
```bash
python app.py
```
Open your browser to: **`http://127.0.0.1:8080`**

### 6. Run Test Suite
```bash
pytest
```

---

## 📁 Project Structure

```text
05-Cat-vs-Dog-Image-Classifier-AI/
│
├── app.py                      # NiceGUI Web Application Entrypoint
├── train.py                    # Root Training Entrypoint
├── predict.py                  # Standalone CLI Predictor
├── download_data.py            # Automatic Dataset Downloader & Validator
├── package_project.py          # Distribution Packager & Integrity Verifier
├── requirements.txt            # Pinned CPU-compatible Dependencies
├── README.md                   # Comprehensive Project Documentation
├── run.bat                     # 1-Click Windows Launch Script
├── train_model.bat             # 1-Click Retraining Script
├── .gitignore                  # Git Ignore Specifications
│
├── src/
│   ├── __init__.py             # Source Package Init
│   ├── config.py               # Constants, Paths & Hyperparameters
│   ├── data_loader.py          # Data Splitting, Pipelines & Augmentation
│   ├── image_preprocessing.py  # Loading, Resizing, EXIF & Normalization
│   ├── model.py                # MobileNetV2 Architecture & Fine-Tuning Setup
│   ├── model_training.py       # Training Loop, Callbacks & Checkpointing
│   ├── predictor.py            # Singleton Predictor Engine & Hashing
│   ├── evaluation.py           # Evaluation Metrics & JSON Serialization
│   └── ui_components.py        # CSS Design Tokens, Theme Toggle & Widgets
│
├── data/
│   ├── raw/                    # Downloaded and Extracted Dataset
│   └── processed/              # Processed Cache
│
├── models/
│   └── cat_dog_mobilenetv2.keras # Trained Keras Model Artifact
│
├── reports/
│   ├── model_metrics.json      # Serialized Test Evaluation Metrics
│   ├── training_history.json   # Epoch Loss and Accuracy Curves
│   └── FINAL_RELEASE_REPORT.md # Official Verification and Release Report
│
├── tests/
│   ├── __init__.py
│   ├── test_image_preprocessing.py # Image Preprocessing & Format Tests
│   ├── test_prediction.py          # Predictor & Output Format Tests
│   ├── test_data_loader.py         # Split & Pipeline Tests
│   ├── test_model_pipeline.py      # Architecture & Forward Pass Tests
│   └── test_app_utils.py           # UI State Management Tests
│
├── assets/                     # UI Assets & Documentation Icons
│
└── dist/
    └── Cat-vs-Dog-Image-Classifier-AI.zip # Verified Portable Release ZIP
```

---

## ⚠️ Important Limitations

- **Binary Closed-Set Classification**: This model is explicitly trained to classify images as either **Cat** or **Dog**. It is **not a universal open-domain object detector**. Images of automobiles, humans, or birds will be forced into a Cat or Dog prediction.
- **Uncertainty Warning**: For out-of-distribution or ambiguous inputs, the model will output balanced scores (<60% top confidence) and trigger a clear **Uncertainty Warning**.

---

## 📜 License
Distributed under the MIT License. Developed for Machine Learning Project 05.

