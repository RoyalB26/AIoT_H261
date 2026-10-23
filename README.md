# Patchcore Industrial Anomaly Detection - Model Baseline Hub

Welcome to the **Patchcore Model Hub**. This directory organizes the entire lifecycle of the **Patchcore Anomaly Detection and Localization** model (*Roth et al., CVPR 2022*) into two distinct, production-ready baselines:

- 🏋️ **`train_baseline/`**: Standalone training, evaluation, validation, and ONNX export pipeline.
- 🔍 **`inference_baseline/`**: High-performance, user-friendly inference API and CLI supporting single-image and batch processing with defect heatmaps.

Both modules are completely self-contained, fully documented with docstrings, and pre-configured for industrial datasets: **MVTec AD**, **VisA**, and **PCB Defect Dataset**.

---

## 📁 Directory Structure

```text
model/
├── train_baseline/             # Modular Training Baseline
│   ├── train.py                # Standalone training script with CLI args
│   └── README.md               # Training guidelines, options, and benchmarks
│
├── inference_baseline/         # Modular Inference Baseline
│   ├── predictor.py            # Object-oriented Python API (PatchcorePredictor)
│   ├── inference.py            # High-level CLI for single & batch inference
│   └── README.md               # Inference quickstart, CLI options & API examples
│
└── README.md                   # Model hub overview and directory navigation (this file)
```

---

## 📦 Baselines Overview

### 1. `train_baseline/`
The training baseline provides an end-to-end training and evaluation script:
- **Dataset Agnostic**: No restrictions to specific datasets. Users supply `--root_dir` and `--category`. The pipeline automatically validates folder integrity before execution.
- **Fast Training**: Employs optimized memory bank indexing with representative coreset selection, reducing training time to **8–40 seconds** per category while retaining state-of-the-art pixel localization accuracy.
- **Auto Checkpointing & ONNX Export**: Saves PyTorch Lightning `.ckpt` checkpoints and optionally exports deployable ONNX models.

👉 **Quick command**:
```bash
python train_baseline/train.py --root_dir "/path/to/dataset" --category bottle --export_onnx
```
*For complete instructions, dataset flags, and parameter customization, refer to [train_baseline/README.md](train_baseline/README.md).*

---

### 2. `inference_baseline/`
The inference baseline is designed for straightforward deployment in manufacturing and quality inspection environments:
- **Plug-and-Play API**: Clean `PatchcorePredictor` class with high-level methods (`predict_image`, `visualize_and_save`, `predict_batch`).
- **Rich Heatmap Overlays**: Computes normalized pixel anomaly heatmaps and blends them with the original image using a JET colormap, alongside status banners and anomaly scores.
- **Batch Processing**: Automatically scans input folders, classifies defective vs. normal items, and logs results to a summary CSV.
- **CPU & GPU Ready**: Seamlessly runs on Intel/AMD CPUs (~25–38 FPS) or NVIDIA GPUs (>60 FPS).

👉 **Quick command**:
```bash
python inference_baseline/inference.py \
    --dataset mvtec \
    --category bottle \
    --image "D:/apps/Datasets/MVTec AD/bottle/test/broken_large/000.png" \
    --output_dir "./inference_output" \
    --device cpu
```
*For Python API snippets and batch evaluation examples, refer to [inference_baseline/README.md](inference_baseline/README.md).*

---

## 🛠️ Environment Prerequisites

Both baselines share standard computer vision and PyTorch dependencies:

```bash
pip install torch torchvision timm anomalib opencv-python pandas pillow tqdm
```

Recommended Python version: **Python 3.10+**.

---

## 📊 Summary Performance Across Datasets

Evaluated across **33 industrial categories** using ResNet-18 feature extraction:

| Dataset | Total Categories | Mean Image AUROC | Mean Pixel AUROC | Mean CPU Latency | Mean CPU FPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MVTec AD** | 15 categories | **85.46%** | **96.17%** | 37.98 ms | **26.33 FPS** |
| **VisA** | 12 categories | **81.74%** | **97.54%** | 40.38 ms | **24.80 FPS** |
| **PCB Defect** | 6 categories | **66.51%** | **62.57%** | 26.27 ms | **38.18 FPS** |


---

## 🔗 Related Documentation
- [Training Module Documentation](train_baseline/README.md): Detailed training baseline guide.
- [Inference Module Documentation](inference_baseline/README.md): Detailed inference baseline guide.
