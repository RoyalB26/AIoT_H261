# Patchcore Training Baseline

Modular training pipeline for **Patchcore Anomaly Detection & Localization** on arbitrary industrial visual inspection datasets.

The training pipeline is fully dataset-agnostic: it does not enforce a hardcoded dataset name or rely on default dataset paths. Instead, you supply your dataset root directory (`--root_dir`) and optionally target a specific category (`--category`). The pipeline automatically validates your folder structure before execution, raising informative errors if the structure is missing required files or subdirectories.

---

## 📌 Features

- **Dataset Agnostic**: No hardcoded datasets or default paths. Supports any custom dataset complying with standard anomaly detection formats (MVTec-style or VisA-style).
- **Automatic Structure Validation**: Verifies directory hierarchy, mandatory normal training images, and test images before running training.
- **Fast Representative Coreset**: Employs an optimized memory bank builder preventing Out-of-Memory and execution bottlenecks.
- **Windows & Unicode Path Safe**: Built-in path sanitizer supporting Unicode directory paths.
- **Automatic Checkpointing & ONNX Export**: Saves PyTorch Lightning checkpoints (`.ckpt`) and exports deployable **ONNX** graphs (`.onnx`).
- **Comprehensive Logging**: Auto-logs training duration, testing latency, Image AUROC, Pixel AUROC, and F1-Scores to a CSV report.

---

## 📂 Directory Structure

```text
train_baseline/
├── train.py        # Main training script with CLI options
└── README.md       # Baseline documentation and usage guide
```

---

## 🚀 Quickstart & Usage

> **Important**: There are no default datasets. Users must provide `--root_dir`.

### 1. Train a Specific Category
```bash
python train_baseline/train.py --root_dir "/path/to/dataset" --category bottle --backbone resnet18 --export_onnx
```

### 2. Automatically Discover and Train All Categories in Root
If `--category` is omitted, the script automatically scans `--root_dir`, validates all category subfolders, and trains them sequentially:
```bash
python train_baseline/train.py --root_dir "/path/to/dataset" --output_dir ./checkpoints --export_onnx
```

### 3. Custom Batch Size and Coreset Ratio
```bash
python train_baseline/train.py --root_dir "/path/to/dataset" --category candle --batch_size 32 --coreset_ratio 0.01
```

---

## 🗂️ Required Dataset Structure

Before starting training, the script checks if your category directory matches one of the supported structures. If any mandatory folder or image is missing, an explicit error message is displayed.

### Format 1: Standard MVTec AD Format (Recommended for custom datasets)

```text
<root_dir>/
├── <category_name>/
│   ├── train/
│   │   └── good/                 # MANDATORY: Normal (defect-free) images for memory bank
│   │       ├── 000.png
│   │       └── ...
│   ├── test/                     # MANDATORY: Evaluation images
│   │   ├── good/                 # (Optional/Recommended) Defect-free test images
│   │   │   └── 000.png
│   │   └── <defect_type>/        # Defective images (e.g. scratch, crack, broken, ...)
│   │       └── 000.png
│   └── ground_truth/             # (Optional) Binary pixel-level masks (.png)
│       └── <defect_type>/
│           └── 000_mask.png      # Mask corresponding to test/<defect_type>/000.png
```

**Requirements**:
- `train/good/` must contain at least one valid image (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`).
- `test/` must contain valid test images in subdirectories.
- `ground_truth/` is optional; if present, pixel AUROC and pixel F1-score are automatically computed.

---

### Format 2: VisA Format

```text
<root_dir>/
├── <category_name>/
│   ├── Data/
│   │   ├── Images/
│   │   │   ├── Normal/           # MANDATORY: Normal images
│   │   │   └── Anomaly/          # MANDATORY: Defective images
│   │   └── Masks/                # (Optional) Pixel ground truth masks
│   │       └── Anomaly/
│   └── image_anno.csv            # (Optional) Split metadata file
```

---

## ⚙️ Command-Line Arguments

| Argument | Type | Required | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `--root_dir` | `str` | **Yes** | - | Path to dataset root directory containing category folders. |
| `--category` | `str` | No | `None` | Target category to train. If omitted, automatically discovers and trains all valid category folders in `root_dir`. |
| `--backbone` | `str` | No | `resnet18` | Pretrained feature extractor backbone (e.g., `resnet18`, `wide_resnet50_2`). |
| `--batch_size` | `int` | No | `32` | DataLoader batch size for training and evaluation. |
| `--coreset_ratio` | `float`| No | `0.01` | Subsampling ratio for representative patch memory bank (0.01 = 1%). |
| `--output_dir` | `str` | No | `./checkpoints` | Directory to save checkpoints, exported ONNX models, and metrics report. |
| `--export_onnx` | `flag`| No | `False` | When specified, exports the trained model to ONNX format. |

---

## 📊 Outputs & Checkpoints

After training completes, outputs are organized as follows:

```text
checkpoints/
├── <category_name>/
│   ├── weights/lightning/model.ckpt  # PyTorch model weights & memory bank
│   └── onnx/model.onnx               # Exported ONNX graph (if --export_onnx is set)
└── training_summary.csv              # Aggregated evaluation metrics for all trained categories
```
