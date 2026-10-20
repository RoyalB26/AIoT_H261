# Patchcore Training Baseline

Modular training pipeline for **Patchcore Anomaly Detection & Localization** on industrial visual quality inspection datasets (**MVTec AD**, **VisA**, and **PCB Defect Dataset**).

---

## 📌 Features

- **Multi-Dataset Support**: Seamlessly trains on MVTec AD (15 categories), VisA (12 categories), and PCB Defect Dataset (6 defect types).
- **Fast Representative Coreset**: Employs an optimized GPU/CPU memory bank builder preventing Out-of-Memory and execution bottlenecks.
- **Windows & Unicode Path Safe**: Built-in path sanitizer supporting Vietnamese and Unicode workspace directories.
- **Automatic Exporting**: Saves PyTorch Lightning checkpoints (`.ckpt`) and exports optimized **ONNX** graph representations (`.onnx`).
- **Comprehensive Logging**: Auto-logs training duration, testing latency, Image AUROC, Pixel AUROC, and F1-Scores to CSV.

---

## 📂 Directory Structure

```text
train_baseline/
├── train.py        # Main training script with CLI options
└── README.md       # Baseline documentation and usage guide
```

---

## 🚀 Quickstart & Usage

### 1. Train a Specific Category (e.g. `bottle` in MVTec AD)
```bash
python train_baseline/train.py --dataset mvtec --category bottle --backbone resnet18 --export_onnx
```

### 2. Train an Entire Dataset (e.g. all 6 PCB Defect categories)
```bash
python train_baseline/train.py --dataset pcb --output_dir ./checkpoints
```

### 3. Train all 33 Categories across all Datasets
```bash
python train_baseline/train.py --dataset all --batch_size 32 --coreset_ratio 0.01
```

---

## ⚙️ Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--dataset` | `str` | `mvtec` | Dataset name: `mvtec`, `visa`, `pcb`, or `all`. |
| `--category` | `str` | `None` | Specific category. If omitted, trains all categories in the dataset. |
| `--data_root` | `str` | `None` | Path to dataset root directory (uses default project path if omitted). |
| `--backbone` | `str` | `resnet18` | Pretrained feature extractor backbone (e.g. `resnet18`, `wide_resnet50_2`). |
| `--batch_size` | `int` | `32` | DataLoader batch size for training and testing. |
| `--coreset_ratio` | `float`| `0.01` | Sampling ratio for representative patch memory bank subsampling. |
| `--output_dir` | `str` | `./checkpoints` | Target directory for saving model checkpoints and export files. |
| `--export_onnx` | `flag`| `False` | When specified, exports the fitted model to ONNX format. |

---

## 📊 Outputs & Artifacts

After training, models and logs are organized as:
```text
checkpoints/
├── mvtec/
│   └── bottle/
│       ├── weights/lightning/model.ckpt  # PyTorch model weights & memory bank
│       └── onnx/model.onnx               # Exported ONNX graph
└── training_summary.csv                  # Aggregated metric results
```
