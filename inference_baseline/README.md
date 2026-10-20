# Patchcore Inference Baseline

User-friendly, high-performance inference engine for **Patchcore Anomaly Detection & Defect Localization**.

---

## 📌 Features

- **Intuitive Python API**: Import `PatchcorePredictor` with 3 lines of code.
- **CLI & Batch Support**: Seamlessly inspect a single image or an entire folder of thousands of inspection frames.
- **CPU & GPU Ready**: Real-time evaluation on pure CPU (~25–38 FPS) or high-throughput GPU.
- **Defect Heatmap Rendering**: Automatically generates colorful JET colormap defect overlays with anomaly scores and diagnostic bounding banners.
- **Automated Checkpoint Discovery**: Finds existing trained checkpoints in the project structure automatically.

---

## 📂 Directory Structure

```text
inference_baseline/
├── predictor.py    # PatchcorePredictor class (Object-Oriented API)
├── inference.py    # Command-Line Interface (CLI) for single & batch inference
└── README.md       # Inference documentation and code snippets
```

---

## 🚀 Quickstart & Usage

### 1. Command-Line Interface (CLI)

#### A. Predict a single image on CPU (Default)
```bash
python inference_baseline/inference.py --dataset mvtec --category bottle --image "D:/apps/Datasets/MVTec AD/bottle/test/broken_large/000.png"
```

#### B. Predict using a specific checkpoint on GPU
```bash
python inference_baseline/inference.py --checkpoint "results/mvtec/bottle/Patchcore/MVTecAD/bottle/v1/weights/lightning/model.ckpt" --image "path/to/test.png" --device cuda
```

#### C. Batch evaluate an entire folder of inspection images
```bash
python inference_baseline/inference.py --dataset visa --category candle --image_dir "D:/Study/University/H261/Đa ngành/VisA_20220922/candle/Data/Images" --output_dir "./candle_predictions"
```

---

### 2. Python API (Embed in Your Own Application)

```python
from pathlib import Path
from inference_baseline.predictor import PatchcorePredictor

# 1. Initialize predictor with trained checkpoint
predictor = PatchcorePredictor(
    checkpoint_path="results/mvtec/bottle/Patchcore/MVTecAD/bottle/v1/weights/lightning/model.ckpt",
    device="cpu",  # or "cuda"
    threshold=0.5
)

# 2. Predict on an image (path, PIL Image, or NumPy array)
result = predictor.predict("sample_image.png")
print("Is Defective :", result["is_anomaly"])
print("Anomaly Score:", result["anomaly_score"])
print("Latency (ms) :", result["latency_ms"])
print("Inference FPS:", result["fps"])

# 3. Predict & export visual anomaly heatmap overlay
predictor.visualize_and_save(
    image_input="sample_image.png",
    output_path="defect_heatmap.png",
    alpha=0.4
)
```

---

## ⚙️ CLI Options Reference

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--checkpoint` | `str` | `None` | Path to PyTorch Lightning checkpoint (`.ckpt`). If omitted, auto-discovers. |
| `--dataset` | `str` | `mvtec` | Dataset name used for automatic checkpoint discovery. |
| `--category` | `str` | `bottle` | Category name used for automatic checkpoint discovery. |
| `--image` | `str` | `None` | Path to a single image file for immediate inference. |
| `--image_dir` | `str` | `None` | Directory path containing images for batch inference. |
| `--output_dir`| `str` | `./inference_outputs` | Output folder to store heatmaps and batch result CSV. |
| `--device` | `str` | `cpu` | Inference hardware device: `cpu` or `cuda`. |
| `--threshold` | `float`| `0.5` | Threshold above which an image is flagged as anomalous. |
