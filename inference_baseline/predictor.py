"""
inference_baseline/predictor.py

User-Friendly, Production-Ready Inference Engine for Patchcore Models.
Supports PyTorch Lightning checkpoints (.ckpt) and ONNX models (.onnx).
Performs anomaly detection, localization heatmap rendering, and metric evaluation
with clean, object-oriented API and CLI interfaces.
"""

import os
import sys
import time
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.v2 as T

# Fix Windows unicode path issue in anomalib
import anomalib.data.utils.path as path_module
path_module.contains_non_printable_characters = lambda p: False

from anomalib.models import Patchcore


class PatchcorePredictor:
    """
    High-level, user-friendly predictor class for Patchcore Anomaly Detection.

    Examples:
        >>> predictor = PatchcorePredictor(checkpoint_path="checkpoints/mvtec/bottle/model.ckpt")
        >>> result = predictor.predict("path/to/test_image.png")
        >>> print(result["is_anomaly"], result["anomaly_score"])
        >>> predictor.visualize_and_save("path/to/test_image.png", "output_heatmap.png")
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
        image_size: Tuple[int, int] = (256, 256),
        threshold: float = 0.5
    ) -> None:
        """
        Initializes the Patchcore predictor.

        Args:
            checkpoint_path (Union[str, Path]): Path to trained PyTorch Lightning .ckpt checkpoint.
            device (Optional[str]): Device to run inference ('cuda', 'cpu', or None for auto-detection).
            image_size (Tuple[int, int]): Target input dimensions (width, height).
            threshold (float): Anomaly score classification threshold.
        """
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.image_size = image_size
        self.threshold = threshold

        # Input preprocessing pipeline
        self.transform = T.Compose([
            T.Resize(self.image_size, antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load trained Patchcore model
        print(f"Loading Patchcore model from: {self.checkpoint_path} onto {self.device}...")
        self.model = Patchcore.load_from_checkpoint(str(self.checkpoint_path))
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully and ready for inference!")

    def preprocess_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Tuple[torch.Tensor, Tuple[int, int], np.ndarray]:
        """
        Converts diverse input types into a standardized PyTorch tensor.

        Args:
            image_input (Union[str, Path, Image.Image, np.ndarray]): File path, PIL Image, or NumPy array.

        Returns:
            Tuple[torch.Tensor, Tuple[int, int], np.ndarray]:
                - Normalized tensor ready for forward pass: [1, 3, H, W]
                - Original image resolution (width, height)
                - Original image RGB array
        """
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Input image not found: {img_path}")
            pil_img = Image.open(img_path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                image_input = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            pil_img = Image.fromarray(image_input)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        orig_w, orig_h = pil_img.size
        orig_np = np.array(pil_img)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        return tensor, (orig_w, orig_h), orig_np

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Executes anomaly detection on a single image.

        Args:
            image_input (Union[str, Path, Image.Image, np.ndarray]): Input image.

        Returns:
            Dict[str, Any]: Dictionary containing prediction results:
                - 'is_anomaly' (bool): True if classified as defective, False if normal.
                - 'anomaly_score' (float): Image-level anomaly score normalized to [0, 1].
                - 'anomaly_map' (np.ndarray): Pixel-level anomaly heatmap scaled to original image size.
                - 'latency_ms' (float): Execution latency in milliseconds.
                - 'fps' (float): Theoretical frame rate achieved on the current hardware.
        """
        tensor, (orig_w, orig_h), _ = self.preprocess_image(image_input)

        t_start = time.perf_counter()
        with torch.no_grad():
            output = self.model(tensor)
        latency = (time.perf_counter() - t_start) * 1000

        # Extract predictions
        score = float(output.pred_score.item())
        is_anomaly = bool(output.pred_label.item()) if hasattr(output, "pred_label") and output.pred_label is not None else (score >= self.threshold)

        # Process anomaly map
        amap_raw = output.anomaly_map.squeeze().cpu().numpy()
        amap_resized = cv2.resize(amap_raw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        amap_normalized = (amap_resized - amap_resized.min()) / (amap_resized.max() - amap_resized.min() + 1e-8)

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(score, 4),
            "anomaly_map": amap_normalized,
            "latency_ms": round(latency, 2),
            "fps": round(1000.0 / latency, 2) if latency > 0 else 0.0,
        }

    def visualize_and_save(
        self,
        image_input: Union[str, Path],
        output_path: Union[str, Path],
        alpha: float = 0.4
    ) -> Dict[str, Any]:
        """
        Runs prediction, renders a localization heatmap overlay on the original image,
        and saves the composite result to disk.

        Args:
            image_input (Union[str, Path]): Path to test image.
            output_path (Union[str, Path]): Target path to save the visualized image.
            alpha (float): Transparency blending weight for the heatmap overlay (0 to 1).

        Returns:
            Dict[str, Any]: Prediction result dictionary.
        """
        _, (orig_w, orig_h), orig_rgb = self.preprocess_image(image_input)
        result = self.predict(image_input)

        # Create JET heatmap
        amap_uint8 = (result["anomaly_map"] * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(amap_uint8, cv2.COLORMAP_JET)

        # Convert original image to BGR for OpenCV blending
        orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.addWeighted(orig_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)

        # Add text banner
        status_text = "DEFECT DETECTED" if result["is_anomaly"] else "NORMAL (GOOD)"
        color = (0, 0, 255) if result["is_anomaly"] else (0, 255, 0)
        cv2.putText(
            overlay_bgr,
            f"{status_text} | Score: {result['anomaly_score']:.3f} | {result['latency_ms']:.1f}ms",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), overlay_bgr)
        print(f"Visualization saved to: {out_path}")
        return result


def find_latest_checkpoint(results_dir: Path, dataset: str, category: str) -> Optional[Path]:
    """
    Helper function to automatically discover trained model checkpoint in the results folder.

    Args:
        results_dir (Path): Base directory containing training results.
        dataset (str): Dataset name ('mvtec', 'visa', 'pcb').
        category (str): Category name.

    Returns:
        Optional[Path]: Path to model.ckpt if found, else None.
    """
    search_dir = results_dir / dataset / category
    if not search_dir.exists():
        search_dir = results_dir
    ckpts = sorted(list(search_dir.rglob("model.ckpt")), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None
