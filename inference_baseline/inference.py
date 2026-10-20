"""
inference_baseline/inference.py

Command-line and Batch Inference CLI for Patchcore Anomaly Detection.
Supports single image prediction, entire folder batch processing,
CPU/GPU device switching, and automated defect heatmap overlay export.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from predictor import PatchcorePredictor, find_latest_checkpoint


def run_cli_inference():
    """
    Parses command-line arguments and coordinates single or batch inference.
    """
    parser = argparse.ArgumentParser(description="Patchcore Anomaly Detection Inference Tool")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint (.ckpt). If omitted, searches in --results_dir.")
    parser.add_argument("--dataset", type=str, default="mvtec", help="Dataset name to help locate checkpoint.")
    parser.add_argument("--category", type=str, default="bottle", help="Category name to help locate checkpoint.")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory where models were trained.")
    parser.add_argument("--image", type=str, default=None, help="Path to a single test image.")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images for batch inference.")
    parser.add_argument("--output_dir", type=str, default="./inference_outputs",
                        help="Output directory to save predictions, heatmaps, and summary CSV.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Execution device ('cpu' or 'cuda'). Defaults to 'cpu' for production compatibility.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly classification decision threshold.")
    args = parser.parse_args()

    # Locate checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_latest_checkpoint(Path(args.results_dir), args.dataset, args.category)

    if not ckpt_path or not ckpt_path.exists():
        print(f"Error: Could not locate checkpoint for {args.dataset}/{args.category}.")
        print("Please specify a valid path using `--checkpoint /path/to/model.ckpt`.")
        sys.exit(1)

    # Initialize Predictor
    predictor = PatchcorePredictor(
        checkpoint_path=ckpt_path,
        device=args.device,
        threshold=args.threshold,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mode 1: Single image inference
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Error: Target image does not exist: {img_path}")
            sys.exit(1)

        vis_path = out_dir / f"{img_path.stem}_heatmap.png"
        res = predictor.visualize_and_save(img_path, vis_path)

        print("\n" + "=" * 50)
        print(f"Prediction for: {img_path.name}")
        print(f"Result        : {'DEFECTIVE [ANOMALY]' if res['is_anomaly'] else 'NORMAL [GOOD]'}")
        print(f"Anomaly Score : {res['anomaly_score']:.4f} (Threshold: {args.threshold})")
        print(f"Device        : {args.device.upper()}")
        print(f"Latency       : {res['latency_ms']:.2f} ms")
        print(f"FPS           : {res['fps']:.2f}")
        print(f"Output Heatmap: {vis_path}")
        print("=" * 50)
        return

    # Mode 2: Batch inference on directory
    target_dir = Path(args.image_dir) if args.image_dir else None
    if not target_dir:
        # Default sample image if nothing specified
        sample_img = Path(r"D:\apps\Datasets\MVTec AD\bottle\test\broken_large\000.png")
        if sample_img.exists():
            print(f"No image or image_dir specified. Running demonstration on sample: {sample_img}")
            vis_path = out_dir / f"{sample_img.stem}_heatmap.png"
            res = predictor.visualize_and_save(sample_img, vis_path)
            print(f"Sample prediction complete! Result: {res['is_anomaly']}, Score: {res['anomaly_score']:.4f}")
            return
        else:
            print("Please specify an image with `--image` or a folder with `--image_dir`.")
            sys.exit(1)

    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    image_files = [f for f in target_dir.rglob("*.*") if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {target_dir}")
        sys.exit(1)

    print(f"\nRunning batch inference on {len(image_files)} images from: {target_dir}")
    batch_records = []

    for img_p in tqdm(image_files, desc="Inferring images"):
        vis_path = out_dir / f"{img_p.stem}_heatmap.png"
        res = predictor.visualize_and_save(img_p, vis_path)
        batch_records.append({
            "Image": img_p.name,
            "Path": str(img_p),
            "Prediction": "Defective" if res["is_anomaly"] else "Normal",
            "Score": res["anomaly_score"],
            "Latency (ms)": res["latency_ms"],
            "FPS": res["fps"],
            "Heatmap_Path": str(vis_path),
        })

    summary_csv = out_dir / "batch_inference_results.csv"
    pd.DataFrame(batch_records).to_csv(summary_csv, index=False)

    avg_latency = sum(r["Latency (ms)"] for r in batch_records) / len(batch_records)
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    print("\n" + "=" * 60)
    print(f"Batch inference complete for {len(image_files)} images!")
    print(f"Average CPU/GPU Latency : {avg_latency:.2f} ms")
    print(f"Average CPU/GPU FPS     : {avg_fps:.2f} FPS")
    print(f"Results CSV saved to    : {summary_csv}")
    print("=" * 60)


if __name__ == "__main__":
    run_cli_inference()
