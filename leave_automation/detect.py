#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    default_weights = Path("runs/detect/train/weights/best.pt")

    parser = argparse.ArgumentParser(
        description="Run YOLO detection on an image, folder, video file, or webcam source."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=default_weights,
        help="Path to model weights (.pt).",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Input source: file, folder, URL, or webcam index (e.g. 0).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device (cpu, mps, cuda:0, etc.).",
    )
    parser.add_argument("--project", default="runs/detect", help="Output project folder.")
    parser.add_argument("--name", default="predict", help="Output run name.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live predictions window when supported.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detections as YOLO text labels.",
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Save confidence in txt labels (requires --save-txt).",
    )
    return parser.parse_args()


def normalize_source(source: str) -> str | int:
    # Treat numeric values as webcam indexes (e.g. "0" -> 0).
    return int(source) if source.isdigit() else source


def main() -> None:
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {args.weights}. Train first or pass --weights path/to/model.pt"
        )

    model = YOLO(str(args.weights))
    results = model.predict(
        source=normalize_source(args.source),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        show=args.show,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )

    if results:
        print(f"Saved predictions to: {results[0].save_dir}")


if __name__ == "__main__":
    main()
