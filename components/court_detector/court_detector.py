"""
Training script for multi-point court detection as tiny boxes.

Each court point from SMALL_COURT_POINTS (with type id) is turned into a small
bounding box; types that are symmetric share the same class id. Model is YOLOv8
detector (yolov8n) trained on this custom detection dataset.
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from court_constants import SMALL_COURT_POINTS
from model import CourtDetectionTrainer


def train_yolo(data_yaml: Path, epochs: int = 20, batch: int = 16, imgsz: int = 640):
    model = YOLO("yolo11s.pt")
    model.train(
        trainer=CourtDetectionTrainer,
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=0 if torch.cuda.is_available() else "cpu",
        project="court_detector",
        name="court_keypoints_detector",
        cache=True,
        exist_ok=True,
        cos_lr=True,
        lr0=0.01,
        lrf=0.001,
        degrees=5,
        bgr=0.2,
    )
    return model


def main():
    data_yaml = Path(__file__).parent / "yolo_combined_data" / "data.yaml"
    train_yolo(data_yaml)


if __name__ == "__main__":
    main()
