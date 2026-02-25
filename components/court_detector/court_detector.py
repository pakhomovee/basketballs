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
from ultralytics import YOLO

from court_constants import SMALL_COURT_POINTS, FIBA_COURT_POINTS, NBA_COURT_POINTS
from trainer import CourtDetectionTrainer
from prepare_dataset import prepare_dataset

from typing import Optional
from collections import defaultdict


def project_homography(points_xy, H):
    """
    This function takes np.array of points and homography matrix H.
    It returns points after applying the homography.
    """
    p = np.vstack([points_xy.T, np.ones(len(points_xy))])
    transformed = H @ p
    return (transformed[:2] / transformed[2]).T


class CourtDetector:
    def __init__(
        self,
        model_path: str | Path = Path(__file__).parent.parent.parent
        / "models"
        / "court_detection_model.pt",
    ):
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.conf = 0.3

    def train(
        self,
        dataset_path: Path = Path(__file__).parent / "yolo_combined_data",
        need_prepare_dataset: bool = False,
        epochs: int = 30,
        batch: int = 16,
        imgsz: int = 640,
        save=True,
        lr0=0.01,
        lrf=0.001,
    ):
        if need_prepare_dataset:
            prepare_dataset(out_root=dataset_path)
        project = "court_detector"
        model_name = "court_keypoints_detector"
        self.model.train(
            trainer=CourtDetectionTrainer,
            data=str(Path(dataset_path) / "data.yaml"),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=0 if torch.cuda.is_available() else "cpu",
            project=project,
            name=model_name,
            cache=False,
            exist_ok=True,
            cos_lr=True,
            lr0=lr0,
            lrf=lrf,
            perspective=0.0007,
            bgr=0.5,
            workers=10,
            optimizer="sgd",
            close_mosaic=0,
            fliplr=0,
        )
        best_model_path = (
            Path(__file__).parent / "runs" / "detect" / project / model_name / "weights" / "best.pt"
        )
        self.model = YOLO(best_model_path)
        if save:
            self.model.save(self.model_path)

    def set_prediction_confedence(self, new_conf):
        self.conf = new_conf

    def predict_keypoints(self, frame_rgb) -> (np.ndarray, np.ndarray):
        preds = self.model.predict(frame_rgb, verbose=False, conf=self.conf)[0]
        if preds.boxes is None:
            return np.empty((0, 3))
        pred_boxes = preds.boxes.xywh.cpu().numpy()
        pred_cls = preds.boxes.cls.cpu().numpy().astype(int)
        pred_centers = pred_boxes[:, :2]
        assert pred_centers.shape[0] == pred_cls.shape[0]
        return pred_centers, pred_cls

    def predict_court_homography(
        self, frame_rgb: np.ndarray, court_points: list[tuple] = NBA_COURT_POINTS
    ) -> (np.ndarray, np.ndarray, Optional[np.ndarray]):
        pred_centers, pred_cls = self.predict_keypoints(frame_rgb)
        cls_to_points = defaultdict(list)
        for px, py, pcls in court_points:
            cls_to_points[pcls].append((px, py))
        frame_points = []
        court_points = []
        for frame_point, point_cls in zip(pred_centers, pred_cls):
            if point_cls not in cls_to_points:
                continue
            court_point = cls_to_points[point_cls][0]
            frame_points.append([frame_point])
            court_points.append([court_point])
        frame_points = np.array(frame_points)
        court_points = np.array(court_points)
        H, _ = cv2.findHomography(frame_points, court_points)
        return pred_centers, pred_cls, H


def main():
    detector = CourtDetector()
    detector.train(epochs=5, lr0=0.01, lrf=0.001, need_prepare_dataset=True)


if __name__ == "__main__":
    main()
