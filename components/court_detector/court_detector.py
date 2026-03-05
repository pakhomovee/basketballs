import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from court_detector.court_constants import (
    SMALL_COURT_POINTS,
    FIBA_COURT_POINTS,
    NBA_COURT_POINTS,
    COURT_TYPE_TO_COURT_POINTS,
)
from court_detector.trainer import CourtDetectionTrainer
from court_detector.prepare_dataset import prepare_dataset
from common.classes.player import PlayersDetections
from common.logger import get_logger

from typing import Optional
from collections import defaultdict

from common.classes import CourtType


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
        model_path: str | Path = Path(__file__).parent.parent.parent / "models" / "court_detection_model.pt",
        conf=0.25,
    ):
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.conf = conf

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
        save_dir = Path(__file__).parent / "runs" / project / model_name
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
            save_dir=str(save_dir),
        )
        best_model_path = Path(__file__).parent / "runs" / "detect" / project / model_name / "weights" / "best.pt"
        self.model = YOLO(best_model_path)
        if save:
            self.model.save(self.model_path)

    def set_prediction_confedence(self, new_conf):
        self.conf = new_conf

    def predict_keypoints(self, frame_rgb) -> (np.ndarray, np.ndarray, np.ndarray):
        preds = self.model.predict(frame_rgb, verbose=False, conf=self.conf)[0]
        if preds.boxes is None:
            return np.empty((0, 3))
        pred_confs = preds.boxes.conf.cpu().numpy()
        pred_boxes = preds.boxes.xywh.cpu().numpy()
        pred_cls = preds.boxes.cls.cpu().numpy().astype(int)
        pred_centers = pred_boxes[:, :2]
        assert pred_centers.shape[0] == pred_cls.shape[0]
        return pred_centers, pred_cls, pred_confs

    def predict_court_homography(
        self,
        frame_rgb: np.ndarray,
        court_type: CourtType = CourtType.NBA,
    ) -> (np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]):
        """
        Estimates homography from frame to court.
        """
        court_points = COURT_TYPE_TO_COURT_POINTS[court_type]
        pred_centers, pred_cls, pred_confs = self.predict_keypoints(frame_rgb)
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
        try:
            H, mask = cv2.findHomography(frame_points, court_points)
        except cv2.error:
            return pred_centers, pred_cls, pred_confs, None
        return pred_centers, pred_cls, pred_confs, H

    def homographies_dist(self, H1, H2, width, height):
        """
        Estimates distance between two homographies.
        """
        H1 = H1.copy()
        H2 = H2.copy()
        H1[:, 0] *= width
        H1[:, 1] *= height
        H2[:, 0] *= width
        H2[:, 1] *= height
        D = np.linalg.inv(H1) @ H2
        _, S, _ = np.linalg.svd(D)
        return np.linalg.cond(D)

    def remove_bad_homographies(self, homographies, width, height, alpha=5, max_skip=5):
        not_none = []
        for i, H in enumerate(homographies):
            if H is not None:
                not_none.append(i)
        n = len(not_none)

        def get_cost(i, j):
            base_cost = alpha * (abs(i - j) - 1)
            if i == 0 or i == n + 1 or j == 0 or j == n + 1:
                return base_cost
            pos_i = not_none[i - 1]
            pos_j = not_none[j - 1]
            # diff = abs(pos_i - pos_j)
            hdist = self.homographies_dist(homographies[pos_i], homographies[pos_j], width, height)
            return base_cost + hdist

        inf = 1e18
        dp = [(inf, -1) for i in range(n + 2)]

        dp[0] = (0, -1)
        for i in range(1, n + 2):
            for j in range(max(0, i - max_skip), i):
                cost = dp[j][0]
                cost += get_cost(j, i)
                dp[i] = min(dp[i], (cost, j))
        remaning = []
        cur = n + 1
        while cur > 0:
            if cur <= n:
                remaning.append(not_none[cur - 1])
            cur = dp[cur][1]
        remaining = list(reversed(remaning))
        new_homographies = [None for i in range(len(homographies))]
        for i in remaining:
            new_homographies[i] = homographies[i]
        # num_removed = len(not_none) - len(remaning)
        # print("Removed: ", num_removed)
        return new_homographies

    def run(self, video_path: str, detections: PlayersDetections) -> None:
        """
        Process video frame-by-frame, compute homography, and enrich each
        :class:`Player` with ``court_position`` (x_m, y_m in meters).
        """

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        homographies = []
        frame_id = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            height, width, _ = frame_bgr.shape
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pred_centers, pred_cls, pred_confs, H = self.predict_court_homography(frame_rgb)
            homographies.append(H)
            frame_id += 1

        cap.release()

        new_homographies = self.remove_bad_homographies(homographies, width, height)

        for frame_id, H in enumerate(new_homographies):
            if H is None:
                continue
            for player in detections.get(frame_id, []):
                x1, y1, x2, y2 = player.bbox
                cx = (x1 + x2) / 2.0
                cy = float(y2)  # bottom middle of bbox
                pts = project_homography(np.array([[cx, cy]]), H)
                if pts.size >= 2:
                    player.court_position = (float(pts[0, 0]), float(pts[0, 1]))


def main():
    detector = CourtDetector()
    detector.train(epochs=5, lr0=0.01, lrf=0.001, need_prepare_dataset=True)


if __name__ == "__main__":
    main()
