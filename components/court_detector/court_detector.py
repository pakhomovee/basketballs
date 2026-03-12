import json
import shutil
from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from court_detector.court_constants import CourtConstants
from court_detector.trainer import CourtDetectionTrainer
from court_detector.prepare_dataset import prepare_dataset
from common.classes.player import PlayersDetections
from common.logger import get_logger

from typing import Any, Optional
from collections import defaultdict

from common.classes import CourtType
from tqdm import tqdm


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
        court_constants: CourtConstants,
    ) -> (np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]):
        """
        Estimates homography from frame to court.
        """
        court_points = court_constants.court_points
        frame_h, frame_w = frame_rgb.shape[:2]
        frame_size = np.array([frame_w, frame_h])
        court_size = np.array(court_constants.court_size)
        pred_centers, pred_cls, pred_confs = self.predict_keypoints(frame_rgb)
        frame_points = []
        court_points = []
        for frame_point, point_cls in zip(pred_centers, pred_cls):
            if point_cls not in court_constants.cls_to_points:
                continue
            court_point = court_constants.cls_to_points[point_cls][0]
            frame_points.append(np.array(frame_point) / frame_size)
            court_points.append(np.array(court_point) / court_size)
        frame_points = np.array(frame_points)
        court_points = np.array(court_points)
        try:
            H, mask = cv2.findHomography(frame_points, court_points, method=cv2.RANSAC)
        except cv2.error:
            return pred_centers, pred_cls, pred_confs, None
        return pred_centers, pred_cls, pred_confs, H

    def cosine_similarity(self, v1, v2, eps=1e-9):
        lib = torch
        if isinstance(v1, np.ndarray):
            lib = np
        return lib.dot(v1, v2) / (lib.linalg.norm(v1) * lib.linalg.norm(v2) + eps)

    def homographies_dist2(self, H1, H2):
        """
        Estimates distance between two homographies.
        """
        vecs = []
        for x in [0, 0.5, 1]:
            for y in [0, 0.5, 1]:
                vecs.append((x, y, 1))
        lib = torch
        if isinstance(H1, np.ndarray):
            lib = np
        D = H2 @ lib.linalg.inv(H1)
        Dinv = H1 @ lib.linalg.inv(H2)
        res = 0
        for v in vecs:
            if isinstance(H1, np.ndarray):
                v = np.array(v, dtype=H1.dtype)
            else:
                v = torch.tensor(v, dtype=H1.dtype, device=H1.device)
            for H in [D, Dinv]:
                Hv = H @ v
                res = res + 1 - self.cosine_similarity(Hv, v)
        return res

    def homographies_dist(self, H1, H2):
        """
        Estimates distance between two homographies.
        """
        vecs = []
        for x in [0, 0.5, -0.5]:
            for y in [0, 0.5, -0.5]:
                vecs.append((x, y, 1))
        lib = torch
        if isinstance(H1, np.ndarray):
            lib = np
        D = lib.linalg.inv(H1) @ H2
        Dinv = lib.linalg.inv(H2) @ H1
        res = 0
        for v in vecs:
            if isinstance(H1, np.ndarray):
                v = np.array(v, dtype=H1.dtype)
            else:
                v = torch.tensor(v, dtype=H1.dtype, device=H1.device)
            for H in [D, Dinv]:
                Hv = H @ v
                res = res + 1 - self.cosine_similarity(Hv, v)
        return res + self.homographies_dist2(H1, H2)

    def remove_bad_homographies(self, homographies, alpha=0.1, max_skip=10):
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
            hdist = self.homographies_dist(homographies[pos_i], homographies[pos_j])
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
        num_removed = len(not_none) - len(remaning)
        print(f"Removed {num_removed} homographies")
        return new_homographies

    def smooth_homographies(
        self,
        homographies,
        keypoints_detections,
        frames_sizes,
        court_constants,
        alpha=1,
        dtype=torch.float32,
        device="cpu",
    ):
        if len(homographies) == 0:
            return homographies, None
        homographies = deepcopy(homographies)
        for i in range(len(homographies) - 1):
            if homographies[i + 1] is None:
                homographies[i + 1] = homographies[i]
        for i in range(len(homographies) - 1, 0, -1):
            if homographies[i - 1] is None:
                homographies[i - 1] = homographies[i]
        if homographies[0] is None:
            return homographies, None
        for i, H in enumerate(homographies):
            norm = np.sqrt((H * H).sum())
            H /= norm
            homographies[i] = torch.tensor(H, dtype=dtype, device=device, requires_grad=True)

        # Pre-compute per-frame keypoint tensors on device
        frame_kp_data = []
        for i in range(len(homographies)):
            centers, classes = keypoints_detections[i]
            kp_centers = []
            kp_needed = []
            for kc, kcls in zip(centers, classes):
                if kcls not in court_constants.cls_to_points:
                    continue
                cp = court_constants.cls_to_points[kcls][0]
                kp_centers.append(
                    torch.tensor(
                        [kc[0] / frames_sizes[i][0], kc[1] / frames_sizes[i][1], 1.0],
                        dtype=dtype,
                        device=device,
                    )
                )
                kp_needed.append(
                    torch.tensor(
                        [cp[0] / court_constants.court_size[0], cp[1] / court_constants.court_size[1], 1.0],
                        dtype=dtype,
                        device=device,
                    )
                )
            frame_kp_data.append((kp_centers, kp_needed))

        def calc_ith_loss(i):
            loss = torch.tensor(0, dtype=dtype, device=device)
            if i < len(homographies) - 1:
                loss += self.homographies_dist(homographies[i], homographies[i + 1]) * alpha
            kp_centers, kp_needed = frame_kp_data[i]
            for kc, kn in zip(kp_centers, kp_needed):
                projected = homographies[i] @ kc
                loss += 1 - self.cosine_similarity(projected, kn)
            return loss

        def calc_loss():
            loss = torch.tensor(0, dtype=dtype, device=device)
            for i in range(len(homographies)):
                loss += calc_ith_loss(i)
            return loss

        # print(f"Initial loss:{calc_loss():.3f}")
        num_epochs = 30
        optimizer = torch.optim.AdamW(homographies, lr=0.005)
        batch_size = 32
        print("Smoothing homographies...")
        for i in tqdm(range(num_epochs)):
            frames_ids = list(range(len(homographies)))
            np.random.shuffle(frames_ids)
            full_loss = 0
            for start in range(0, len(frames_ids), batch_size):
                optimizer.zero_grad()
                batch_frames_ids = frames_ids[start : start + batch_size]
                tot_loss = torch.tensor(0, dtype=dtype, device=device)
                for j in batch_frames_ids:
                    tot_loss = tot_loss + calc_ith_loss(j)
                tot_loss.backward()
                optimizer.step()
                full_loss = full_loss + tot_loss.item()
            with torch.no_grad():
                for H in homographies:
                    H /= torch.sqrt((H * H).sum())
            # print(f"Loss after {i} iterations:{full_loss:.3f}")
        with torch.no_grad():
            losses = [calc_ith_loss(i).item() for i in range(len(homographies))]
        for i, H in enumerate(homographies):
            homographies[i] = H.detach().cpu().numpy()
        return homographies, losses

    def extract_homographies_from_video(self, video_path: str, court_constants: CourtConstants):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        homographies = []
        frames_sizes = []
        keypoints_detections = []
        frame_id = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            height, width, _ = frame_bgr.shape
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pred_centers, pred_cls, pred_confs, H = self.predict_court_homography(frame_rgb, court_constants)
            keypoints_detections.append((pred_centers, pred_cls))
            homographies.append(H)
            frames_sizes.append((width, height))
            frame_id += 1

        cap.release()

        homographies = self.remove_bad_homographies(homographies)
        new_homographies, losses = self.smooth_homographies(
            homographies, keypoints_detections, frames_sizes, court_constants
        )

        return new_homographies, frames_sizes, keypoints_detections, losses

    def run(self, video_path: str, detections: PlayersDetections, court_type: CourtType = CourtType.NBA) -> None:
        """
        Process video frame-by-frame, compute homography, and enrich each
        :class:`Player` with ``court_position`` (x_m, y_m in meters).
        """

        court_constants = CourtConstants(court_type)
        homographies, frames_sizes, keypoint_detections, losses = self.extract_homographies_from_video(
            video_path, court_constants
        )

        court_w, court_h = court_constants.court_size
        for frame_id, H in enumerate(homographies):
            if H is None:
                continue
            # if frame_id + 1 < len(homographies) and homographies[frame_id + 1] is not None:
            #     H2 = homographies[frame_id + 1]
            #     dst = self.homographies_dist(H, H2)
            #     get_logger().log(frame_id, f"Homo dist: {dst}")
            frame_w, frame_h = frames_sizes[frame_id]
            for player in detections.get(frame_id, []):
                x1, y1, x2, y2 = player.bbox
                cx = (x1 + x2) / 2.0 / frame_w
                cy = float(y2) / frame_h
                pts = project_homography(np.array([[cx, cy]]), H)
                if pts.size >= 2:
                    player.court_position = (float(pts[0, 0]) * court_w, float(pts[0, 1] * court_h))


def main():
    detector = CourtDetector()
    detector.train(epochs=5, lr0=0.01, lrf=0.001, need_prepare_dataset=True)


if __name__ == "__main__":
    main()
