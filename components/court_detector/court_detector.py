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
from court_detector.homography import find_homography_ransac
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
        for x in [-0.5, 0, 0.5]:
            for y in [-0.5, 0, 0.5]:
                vecs.append((x, y, 1))
        lib = torch
        if isinstance(H1, np.ndarray):
            lib = np
        D = H2 @ lib.linalg.pinv(H1)
        Dinv = H1 @ lib.linalg.pinv(H2)
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
        for x in [0, 0.5, 1]:
            for y in [0, 0.5, 1]:
                vecs.append((x, y, 1))
        lib = torch
        if isinstance(H1, np.ndarray):
            lib = np
        D = lib.linalg.pinv(H1) @ H2
        Dinv = lib.linalg.pinv(H2) @ H1
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

    def extract_homographies_from_video_v2(
        self,
        video_path: str,
        court_constants: CourtConstants,
        alpha: float = 200.0,
        eps: float = 0.003,
        smooth_num_epochs: int = 300,
        smooth_lr: float = 0.001,
        device: Optional[str] = None,
        smooth_dtype=torch.float32,
    ):
        """
        Optical-flow + D&C greedy optimisation.

        1) Single pass: detect keypoints, compute inter-frame flow homographies.
        2) Optimise:  sum_i reward_i  -  alpha * sum_i smooth_cost_i
           where reward uses exp((cos_sim - 1) / eps) and smooth_cost is
           homographies_dist between H_{i+1} and H_i propagated through flow.
        3) Greedy D&C: at each level, take children answers, then try a single
           level-wide RANSAC; keep whichever gives a better objective.
        """
        # choose device once and use it for both D&C and smoothing
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frames_sizes: list[tuple[int, int]] = []
        keypoints_detections: list[tuple[np.ndarray, np.ndarray]] = []
        forward_H_norm: list[Optional[np.ndarray]] = []

        court_size = np.array(court_constants.court_size, dtype=np.float64)

        per_frame_src_h: list[np.ndarray] = []
        per_frame_dst_h: list[np.ndarray] = []

        prev_gray: Optional[np.ndarray] = None
        prev_size: Optional[tuple[int, int]] = None
        frame_idx = 0

        print("Reading frames, detecting keypoints and computing optical flow...")
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            h, w, _ = frame_bgr.shape
            frame_size = np.array([w, h], dtype=np.float64)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            pred_centers, pred_cls, pred_confs = self.predict_keypoints(frame_rgb)
            keypoints_detections.append((pred_centers, pred_cls))
            frames_sizes.append((w, h))

            src_pts = []
            dst_pts = []
            for frame_pt, cls_id in zip(pred_centers, pred_cls):
                if cls_id not in court_constants.cls_to_points:
                    continue
                court_pt = court_constants.cls_to_points[cls_id][0]
                src_pts.append(np.array(frame_pt) / frame_size)
                dst_pts.append(np.array(court_pt) / court_size)
            src_arr = np.array(src_pts, dtype=np.float64).reshape(-1, 2)
            dst_arr = np.array(dst_pts, dtype=np.float64).reshape(-1, 2)
            if len(src_arr) > 0:
                ones = np.ones((len(src_arr), 1), dtype=np.float64)
                per_frame_src_h.append(np.hstack([src_arr, ones]))
                per_frame_dst_h.append(np.hstack([dst_arr, ones]))
            else:
                per_frame_src_h.append(np.empty((0, 3), dtype=np.float64))
                per_frame_dst_h.append(np.empty((0, 3), dtype=np.float64))

            if frame_idx > 0:
                H_flow = self._compute_flow_homography_norm(prev_gray, gray, prev_size, (w, h))
                forward_H_norm.append(H_flow)

            prev_gray = gray
            prev_size = (w, h)
            frame_idx += 1

        cap.release()

        n = len(frames_sizes)
        if n == 0:
            return [], [], [], None

        # ---- precompute chains and inverses --------------------------------
        H_norm_0_to: list[Optional[np.ndarray]] = [None] * n
        H_norm_0_to[0] = np.eye(3, dtype=np.float64)
        for i in range(1, n):
            if H_norm_0_to[i - 1] is None or forward_H_norm[i - 1] is None:
                H_norm_0_to[i] = H_norm_0_to[i - 1]
            else:
                H_norm_0_to[i] = forward_H_norm[i - 1] @ H_norm_0_to[i - 1]

        H_norm_i_to_0: list[Optional[np.ndarray]] = [None] * n
        for i in range(n):
            if H_norm_0_to[i] is not None:
                try:
                    H_norm_i_to_0[i] = np.linalg.pinv(H_norm_0_to[i])
                except np.linalg.LinAlgError:
                    pass

        forward_H_norm_inv: list[Optional[np.ndarray]] = [None] * len(forward_H_norm)
        for i in range(len(forward_H_norm)):
            if forward_H_norm[i] is not None:
                try:
                    forward_H_norm_inv[i] = np.linalg.pinv(forward_H_norm[i])
                except np.linalg.LinAlgError:
                    pass

        # ---- padded arrays for vectorised objective (NumPy) -----------------
        max_K = max((len(s) for s in per_frame_src_h), default=0)
        max_K = max(max_K, 1)
        src_pad = np.zeros((n, max_K, 3), dtype=np.float64)
        dst_pad = np.zeros((n, max_K, 3), dtype=np.float64)
        kp_mask = np.zeros((n, max_K), dtype=np.float64)
        for i in range(n):
            k = len(per_frame_src_h[i])
            if k > 0:
                src_pad[i, :k] = per_frame_src_h[i]
                dst_pad[i, :k] = per_frame_dst_h[i]
                kp_mask[i, :k] = 1.0

        dst_pad_T = dst_pad.transpose(0, 2, 1)  # (N, 3, K)
        dst_norm = np.linalg.norm(dst_pad_T, axis=1)  # (N, K)

        flow_inv_arr = np.zeros((max(n - 1, 1), 3, 3), dtype=np.float64)
        flow_mask_arr = np.zeros(max(n - 1, 1), dtype=np.float64)
        for i in range(n - 1):
            if forward_H_norm_inv[i] is not None:
                flow_inv_arr[i] = forward_H_norm_inv[i]
                flow_mask_arr[i] = 1.0
            else:
                flow_inv_arr[i] = np.eye(3, dtype=np.float64)

        vecs1_np = np.array(
            [[x, y, 1.0] for x in [0, 0.5, 1.0] for y in [0, 0.5, 1.0]],
            dtype=np.float64,
        ).T  # (3, 9)
        vecs2_np = np.array(
            [[x, y, 1.0] for x in [-0.5, 0, 0.5] for y in [-0.5, 0, 0.5]],
            dtype=np.float64,
        ).T  # (3, 9)
        v1_norm = np.linalg.norm(vecs1_np, axis=0)  # (9,)
        v2_norm = np.linalg.norm(vecs2_np, axis=0)  # (9,)

        # ---- vectorised NumPy helpers ---------------------------------------
        def _np_batch_cs(Mv, v_ref, v_norm):
            dot = (Mv * v_ref[None]).sum(axis=1)
            return dot / (np.linalg.norm(Mv, axis=1) * v_norm[None] + 1e-12)

        def _np_batch_hdist(H1, H2):
            H1i = np.linalg.pinv(H1)
            H2i = np.linalg.pinv(H2)
            D = H1i @ H2
            Di = H2i @ H1
            d1 = (1 - _np_batch_cs(D @ vecs1_np, vecs1_np, v1_norm)).sum(1) + (
                1 - _np_batch_cs(Di @ vecs1_np, vecs1_np, v1_norm)
            ).sum(1)
            D2 = H2 @ H1i
            D2i = H1 @ H2i
            d2 = (1 - _np_batch_cs(D2 @ vecs2_np, vecs2_np, v2_norm)).sum(1) + (
                1 - _np_batch_cs(D2i @ vecs2_np, vecs2_np, v2_norm)
            ).sum(1)
            return d1 + d2

        def _seg_rewards(Hs_list, left, right):
            H_arr = np.stack([H if H is not None else np.eye(3) for H in Hs_list])
            h_mask = np.array([float(H is not None) for H in Hs_list])
            proj = H_arr @ src_pad[left:right].transpose(0, 2, 1)  # (S, 3, K)
            dot = (proj * dst_pad_T[left:right]).sum(axis=1)  # (S, K)
            cs = dot / (np.linalg.norm(proj, axis=1) * dst_norm[left:right] + 1e-12)
            return (np.exp((cs - 1.0) / eps) * kp_mask[left:right]).sum(axis=1) * h_mask

        def _seg_smooth(Hs_list, left, right):
            if right - left <= 1:
                return np.empty(0)
            H_arr = np.stack([H if H is not None else np.eye(3) for H in Hs_list])
            h_mask = np.array([float(H is not None) for H in Hs_list])
            pair_mask = h_mask[:-1] * h_mask[1:] * flow_mask_arr[left : right - 1]
            predicted = H_arr[:-1] @ flow_inv_arr[left : right - 1]
            return _np_batch_hdist(H_arr[1:], predicted) * pair_mask

        def _seg_objective(Hs_list, left, right):
            return float(_seg_rewards(Hs_list, left, right).sum() - alpha * _seg_smooth(Hs_list, left, right).sum())

        def h_i_to_ref(i: int, ref: int) -> Optional[np.ndarray]:
            if H_norm_i_to_0[i] is None or H_norm_0_to[ref] is None:
                return None
            return H_norm_0_to[ref] @ H_norm_i_to_0[i]

        # ---- iterative bottom-up D&C (NumPy + RANSAC) -----------------------
        print("Running D&C optimisation...")
        homographies = [None] * n
        for i in range(n):
            if len(per_frame_src_h[i]) >= 4:
                H_i = find_homography_ransac(per_frame_src_h[i], per_frame_dst_h[i])
                if isinstance(H_i, torch.Tensor):
                    H_i = H_i.detach().cpu().numpy()
                homographies[i] = H_i

        seg_size = 2
        while seg_size <= 2 * n:
            for seg_start in range(0, n, seg_size):
                left = seg_start
                right = min(left + seg_size, n)
                if right - left <= 1:
                    continue

                children_obj = _seg_objective(homographies[left:right], left, right)

                all_src: list[np.ndarray] = []
                all_dst: list[np.ndarray] = []
                for i in range(left, right):
                    if len(per_frame_src_h[i]) == 0:
                        continue
                    T = h_i_to_ref(i, left)
                    if T is None:
                        continue
                    all_src.append((T @ per_frame_src_h[i].T).T)
                    all_dst.append(per_frame_dst_h[i])

                if not all_src:
                    continue
                src_combined = np.concatenate(all_src, axis=0)
                dst_combined = np.concatenate(all_dst, axis=0)
                if len(src_combined) < 4:
                    continue

                H_l = find_homography_ransac(src_combined, dst_combined)
                if H_l is None:
                    continue
                if isinstance(H_l, torch.Tensor):
                    H_l_np = H_l.detach().cpu().numpy()
                else:
                    H_l_np = H_l

                candidate_Hs: list[Optional[np.ndarray]] = []
                for i in range(left, right):
                    T = h_i_to_ref(i, left)
                    candidate_Hs.append(H_l_np @ T if T is not None else None)

                if _seg_objective(candidate_Hs, left, right) > children_obj:
                    homographies[left:right] = candidate_Hs

            seg_size *= 2

        homographies, losses = self.smooth_homographies_v2(
            homographies,
            keypoints_detections,
            frames_sizes,
            court_constants,
            forward_H_norm_inv,
            alpha=alpha,
            eps=eps,
            num_epochs=smooth_num_epochs,
            lr=smooth_lr,
            device=device,
            dtype=smooth_dtype,
        )

        return homographies, frames_sizes, keypoints_detections, losses

    def smooth_homographies_v2(
        self,
        homographies: list[Optional[np.ndarray]],
        keypoints_detections: list[tuple[np.ndarray, np.ndarray]],
        frames_sizes: list[tuple[int, int]],
        court_constants: CourtConstants,
        forward_H_norm_inv: list[Optional[np.ndarray]],
        alpha: float = 200.0,
        eps: float = 0.003,
        num_epochs: int = 300,
        lr: float = 0.001,
        device: Optional[str] = None,
        dtype=torch.float32,
    ):
        """
        Optimise the same objective as extract_homographies_from_video_v2 using Adam.

        objective = sum_i reward_i  -  alpha * sum_i smooth_i
        reward_i  = sum_k exp((cos_sim(H_i @ src_ik, dst_ik) - 1) / eps)
        smooth_i  = homographies_dist(H_{i+1}, H_i @ inv(flow_i))

        Fully vectorised: all keypoints and smoothness terms are computed in
        batched tensor operations (bmm / broadcasting) with no Python loops
        inside the training loop.
        """
        n = len(homographies)
        if n == 0:
            return homographies, None

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---- fill None homographies by propagation -------------------------
        homographies = deepcopy(homographies)
        for i in range(n - 1):
            if homographies[i + 1] is None:
                homographies[i + 1] = homographies[i]
        for i in range(n - 1, 0, -1):
            if homographies[i - 1] is None:
                homographies[i - 1] = homographies[i]
        if homographies[0] is None:
            return homographies, None

        # ---- normalise & stack into (N, 3, 3) tensor -----------------------
        H_np = np.stack([H / np.sqrt((H * H).sum()) for H in homographies], axis=0)
        H = torch.tensor(H_np, dtype=dtype, device=device, requires_grad=True)

        # ---- padded keypoint tensors (N, max_K, 3) -------------------------
        court_size = court_constants.court_size
        per_src: list[np.ndarray] = []
        per_dst: list[np.ndarray] = []
        counts: list[int] = []
        for i in range(n):
            centers, classes = keypoints_detections[i]
            w, h = frames_sizes[i]
            s, d = [], []
            for kc, kcls in zip(centers, classes):
                if kcls not in court_constants.cls_to_points:
                    continue
                cp = court_constants.cls_to_points[kcls][0]
                s.append([kc[0] / w, kc[1] / h, 1.0])
                d.append([cp[0] / court_size[0], cp[1] / court_size[1], 1.0])
            counts.append(len(s))
            per_src.append(np.array(s, dtype=np.float64) if s else np.empty((0, 3)))
            per_dst.append(np.array(d, dtype=np.float64) if d else np.empty((0, 3)))

        max_K = max(max(counts), 1)
        src_pad = np.zeros((n, max_K, 3), dtype=np.float64)
        dst_pad = np.zeros((n, max_K, 3), dtype=np.float64)
        kp_mask = np.zeros((n, max_K), dtype=np.float64)
        for i in range(n):
            k = counts[i]
            if k > 0:
                src_pad[i, :k] = per_src[i]
                dst_pad[i, :k] = per_dst[i]
                kp_mask[i, :k] = 1.0

        src_t = torch.tensor(src_pad, dtype=dtype, device=device)  # (N, K, 3)
        dst_t = torch.tensor(dst_pad, dtype=dtype, device=device)  # (N, K, 3)
        mask_t = torch.tensor(kp_mask, dtype=dtype, device=device)  # (N, K)

        # ---- flow inverse tensor & mask (N-1, 3, 3) -----------------------
        if n > 1:
            fi_list, fm_list = [], []
            for i in range(n - 1):
                if forward_H_norm_inv[i] is not None:
                    fi_list.append(forward_H_norm_inv[i])
                    fm_list.append(1.0)
                else:
                    fi_list.append(np.eye(3, dtype=np.float64))
                    fm_list.append(0.0)
            flow_inv_t = torch.tensor(np.stack(fi_list), dtype=dtype, device=device)
            flow_mask_t = torch.tensor(fm_list, dtype=dtype, device=device)
        else:
            flow_inv_t = torch.empty((0, 3, 3), dtype=dtype, device=device)
            flow_mask_t = torch.empty((0,), dtype=dtype, device=device)

        # ---- test vectors for homographies_dist (transposed: 3×9) ----------
        v1 = torch.tensor(
            [[x, y, 1.0] for x in [0, 0.5, 1.0] for y in [0, 0.5, 1.0]],
            dtype=dtype,
            device=device,
        ).T  # (3, 9)
        v2 = torch.tensor(
            [[x, y, 1.0] for x in [-0.5, 0, 0.5] for y in [-0.5, 0, 0.5]],
            dtype=dtype,
            device=device,
        ).T  # (3, 9)

        def _batch_cs(Mv, v_ref):
            """Batched cosine similarity: (M,3,9) vs (3,9) -> (M,9)."""
            dot = (Mv * v_ref.unsqueeze(0)).sum(dim=1)
            return dot / (Mv.norm(dim=1) * v_ref.norm(dim=0).unsqueeze(0) + 1e-12)

        def _batch_hdist(H1, H2):
            """Vectorised homographies_dist + dist2 for (M,3,3) pairs."""
            H1i = torch.linalg.pinv(H1)
            H2i = torch.linalg.pinv(H2)

            # dist: vecs1
            D = H1i @ H2
            Di = H2i @ H1
            d1 = (1 - _batch_cs(D @ v1, v1)).sum(1) + (1 - _batch_cs(Di @ v1, v1)).sum(1)

            # dist2: vecs2
            D2 = H2 @ H1i
            D2i = H1 @ H2i
            d2 = (1 - _batch_cs(D2 @ v2, v2)).sum(1) + (1 - _batch_cs(D2i @ v2, v2)).sum(1)

            return d1 + d2  # (M,)

        # ---- loss function (fully vectorised) ------------------------------
        def calc_loss():
            # reward: project all keypoints through their frame's H
            proj = torch.bmm(H, src_t.transpose(1, 2))  # (N, 3, K)
            dT = dst_t.transpose(1, 2)  # (N, 3, K)
            dot = (proj * dT).sum(dim=1)  # (N, K)
            cs = dot / (proj.norm(dim=1) * dT.norm(dim=1) + 1e-12)
            reward = (torch.exp((cs - 1.0) / eps) * mask_t).sum()

            # smoothness: compare H_{i+1} with H_i propagated through flow
            if n > 1:
                predicted = torch.bmm(H[:-1], flow_inv_t)  # (N-1, 3, 3)
                dists = _batch_hdist(H[1:], predicted)  # (N-1,)
                smooth = (dists * flow_mask_t).sum()
            else:
                smooth = torch.tensor(0.0, dtype=dtype, device=device)

            return -(reward - alpha * smooth)

        # ---- optimisation loop ---------------------------------------------
        optimizer = torch.optim.Adam([H], lr=lr)
        print("Smoothing homographies (v2)...")
        for epoch in tqdm(range(num_epochs)):
            optimizer.zero_grad()
            loss = calc_loss()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                H.div_(torch.sqrt((H * H).sum(dim=(1, 2), keepdim=True)))
            # print(f"Loss after {epoch} iterations: {loss.item():.3f}")

        # ---- per-frame losses & convert back to numpy ----------------------
        with torch.no_grad():
            proj = torch.bmm(H, src_t.transpose(1, 2))
            dT = dst_t.transpose(1, 2)
            dot = (proj * dT).sum(dim=1)
            cs = dot / (proj.norm(dim=1) * dT.norm(dim=1) + 1e-12)
            per_reward = (torch.exp((cs - 1.0) / eps) * mask_t).sum(dim=1)  # (N,)
            losses = (-per_reward).cpu().numpy().tolist()

        result = [H[i].detach().cpu().numpy() for i in range(n)]
        return result, losses

    @staticmethod
    def _compute_flow_homography_norm(
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_size: tuple[int, int],
        curr_size: tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Compute a normalized homography (prev_norm -> curr_norm) between two
        consecutive frames using optical flow.
        """
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=7,
            useHarrisDetector=False,
            k=0.04,
        )
        if prev_pts is None:
            return None
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        if curr_pts is None or status is None:
            return None
        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]
        if len(good_prev) < 4:
            return None

        pw, ph = prev_size
        cw, ch = curr_size
        prev_norm = good_prev.reshape(-1, 2) / np.array([pw, ph], dtype=np.float64)
        curr_norm = good_curr.reshape(-1, 2) / np.array([cw, ch], dtype=np.float64)

        H, inliers = cv2.findHomography(prev_norm, curr_norm, cv2.RANSAC, 3.0 / max(pw, ph, cw, ch))
        if H is None or inliers is None:
            return None
        return H

    def run(self, video_path: str, detections: PlayersDetections, court_type: CourtType = CourtType.NBA) -> None:
        """
        Process video frame-by-frame, compute homography, and enrich each
        :class:`Player` with ``court_position`` (x_m, y_m in meters).
        """

        print(f"Running court detector on {video_path}...")

        court_constants = CourtConstants(court_type)
        homographies, frames_sizes, keypoint_detections, losses = self.extract_homographies_from_video_v2(
            video_path, court_constants
        )

        court_w, court_h = court_constants.court_size
        for frame_id, H in enumerate(homographies):
            if H is None:
                continue
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
