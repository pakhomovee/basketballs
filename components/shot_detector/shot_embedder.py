from __future__ import annotations

import numpy as np

from common.classes import CourtType
from common.classes.ball import Ball
from common.classes.detections import Detection
from court_detector.court_constants import CourtConstants, SYMMETRIC_MAPPING


class ShotEmbedder:
    """
    Build shot-model input embeddings from ball/rim detections and homographies.

    Object slots:
      - 1 ball
      - 2 rims (left + right)
      - 33 court keypoints

    Each slot begins with "present" flag.
    - ball/keypoint slot: [present, x_rel, y_rel]
    - rim slot: [present, x1_rel, y1_rel, x2_rel, y2_rel]
    """

    NUM_KEYPOINTS = 33
    BALL_DIM = 3
    RIM_DIM = 5
    KEYPOINT_DIM = 3
    FLAT_DIM = BALL_DIM + 2 * RIM_DIM + NUM_KEYPOINTS * KEYPOINT_DIM

    def __init__(
        self,
        court_type: CourtType = CourtType.NBA,
        *,
        fliplr: bool = False,
        random_scale: float = 1.0,
        random_shift: float = 0.0,
        random_rotate: float = 0.0,
    ):
        self.court_constants = CourtConstants(court_type)
        self.court_w, self.court_h = self.court_constants.court_size
        self.fliplr = fliplr
        self.random_scale = max(float(random_scale), 1e-8)
        self.random_shift = max(float(random_shift), 0.0)
        self.random_rotate = max(float(random_rotate), 0.0)

    @staticmethod
    def _project(H: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
        v = H @ np.array([x, y, 1.0], dtype=np.float32)
        if abs(float(v[2])) < 1e-8:
            return None
        return float(v[0] / v[2]), float(v[1] / v[2])

    @staticmethod
    def _pick_best_ball(ball_or_list: Ball | list[Ball] | None) -> Ball | None:
        if ball_or_list is None:
            return None
        if isinstance(ball_or_list, list):
            if not ball_or_list:
                return None
            return max(ball_or_list, key=lambda b: float(b.confidence or 0.0))
        return ball_or_list

    @staticmethod
    def _pick_left_right_rims(
        rims: list[Detection],
        H: np.ndarray | None,
        w: float,
        h: float,
    ) -> tuple[Detection | None, Detection | None]:
        left_best: Detection | None = None
        right_best: Detection | None = None
        left_score = -1.0
        right_score = -1.0

        for d in rims:
            x1, y1, x2, y2 = [float(v) for v in d.get_bbox()]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            x_rel = cx / max(w, 1.0)
            y_rel = cy / max(h, 1.0)

            is_left = None
            if H is not None:
                court_xy = ShotEmbedder._project(H, x_rel, y_rel)
                if court_xy is not None:
                    is_left = court_xy[0] < 0.0
            if is_left is None:
                is_left = x_rel < 0.5

            score = float(getattr(d, "confidence", 0.0))
            if is_left:
                if score > left_score:
                    left_best = d
                    left_score = score
            else:
                if score > right_score:
                    right_best = d
                    right_score = score

        return left_best, right_best

    def _build_keypoint_slots(self, H: np.ndarray | None) -> np.ndarray:
        kp = np.zeros((self.NUM_KEYPOINTS, self.KEYPOINT_DIM), dtype=np.float32)
        if H is None:
            return kp

        try:
            H_inv = np.linalg.pinv(H)
        except np.linalg.LinAlgError:
            return kp

        for x_court, y_court, cls_id in self.court_constants.court_points:
            if cls_id < 0 or cls_id >= self.NUM_KEYPOINTS:
                continue
            x_norm = float(x_court) / float(self.court_w)
            y_norm = float(y_court) / float(self.court_h)
            frame_xy = self._project(H_inv, x_norm, y_norm)
            if frame_xy is None:
                continue
            x_rel, y_rel = frame_xy
            if 0.0 <= x_rel <= 1.0 and 0.0 <= y_rel <= 1.0:
                kp[cls_id, 0] = 1.0
                kp[cls_id, 1] = x_rel
                kp[cls_id, 2] = y_rel
        return kp

    @staticmethod
    def _apply_affine_to_points(
        points_xy: np.ndarray, scale: float, tx: float, ty: float, angle_rad: float
    ) -> np.ndarray:
        """
        Apply geometric augmentation in normalized coordinates.
        Transform is applied around frame center (0.5, 0.5):
          p' = R * (scale * (p - c)) + c + t
        """
        if points_xy.size == 0:
            return points_xy
        c = np.array([0.5, 0.5], dtype=np.float32)
        p = points_xy.astype(np.float32) - c[None, :]
        p = p * np.float32(scale)
        ca = np.float32(np.cos(angle_rad))
        sa = np.float32(np.sin(angle_rad))
        R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
        p = p @ R.T
        p = p + c[None, :]
        p[:, 0] += np.float32(tx)
        p[:, 1] += np.float32(ty)
        return p

    def _augment_sequence(
        self,
        ball_arr: np.ndarray,
        left_rim_arr: np.ndarray,
        right_rim_arr: np.ndarray,
        kp_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply random augmentations to all frames in the sequence consistently.
        """
        T = ball_arr.shape[0]
        if T == 0:
            return ball_arr, left_rim_arr, right_rim_arr, kp_arr

        # Sample one transform per sequence.
        scale = 1.0
        if self.random_scale > 0 and abs(self.random_scale - 1.0) > 1e-8:
            lo = 1.0 / self.random_scale
            hi = self.random_scale
            if lo > hi:
                lo, hi = hi, lo
            scale = float(np.random.uniform(lo, hi))

        tx = float(np.random.uniform(-self.random_shift, self.random_shift)) if self.random_shift > 0 else 0.0
        ty = float(np.random.uniform(-self.random_shift, self.random_shift)) if self.random_shift > 0 else 0.0
        angle_deg = float(np.random.uniform(-self.random_rotate, self.random_rotate)) if self.random_rotate > 0 else 0.0
        angle_rad = np.deg2rad(angle_deg)

        # Ball points
        ball_mask = ball_arr[:, 0] > 0.5
        if np.any(ball_mask):
            pts = ball_arr[ball_mask, 1:3]
            ball_arr[ball_mask, 1:3] = self._apply_affine_to_points(pts, scale, tx, ty, angle_rad)

        # Keypoint points
        kp_mask = kp_arr[:, :, 0] > 0.5
        if np.any(kp_mask):
            pts = kp_arr[:, :, 1:3].reshape(-1, 2)
            mask_flat = kp_mask.reshape(-1)
            pts_sel = pts[mask_flat]
            pts_sel = self._apply_affine_to_points(pts_sel, scale, tx, ty, angle_rad)
            pts[mask_flat] = pts_sel
            kp_arr[:, :, 1:3] = pts.reshape(T, self.NUM_KEYPOINTS, 2)

        # Rim bboxes: transform all 4 corners, rebuild axis-aligned bbox.
        def _augment_rim_bboxes(arr: np.ndarray) -> np.ndarray:
            mask = arr[:, 0] > 0.5
            if not np.any(mask):
                return arr
            b = arr[mask, 1:5]  # x1,y1,x2,y2
            x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            corners = np.stack(
                [
                    np.stack([x1, y1], axis=1),
                    np.stack([x2, y1], axis=1),
                    np.stack([x2, y2], axis=1),
                    np.stack([x1, y2], axis=1),
                ],
                axis=1,
            )  # (N,4,2)
            corners2 = self._apply_affine_to_points(corners.reshape(-1, 2), scale, tx, ty, angle_rad).reshape(-1, 4, 2)
            new_x1 = corners2[:, :, 0].min(axis=1)
            new_y1 = corners2[:, :, 1].min(axis=1)
            new_x2 = corners2[:, :, 0].max(axis=1)
            new_y2 = corners2[:, :, 1].max(axis=1)
            arr[mask, 1:5] = np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)
            return arr

        left_rim_arr = _augment_rim_bboxes(left_rim_arr)
        right_rim_arr = _augment_rim_bboxes(right_rim_arr)

        # Optional horizontal flip with p=0.5.
        do_flip = self.fliplr and (np.random.rand() < 0.5)
        if do_flip:
            # Ball / keypoints points
            if np.any(ball_mask):
                ball_arr[ball_mask, 1] = 1.0 - ball_arr[ball_mask, 1]

            if np.any(kp_mask):
                pts = kp_arr[:, :, 1:3].reshape(-1, 2)
                mask_flat = kp_mask.reshape(-1)
                pts[mask_flat, 0] = 1.0 - pts[mask_flat, 0]
                kp_arr[:, :, 1:3] = pts.reshape(T, self.NUM_KEYPOINTS, 2)

            # Rim boxes: x -> 1-x, then reorder x1<=x2
            def _flip_rim_boxes(arr: np.ndarray) -> np.ndarray:
                m = arr[:, 0] > 0.5
                if not np.any(m):
                    return arr
                x1 = arr[m, 1].copy()
                x2 = arr[m, 3].copy()
                arr[m, 1] = 1.0 - x2
                arr[m, 3] = 1.0 - x1
                return arr

            left_rim_arr = _flip_rim_boxes(left_rim_arr)
            right_rim_arr = _flip_rim_boxes(right_rim_arr)

            # Swap left/right rim semantic slots.
            left_rim_arr, right_rim_arr = right_rim_arr.copy(), left_rim_arr.copy()

            # Remap keypoint ids by court symmetry.
            kp_new = np.zeros_like(kp_arr)
            for i in range(self.NUM_KEYPOINTS):
                j = SYMMETRIC_MAPPING.get(i, i)
                if 0 <= j < self.NUM_KEYPOINTS:
                    kp_new[:, j, :] = kp_arr[:, i, :]
            kp_arr = kp_new

        return ball_arr, left_rim_arr, right_rim_arr, kp_arr

    def build_embedding(
        self,
        ball_detections: dict[int, Ball | list[Ball]],
        rim_detections: dict[int, list[Detection]],
        homographies: list[np.ndarray | None] | dict[int, np.ndarray | None],
        *,
        frame_width: float,
        frame_height: float,
        num_frames: int | None = None,
    ) -> np.ndarray:
        """
        Build flattened frame-wise embedding from detections + homographies.

        *frame_width* / *frame_height* must match the video (e.g. from
        ``VideoCapture.get(cv2.CAP_PROP_FRAME_WIDTH/HEIGHT)`` or stored clip size).

        Returns
        -------
        np.ndarray
            Shape (T, FLAT_DIM), where FLAT_DIM = 112.
        """

        def h_get(i: int) -> np.ndarray | None:
            if isinstance(homographies, dict):
                return homographies.get(i)
            if 0 <= i < len(homographies):
                return homographies[i]
            return None

        if isinstance(homographies, dict):
            max_h_idx = max(homographies.keys(), default=-1)
        else:
            max_h_idx = len(homographies) - 1

        max_ball_idx = max(ball_detections.keys(), default=-1)
        max_rim_idx = max(rim_detections.keys(), default=-1)
        T = num_frames if num_frames is not None else (max(max_h_idx, max_ball_idx, max_rim_idx) + 1)
        T = max(int(T), 0)

        ball_arr = np.zeros((T, self.BALL_DIM), dtype=np.float32)
        left_rim_arr = np.zeros((T, self.RIM_DIM), dtype=np.float32)
        right_rim_arr = np.zeros((T, self.RIM_DIM), dtype=np.float32)
        kp_arr = np.zeros((T, self.NUM_KEYPOINTS, self.KEYPOINT_DIM), dtype=np.float32)

        w_use = float(max(frame_width, 1.0))
        h_use = float(max(frame_height, 1.0))

        for f in range(T):
            H = h_get(f)
            w, h = w_use, h_use

            ball = self._pick_best_ball(ball_detections.get(f))
            if ball is not None and ball.bbox and len(ball.bbox) >= 4:
                x1, y1, x2, y2 = [float(v) for v in ball.bbox[:4]]
                cx = 0.5 * (x1 + x2) / w
                cy = 0.5 * (y1 + y2) / h
                ball_arr[f, 0] = 1.0
                ball_arr[f, 1] = cx
                ball_arr[f, 2] = cy

            rims = rim_detections.get(f, [])
            left_rim, right_rim = self._pick_left_right_rims(rims, H, w, h)

            if left_rim is not None:
                x1, y1, x2, y2 = [float(v) for v in left_rim.get_bbox()]
                left_rim_arr[f, 0] = 1.0
                left_rim_arr[f, 1:] = np.array([x1 / w, y1 / h, x2 / w, y2 / h], dtype=np.float32)

            if right_rim is not None:
                x1, y1, x2, y2 = [float(v) for v in right_rim.get_bbox()]
                right_rim_arr[f, 0] = 1.0
                right_rim_arr[f, 1:] = np.array([x1 / w, y1 / h, x2 / w, y2 / h], dtype=np.float32)

            kp_arr[f] = self._build_keypoint_slots(H)

        ball_arr, left_rim_arr, right_rim_arr, kp_arr = self._augment_sequence(
            ball_arr=ball_arr,
            left_rim_arr=left_rim_arr,
            right_rim_arr=right_rim_arr,
            kp_arr=kp_arr,
        )

        flat = np.concatenate(
            [
                ball_arr,
                left_rim_arr,
                right_rim_arr,
                kp_arr.reshape(T, self.NUM_KEYPOINTS * self.KEYPOINT_DIM),
            ],
            axis=1,
        ).astype(np.float32)
        return flat
