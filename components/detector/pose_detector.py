"""
Lightweight wrapper around a YOLO-pose model for player pose detection on basketball videos.

This module is intentionally self-contained and does NOT modify any existing code.
It can be imported and used independently alongside the existing detector pipeline.

Example:
    from detector.pose_detector import PoseDetector, detect_video_poses

    poses_by_frame = detect_video_poses("input.mp4")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Iterable, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from common.distances import bbox_iou
from detector import (
    Detector,
    get_video_players_detections,
    get_video_ball_detections,
    get_video_rim_detections,
)


@dataclass
class PoseDetection:
    """
    Single pose detection for one person in a frame.

    Attributes:
        bbox: [x1, y1, x2, y2] in pixels.
        keypoints: np.ndarray of shape (K, 3) with (x, y, confidence) per keypoint.
        score: overall detection confidence.
        class_id: integer class ID from the pose model (e.g. 0 for person).
    """

    bbox: List[float] = field(default_factory=list)
    keypoints: np.ndarray | None = None
    score: float | None = None
    class_id: int | None = None


FramePoses = Dict[int, List[PoseDetection]]  # frame_id -> list of poses


class PoseDetector:
    """
    YOLO-pose based detector for player poses in basketball videos.

    The model is loaded once and then applied frame-by-frame.
    By default it expects a pose checkpoint in the repository's models/ directory,
    but a custom path can be passed explicitly.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = 0.1,
    ):
        """
        Args:
            model_path: Path to YOLO-pose checkpoint. If None, uses
                <repo_root>/models/yolov8n-pose.pt by default.
            conf_threshold: Minimum confidence for keeping a pose detection.
        """
        repo_root = Path(__file__).resolve().parent.parent.parent
        default_model = repo_root / "models" / "yolov8m-pose.pt"
        self.model_path = Path(model_path) if model_path is not None else default_model
        self.conf_threshold = conf_threshold
        self.model = YOLO(str(self.model_path))

    def detect_frame(self, frame: np.ndarray) -> List[PoseDetection]:
        """
        Run pose detection on a single frame.

        Returns:
            List of PoseDetection for this frame.
        """
        results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
        poses: List[PoseDetection] = []

        # boxes.xyxy: (N, 4), boxes.conf: (N,), boxes.cls: (N,)
        # keypoints.xy: (N, K, 2), keypoints.conf: (N, K)
        boxes = getattr(results, "boxes", None)
        keypoints = getattr(results, "keypoints", None)

        if boxes is None or keypoints is None:
            return poses

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        k_xy = keypoints.xy.cpu().numpy()  # (N, K, 2)
        k_conf = keypoints.conf.cpu().numpy()  # (N, K)

        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i].tolist()
            score = float(confs[i])
            class_id = int(clses[i])
            # Combine keypoint coordinates and confidences into (K, 3)
            kp_xy = k_xy[i]  # (K, 2)
            kp_c = k_conf[i][:, None]  # (K, 1)
            kp = np.concatenate([kp_xy, kp_c], axis=1)  # (K, 3)
            poses.append(
                PoseDetection(
                    bbox=[x1, y1, x2, y2],
                    keypoints=kp,
                    score=score,
                    class_id=class_id,
                )
            )

        return poses

    def detect_video(self, video_path: str) -> FramePoses:
        """
        Run pose detection on all frames of the given video.

        Args:
            video_path: Path to the input basketball video.

        Returns:
            Dictionary mapping frame_id -> list[PoseDetection].
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        poses_by_frame: FramePoses = {}

        try:
            for frame_id in tqdm(range(frame_count), desc="PoseDetector", unit="frame"):
                ret, frame = cap.read()
                if not ret:
                    break
                poses_by_frame[frame_id] = self.detect_frame(frame)
        finally:
            cap.release()

        return poses_by_frame


def detect_video_poses(
    video_path: str,
    model_path: str | Path | None = None,
    conf_threshold: float = 0.25,
) -> FramePoses:
    """
    Convenience function: run YOLO-pose on a video and return per-frame poses.

    This is a thin wrapper around PoseDetector for quick one-shot usage.
    Existing code is not modified; this function is fully standalone.
    """
    detector = PoseDetector(model_path=model_path, conf_threshold=conf_threshold)
    return detector.detect_video(video_path)


def _draw_pose_skeleton(
    frame: np.ndarray,
    pose: PoseDetection,
    *,
    keypoint_color: Tuple[int, int, int] = (0, 255, 0),
    limb_color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 3,
    thickness: int = 2,
    conf_threshold: float = 0.3,
) -> None:
    """
    Draw keypoints and a simple skeleton on the frame for a single PoseDetection.

    The exact connectivity depends on the YOLO-pose keypoint ordering; here we use
    a generic subset of COCO-style limbs which works reasonably well for debugging.
    """
    if pose.keypoints is None or pose.keypoints.size == 0:
        return

    kps = pose.keypoints  # (K, 3) -> (x, y, c)
    num_kp = kps.shape[0]

    # Draw joints
    for idx in range(num_kp):
        x, y, c = kps[idx]
        if c < conf_threshold:
            continue
        cv2.circle(frame, (int(x), int(y)), radius, keypoint_color, -1, lineType=cv2.LINE_AA)

    # Simple skeleton connections (indices for common 17-kp layout; robust to missing indices)
    skeleton: Iterable[Tuple[int, int]] = [
        (5, 6),  # shoulders
        (5, 7),
        (7, 9),  # left arm
        (6, 8),
        (8, 10),  # right arm
        (11, 12),  # hips
        (11, 13),
        (13, 15),  # left leg
        (12, 14),
        (14, 16),  # right leg
    ]
    for i, j in skeleton:
        if i >= num_kp or j >= num_kp:
            continue
        x1, y1, c1 = kps[i]
        x2, y2, c2 = kps[j]
        if c1 < conf_threshold or c2 < conf_threshold:
            continue
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), limb_color, thickness, lineType=cv2.LINE_AA)


def write_video_with_poses(
    input_video_path: str,
    output_video_path: str | None = None,
    *,
    model_path: str | Path | None = None,
    conf_threshold: float = 0.25,
) -> str:
    """
    Run YOLO-pose on a basketball video and save a new video with drawn poses.

    This is a high-level convenience wrapper that:
      1) Opens the input video
      2) Runs PoseDetector frame-by-frame
      3) Draws keypoints + skeleton for each detected person
      4) Saves the result to a new file

    Returns:
        Path to the saved output video.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if output_video_path is None:
        p = Path(input_video_path)
        output_video_path = str(p.parent / f"{p.stem}_poses{p.suffix}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    detector = PoseDetector(model_path=model_path, conf_threshold=conf_threshold)

    try:
        frame_id = 0
        # Total frames can be 0 / unknown, so don't rely on tqdm(total=...)
        for _ in tqdm(iter(int, 1), desc="Pose video", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            poses = detector.detect_frame(frame)
            for pose in poses:
                _draw_pose_skeleton(frame, pose)

            writer.write(frame)
            frame_id += 1
    finally:
        cap.release()
        writer.release()

    return output_video_path


def write_video_with_player_poses(
    input_video_path: str,
    output_video_path: str | None = None,
    *,
    pose_model_path: str | Path | None = None,
    pose_conf_threshold: float = 0.25,
    player_conf_threshold: float = 0.1,
) -> str:
    """
    Run player detector first, then apply YOLO-pose only on player crops.

    Pipeline:
      1) Detector() -> detect_video(input_video_path) -> VideoDetections
      2) get_video_players_detections(...) -> players per frame
      3) For each frame: crop each player bbox and run PoseDetector on the crop
      4) Map keypoints back to full-frame coordinates and draw the skeleton

    This produces poses for players only and reduces pose-model work on background regions.
    """
    # 1) Get player detections for the whole video
    det = Detector()
    video_detections = det.detect_video(input_video_path)
    players_detections = get_video_players_detections(video_detections, conf_threshold=player_conf_threshold)

    # 2) Prepare video I/O
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if output_video_path is None:
        p = Path(input_video_path)
        output_video_path = str(p.parent / f"{p.stem}_player_poses{p.suffix}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    pose_detector = PoseDetector(model_path=pose_model_path, conf_threshold=pose_conf_threshold)

    try:
        frame_id = 0
        for _ in tqdm(iter(int, 1), desc="Player pose video", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            players_in_frame = players_detections.get(frame_id, [])

            for player in players_in_frame:
                if not player.bbox:
                    continue
                x1, y1, x2, y2 = map(int, player.bbox)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w - 1, x2)
                y2 = min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                poses = pose_detector.detect_frame(crop)
                # Map keypoints back to full-frame coordinates and draw
                for pose in poses:
                    if pose.keypoints is not None and pose.keypoints.size > 0:
                        pose.keypoints[:, 0] += x1
                        pose.keypoints[:, 1] += y1
                    _draw_pose_skeleton(frame, pose)

            writer.write(frame)
            frame_id += 1
    finally:
        cap.release()
        writer.release()

    return output_video_path


def _match_poses_to_players(
    players,
    poses: List[PoseDetection],
    iou_threshold: float = 0.3,
):
    """
    Match YOLO-pose detections to existing Player objects by IoU of bboxes.

    Returns:
        List of (player, PoseDetection) pairs
    """
    matched: list[tuple] = []
    for player in players:
        if not player.bbox:
            continue
        best_iou = 0.0
        best_pose: PoseDetection | None = None
        for pose in poses:
            iou = bbox_iou(player.bbox, pose.bbox)
            if iou > best_iou:
                best_iou = iou
                best_pose = pose
        if best_pose is not None and best_iou >= iou_threshold:
            matched.append((player, best_pose))
    return matched


def _hand_centers_from_pose(
    pose: PoseDetection,
    *,
    left_wrist_idx: int = 9,
    right_wrist_idx: int = 10,
    conf_threshold: float = 0.1,
) -> List[Tuple[float, float]]:
    """
    Return list of hand points (left/right wrists) for pose.

    Each hand is considered separately; may return 0, 1 or 2 points.
    """
    if pose.keypoints is None or pose.keypoints.size == 0:
        return []
    kps = pose.keypoints
    num_kp = kps.shape[0]

    points: List[Tuple[float, float]] = []
    for idx in (left_wrist_idx, right_wrist_idx):
        if idx < num_kp:
            x, y, c = kps[idx]
            if c >= conf_threshold:
                points.append((float(x), float(y)))
    return points


def write_video_with_ball_handler_poses(
    input_video_path: str,
    output_video_path: str | None = None,
    *,
    pose_model_path: str | Path | None = None,
    pose_conf_threshold: float = 0.1,
    player_conf_threshold: float = 0.1,
    max_hand_ball_dist: float = 30.0,
) -> str:
    """
    Detect players + ball with your Detector, run YOLO-pose on full frames,
    match poses to players, and highlight the ball handler with an arrow.

    "Ball handler" is the player whose hand (either wrist keypoint) is closest
    to the center of the ball bbox.
    """
    # 1) Player/ball detections for the whole video
    det = Detector()
    video_detections = det.detect_video(input_video_path)
    players_detections = get_video_players_detections(video_detections, conf_threshold=player_conf_threshold)
    ball_detections = get_video_ball_detections(video_detections)
    rim_detections = get_video_rim_detections(video_detections, conf_threshold=0.1)

    # 2) Prepare video I/O
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if output_video_path is None:
        p = Path(input_video_path)
        output_video_path = str(p.parent / f"{p.stem}_ball_handler_poses{p.suffix}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    pose_detector = PoseDetector(model_path=pose_model_path, conf_threshold=pose_conf_threshold)

    arrow_color = (0, 0, 255)  # red
    arrow_thickness = 3

    try:
        frame_id = 0
        for _ in tqdm(iter(int, 1), desc="Ball handler pose video", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            players_in_frame = players_detections.get(frame_id, [])
            balls_in_frame = ball_detections.get(frame_id, [])
            rims_in_frame = rim_detections.get(frame_id, [])

            poses = pose_detector.detect_frame(frame)
            matched = _match_poses_to_players(players_in_frame, poses)

            # Draw skeletons and player boxes
            for player, pose in matched:
                _draw_pose_skeleton(frame, pose)
                if player.bbox:
                    x1, y1, x2, y2 = map(int, player.bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)

            # Determine ball handler
            ball_center: Tuple[float, float] | None = None
            if balls_in_frame:
                # Take the most confident ball detection
                best_ball = max(balls_in_frame, key=lambda b: b.confidence or 0.0)
                bx1, by1, bx2, by2 = best_ball.bbox
                ball_center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
                cv2.circle(frame, (int(ball_center[0]), int(ball_center[1])), 6, (0, 165, 255), -1)

            best_player_for_ball = None
            best_dist = float("inf")
            if ball_center is not None:
                bx, by = ball_center
                for player, pose in matched:
                    hand_points = _hand_centers_from_pose(pose)
                    if not hand_points:
                        continue
                    for hx, hy in hand_points:
                        dist2 = (hx - bx) ** 2 + (hy - by) ** 2
                        if dist2 < best_dist:
                            best_dist = dist2
                            best_player_for_ball = player

            # Draw arrow above ball handler only if the hand is close enough to the ball
            if best_player_for_ball is not None and best_player_for_ball.bbox and best_dist <= max_hand_ball_dist**2:
                x1, y1, x2, y2 = map(int, best_player_for_ball.bbox)
                cx = (x1 + x2) // 2
                top_y = max(0, y1 - 40)
                # Arrow down to the player
                cv2.arrowedLine(
                    frame,
                    (cx, top_y),
                    (cx, y1),
                    arrow_color,
                    arrow_thickness,
                    tipLength=0.3,
                )

            # Draw rim (class 10) from the detector
            for rim_det in rims_in_frame:
                rx1, ry1, rx2, ry2 = rim_det.get_bbox()
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)

            writer.write(frame)
            frame_id += 1
    finally:
        cap.release()
        writer.release()

    return output_video_path


write_video_with_ball_handler_poses("test_nba3.mp4", "megakek4.mp4")
