from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch
from tqdm import tqdm

from common.classes.detections import Detection, FrameDetections, VideoDetections
from common.classes.ball import Ball, BallDetections
from common.classes.player import Player, PlayersDetections
from common.distances import bbox_iou, bbox_overlap_ratio
from rfdetr import RFDETRNano, RFDETRLarge


class DetectorRFDETR:
    """
    RF-DETR-based detector with the same interface as `detector.Detector`.

    Expects weights in `models/` (or a custom path) and returns project
    `Detection` objects compatible with the rest of the pipeline.
    """

    def __init__(
        self,
        model_path: str | Path = Path(__file__).parent.parent.parent / "models" / "checkpoint_best_ema.pth",
        conf_threshold: float = 0.1,
    ) -> None:
        # try:

        # except Exception as e:
        #     raise ImportError(
        #         "rfdetr is not installed. Install it with `pip install rfdetr` "
        #         "before using DetectorRFDETR."
        #     ) from e

        self._converted_checkpoint_path: Path | None = None
        model_path = Path(model_path)
        load_path = self._prepare_checkpoint_path(model_path)
        try:
            self.model = RFDETRLarge(pretrain_weights=str(load_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load RF-DETR weights from '{model_path}'. Expected a compatible checkpoint (.ckpt or .pth)."
            ) from e
        self.conf_threshold = conf_threshold

    def _prepare_checkpoint_path(self, model_path: Path) -> Path:
        """
        Make RF-DETR-compatible checkpoint path.

        RF-DETR inference expects a checkpoint dict with a top-level "model" key.
        Lightning .ckpt files often store weights under "state_dict".
        """
        checkpoint = torch.load(str(model_path), map_location="cpu")
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
            return model_path

        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
        if not isinstance(state_dict, dict):
            return model_path

        prefixes = ("model.", "module.model.", "module.", "_orig_mod.model.", "_orig_mod.")
        converted_model: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key
            for pref in prefixes:
                if new_key.startswith(pref):
                    new_key = new_key[len(pref) :]
                    break
            converted_model[new_key] = value

        with tempfile.NamedTemporaryFile(prefix="rfdetr_converted_", suffix=".pth", delete=False) as tmp:
            temp_path = Path(tmp.name)
        torch.save({"model": converted_model}, str(temp_path))
        self._converted_checkpoint_path = temp_path
        return temp_path

    def _normalize_predictions(self, preds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert RF-DETR predictions to arrays: xyxy, confidence, class_id.

        RF-DETR `predict` may return either a single supervision `Detections`
        object or a list with one element for a single image.
        """
        if isinstance(preds, list):
            if not preds:
                return np.empty((0, 4)), np.empty((0,)), np.empty((0,))
            preds = preds[0]

        xyxy = np.asarray(getattr(preds, "xyxy", np.empty((0, 4))))
        confidence = np.asarray(getattr(preds, "confidence", np.empty((0,))))
        class_id = np.asarray(getattr(preds, "class_id", np.empty((0,))))
        return xyxy, confidence, class_id

    def detect_frame(self, frame: np.ndarray) -> list[Detection]:
        preds = self.model.predict(frame, threshold=self.conf_threshold)
        xyxy, confidence, class_id = self._normalize_predictions(preds)

        detections: list[Detection] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    class_id=int(class_id[i]),
                    confidence=float(confidence[i]),
                )
            )
        return detections

    def detect_video(self, video_path: str) -> VideoDetections:
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_detections: VideoDetections = []

        for i in tqdm(range(frame_count), desc="Detector (RF-DETR)"):
            ret, frame = video.read()
            if not ret:
                break
            detections = self.detect_frame(frame)
            frame_detections.append(FrameDetections(i, detections))

        video.release()
        return frame_detections


def _nms_detections(detections: list[Detection], iou_threshold: float = 0.9) -> list[Detection]:
    """Simple NMS compatible with detector.py logic."""
    if len(detections) <= 1:
        return detections

    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for det in sorted_dets:
        bbox = det.get_bbox()
        is_duplicate = False
        for k in kept:
            k_bbox = k.get_bbox()
            iou = bbox_iou(bbox, k_bbox)
            overlap = bbox_overlap_ratio(bbox, k_bbox)
            if iou >= iou_threshold or overlap >= iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(det)
    return kept


def get_frame_players_detections(
    frame_detections: FrameDetections,
    conf_threshold: float = 0.2,
    nms_iou_threshold: float = 0.9,
) -> list[Player]:
    """
    Same signature/behavior shape as detector.get_frame_players_detections.
    """
    # RF-DETR classes: player class_id is 3. Dribble class is not used.
    player_detections = [d for d in frame_detections.detections if d.class_id == 1 and d.confidence >= conf_threshold]
    player_detections = _nms_detections(player_detections, iou_threshold=nms_iou_threshold)
    player_detections.sort(key=lambda x: -x.confidence)
    player_detections = player_detections[:10]

    return [
        Player(
            bbox=d.get_bbox(),
            player_id=i,
            confidence=d.confidence,
            is_dribble=False,
        )
        for i, d in enumerate(player_detections)
    ]


def get_video_players_detections(video_detections: VideoDetections, conf_threshold=0.2) -> PlayersDetections:
    """
    Same signature as detector.get_video_players_detections.
    """
    players_detections: PlayersDetections = {}
    for frame_detection in video_detections:
        players_detections[frame_detection.frame_id] = get_frame_players_detections(
            frame_detection, conf_threshold=conf_threshold
        )
    return players_detections


def get_players_detections(video_detections: VideoDetections, conf_threshold=0.2) -> PlayersDetections:
    """
    Convenience alias for RF-DETR player detections.
    """
    return get_video_players_detections(video_detections, conf_threshold=conf_threshold)


def get_frame_ball_detections(
    frame_detections: FrameDetections,
    conf_threshold: float = 0.1,
    nms_iou_threshold: float = 0.9,
) -> list[Ball]:
    """
    Same signature/behavior shape as detector.get_frame_ball_detections.
    """
    BALL_CLASS_ID = 0
    ball_detections = [
        d for d in frame_detections.detections if d.class_id == BALL_CLASS_ID and d.confidence >= conf_threshold
    ]
    ball_detections = _nms_detections(ball_detections, iou_threshold=nms_iou_threshold)
    return [Ball(bbox=d.get_bbox(), confidence=d.confidence) for d in ball_detections]


def get_video_ball_detections(video_detections: VideoDetections, conf_threshold: float = 0.1) -> BallDetections:
    """
    Same signature as detector.get_video_ball_detections.
    """
    ball_detections: BallDetections = {}
    for frame_detection in video_detections:
        ball_detections[frame_detection.frame_id] = get_frame_ball_detections(
            frame_detection, conf_threshold=conf_threshold
        )
    return ball_detections


def get_frame_rim_detections(
    frame_detections: FrameDetections,
    conf_threshold: float = 0.25,
    nms_iou_threshold: float = 0.9,
) -> list[Detection]:
    """
    Same signature as detector.get_frame_rim_detections.
    """
    RIM_CLASS_ID = 5
    rim_detections = [
        d for d in frame_detections.detections if d.class_id == RIM_CLASS_ID and d.confidence >= conf_threshold
    ]
    rim_detections = _nms_detections(rim_detections, iou_threshold=nms_iou_threshold)
    return rim_detections


def get_video_rim_detections(
    video_detections: VideoDetections,
    conf_threshold: float = 0.25,
) -> dict[int, list[Detection]]:
    """
    frame_id -> list of rim detections.
    """
    rim_by_frame: dict[int, list[Detection]] = {}
    for frame_detection in video_detections:
        rim_by_frame[frame_detection.frame_id] = get_frame_rim_detections(
            frame_detection, conf_threshold=conf_threshold
        )
    return rim_by_frame


def visualize_players_detections(
    input_path: str,
    output_path: str,
    *,
    detector: DetectorRFDETR | None = None,
    conf_threshold: float = 0.2,
) -> str:
    """
    Visualize only player bounding boxes on video using RF-DETR detections.
    """
    model = detector if detector is not None else DetectorRFDETR(conf_threshold=conf_threshold)
    video_detections = model.detect_video(input_path)
    player_detections = get_video_players_detections(video_detections, conf_threshold=conf_threshold)
    # rim_detections = get_video_rim_detections(video_detections, conf_threshold=conf_threshold)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    color_player = (255, 165, 0)  # BGR
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    try:
        for frame_id in tqdm(range(total_frames), desc="Visualize players (RF-DETR)"):
            ret, frame = cap.read()
            if not ret:
                break
            for player in player_detections.get(frame_id, []):
                x1, y1, x2, y2 = map(int, player.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_player, thickness)

                label = f"id:{player.player_id}"
                if player.confidence is not None:
                    label += f" {player.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color_player, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 2), font, font_scale, (0, 0, 0), font_thickness)

            # for rim in rim_detections.get(frame_id, []):
            #     x1, y1, x2, y2 = map(int, rim.get_bbox())
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), color_rim, thickness)

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return output_path


def visualize_all_detections(
    input_path: str,
    output_path: str,
    *,
    detector: DetectorRFDETR | None = None,
    conf_threshold: float = 0.1,
) -> str:
    """
    Visualize all RF-DETR detections with class_id and confidence labels.
    """
    model = detector if detector is not None else DetectorRFDETR(conf_threshold=conf_threshold)
    video_detections = model.detect_video(input_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    def _class_color(class_id: int) -> tuple[int, int, int]:
        # Deterministic color per class id (BGR).
        b = (37 * class_id + 89) % 256
        g = (17 * class_id + 149) % 256
        r = (53 * class_id + 197) % 256
        return int(b), int(g), int(r)

    try:
        for frame_id in tqdm(range(total_frames), desc="Visualize all detections (RF-DETR)"):
            ret, frame = cap.read()
            if not ret:
                break

            for det in video_detections[frame_id].detections:
                x1, y1, x2, y2 = det.get_bbox()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                color = _class_color(det.class_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                label = f"cls:{det.class_id} conf:{det.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                top = max(0, y1 - th - 4)
                cv2.rectangle(frame, (x1, top), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 2), font, font_scale, (0, 0, 0), font_thickness)

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return output_path
