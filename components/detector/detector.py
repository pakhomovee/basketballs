from common.classes.detections import FrameDetections, Detection, VideoDetections
from common.classes.ball import BallDetections, Ball
from common.distances import bbox_iou, bbox_overlap_ratio
from ultralytics import YOLO
import numpy as np
import cv2
from typing import List
from common.classes.player import PlayersDetections, Player
from common.classes.number import NumberDetections, Number
from tqdm import tqdm
from pathlib import Path

from detector.number_recognizer import recognize_numbers_in_frame

# names: ['ball', 'number', 'player', 'player-dribble', 'player-fall', 'player-jump-shot', 'player-layup',
# 'player-screen', 'player-shot-block', 'referee', 'rim']


class Detector:
    def __init__(
        self,
        model_path: str | Path = Path(__file__).parent.parent.parent / "models" / "best-4.pt",
        conf_threshold: float = 0.1,
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_frame(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().reshape(-1))

            class_id = int(box.cls.item())
            detections.append(Detection(x1, y1, x2, y2, class_id, float(box.conf.item())))
        return detections

    def detect_video(self, video_path: str) -> VideoDetections:
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_detections = []
        for i in tqdm(range(frame_count), desc="Detector"):
            ret, frame = video.read()
            if not ret:
                break
            detections = self.detect_frame(frame)
            frame_detections.append(FrameDetections(i, detections))
        return VideoDetections(frame_detections)


def _nms_detections(detections: List[Detection], iou_threshold: float = 0.9) -> List[Detection]:
    """
    Non-maximum suppression with high IoU threshold.
    Suppresses duplicates when boxes are nearly identical or one is fully inside the other.
    Keeps the detection with highest confidence in each cluster.
    """
    if len(detections) <= 1:
        return detections
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept = []
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
    conf_threshold: float = 0.25,
    nms_iou_threshold: float = 0.9,
) -> list[Player]:
    player_detections = [
        d for d in frame_detections.detections if 2 <= d.class_id <= 8 and d.confidence >= conf_threshold
    ]
    player_detections = _nms_detections(player_detections, iou_threshold=nms_iou_threshold)

    player_detections.sort(key=lambda x: -x.confidence)
    player_detections = player_detections[:10]
    return [Player(bbox=d.get_bbox(), player_id=i, confidence=d.confidence) for i, d in enumerate(player_detections)]


def get_video_players_detections(video_detections: VideoDetections, conf_threshold=0.25) -> PlayersDetections:
    players_detections: PlayersDetections = {}
    for frame_detection in video_detections:
        players_detections[frame_detection.frame_id] = get_frame_players_detections(
            frame_detection, conf_threshold=conf_threshold
        )
    return players_detections


def get_frame_ball_detections(
    frame_detections: FrameDetections, conf_threshold=0.5, nms_iou_threshold: float = 0.9
) -> list[Ball]:
    ball_detections = [d for d in frame_detections.detections if d.class_id == 0 and d.confidence >= conf_threshold]
    ball_detections = _nms_detections(ball_detections, iou_threshold=nms_iou_threshold)

    return [Ball(bbox=d.get_bbox(), confidence=d.confidence) for d in ball_detections]


def get_video_ball_detections(video_detections: VideoDetections) -> BallDetections:
    ball_detections: BallDetections = {}
    for frame_detection in video_detections:
        ball_detections[frame_detection.frame_id] = get_frame_ball_detections(frame_detection)
    return ball_detections


def get_frame_number_detections(
    frame_detections: FrameDetections,
    frame: np.ndarray | None = None,
    conf_threshold: float = 0.5,
    nms_iou_threshold: float = 0.9,
    ocr_conf_threshold: float = 0.9,
) -> list[Number]:
    number_detections = [d for d in frame_detections.detections if d.class_id == 1 and d.confidence >= conf_threshold]
    number_detections = _nms_detections(number_detections, iou_threshold=nms_iou_threshold)
    number_detections = [Number(bbox=d.get_bbox(), confidence=d.confidence) for d in number_detections]
    if frame is not None and number_detections:
        recognize_numbers_in_frame(frame, number_detections, ocr_conf_threshold=ocr_conf_threshold)
    return number_detections


def get_video_number_detections(video_detections: VideoDetections, conf_threshold: float = 0.5) -> NumberDetections:
    number_detections: NumberDetections = {}
    for frame_detection in video_detections:
        number_detections[frame_detection.frame_id] = get_frame_number_detections(
            frame_detection, conf_threshold=conf_threshold
        )
    return number_detections
