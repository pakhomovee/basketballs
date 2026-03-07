from common.classes.detections import FrameDetections, Detection, VideoDetections
from common.distances import bbox_iou, bbox_overlap_ratio
from ultralytics import YOLO
import numpy as np
import cv2
from typing import List
from common.classes.player import PlayersDetections, Player
from tqdm import tqdm
from pathlib import Path

# names: ['ball', 'number', 'player', 'player-dribble', 'player-fall', 'player-jump-shot', 'player-layup',
# 'player-screen', 'player-shot-block', 'referee', 'rim']


class Detector:
    def __init__(
        self,
        model_path: str | Path = Path(__file__).parent.parent.parent / "models" / "yolo26m_object_detection.pt",
        conf_threshold: float = 0.1,
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_frame(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().reshape(-1))

            # x1 /= frame.shape[1]
            # x2 /= frame.shape[1]
            # y1 /= frame.shape[0]
            # y2 /= frame.shape[0]

            class_id = int(box.cls.item())
            detections.append(Detection(x1, y1, x2, y2, class_id, float(box.conf.item())))
        return detections

    def detect_video(self, video_path: str) -> VideoDetections:
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_detections = []
        for i in tqdm(range(frame_count), desc='Detector'):
            ret, frame = video.read()
            if not ret:
                break
            detections = self.detect_frame(frame)
            frame_detections.append(FrameDetections(i, detections))
        return VideoDetections(frame_detections)


def _nms_player_detections(detections: List[Detection], iou_threshold: float = 0.9) -> List[Detection]:
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
    conf_threshold: float = 0.5,
    nms_iou_threshold: float = 0.9,
) -> list[Player]:
    player_detections = [
        d for d in frame_detections.detections
        if 2 <= d.class_id <= 8 and d.confidence >= conf_threshold
    ]
    player_detections = _nms_player_detections(player_detections, iou_threshold=nms_iou_threshold)
    return [Player(bbox=d.get_bbox(), player_id=i) for i, d in enumerate(player_detections)]


def get_video_players_detections(video_detections: VideoDetections) -> PlayersDetections:
    players_detections: PlayersDetections = {}
    for frame_detection in video_detections:
        players_detections[frame_detection.frame_id] = get_frame_players_detections(frame_detection)
    return players_detections
