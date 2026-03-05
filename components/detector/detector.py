from common.classes.detections import FrameDetections, Detection, VideoDetections
from ultralytics import YOLO
import numpy as np
import cv2
from typing import List
from common.classes.player import PlayersDetections, Player
from tqdm import tqdm
from pathlib import Path

# names: ['ball', 'number', 'player', 'player-dribble', 'player-fall', 'player-jump-shot', 'player-layup', 'player-screen', 'player-shot-block', 'referee', 'rim']


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
        for i in tqdm(range(frame_count)):
            ret, frame = video.read()
            if not ret:
                break
            detections = self.detect_frame(frame)
            frame_detections.append(FrameDetections(i, detections))
        return VideoDetections(frame_detections)


def get_frame_players_detections(frame_detections: FrameDetections, conf_threshold=0.5) -> list[Player]:
    detections = []
    idx = 0
    for detection in frame_detections.detections:
        if detection.class_id == 2 and detection.confidence >= conf_threshold:
            detections.append(Player(bbox=detection.get_bbox(), player_id=idx))
            idx += 1
    return detections


def get_video_players_detections(video_detections: VideoDetections) -> PlayersDetections:
    players_detections: PlayersDetections = {}
    for frame_detection in video_detections:
        players_detections[frame_detection.frame_id] = get_frame_players_detections(frame_detection)
    return players_detections
