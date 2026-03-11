from common.classes.detections import FrameDetections, Detection, VideoDetections
from common.classes.ball import BallDetections, Ball
from common.distances import bbox_iou, bbox_overlap_ratio
from ultralytics import YOLO
import numpy as np
import cv2
from typing import List
from common.classes.player import PlayersDetections, Player
from common.classes.referee import RefereesDetections, Referee
from common.classes.number import NumberDetections, Number
from tqdm import tqdm
from pathlib import Path

from detector.number_recognizer_parseq import recognize_numbers_in_frame

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
    conf_threshold: float = 0.1,
    nms_iou_threshold: float = 0.9,
) -> list[Player]:
    player_detections = [
        d for d in frame_detections.detections if 2 <= d.class_id <= 8 and d.confidence >= conf_threshold
    ]
    player_detections = _nms_detections(player_detections, iou_threshold=nms_iou_threshold)

    player_detections.sort(key=lambda x: -x.confidence)
    player_detections = player_detections[:10]
    return [Player(bbox=d.get_bbox(), player_id=i, confidence=d.confidence) for i, d in enumerate(player_detections)]


def get_video_players_detections(video_detections: VideoDetections, conf_threshold=0.1) -> PlayersDetections:
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


# referee class_id = 9 in model names
REFEREE_CLASS_ID = 9


def get_frame_referee_detections(
    frame_detections: FrameDetections,
    conf_threshold: float = 0.25,
    nms_iou_threshold: float = 0.9,
) -> list[Referee]:
    referee_detections = [
        d for d in frame_detections.detections if d.class_id == REFEREE_CLASS_ID and d.confidence >= conf_threshold
    ]
    referee_detections = _nms_detections(referee_detections, iou_threshold=nms_iou_threshold)
    referee_detections.sort(key=lambda x: -x.confidence)
    referee_detections = referee_detections[:5]
    return [Referee(bbox=d.get_bbox(), confidence=d.confidence) for i, d in enumerate(referee_detections)]


def get_video_referee_detections(
    video_detections: VideoDetections,
    conf_threshold: float = 0.25,
) -> RefereesDetections:
    referees_detections: RefereesDetections = {}
    for frame_detection in video_detections:
        referees_detections[frame_detection.frame_id] = get_frame_referee_detections(
            frame_detection, conf_threshold=conf_threshold
        )
    return referees_detections


def get_frame_number_detections(
    frame_detections: FrameDetections,
    frame: np.ndarray | None = None,
    conf_threshold: float = 0.5,
    nms_iou_threshold: float = 0.9,
    ocr_conf_threshold: float = 0.999,
    save_crops_dir: str | Path | None = None,
) -> list[Number]:
    number_detections = [d for d in frame_detections.detections if d.class_id == 1 and d.confidence >= conf_threshold]
    number_detections = _nms_detections(number_detections, iou_threshold=nms_iou_threshold)
    number_detections = [Number(bbox=d.get_bbox(), confidence=d.confidence) for d in number_detections]
    if frame is not None and number_detections:
        recognize_numbers_in_frame(
            frame,
            number_detections,
            ocr_conf_threshold=ocr_conf_threshold,
            save_crops_dir=save_crops_dir,
            frame_id=frame_detections.frame_id,
        )
    return number_detections


def _bbox_inside(inner: list[int], outer: list[int]) -> bool:
    """True if inner bbox [x1,y1,x2,y2] is entirely inside outer bbox."""
    if len(inner) < 4 or len(outer) < 4:
        return False
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def match_numbers_to_players(
    players_detections: PlayersDetections,
    number_detections: NumberDetections,
    referees_detections: RefereesDetections,
) -> None:
    """
    For each number, assign it to a player only if exactly one player bbox contains it.
    If the number lies inside several players, it is not assigned to anyone.
    If the number is inside any referee bbox, it is not assigned to any player.
    Modifies players in place.
    """
    for frame_id, numbers in number_detections.items():
        players = players_detections.get(frame_id, [])
        referees = referees_detections.get(frame_id, [])
        for player in players:
            player.number = None
        for number in numbers:
            if any(_bbox_inside(number.bbox, r.bbox) for r in referees):
                continue
            containing: list[Player] = []
            for player in players:
                if _bbox_inside(number.bbox, player.bbox):
                    containing.append(player)
            if not containing:
                continue
            if len(containing) > 1:
                continue
            p = containing[0]
            if p.number is not None:
                continue
            p.number = number


def enrich_detections_with_numbers(
    video_path: str,
    video_detections: VideoDetections,
    *,
    player_conf_threshold: float = 0.1,
    referee_conf_threshold: float = 0.25,
    number_conf_threshold: float = 0.25,
    ocr_conf_threshold: float = 0.999,
    save_crops_dir: str | Path | None = None,
) -> tuple[PlayersDetections, RefereesDetections, NumberDetections]:
    """
    Run number recognition on each frame and assign numbers to players.
    Reads video frames, runs OCR on number bboxes via recognize_numbers_in_frame (inside
    get_frame_number_detections), then match_numbers_to_players. Mutates and returns
    players_detections with .number set where a number was assigned.
    """
    players_detections = get_video_players_detections(video_detections, conf_threshold=player_conf_threshold)
    referees_detections = get_video_referee_detections(video_detections, conf_threshold=referee_conf_threshold)
    number_detections: NumberDetections = {}
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        idx = 0
        for i in tqdm(range(frame_count), desc="Enrich numbers"):
            ret, frame = cap.read()
            if not ret:
                break
            if idx == len(video_detections):
                break
            if video_detections[idx].frame_id != i:
                continue

            number_detections[i] = get_frame_number_detections(
                video_detections[idx],
                frame=frame,
                conf_threshold=number_conf_threshold,
                ocr_conf_threshold=ocr_conf_threshold,
                save_crops_dir=save_crops_dir,
            )
            idx += 1
    finally:
        cap.release()
    match_numbers_to_players(players_detections, number_detections, referees_detections)
    return players_detections, referees_detections, number_detections
