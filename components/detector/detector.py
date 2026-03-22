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
        model_path: str | Path = Path(__file__).parent.parent.parent / "models" / "detector_model.pt",
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
    # class_id mapping from dataset yaml:
    # 0: ball, 1: number, 2: player, 3: player-dribble, ...
    DRIBBLE_CLASS_ID = 3
    player_detections = [
        d for d in frame_detections.detections if 2 <= d.class_id <= 8 and d.confidence >= conf_threshold
    ]
    player_detections = _nms_detections(player_detections, iou_threshold=nms_iou_threshold)

    player_detections.sort(key=lambda x: -x.confidence)
    player_detections = player_detections[:10]
    return [
        Player(
            bbox=d.get_bbox(),
            player_id=i,
            confidence=d.confidence,
            is_dribble=(d.class_id == DRIBBLE_CLASS_ID),
        )
        for i, d in enumerate(player_detections)
    ]


def get_video_players_detections(video_detections: VideoDetections, conf_threshold=0.1) -> PlayersDetections:
    players_detections: PlayersDetections = {}
    for frame_detection in video_detections:
        players_detections[frame_detection.frame_id] = get_frame_players_detections(
            frame_detection, conf_threshold=conf_threshold
        )
    return players_detections


class PoseDetection:
    """Internal pose detection: bbox + keypoints for one person."""

    def __init__(self, bbox: list[float], keypoints: np.ndarray, confidence: float | None = None):
        self.bbox = bbox  # [x1,y1,x2,y2] float
        self.keypoints = keypoints  # (K,3) float32
        self.confidence = confidence


_POSE_MODEL = None


def _get_pose_model(model_path: str | Path | None = None):
    global _POSE_MODEL
    if _POSE_MODEL is not None:
        return _POSE_MODEL
    repo_root = Path(__file__).resolve().parent.parent.parent
    default_model = repo_root / "models" / "yolov8m-pose.pt"
    path = Path(model_path) if model_path is not None else default_model
    _POSE_MODEL = YOLO(str(path))
    return _POSE_MODEL


def get_frame_pose_detections(
    frame: np.ndarray,
    *,
    model_path: str | Path | None = None,
    conf_threshold: float = 0.15,
) -> list[PoseDetection]:
    """
    Run YOLO-pose on a single frame and return pose detections.
    """
    model = _get_pose_model(model_path=model_path)
    results = model(frame, verbose=False, conf=conf_threshold)[0]
    boxes = getattr(results, "boxes", None)
    keypoints = getattr(results, "keypoints", None)
    if boxes is None or keypoints is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    k_xy = keypoints.xy.cpu().numpy()  # (N,K,2)
    k_conf = keypoints.conf.cpu().numpy()  # (N,K)

    pose_dets: list[PoseDetection] = []
    for i in range(xyxy.shape[0]):
        x1, y1, x2, y2 = xyxy[i].tolist()
        kp_xy = k_xy[i]
        kp_c = k_conf[i][:, None]
        kp = np.concatenate([kp_xy, kp_c], axis=1)
        pose_dets.append(PoseDetection(bbox=[x1, y1, x2, y2], keypoints=kp, confidence=float(confs[i])))
    return pose_dets


def get_frame_ball_detections(
    frame_detections: FrameDetections, conf_threshold=0.2, nms_iou_threshold: float = 0.9
) -> list[Ball]:
    ball_detections = [d for d in frame_detections.detections if d.class_id == 0 and d.confidence >= conf_threshold]
    ball_detections = _nms_detections(ball_detections, iou_threshold=nms_iou_threshold)

    return [Ball(bbox=d.get_bbox(), confidence=d.confidence) for d in ball_detections]


def get_video_ball_detections(video_detections: VideoDetections) -> BallDetections:
    ball_detections: BallDetections = {}
    for frame_detection in video_detections:
        ball_detections[frame_detection.frame_id] = get_frame_ball_detections(frame_detection)
    return ball_detections


# rim class_id = 10 in model names
RIM_CLASS_ID = 10


def get_frame_rim_detections(
    frame_detections: FrameDetections,
    conf_threshold: float = 0.25,
    nms_iou_threshold: float = 0.9,
) -> list[Detection]:
    """Return raw Detection objects for rim (class 10) on a single frame."""
    rim_detections = [
        d for d in frame_detections.detections if d.class_id == RIM_CLASS_ID and d.confidence >= conf_threshold
    ]
    rim_detections = _nms_detections(rim_detections, iou_threshold=nms_iou_threshold)
    return rim_detections


def get_video_rim_detections(
    video_detections: VideoDetections,
    conf_threshold: float = 0.25,
) -> dict[int, list[Detection]]:
    """frame_id -> list of rim detections."""
    rim_by_frame: dict[int, list[Detection]] = {}
    for frame_detection in video_detections:
        rim_by_frame[frame_detection.frame_id] = get_frame_rim_detections(
            frame_detection, conf_threshold=conf_threshold
        )
    return rim_by_frame


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
) -> list[Number]:
    number_detections = [d for d in frame_detections.detections if d.class_id == 1 and d.confidence >= conf_threshold]
    number_detections = _nms_detections(number_detections, iou_threshold=nms_iou_threshold)
    number_detections = [Number(bbox=d.get_bbox(), confidence=d.confidence) for d in number_detections]
    if frame is not None and number_detections:
        recognize_numbers_in_frame(
            frame,
            number_detections,
            ocr_conf_threshold=ocr_conf_threshold,
        )
    return number_detections
