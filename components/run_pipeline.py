from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Any

import cv2

from ball_detector.detector import WASBBallDetector
from court_detector.court_detector import CourtDetector
from detector import Detector, enrich_detections_with_numbers, enrich_players_with_pose
from actions.ball_possession import (
    assign_ball_possession_soft_dribble,
    apply_possession_segments,
    greedy_possession_segments_soft_dribble,
)
from actions.passes import find_team_passes
from common.classes import CourtType
from common.utils.models import ensure_models, get_model_paths
from common.utils.utils import get_device
from reidentification import extract_reid_embeddings
from smoother import smooth_detection_coordinates
from team_clustering.embedding import PlayerEmbedder
from team_clustering.team_clustering import TeamClustering
from tracking import FlowTracker

from config import AppConfig


TOTAL_STAGES = 10


class StageLogger(Protocol):
    total_stages: int

    def set_stage(self, label: str, stage_idx: int) -> None: ...


class NullStageLogger:
    def __init__(self, total_stages: int):
        self.total_stages = total_stages

    def set_stage(self, label: str, stage_idx: int) -> None:
        return


def _get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 25.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if fps > 0 else 25.0


def _get_video_frame_width(video_path: str) -> float | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return width if width > 0 else None


def _get_video_meta(video_path: str) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"fps": 25.0, "width": 0, "height": 0, "total_frames": 0, "video_name": Path(video_path).name}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "video_name": Path(video_path).name,
    }


@dataclass(frozen=True)
class PipelineResult:
    players_detections: Any
    ball_detections: dict[int, list[Any]]
    possession_segments: Any
    pass_events: Any
    court_type: CourtType
    video_fps: float
    frame_width: float | None
    video_meta: dict[str, Any]


def run_pipeline(
    video_path: str,
    cfg: AppConfig,
    stage_logger: Any = None,
) -> PipelineResult:
    """
    Run the full analytics pipeline (shared by CLI and web backend).

    This function follows the orchestration in `components/main.py` and reports
    coarse stage progress via `stage_logger`.
    """
    if stage_logger is None:
        stage_logger = NullStageLogger(total_stages=TOTAL_STAGES)

    stage_logger.set_stage("Checking models…", 0)
    ensure_models(cfg)
    paths = get_model_paths(cfg)

    main_cfg = cfg.main
    court_type = CourtType.NBA if main_cfg.court_type == "nba" else CourtType.FIBA

    stage_logger.set_stage("Detecting players…", 1)
    detector = Detector(model_path=str(paths.detector), conf_threshold=cfg.detector.initial_threshold)
    all_detections = detector.detect_video(video_path)
    players_detections, referees_detections, numbers_detections = enrich_detections_with_numbers(
        video_path,
        all_detections,
        player_conf_threshold=cfg.detector.player_conf_threshold,
        referee_conf_threshold=cfg.detector.referee_conf_threshold,
        number_conf_threshold=cfg.detector.number_conf_threshold,
        ocr_conf_threshold=cfg.detector.ocr_conf_threshold,
    )

    stage_logger.set_stage("Estimating poses…", 2)
    if main_cfg.with_pose:
        enrich_players_with_pose(video_path, players_detections)

    stage_logger.set_stage("Detecting court…", 3)
    court_detector = CourtDetector(model_path=str(paths.court_detection), cfg=cfg)
    court_detector.run(video_path, players_detections)

    stage_logger.set_stage("Detecting ball…", 4)
    wasb_detector = WASBBallDetector(weights_path=str(paths.wasb), cfg=cfg)
    sparse_ball = wasb_detector.detect_video(video_path)  # dict[int, Ball] (already interpolated)
    ball_detections = {fid: [b] for fid, b in sparse_ball.items()}

    stage_logger.set_stage("Extracting colour embeddings & masks…", 5)
    PlayerEmbedder(cfg=cfg).extract_player_embeddings(video_path, players_detections)

    stage_logger.set_stage("Extracting ReID embeddings…", 6)
    if (not main_cfg.no_reid) and Path(paths.reid).is_file():
        extract_reid_embeddings(video_path, players_detections, str(paths.reid), device=get_device())

    stage_logger.set_stage("Running tracker…", 7)
    video_fps = _get_video_fps(video_path)
    frame_width = _get_video_frame_width(video_path)
    tracker = FlowTracker(cfg=cfg, frame_width=frame_width, fps=video_fps)
    tracker.track(players_detections)

    stage_logger.set_stage("Clustering teams…", 8)
    team_clustering = TeamClustering(cfg=cfg)
    team_clustering.run(players_detections)

    stage_logger.set_stage("Possession & smoothing…", 9)
    assign_ball_possession_soft_dribble(players_detections, ball_detections)
    possession_segments = greedy_possession_segments_soft_dribble(players_detections, fps=video_fps)
    apply_possession_segments(players_detections, possession_segments)
    pass_events = find_team_passes(possession_segments, players_detections)

    if main_cfg.enable_smoothing:
        smooth_detection_coordinates(players_detections)

    video_meta = _get_video_meta(video_path)
    stage_logger.set_stage("Done", TOTAL_STAGES)
    return PipelineResult(
        players_detections=players_detections,
        ball_detections=ball_detections,
        possession_segments=possession_segments,
        pass_events=pass_events,
        court_type=court_type,
        video_fps=video_fps,
        frame_width=frame_width,
        video_meta=video_meta,
    )
