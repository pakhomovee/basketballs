from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Any

from ball_detector.detector import WASBBallDetector
from court_detector.court_detector import CourtDetector
from detector import Detector, enrich_detections_with_numbers, enrich_players_with_pose, get_video_rim_detections
from detector.enrich import propagate_track_numbers
from actions.ball_possession import BallPossession
from common.classes import CourtType, ShotEvent
from common.utils.models import ensure_models, get_model_paths
from common.utils.utils import get_device
from reidentification import extract_reid_embeddings
from smoother import smooth_detection_coordinates
from team_clustering.embedding import PlayerEmbedder
from team_clustering.team_clustering import TeamClustering
from tracking import FlowTracker
from video_reader import VideoReader
from shot_detector.shot_detector import ShotDetector

from config import AppConfig


TOTAL_STAGES = 11


class StageLogger(Protocol):
    total_stages: int

    def set_stage(self, label: str, stage_idx: int) -> None: ...


class NullStageLogger:
    def __init__(self, total_stages: int):
        self.total_stages = total_stages

    def set_stage(self, label: str, stage_idx: int) -> None:
        return


@dataclass(frozen=True)
class PipelineResult:
    players_detections: Any
    ball_detections: dict[int, list[Any]]
    possession_segments: Any
    pass_events: Any
    shot_events: list[ShotEvent]
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

    with VideoReader(video_path, target_fps=main_cfg.target_fps) as vr:
        video_fps = vr.fps

        stage_logger.set_stage("Detecting players…", 1)
        detector = Detector(model_path=str(paths.detector), conf_threshold=cfg.detector.initial_threshold)
        all_detections = detector.detect_video(vr)
        players_detections, referees_detections, numbers_detections = enrich_detections_with_numbers(
            vr,
            all_detections,
            player_conf_threshold=cfg.detector.player_conf_threshold,
            referee_conf_threshold=cfg.detector.referee_conf_threshold,
            number_conf_threshold=cfg.detector.number_conf_threshold,
            ocr_conf_threshold=cfg.detector.ocr_conf_threshold,
            number_recognizer_n_votes=cfg.detector.number_recognizer_n_votes,
            number_recognizer_vote_mode=cfg.detector.number_recognizer_vote_mode,
        )

        stage_logger.set_stage("Estimating poses…", 2)
        if main_cfg.with_pose:
            enrich_players_with_pose(vr, players_detections)

        stage_logger.set_stage("Detecting court…", 3)
        court_detector = CourtDetector(model_path=str(paths.court_detection), cfg=cfg)
        homographies = court_detector.run(vr, players_detections)

        stage_logger.set_stage("Detecting ball…", 4)
        wasb_detector = WASBBallDetector(weights_path=str(paths.wasb), cfg=cfg)
        sparse_ball = wasb_detector.detect_video(vr)  # dict[int, Ball] (already interpolated)
        sampled_frames = set(players_detections.keys())
        ball_detections = {fid: [b] for fid, b in sparse_ball.items() if fid in sampled_frames}

        stage_logger.set_stage("Detecting shots…", 5)
        rim_by_frame = get_video_rim_detections(all_detections, conf_threshold=cfg.detector.initial_threshold)
        shot_detector = ShotDetector(cfg=cfg)
        shot_events = shot_detector.predict_events(
            sparse_ball,
            rim_by_frame,
            homographies,
            frame_width=float(vr.width),
            frame_height=float(vr.height),
            num_frames=vr.total_frames,
        )

        stage_logger.set_stage("Extracting colour embeddings & masks…", 6)
        PlayerEmbedder(cfg=cfg).extract_player_embeddings(vr, players_detections)

        stage_logger.set_stage("Extracting ReID embeddings…", 7)
        if (not main_cfg.no_reid) and Path(paths.reid).is_file():
            extract_reid_embeddings(vr, players_detections, str(paths.reid), device=get_device())

        stage_logger.set_stage("Running tracker…", 8)
        tracker = FlowTracker(cfg=cfg, frame_width=float(vr.width), fps=video_fps)
        tracker.track(players_detections)

        stage_logger.set_stage("Clustering teams…", 9)
        team_clustering = TeamClustering(cfg=cfg)
        team_clustering.run(players_detections)

        act_cfg = cfg.actions
        propagate_track_numbers(players_detections, min_share=act_cfg.track_number_min_share)

        stage_logger.set_stage("Possession & smoothing…", 10)
        ball_possession = BallPossession()
        ball_possession.run(
            players_detections,
            ball_detections,
            fps=video_fps,
            bbox_expand_ratio=act_cfg.bbox_expand_ratio,
            min_expand_px=act_cfg.min_expand_px,
            other_max_share=act_cfg.other_max_share,
            min_owner_share=act_cfg.min_owner_share,
            max_pass_gap_seconds=act_cfg.max_pass_gap_seconds,
        )
        possession_segments = ball_possession.segments
        pass_events = ball_possession.pass_events

        if main_cfg.enable_smoothing:
            smooth_detection_coordinates(players_detections, cfg=cfg)

        video_meta = {
            "fps": round(video_fps, 2),
            "width": vr.width,
            "height": vr.height,
            "total_frames": vr.total_frames,
            "video_name": Path(video_path).name,
        }

    stage_logger.set_stage("Done", TOTAL_STAGES)
    return PipelineResult(
        players_detections=players_detections,
        ball_detections=ball_detections,
        possession_segments=possession_segments,
        pass_events=pass_events,
        shot_events=shot_events,
        court_type=court_type,
        video_fps=video_fps,
        frame_width=video_meta["width"],
        video_meta=video_meta,
    )
