from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class MainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_path: str | None = None
    output_2d_path: str | None = None
    output_both: str | None = None

    court_type: Literal["nba", "fiba"] = "nba"
    with_pose: bool = False
    enable_smoothing: bool = True
    no_reid: bool = False


class CourtDetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detection_threshold: float = 0.25
    smoothness_cost: float = 200.0
    keypoint_eps: float = 0.003
    smoothing_num_epochs: int = 300
    smoothing_lr: float = 0.001


class BallDetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_threshold: float = 0.3
    max_disp_ratio: float = 1 / 3


class DetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_threshold: float = 0.1
    player_conf_threshold: float = 0.1
    referee_conf_threshold: float = 0.25
    number_conf_threshold: float = 0.25
    ocr_conf_threshold: float = 0.999


class TrackerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_tracks: int = 10
    max_skip: int = 15
    bbox_scale: float = 20.0
    w_spatial: float = 0.2
    w_app: float = 0.6
    w_iou: float = 0.2
    w_extended: float = 1.5
    skip_penalty: float = 1.0
    detection_reward: float = 1.0
    enter_cost: float = 2.0
    exit_cost: float = 2.0
    edge_margin: float = 5.0
    k_warmup_frames: int = 10
    last_frames: int = 10
    n_passes: int = 5
    lookback: int = 5
    oof_entry_cost: float = 0.5
    w_color: float = 0.3
    max_occlusion: int = 45
    occlusion_gate: float = 50.0
    occlusion_penalty: float = 0.1
    occlusion_start_pass: int = 2


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hue_bins: int = 12
    saturation_bins: int = 8
    value_bins: int = 8
    lab_bins: int = 12
    min_saturation_for_hue: int = 40
    torso_center_x: float = 0.5
    torso_center_y: float = 0.35
    torso_sigma_x: float = 0.18
    torso_sigma_y: float = 0.18
    skin_weight: float = 0.15


class TeamClusteringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    embedding: EmbeddingConfig = EmbeddingConfig()


class ModelConfig(BaseModel):
    """Filesystem paths and download URLs for all required model files."""

    model_config = ConfigDict(extra="forbid")

    # Optional override for the models directory (defaults to <repo_root>/models)
    models_dir: str | None = None

    court_detection_filename: str = "court_detection_model.pt"
    court_detection_url: str = "https://disk.yandex.ru/d/VRabl680FfKBog"

    detector_filename: str = "detector_model.pt"
    detector_url: str = "https://disk.yandex.ru/d/KdtL0zaQQlI5fg"

    parseq_filename: str = "parseq_flex.ckpt"
    parseq_url: str = "https://disk.yandex.ru/d/QucoCUmnUbLMHw"

    reid_filename: str = "reid_model.pth"
    reid_url: str = "https://disk.yandex.ru/d/Ak2skkMBdVCqmQ"

    seg_filename: str = "seg-model.pt"
    seg_url: str = "https://disk.yandex.ru/d/dpjzmKkadg-nZg"

    wasb_filename: str = "wasb_basketball_best.pth.tar"
    wasb_url: str = "https://disk.yandex.ru/d/JZQN5HEOKOegog"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    main: MainConfig = MainConfig()
    court_detector: CourtDetectorConfig = CourtDetectorConfig()
    ball_detector: BallDetectorConfig = BallDetectorConfig()
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    team_clustering: TeamClusteringConfig = TeamClusteringConfig()
    models: ModelConfig = ModelConfig()


def load_app_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return AppConfig.model_validate(data)


def load_default_config() -> AppConfig:
    default_path = Path(__file__).resolve().parent / "configs" / "main.yaml"
    return load_app_config(default_path)
