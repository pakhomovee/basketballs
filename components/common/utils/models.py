"""Model path resolution and automatic download utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from common.utils.utils import download

if TYPE_CHECKING:
    from config import AppConfig


@dataclass
class ModelPaths:
    court_detection: Path
    detector: Path
    parseq: Path
    reid: Path
    seg: Path
    wasb: Path


def get_models_dir(cfg: "AppConfig") -> Path:
    """Return the resolved models directory."""
    if cfg.models.models_dir:
        return Path(cfg.models.models_dir)
    # components/common/utils/models.py → parents[3] = repo root
    return Path(__file__).resolve().parents[3] / "models"


def get_model_paths(cfg: "AppConfig") -> ModelPaths:
    """Return typed paths for all model files."""
    d = get_models_dir(cfg)
    m = cfg.models
    return ModelPaths(
        court_detection=d / m.court_detection_filename,
        detector=d / m.detector_filename,
        parseq=d / m.parseq_filename,
        reid=d / m.reid_filename,
        seg=d / m.seg_filename,
        wasb=d / m.wasb_filename,
    )


def ensure_models(cfg: "AppConfig") -> None:
    """Download any missing model files to the configured models directory."""
    models_dir = get_models_dir(cfg)
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = get_model_paths(cfg)
    m = cfg.models
    for path, url in [
        (paths.court_detection, m.court_detection_url),
        (paths.detector, m.detector_url),
        (paths.parseq, m.parseq_url),
        (paths.reid, m.reid_url),
        (paths.seg, m.seg_url),
        (paths.wasb, m.wasb_url),
    ]:
        if not path.exists():
            download(url, path.name, str(models_dir))
