"""Segmentation model training"""

from __future__ import annotations

import sys
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_DIR = REPO_ROOT / "output" / "segmentation"
DEFAULT_DATASET_YAML = REPO_ROOT / "dataset" / "basketball_player_segmentation" / "data.yaml"


def train_segmentation_model(
    dataset_yaml: str | Path,
    model: str = "yolo26s-seg.pt",
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 16,
    device: str | int | None = None,
    project: str | Path | None = None,
    name: str = "basketball-player-seg-yolo26s",
    workers: int = 8,
    patience: int = 20,
) -> None:
    """Fine-tune a segmentation checkpoint on the basketball player dataset."""
    project_dir = Path(project or DEFAULT_PROJECT_DIR).resolve()
    dataset_yaml = Path(dataset_yaml).resolve()

    trainer = YOLO(model)
    trainer.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        project=str(project_dir),
        name=name,
    )


def main(dataset_yaml: str | Path | None = None) -> None:
    train_segmentation_model(
        dataset_yaml=dataset_yaml or DEFAULT_DATASET_YAML,
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
