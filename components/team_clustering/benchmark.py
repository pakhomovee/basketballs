"""
Team clustering benchmark: generate baseline predictions on images from the dataset.

Usage:
    python -m components.team_clustering.benchmark [--max-images N]

Output: JSON file with image_path, bboxes, baseline team labels for each detection.
Use the annotation tool to correct labels and create ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from tqdm.auto import tqdm

from common.classes.detections import FrameDetections
from common.utils.utils import download
from detector.detector import Detector, get_frame_players_detections
from team_clustering.embedding import PlayerEmbedder, extract_embeddings_from_image
from team_clustering.team_clustering import TeamClustering


def run_benchmark(
    images_dir: str | Path,
    output_path: str | Path,
    max_images: int = 200,
) -> None:
    """
    Run detector + team clustering on images and save baseline predictions.

    Args:
        images_dir: Directory containing images.
        output_path: Path to save predictions JSON (input for annotation tool).
        max_images: Maximum number of images to process.
    """
    images_dir = Path(images_dir)
    output_path = Path(output_path)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in image_extensions
    )[:max_images]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    _root = Path(__file__).resolve().parent.parent.parent
    models_dir = _root / "models"
    default_model = models_dir / "best-4.pt"
    if not default_model.exists():
        download("https://disk.yandex.ru/d/MAGAbYxRFEvX6w", "best-4.pt", str(models_dir))
    detector = Detector(model_path=default_model, conf_threshold=0.25)
    embedder = PlayerEmbedder(device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    tc = TeamClustering(n_clusters=2)

    results: list[dict] = []

    for img_path in tqdm(image_paths, desc="Benchmark"):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        dets = detector.detect_frame(frame)
        frame_dets = FrameDetections(0, dets)
        players = get_frame_players_detections(frame_dets, conf_threshold=0.25)

        if len(players) < 2:
            continue

        extract_embeddings_from_image(frame, players, embedder=embedder)
        clusters = tc.cluster_per_frame(players)

        bboxes = [p.bbox for p in players]
        team_labels: list[int | None] = [None] * len(players)
        for idx, label in clusters.items():
            team_labels[idx] = label

        try:
            rel_path = img_path.resolve().relative_to(_root)
        except ValueError:
            rel_path = img_path
        results.append({
            "image_path": str(rel_path).replace("\\", "/"),
            "bboxes": bboxes,
            "team_labels": team_labels,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} image predictions to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate team clustering baseline predictions")
    parser.add_argument("--max-images", type=int, default=200, help="Max images to process")
    args = parser.parse_args()

    _root = Path(__file__).resolve().parent.parent.parent
    run_benchmark(
        images_dir=_root / "dataset" / "basketball-player-detection-2" / "train" / "images",
        output_path=_root / "dataset" / "team_clustering_benchmark" / "predictions.json",
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
