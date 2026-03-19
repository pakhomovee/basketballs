from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from team_clustering.team_clustering import _cluster_track_embeddings

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEG_MODEL = REPO_ROOT / "models" / "seg-model.pt"


def empty_accuracy(mapping: str = "none") -> dict[str, int | float | str]:
    return {
        "accuracy": 0.0,
        "correct": 0,
        "total": 0,
        "mapping": mapping,
    }


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (REPO_ROOT / path).resolve()


def clip_bbox(bbox: list[int] | tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    return x1, y1, x2, y2


def read_image(image_path: str | Path):
    image_path = resolve_repo_path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image, image_path


def compute_team_split_accuracy(
    predicted_labels: list[int | None],
    ground_truth_labels: list[int | str],
) -> dict[str, int | float | str]:
    comparable_pairs = [
        (predicted, int(target)) for predicted, target in zip(predicted_labels, ground_truth_labels) if target in (0, 1)
    ]
    if not comparable_pairs:
        return empty_accuracy()

    direct = sum(predicted == target for predicted, target in comparable_pairs)
    swapped = sum(predicted in (0, 1) and 1 - predicted == target for predicted, target in comparable_pairs)
    correct = max(direct, swapped)
    mapping = "swapped" if swapped > direct else "direct"

    return {
        "accuracy": correct / len(comparable_pairs),
        "correct": correct,
        "total": len(comparable_pairs),
        "mapping": mapping,
    }


def extract_track_embeddings_from_image(
    image: np.ndarray,
    track_bboxes: list[tuple[int, list[int] | tuple[int, int, int, int]]],
    embedder,
) -> dict[int, list[np.ndarray]]:
    valid_entries: list[tuple[int, tuple[int, int, int, int]]] = []
    height, width = image.shape[:2]

    for track_id, bbox in track_bboxes:
        clipped_bbox = clip_bbox(bbox, width, height)
        if clipped_bbox is None:
            continue

        valid_entries.append((track_id, clipped_bbox))

    if not valid_entries:
        return {}

    masks = embedder.get_player_masks_for_frame(image, [bbox for _, bbox in valid_entries])
    track_embeddings: dict[int, list[np.ndarray]] = {}
    for (track_id, (x1, y1, x2, y2)), mask in zip(valid_entries, masks):
        crop = image[y1:y2, x1:x2]
        embedding = embedder.extract_embedding(crop, mask).astype(np.float32)
        track_embeddings.setdefault(track_id, []).append(embedding)
    return track_embeddings


def predict_team_labels(
    track_embeddings: dict[int, list[np.ndarray]],
    track_ids: list[int],
) -> list[int | None]:
    clusters = _cluster_track_embeddings(track_embeddings, n_clusters=2)
    return [clusters.get(track_id) for track_id in track_ids]


def evaluate_team_split(
    track_embeddings: dict[int, list[np.ndarray]],
    track_ids: list[int],
    ground_truth_labels: list[int | str],
) -> dict[str, int | float | str]:
    predicted_labels = predict_team_labels(track_embeddings, track_ids)
    return compute_team_split_accuracy(predicted_labels, ground_truth_labels)
