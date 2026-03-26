from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_repo_root))
    sys.path.insert(0, str(_repo_root / "components"))

import numpy as np

from common.utils.utils import get_device
from team_clustering.embedding import PlayerEmbedder
from team_clustering.shared import (
    DEFAULT_SEG_MODEL,
    REPO_ROOT,
    clip_bbox,
    empty_accuracy,
    evaluate_team_split,
    extract_track_embeddings_from_image,
    predict_team_labels,
    read_image,
    resolve_repo_path,
)

DEFAULT_MULTISHOT_ROOT = REPO_ROOT / "dataset" / "team_clustering_benchmark" / "multishot"


def _resolve_track_team_label(track_id: int) -> int:
    return 0 if track_id < 5 else 1


def _yolo_bbox_to_xyxy(parts: list[str], width: int, height: int) -> tuple[int, int, int, int] | None:
    _, x_center, y_center, bbox_width, bbox_height = parts
    bbox = [
        int((float(x_center) - float(bbox_width) / 2) * width),
        int((float(y_center) - float(bbox_height) / 2) * height),
        int((float(x_center) + float(bbox_width) / 2) * width),
        int((float(y_center) + float(bbox_height) / 2) * height),
    ]
    return clip_bbox(bbox, width, height)


def _load_frame_track_bboxes(
    label_path: Path, width: int, height: int
) -> tuple[list[int], list[tuple[int, tuple[int, int, int, int]]]]:
    track_ids: set[int] = set()
    track_bboxes: list[tuple[int, tuple[int, int, int, int]]] = []
    for raw_line in label_path.read_text().splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue
        track_id = int(parts[0])
        track_ids.add(track_id)
        bbox = _yolo_bbox_to_xyxy(parts, width, height)
        if bbox is not None:
            track_bboxes.append((track_id, bbox))
    return sorted(track_ids), track_bboxes


def _load_single_frame_embeddings(image_path: Path, label_path: Path, embedder: PlayerEmbedder):
    image, _ = read_image(image_path)
    height, width = image.shape[:2]
    track_ids, track_bboxes = _load_frame_track_bboxes(label_path, width, height)
    return extract_track_embeddings_from_image(image, track_bboxes, embedder), track_ids


def _load_sample_embeddings(multishot_root: Path, sample_id: str, embedder: PlayerEmbedder):
    image_dir = multishot_root / "images" / sample_id
    label_dir = multishot_root / "labels" / sample_id
    sample_embeddings: dict[int, list[np.ndarray]] = {}
    sample_track_ids: set[int] = set()
    for label_path in sorted(label_dir.glob("*.txt")):
        image, _ = read_image(image_dir / label_path.with_suffix(".jpg").name)
        height, width = image.shape[:2]
        track_ids, track_bboxes = _load_frame_track_bboxes(label_path, width, height)
        sample_track_ids.update(track_ids)
        frame_embeddings = extract_track_embeddings_from_image(image, track_bboxes, embedder)
        for track_id, embeddings in frame_embeddings.items():
            sample_embeddings.setdefault(track_id, []).extend(embeddings)
    return sample_embeddings, sorted(sample_track_ids)


def _sample_ground_truth(track_ids: list[int]) -> list[int]:
    return [_resolve_track_team_label(track_id) for track_id in track_ids]


def _evaluate_sample(multishot_root: Path, sample_id: str, embedder: PlayerEmbedder):
    track_embeddings, track_ids = _load_sample_embeddings(multishot_root, sample_id, embedder)
    if len(track_embeddings) < 2:
        return empty_accuracy(mapping="insufficient_tracks")
    return evaluate_team_split(track_embeddings, track_ids, _sample_ground_truth(track_ids))


def _evaluate_single_frames(multishot_root: Path, sample_id: str, embedder: PlayerEmbedder):
    image_dir = multishot_root / "images" / sample_id
    label_dir = multishot_root / "labels" / sample_id
    metrics = []
    for label_path in sorted(label_dir.glob("*.txt")):
        track_embeddings, track_ids = _load_single_frame_embeddings(
            image_dir / label_path.with_suffix(".jpg").name, label_path, embedder
        )
        if len(track_embeddings) < 2:
            metrics.append(empty_accuracy(mapping="insufficient_tracks"))
            continue
        metrics.append(evaluate_team_split(track_embeddings, track_ids, _sample_ground_truth(track_ids)))
    return metrics


def _summarize_single_shot_metrics(frame_metrics: list[dict[str, int | float | str]]) -> dict[str, int | float]:
    accuracies = [float(metric["accuracy"]) for metric in frame_metrics if int(metric["total"]) > 0]
    return {
        "single_shot_mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "single_shot_median_accuracy": float(np.median(accuracies)) if accuracies else 0.0,
        "single_shot_frames_evaluated": len(accuracies),
        "single_shot_frames_skipped": len(frame_metrics) - len(accuracies),
    }


def _iter_sample_ids(multishot_root: Path) -> list[str]:
    labels_root = multishot_root / "labels"
    return sorted((path.name for path in labels_root.iterdir() if path.is_dir() and path.name.isdigit()), key=int)


def run_multishot_team_clustering_benchmark(
    multishot_root: str | Path = DEFAULT_MULTISHOT_ROOT,
    seg_model: str | Path = DEFAULT_SEG_MODEL,
    limit: int | None = None,
) -> dict[str, int | float]:
    from tqdm.auto import tqdm

    multishot_root = resolve_repo_path(multishot_root)
    sample_ids = _iter_sample_ids(multishot_root)
    if limit is not None:
        sample_ids = sample_ids[:limit]

    embedder = PlayerEmbedder(str(resolve_repo_path(seg_model)), device=get_device())
    single_frame_metrics: list[dict[str, int | float | str]] = []
    results = {
        "accuracy": 0.0,
        "correct": 0,
        "total": 0,
        "samples_evaluated": 0,
        "samples_skipped": 0,
    }

    for sample_id in tqdm(sample_ids, desc="Benchmarking multishot team clustering"):
        single_frame_metrics.extend(_evaluate_single_frames(multishot_root, sample_id, embedder))
        metrics = _evaluate_sample(multishot_root, sample_id, embedder)
        if int(metrics["total"]) == 0:
            results["samples_skipped"] += 1
            continue
        results["samples_evaluated"] += 1
        results["correct"] += int(metrics["correct"])
        results["total"] += int(metrics["total"])

    if results["total"]:
        results["accuracy"] = results["correct"] / results["total"]
    results.update(_summarize_single_shot_metrics(single_frame_metrics))
    return results


def _print_results(results: dict[str, int | float]) -> None:
    print("\n══════════════════════════════════════")
    print("  Multishot Team Clustering Benchmark")
    print("══════════════════════════════════════")
    print(f"  Accuracy:         {results['accuracy']:.4f}")
    print(f"  Correct labels:   {results['correct']}")
    print(f"  Evaluated labels: {results['total']}")
    print(f"  Samples used:     {results['samples_evaluated']}")
    print(f"  Samples skipped:  {results['samples_skipped']}")
    print(f"  Single-shot mean:   {results['single_shot_mean_accuracy']:.4f}")
    print(f"  Single-shot median: {results['single_shot_median_accuracy']:.4f}")
    print(f"  Single-shot frames: {results['single_shot_frames_evaluated']}")
    print("══════════════════════════════════════")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run team clustering benchmark on multishot samples")
    parser.add_argument("multishot_root", nargs="?", default=str(DEFAULT_MULTISHOT_ROOT))
    parser.add_argument("--seg-model", default=str(DEFAULT_SEG_MODEL))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    _print_results(
        run_multishot_team_clustering_benchmark(
            multishot_root=args.multishot_root,
            seg_model=args.seg_model,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
