from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.utils.utils import get_device
from team_clustering.shared import (
    DEFAULT_SEG_MODEL,
    evaluate_team_split,
    extract_track_embeddings_from_image,
    read_image,
    resolve_repo_path,
)


def _load_entries(ground_truth_path, limit: int | None = None) -> list[dict]:
    ground_truth_path = resolve_repo_path(ground_truth_path)
    with ground_truth_path.open("r") as file:
        entries = json.load(file)
    return entries[:limit] if limit is not None else entries


def _evaluate_entry(entry: dict, embedder) -> dict[str, int | float | str]:
    image, _ = read_image(entry["image_path"])

    track_ids = list(range(len(entry["bboxes"])))
    track_bboxes = list(zip(track_ids, entry["bboxes"], strict=False))
    track_embeddings = extract_track_embeddings_from_image(image, track_bboxes, embedder)
    return evaluate_team_split(track_embeddings, track_ids, entry["team_labels"])


def run_team_clustering_benchmark(
    ground_truth_path: str | Path,
    seg_model: str | Path = DEFAULT_SEG_MODEL,
    limit: int | None = None,
) -> dict[str, int | float]:
    """Run team clustering benchmark on per-image ground truth annotations."""
    from tqdm.auto import tqdm

    from team_clustering.embedding import PlayerEmbedder

    entries = _load_entries(ground_truth_path, limit=limit)
    embedder = PlayerEmbedder(str(resolve_repo_path(seg_model)), get_device())
    results = {
        "accuracy": 0.0,
        "correct": 0,
        "total": 0,
        "images_evaluated": 0,
        "images_skipped": 0,
    }

    for entry in tqdm(entries, desc="Benchmarking team clustering"):
        metrics = _evaluate_entry(entry, embedder)

        if metrics["total"] == 0:
            results["images_skipped"] += 1
            continue

        results["images_evaluated"] += 1
        results["correct"] += int(metrics["correct"])
        results["total"] += int(metrics["total"])

    if results["total"]:
        results["accuracy"] = results["correct"] / results["total"]

    return results


def _print_results(results: dict[str, int | float]) -> None:
    print("\n══════════════════════════════════════")
    print("     Team Clustering Benchmark")
    print("══════════════════════════════════════")
    print(f"  Accuracy:         {results['accuracy']:.4f}")
    print(f"  Correct labels:   {results['correct']}")
    print(f"  Evaluated labels: {results['total']}")
    print(f"  Images used:      {results['images_evaluated']}")
    print(f"  Images skipped:   {results['images_skipped']}")
    print("══════════════════════════════════════")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run team clustering benchmark from image ground truth")
    parser.add_argument("ground_truth", help="Path to JSON file with image_path, bboxes, and team_labels")
    parser.add_argument("--seg-model", default=str(DEFAULT_SEG_MODEL), help="YOLO segmentation checkpoint")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N entries")
    args = parser.parse_args()

    results = run_team_clustering_benchmark(
        ground_truth_path=args.ground_truth,
        seg_model=args.seg_model,
        limit=args.limit,
    )

    _print_results(results)


if __name__ == "__main__":
    main()
