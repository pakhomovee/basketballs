"""
CLI entry point: python -m components.tracking <tracking_dir> [options]
"""

import argparse
import logging

from . import run_benchmark

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Run basketball tracker benchmark")
    parser.add_argument("tracking_dir", help="Directory containing segment folders")
    parser.add_argument("--dist-threshold", type=float, default=6.0)
    parser.add_argument("--max-age", type=int, default=8)
    parser.add_argument("--eval-dist", type=float, default=2.0)
    parser.add_argument(
        "--reid-weights",
        default=None,
        help="Path to ReID model weights (when video is present in entry folder)",
    )
    args = parser.parse_args()

    results = run_benchmark(
        args.tracking_dir,
        distance_threshold=args.dist_threshold,
        max_time_since_update=args.max_age,
        eval_distance=args.eval_dist,
        reid_weights=args.reid_weights,
    )

    print("\n══════════════════════════════════════")
    print("         Aggregate Results")
    print("══════════════════════════════════════")
    print(f"  MOTA:     {results['MOTA']:.4f}")
    print(f"  IDF1:     {results['IDF1']:.4f}")
    print(f"  TP:       {results['TP']}")
    print(f"  FP:       {results['FP']}")
    print(f"  FN:       {results['FN']}")
    print(f"  IDSW:     {results['IDSW']}")
    print(f"  Total GT: {results['total_gt']}")
    print("══════════════════════════════════════")


if __name__ == "__main__":
    main()
