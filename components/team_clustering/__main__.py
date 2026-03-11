"""
CLI for team clustering benchmark: generate, annotate, evaluate.

Usage:
    python -m components.team_clustering benchmark [options]   # Generate baseline predictions
    python -m components.team_clustering annotate [options]    # Fix labels, save ground truth
    python -m components.team_clustering evaluate [options]    # Compare to ground truth
"""

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("benchmark", "annotate", "evaluate", "visualize"):
        print("Usage: python -m components.team_clustering <benchmark|annotate|evaluate|visualize> [options]")
        sys.exit(1)

    cmd = sys.argv[1]
    sys.argv = ["team_clustering_" + cmd] + sys.argv[2:]

    if cmd == "benchmark":
        from .benchmark import main as _main
    elif cmd == "annotate":
        from .annotate import main as _main
    elif cmd == "visualize":
        from .visualize_eval import main as _main
    else:
        from .evaluate import main as _main

    _main()


if __name__ == "__main__":
    main()
