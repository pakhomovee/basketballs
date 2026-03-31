"""
Benchmark court homography extraction against manual keypoint + homography JSON.

For each annotated video, runs:
  - ``extract_homographies_from_video_v1``
  - ``extract_homographies_from_video_v2`` with ``smoothing_num_epochs=0``
  - ``extract_homographies_from_video_v2`` (default smoothing)

Metrics (per algorithm, per video; also pooled over all videos):
  1) Sample a 20×20 grid on the GT court plane (normalized [-0.5, 0.5]²).
  2) Keep only points visible in-frame under GT homography.
  3) Project each kept court point to frame via GT inverse homography.
  4) Predict court coordinates from that frame point via model homography.
  5) Measure Euclidean distance in meters between GT and predicted court points.
  Report RMSE, mean, median over all such distances.

Run from ``components/`` with e.g.:
    PYTHONPATH=. python court_detector/benchmark.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_COMPONENTS_ROOT = Path(__file__).resolve().parent.parent
if str(_COMPONENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMPONENTS_ROOT))

from common.classes import CourtType  # noqa: E402
from config import load_default_config  # noqa: E402
from court_detector.annotation_tool import (  # noqa: E402
    FrameAnnotation,
    homography_ransac_from_annotations,
)
from court_detector.court_constants import CourtConstants  # noqa: E402
from court_detector.court_detector import CourtDetector  # noqa: E402

GRID_N = 20
NUM_GRID = GRID_N * GRID_N

ALGO_KEYS = ("v1", "v2_no_smooth", "v2")


@dataclass
class AlgorithmMetrics:
    name: str
    rmse_m: float
    mean_m: float
    median_m: float
    n_samples: int


def _json_float(v: float) -> float | None:
    return float(v) if math.isfinite(float(v)) else None


def _metrics_to_json_dict(m: AlgorithmMetrics) -> dict[str, float | int | str | None]:
    return {
        "name": m.name,
        "rmse_m": _json_float(m.rmse_m),
        "mean_m": _json_float(m.mean_m),
        "median_m": _json_float(m.median_m),
        "n_samples": int(m.n_samples),
    }


def _resolve_gt_homography(fa: FrameAnnotation, court_constants: CourtConstants) -> np.ndarray | None:
    H = fa.homography_numpy()
    if H is not None:
        return np.asarray(H, dtype=np.float64)
    H = homography_ransac_from_annotations(fa, court_constants, fa.width, fa.height)
    return None if H is None else np.asarray(H, dtype=np.float64)


def grid_errors_meters(
    H_gt: np.ndarray,
    H_pred: np.ndarray | None,
    court_size: tuple[float, float],
) -> np.ndarray:
    """
    Distances in meters using:
      GT court point -> frame via inv(H_gt) -> predicted court via H_pred.
    Keep only GT court points visible in-frame under H_gt.
    Both H map normalized frame coords -> normalized court coords.
    """
    if H_pred is None:
        return np.array([], dtype=np.float64)
    H_gt = np.asarray(H_gt, dtype=np.float64)
    H_pred = np.asarray(H_pred, dtype=np.float64)
    try:
        Hg_inv = np.linalg.inv(H_gt)
    except np.linalg.LinAlgError:
        return np.array([], dtype=np.float64)

    u = np.linspace(-0.5, 0.5, GRID_N, dtype=np.float64)
    v = np.linspace(-0.5, 0.5, GRID_N, dtype=np.float64)
    U, V = np.meshgrid(u, v, indexing="xy")
    P = np.stack([U.ravel(), V.ravel(), np.ones(NUM_GRID, dtype=np.float64)], axis=1)

    Qg = (Hg_inv @ P.T).T
    w_g = Qg[:, 2:3]
    w_g = np.where(np.abs(w_g) < 1e-12, 1e-12, w_g)
    q_g = Qg[:, :2] / w_g
    inside = (q_g[:, 0] >= 0.0) & (q_g[:, 0] <= 1.0) & (q_g[:, 1] >= 0.0) & (q_g[:, 1] <= 1.0)
    if not np.any(inside):
        return np.array([], dtype=np.float64)

    P_in = P[inside]
    Q_gt = (Hg_inv @ P_in.T).T
    w_q = Q_gt[:, 2:3]
    w_q = np.where(np.abs(w_q) < 1e-12, 1e-12, w_q)
    q_gt = Q_gt[:, :2] / w_q
    ones = np.ones((q_gt.shape[0], 1), dtype=np.float64)
    q_gt_h = np.hstack([q_gt, ones])

    P_pred = (H_pred @ q_gt_h.T).T
    w_p = P_pred[:, 2:3]
    w_p = np.where(np.abs(w_p) < 1e-12, 1e-12, w_p)
    u_hat = P_pred[:, 0] / w_p[:, 0]
    v_hat = P_pred[:, 1] / w_p[:, 0]

    u0 = P_in[:, 0]
    v0 = P_in[:, 1]
    Wc, Hc = float(court_size[0]), float(court_size[1])
    dx = (u0 - u_hat) * Wc
    dy = (v0 - v_hat) * Hc
    return np.hypot(dx, dy)


def _rmse(vals: np.ndarray) -> float:
    if vals.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(vals))))


def _homography_list_to_numpy(h) -> np.ndarray | None:
    if h is None:
        return None
    if hasattr(h, "detach"):
        h = h.detach().cpu().numpy()
    return np.asarray(h, dtype=np.float64)


def _errors_for_homographies(
    frames_meta: list[dict],
    homographies: list,
    court_constants: CourtConstants,
    court_size: tuple[float, float],
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    n_h = len(homographies)
    for fd in frames_meta:
        fa = FrameAnnotation.from_json_dict(fd)
        H_gt = _resolve_gt_homography(fa, court_constants)
        if H_gt is None:
            continue
        fidx = int(fd["physical_frame"])
        if fidx < 0 or fidx >= n_h:
            continue
        H_pred = _homography_list_to_numpy(homographies[fidx])
        e = grid_errors_meters(H_gt, H_pred, court_size)
        if e.size:
            chunks.append(e)
    if not chunks:
        return np.array([], dtype=np.float64)
    return np.concatenate(chunks)


def benchmark_one_video(
    detector: CourtDetector,
    json_path: Path,
    court_constants: CourtConstants,
) -> dict[str, tuple[AlgorithmMetrics, np.ndarray]]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    video_path = Path(data.get("video_path", ""))
    if not video_path.is_file():
        stem = data.get("video_file", json_path.stem + ".mp4")
        alt = json_path.parent.parent / "videos" / stem
        if alt.is_file():
            video_path = alt
        else:
            raise FileNotFoundError(f"No video for {json_path}: tried {video_path} and {alt}")

    court_size = court_constants.court_size
    frames_meta = data.get("frames", [])
    if not frames_meta:
        raise ValueError(f"No frames in {json_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        homographies_v1, _, _ = detector.extract_homographies_from_video_v1(cap, court_constants)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        homographies_v2_ns, _, _ = detector.extract_homographies_from_video_v2(
            cap, court_constants, smoothing_num_epochs=0
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        homographies_v2, _, _ = detector.extract_homographies_from_video_v2(cap, court_constants)
    finally:
        cap.release()

    out: dict[str, tuple[AlgorithmMetrics, np.ndarray]] = {}
    for key, homogs, label in (
        ("v1", homographies_v1, "v1"),
        ("v2_no_smooth", homographies_v2_ns, "v2_no_smooth"),
        ("v2", homographies_v2, "v2"),
    ):
        all_e = _errors_for_homographies(frames_meta, homogs, court_constants, court_size)
        n = int(all_e.size)
        rmse_m = _rmse(all_e)
        mean_m = float(np.mean(all_e)) if n else float("nan")
        median_m = float(np.median(all_e)) if n else float("nan")
        out[key] = (
            AlgorithmMetrics(
                name=label,
                rmse_m=rmse_m,
                mean_m=mean_m,
                median_m=median_m,
                n_samples=n,
            ),
            all_e,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark homography extraction vs annotated JSON.")
    parser.add_argument(
        "--keypoints-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "annotated" / "keypoints",
        help="Directory with *.json annotations",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Court detector weights (default: from config)",
    )
    parser.add_argument(
        "--results-out",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_results",
        help="Path to write benchmark results JSON (default: court_detector/benchmark_results)",
    )
    args = parser.parse_args()

    keypoints_dir = args.keypoints_dir
    if not keypoints_dir.is_dir():
        print(f"Not a directory: {keypoints_dir}", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(keypoints_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files in {keypoints_dir}", file=sys.stderr)
        sys.exit(1)

    cfg = load_default_config()
    detector = CourtDetector(str(args.model) if args.model else None, cfg=cfg)
    court_constants = CourtConstants(CourtType.NBA)

    # Store every computed distance (meters) across all successful videos.
    pooled: dict[str, list[float]] = {k: [] for k in ALGO_KEYS}
    per_video_results: dict[str, dict[str, dict[str, float | int | str | None]]] = {}
    failed_videos: list[dict[str, str]] = []

    print(f"JSON annotations: {len(json_files)}")
    print(
        "Grid 20×20 on normalized court [-0.5,0.5]²; keep points visible in-frame under GT H; "
        "GT court -> frame via inv(H_gt) -> predicted court via H_pred; "
        "distance = ||court_gt - court_pred|| (meters).\n"
    )

    for jp in json_files:
        try:
            res = benchmark_one_video(detector, jp, court_constants)
        except Exception as ex:
            print(f"{jp.name}: ERROR {ex}")
            failed_videos.append({"json": jp.name, "error": str(ex)})
            continue
        print(f"=== {jp.name} ===")
        per_video_results[jp.name] = {}
        for key in ALGO_KEYS:
            m, arr = res[key]
            print(
                f"  {m.name:16s}  rmse={m.rmse_m:.4f} m  mean={m.mean_m:.4f} m  "
                f"median={m.median_m:.4f} m  n={m.n_samples}"
            )
            per_video_results[jp.name][key] = _metrics_to_json_dict(m)
            if arr.size:
                pooled[key].extend([float(x) for x in arr])
        print()

    print("--- Pooled (all successful videos) ---")
    pooled_results: dict[str, dict[str, float | int | str | None]] = {}
    for key in ALGO_KEYS:
        vals = pooled[key]
        if not vals:
            print(f"  {key:16s}  (no samples)")
            pooled_results[key] = {
                "name": key,
                "rmse_m": None,
                "mean_m": None,
                "median_m": None,
                "n_samples": 0,
            }
            continue
        a = np.asarray(vals, dtype=np.float64)
        pooled_rmse = _rmse(a)
        pooled_mean = float(np.mean(a))
        pooled_median = float(np.median(a))
        print(
            f"  {key:16s}  rmse={pooled_rmse:.4f} m  mean={pooled_mean:.4f} m  median={pooled_median:.4f} m  n={a.size}"
        )
        pooled_results[key] = {
            "name": key,
            "rmse_m": _json_float(pooled_rmse),
            "mean_m": _json_float(pooled_mean),
            "median_m": _json_float(pooled_median),
            "n_samples": int(a.size),
        }

    out_payload = {
        "grid_n": GRID_N,
        "algorithms": list(ALGO_KEYS),
        "total_json_files": len(json_files),
        "successful_videos": len(per_video_results),
        "failed_videos": failed_videos,
        "per_video": per_video_results,
        "pooled": pooled_results,
    }
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_out, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved benchmark results -> {args.results_out}")


if __name__ == "__main__":
    main()
