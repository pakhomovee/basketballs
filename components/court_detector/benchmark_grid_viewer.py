"""
Viewer for annotated frames with the benchmark 20×20 grid drawn as red points.

The benchmark samples a 20×20 grid on the *court plane* in normalized coordinates
[-0.5, 0.5]², then maps those points to frame space via the inverse of the GT
homography (court -> frame) and keeps only points visible in-frame.

This tool lets you step through annotated frames forward/backward and also move
between different annotated videos.

Run from repo `components/` root, e.g.:
    PYTHONPATH=. python court_detector/benchmark_grid_viewer.py
    PYTHONPATH=. python court_detector/benchmark_grid_viewer.py --video nba_long_01.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Allow running as script without installing package
_COMPONENTS_ROOT = Path(__file__).resolve().parent.parent
if str(_COMPONENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMPONENTS_ROOT))

from common.classes import CourtType  # noqa: E402
from court_detector.annotation_tool import (  # noqa: E402
    FrameAnnotation,
    homography_ransac_from_annotations,
)
from court_detector.court_constants import CourtConstants  # noqa: E402
from video_reader import VideoReader  # noqa: E402

WINDOW_NAME = "benchmark grid viewer (20x20)"
GRID_N = 20
NUM_GRID = GRID_N * GRID_N

# Arrow keys from cv2.waitKeyEx() (GTK/Qt)
_KEY_ARROW_LEFT = 65361
_KEY_ARROW_UP = 65362
_KEY_ARROW_RIGHT = 65363
_KEY_ARROW_DOWN = 65364
# Some builds / platforms use these (waitKeyEx high word).
_WIN_ARROW_LEFT = 2424832
_WIN_ARROW_UP = 2555904
_WIN_ARROW_RIGHT = 2621440
_WIN_ARROW_DOWN = 2690048


def _arrow_delta(key_raw: int) -> int | None:
    if key_raw in (_KEY_ARROW_LEFT, _KEY_ARROW_UP, _WIN_ARROW_LEFT, _WIN_ARROW_UP):
        return -1
    if key_raw in (_KEY_ARROW_RIGHT, _KEY_ARROW_DOWN, _WIN_ARROW_RIGHT, _WIN_ARROW_DOWN):
        return 1
    return None


def _resolve_video_path(payload: dict, json_path: Path, videos_dir: Path) -> Path:
    vp = Path(str(payload.get("video_path", "")))
    if vp.is_file():
        return vp
    stem = str(payload.get("video_file", "")) or f"{json_path.stem}.mp4"
    cand = videos_dir / stem
    if cand.is_file():
        return cand
    # fallback: sibling annotated/videos (matching benchmark.py behavior)
    alt = json_path.parent.parent / "videos" / stem
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"Video not found for {json_path.name}: tried {vp}, {cand}, {alt}")


def _sorted_frame_dicts(payload: dict) -> list[dict]:
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        return []
    out = [fd for fd in frames if isinstance(fd, dict) and "physical_frame" in fd]
    out.sort(key=lambda d: int(d.get("physical_frame", 0)))
    return out


def _grid_points_court_norm() -> np.ndarray:
    u = np.linspace(-0.5, 0.5, GRID_N, dtype=np.float64)
    v = np.linspace(-0.5, 0.5, GRID_N, dtype=np.float64)
    U, V = np.meshgrid(u, v, indexing="xy")
    P = np.stack([U.ravel(), V.ravel(), np.ones(NUM_GRID, dtype=np.float64)], axis=1)
    return P


def _project_court_grid_to_frame_px(
    H_frame_to_court: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """
    Return (M,2) float pixel coords for grid points that land inside the frame.
    H maps frame_norm -> court_norm, so we use inv(H) to map court_norm -> frame_norm.
    """
    if frame_w <= 0 or frame_h <= 0:
        return np.empty((0, 2), dtype=np.float64)
    H = np.asarray(H_frame_to_court, dtype=np.float64)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.empty((0, 2), dtype=np.float64)
    P = _grid_points_court_norm()  # (400,3)
    Q = (H_inv @ P.T).T
    w = Q[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    q = Q[:, :2] / w  # frame_norm
    inside = (q[:, 0] >= 0.0) & (q[:, 0] <= 1.0) & (q[:, 1] >= 0.0) & (q[:, 1] <= 1.0)
    if not np.any(inside):
        return np.empty((0, 2), dtype=np.float64)
    q_in = q[inside]
    px = q_in * np.array([[float(frame_w), float(frame_h)]], dtype=np.float64)
    return px


def _draw_help(img: np.ndarray, lines: list[str]) -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 22 * len(lines) - 10), (w, h), (20, 20, 20), -1)
    y0 = h - 10 - 22 * len(lines) + 16
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, y0 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)


@dataclass
class VideoItem:
    json_path: Path
    video_path: Path
    frames_meta: list[dict]


def _collect_items(keypoints_dir: Path, videos_dir: Path, video_name: str | None) -> list[VideoItem]:
    if not keypoints_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {keypoints_dir}")
    json_files = sorted(keypoints_dir.glob("*.json"))
    items: list[VideoItem] = []
    for jp in json_files:
        with open(jp, encoding="utf-8") as f:
            payload = json.load(f)
        vp = _resolve_video_path(payload, jp, videos_dir)
        if video_name and vp.name != video_name:
            continue
        frames_meta = _sorted_frame_dicts(payload)
        if not frames_meta:
            continue
        items.append(VideoItem(json_path=jp, video_path=vp, frames_meta=frames_meta))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Show annotated frames with benchmark 20x20 grid points.")
    parser.add_argument(
        "--keypoints-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "annotated" / "keypoints",
        help="Directory with *.json annotations",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "annotated" / "videos",
        help="Directory with input videos (used when JSON lacks an absolute video_path)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Only show this video file name (e.g. nba_long_01.mp4)",
    )
    parser.add_argument(
        "--court-type",
        type=str,
        default="nba",
        choices=("nba", "fiba"),
        help="Court type for homography recomputation when JSON lacks homography",
    )
    args = parser.parse_args()

    court_type = CourtType.NBA if args.court_type.lower() == "nba" else CourtType.FIBA
    court_constants = CourtConstants(court_type)

    items = _collect_items(args.keypoints_dir, args.videos_dir, args.video)
    if not items:
        print("No annotated videos found for the given filters.", file=sys.stderr)
        sys.exit(1)

    help_lines = [
        "←/→ or ↑/↓ (or p/n) : prev/next annotated frame",
        "a/d : prev/next video    r : recompute homography from keypoints (RANSAC) for this frame",
        "q / ESC : quit",
    ]

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    vid_i = 0
    frame_i = 0
    vr: VideoReader | None = None

    def open_video(i: int) -> VideoReader:
        nonlocal vr
        if vr is not None:
            vr.release()
        vr = VideoReader(str(items[i].video_path), target_fps=30)
        return vr

    open_video(vid_i)

    while True:
        item = items[vid_i]
        frame_i = max(0, min(frame_i, len(item.frames_meta) - 1))
        fd = item.frames_meta[frame_i]
        fa = FrameAnnotation.from_json_dict(fd)

        assert vr is not None
        logical = vr.nearest_logical_for_physical(int(fa.physical_frame))
        vr.set(cv2.CAP_PROP_POS_FRAMES, float(logical))
        ret, frame_bgr = vr.read()
        if not ret or frame_bgr is None:
            # skip unreadable frames
            frame_i = min(frame_i + 1, len(item.frames_meta) - 1)
            continue

        H = fa.homography_numpy()
        if H is None:
            H = homography_ransac_from_annotations(fa, court_constants, int(fa.width), int(fa.height))

        vis = frame_bgr.copy()
        if H is not None:
            pts_px = _project_court_grid_to_frame_px(H, vis.shape[1], vis.shape[0])
            for x, y in pts_px:
                cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1, cv2.LINE_AA)

        title = (
            f"{item.video_path.name} | {item.json_path.name} | "
            f"frame {frame_i + 1}/{len(item.frames_meta)} phys={fa.physical_frame} log={logical} | "
            f"grid={GRID_N}x{GRID_N} ({NUM_GRID})"
        )
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(vis, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        _draw_help(vis, help_lines)

        cv2.imshow(WINDOW_NAME, vis)
        k_raw = cv2.waitKeyEx(0)
        key = k_raw & 0xFF

        if key in (27, ord("q")):
            break

        if (delta := _arrow_delta(k_raw)) is not None:
            frame_i = max(0, min(frame_i + delta, len(item.frames_meta) - 1))
            continue
        if key == ord("n"):
            frame_i = min(frame_i + 1, len(item.frames_meta) - 1)
            continue
        if key == ord("p"):
            frame_i = max(frame_i - 1, 0)
            continue

        if key == ord("d"):
            vid_i = (vid_i + 1) % len(items)
            frame_i = 0
            open_video(vid_i)
            continue
        if key == ord("a"):
            vid_i = (vid_i - 1) % len(items)
            frame_i = 0
            open_video(vid_i)
            continue

        if key == ord("r"):
            # Recompute for current display only (does not write JSON).
            # Next loop iteration will recompute again; this key is mainly for parity with the annotation tool.
            pass

    if vr is not None:
        vr.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
