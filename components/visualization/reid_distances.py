"""
Visualize pairwise ReID embedding distances between players.

Generates a video where each frame shows:
  - Left: original frame with labelled player bboxes
  - Right: two heatmaps — intra-frame distances and cross-frame distances
    (current frame vs previous sampled frame), plus player crops along axes.

Usage (standalone):
    python -m visualization.reid_distances <video_path> [--sample-every 30] [-o output.mp4]
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from common.classes.player import Player, PlayersDetections
from common.distances import cosine_dist


# ── Colours / layout ────────────────────────────────────────────────────────

_CROP_SIZE = 48          # thumbnail size alongside heatmap axes
_CELL_SIZE = 48          # heatmap cell size
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_FONT_THICK = 1
_ID_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
    (255, 128, 0), (128, 0, 255), (0, 255, 128), (255, 0, 128),
]


def _id_color(player_id: int) -> tuple[int, int, int]:
    return _ID_COLORS[player_id % len(_ID_COLORS)]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _crop_player(frame: np.ndarray, player: Player) -> np.ndarray:
    """Extract and resize a player crop from the frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (max(0, int(v)) for v in player.bbox[:4])
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((_CROP_SIZE, _CROP_SIZE, 3), dtype=np.uint8)
    return cv2.resize(crop, (_CROP_SIZE, _CROP_SIZE))


def _distance_matrix(players_a: list[Player], players_b: list[Player]) -> np.ndarray:
    """Compute cosine distance matrix between two lists of players (using reid_embedding)."""
    na, nb = len(players_a), len(players_b)
    mat = np.ones((na, nb), dtype=np.float32)
    for i, pa in enumerate(players_a):
        for j, pb in enumerate(players_b):
            emb_a = pa.reid_embedding if pa.reid_embedding is not None else pa.embedding
            emb_b = pb.reid_embedding if pb.reid_embedding is not None else pb.embedding
            mat[i, j] = cosine_dist(emb_a, emb_b)
    return mat


def _colormap_value(val: float) -> tuple[int, int, int]:
    """Map a distance value in [0, 2] to a BGR color (green=close, red=far)."""
    t = np.clip(val / 1.0, 0.0, 1.0)  # clamp to [0, 1] for practical range
    r = int(255 * t)
    g = int(255 * (1 - t))
    return (0, g, r)  # BGR


def _draw_heatmap(
    dist_mat: np.ndarray,
    crops_row: list[np.ndarray],
    crops_col: list[np.ndarray],
    ids_row: list[int],
    ids_col: list[int],
    title: str = "",
) -> np.ndarray:
    """
    Render a distance-matrix heatmap with player thumbnails along axes.

    Layout:
        [title row                     ]
        [          crop_col_0 crop_col_1 ...]
        [crop_row_0  cell      cell     ...]
        [crop_row_1  cell      cell     ...]
    """
    nr, nc = dist_mat.shape
    if nr == 0 or nc == 0:
        placeholder = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No players", (10, 50), _FONT, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        return placeholder

    title_h = 28
    hmap_w = _CROP_SIZE + nc * _CELL_SIZE
    hmap_h = title_h + _CROP_SIZE + nr * _CELL_SIZE
    canvas = np.zeros((hmap_h, hmap_w, 3), dtype=np.uint8)

    # Title
    cv2.putText(canvas, title, (4, title_h - 8), _FONT, _FONT_SCALE, (220, 220, 220), _FONT_THICK, cv2.LINE_AA)

    # Column crops (top row)
    for j in range(nc):
        x0 = _CROP_SIZE + j * _CELL_SIZE
        crop = cv2.resize(crops_col[j], (_CELL_SIZE, _CROP_SIZE))  # (w, h)
        canvas[title_h : title_h + _CROP_SIZE, x0 : x0 + _CELL_SIZE] = crop
        # Player ID
        cv2.putText(canvas, str(ids_col[j]), (x0 + 2, title_h + _CROP_SIZE - 4),
                     _FONT, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # Row crops (left column) + cells
    for i in range(nr):
        y0 = title_h + _CROP_SIZE + i * _CELL_SIZE
        crop = cv2.resize(crops_row[i], (_CROP_SIZE, _CELL_SIZE))  # (w, h)
        canvas[y0 : y0 + _CELL_SIZE, 0 : _CROP_SIZE] = crop
        cv2.putText(canvas, str(ids_row[i]), (2, y0 + _CELL_SIZE - 4),
                     _FONT, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        for j in range(nc):
            x0 = _CROP_SIZE + j * _CELL_SIZE
            val = dist_mat[i, j]
            color = _colormap_value(val)
            cv2.rectangle(canvas, (x0, y0), (x0 + _CELL_SIZE - 1, y0 + _CELL_SIZE - 1), color, -1)
            # Print distance value
            txt = f"{val:.2f}"
            cv2.putText(canvas, txt, (x0 + 3, y0 + _CELL_SIZE // 2 + 5),
                         _FONT, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            # highlight same-ID matches
            if ids_row[i] >= 0 and ids_row[i] == ids_col[j]:
                cv2.rectangle(canvas, (x0, y0), (x0 + _CELL_SIZE - 1, y0 + _CELL_SIZE - 1),
                               (255, 255, 255), 2)

    return canvas


def _draw_labelled_frame(frame: np.ndarray, players: list[Player]) -> np.ndarray:
    """Draw bboxes with player_id labels on a copy of the frame."""
    vis = frame.copy()
    for p in players:
        if not p.bbox or len(p.bbox) < 4:
            continue
        x1, y1, x2, y2 = map(int, p.bbox[:4])
        color = _id_color(p.player_id)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"#{p.player_id}"
        cv2.putText(vis, label, (x1, y1 - 6), _FONT, 0.5, color, 1, cv2.LINE_AA)
    return vis


# ── Main rendering ───────────────────────────────────────────────────────────

def render_reid_distance_frame(
    frame: np.ndarray,
    players: list[Player],
    prev_frame: np.ndarray | None,
    prev_players: list[Player] | None,
    frame_id: int,
) -> np.ndarray:
    """
    Build a single composite visualization frame.

    Left: labelled video frame.
    Right-top: intra-frame distance heatmap.
    Right-bottom: cross-frame distance heatmap (current vs previous).
    """
    # Filter to players that have embeddings
    players = [p for p in players if (p.reid_embedding is not None or p.embedding is not None)]

    vis_frame = _draw_labelled_frame(frame, players)

    # Intra-frame heatmap
    crops = [_crop_player(frame, p) for p in players]
    ids = [p.player_id for p in players]
    intra_mat = _distance_matrix(players, players)
    hmap_intra = _draw_heatmap(intra_mat, crops, crops, ids, ids, f"Intra frame {frame_id}")

    # Cross-frame heatmap
    if prev_frame is not None and prev_players:
        prev_players_f = [p for p in prev_players if (p.reid_embedding is not None or p.embedding is not None)]
        prev_crops = [_crop_player(prev_frame, p) for p in prev_players_f]
        prev_ids = [p.player_id for p in prev_players_f]
        cross_mat = _distance_matrix(prev_players_f, players)
        hmap_cross = _draw_heatmap(cross_mat, prev_crops, crops, prev_ids, ids, "Cross-frame (prev → curr)")
    else:
        hmap_cross = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.putText(hmap_cross, "No previous frame", (10, 50), _FONT, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    # Stack heatmaps vertically
    max_w = max(hmap_intra.shape[1], hmap_cross.shape[1])

    def _pad_w(img: np.ndarray, target_w: int) -> np.ndarray:
        if img.shape[1] >= target_w:
            return img
        pad = np.zeros((img.shape[0], target_w - img.shape[1], 3), dtype=np.uint8)
        return np.hstack([img, pad])

    hmap_intra = _pad_w(hmap_intra, max_w)
    hmap_cross = _pad_w(hmap_cross, max_w)
    # Separator
    sep = np.full((6, max_w, 3), 60, dtype=np.uint8)
    right_panel = np.vstack([hmap_intra, sep, hmap_cross])

    # Scale left panel and right panel to same height
    fh, fw = vis_frame.shape[:2]
    rh, rw = right_panel.shape[:2]
    target_h = max(fh, rh)

    def _pad_h(img: np.ndarray, target_h: int) -> np.ndarray:
        if img.shape[0] >= target_h:
            return img
        pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
        return np.vstack([img, pad])

    vis_frame = _pad_h(vis_frame, target_h)
    right_panel = _pad_h(right_panel, target_h)

    # Vertical separator
    vsep = np.full((target_h, 4, 3), 60, dtype=np.uint8)
    return np.hstack([vis_frame, vsep, right_panel])


def _compute_canvas_size(
    video_path: str,
    detections: PlayersDetections,
    sample_every: int,
    max_players: int,
) -> tuple[int, int]:
    """
    Return (width, height) for the composite output canvas.

    Uses the actual video frame size for the left panel and derives the
    right panel size from max_players so the canvas is stable across frames.
    """
    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Right panel: intra heatmap height + separator + cross heatmap height + titles
    title_h = 28
    # intra heatmap: title_h + CROP_SIZE rows (col header) + max_players * CELL_SIZE rows
    intra_h = title_h + _CROP_SIZE + max_players * _CELL_SIZE
    cross_h = title_h + _CROP_SIZE + max_players * _CELL_SIZE  # same dimensions (prev → curr)
    sep_h = 6
    right_h = intra_h + sep_h + cross_h

    # Right panel width: CROP_SIZE + max_players * CELL_SIZE
    right_w = _CROP_SIZE + max_players * _CELL_SIZE

    total_h = max(frame_h, right_h)
    # 4 px vertical separator between left and right panels
    total_w = frame_w + 4 + right_w
    return (total_w, total_h)


def write_reid_distance_video(
    video_path: str,
    detections: PlayersDetections,
    output_path: str,
    sample_every: int = 30,
) -> None:
    """
    Write a visualisation video showing ReID pairwise distances.

    Parameters
    ----------
    video_path : str
        Source video.
    detections : PlayersDetections
        Detections with reid_embedding (and player_id from tracking).
    output_path : str
        Where to save the output MP4.
    sample_every : int
        Process every N-th frame.
    """
    from tqdm.auto import tqdm

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    sampled_ids = set(range(0, total_frames, sample_every))

    # Pre-compute stable canvas dimensions from max player count in sampled frames.
    max_players = max(
        (len([p for p in detections.get(fid, []) if p.reid_embedding is not None or p.embedding is not None])
         for fid in sampled_ids),
        default=1,
    )
    max_players = max(max_players, 1)
    target_w, target_h = _compute_canvas_size(video_path, detections, sample_every, max_players)

    out_fps = max(1.0, fps / sample_every)
    # avc1 (H.264) is required on macOS — mp4v produces a green screen.  Fall
    # back to mp4v on other platforms where avc1 may not be available.
    for fourcc_str in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (target_w, target_h))
        if writer.isOpened():
            break
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    prev_frame = None
    prev_players: list[Player] | None = None

    cap = cv2.VideoCapture(video_path)
    for frame_id in tqdm(range(total_frames), desc="Reid distance viz"):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id not in sampled_ids:
            continue

        players = detections.get(frame_id, [])
        composite = render_reid_distance_frame(frame, players, prev_frame, prev_players, frame_id)

        # Pad composite to fixed canvas size so every write is the same dimensions
        ch, cw = composite.shape[:2]
        if ch != target_h or cw != target_w:
            canonical = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            copy_h = min(ch, target_h)
            copy_w = min(cw, target_w)
            canonical[:copy_h, :copy_w] = composite[:copy_h, :copy_w]
            composite = canonical

        writer.write(composite)
        prev_frame = frame
        prev_players = players

    cap.release()
    writer.release()
    print(f"Saved ReID distance video to {output_path}")


# ── Statistics summary ───────────────────────────────────────────────────────

def print_reid_distance_stats(
    detections: PlayersDetections,
    sample_every: int = 10,
) -> None:
    """Print aggregate stats: same-ID vs different-ID distances across sampled pairs."""
    same_id_dists: list[float] = []
    diff_id_dists: list[float] = []

    frames = sorted(detections.keys())
    sampled = frames[::sample_every]

    for idx in range(1, len(sampled)):
        fid_prev, fid_curr = sampled[idx - 1], sampled[idx]
        players_prev = [p for p in detections.get(fid_prev, [])
                        if p.reid_embedding is not None or p.embedding is not None]
        players_curr = [p for p in detections.get(fid_curr, [])
                        if p.reid_embedding is not None or p.embedding is not None]
        if not players_prev or not players_curr:
            continue

        for pa in players_prev:
            for pb in players_curr:
                emb_a = pa.reid_embedding if pa.reid_embedding is not None else pa.embedding
                emb_b = pb.reid_embedding if pb.reid_embedding is not None else pb.embedding
                d = cosine_dist(emb_a, emb_b)
                if pa.player_id >= 0 and pb.player_id >= 0:
                    if pa.player_id == pb.player_id:
                        same_id_dists.append(d)
                    else:
                        diff_id_dists.append(d)

    print("\n=== ReID Distance Statistics ===")
    if same_id_dists:
        arr = np.array(same_id_dists)
        print(f"Same-ID pairs    (n={len(arr):>6d}):  "
              f"mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"median={np.median(arr):.4f}  p90={np.percentile(arr, 90):.4f}")
    else:
        print("Same-ID pairs: none found (tracking not yet run?)")

    if diff_id_dists:
        arr = np.array(diff_id_dists)
        print(f"Diff-ID pairs    (n={len(arr):>6d}):  "
              f"mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"median={np.median(arr):.4f}  p10={np.percentile(arr, 10):.4f}")
    else:
        print("Diff-ID pairs: none found")

    if same_id_dists and diff_id_dists:
        gap = np.mean(diff_id_dists) - np.mean(same_id_dists)
        print(f"Mean gap (diff − same): {gap:.4f}")
        overlap_thresh = np.percentile(same_id_dists, 90)
        frac_diff_below = np.mean(np.array(diff_id_dists) < overlap_thresh)
        print(f"Diff-ID pairs below same-ID 90th pct ({overlap_thresh:.4f}): {frac_diff_below:.1%}")
    print()


# ── CLI entry point ──────────────────────────────────────────────────────────

def _cli():
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from cache import load_detections_cache
    from team_clustering.embedding import DEFAULT_SEG_MODEL, extract_player_embeddings
    from reidentification import extract_reid_embeddings
    from tracking import FlowTracker
    from main import _ensure_default_models, DEFAULT_REID_WEIGHTS, _get_video_frame_width

    parser = argparse.ArgumentParser(description="Visualize ReID pairwise distances")
    parser.add_argument("video_path", help="Input video")
    parser.add_argument("-o", "--output", default=None, help="Output video path (default: <video>_reid_viz.mp4)")
    parser.add_argument("--sample-every", type=int, default=30, help="Sample every N-th frame")
    parser.add_argument("--stats-only", action="store_true", help="Only print distance statistics, no video")
    parser.add_argument("--no-cache", action="store_true", help="Ignore detection cache")
    args = parser.parse_args()

    _ensure_default_models()
    from common.utils.utils import get_device

    video_path = args.video_path

    # Load or compute detections
    detections = None
    if not args.no_cache:
        detections = load_detections_cache(video_path, DEFAULT_SEG_MODEL)
    if detections is None:
        from detector import Detector, enrich_detections_with_numbers
        from court_detector.court_detector import CourtDetector
        from common.classes import CourtType

        print("No cache found — running detection pipeline...")
        detector = Detector()
        all_dets = detector.detect_video(video_path)
        detections, _, _ = enrich_detections_with_numbers(video_path, all_dets)
        court = CourtDetector()
        court.run(video_path, detections, CourtType.NBA)

    # Extract embeddings
    extract_player_embeddings(video_path, detections)
    device = get_device()
    extract_reid_embeddings(video_path, detections, DEFAULT_REID_WEIGHTS, device=device)

    # Run tracker to get player_ids
    frame_width = _get_video_frame_width(video_path)
    tracker = FlowTracker(num_tracks=10, frame_width=frame_width)
    tracker.track(detections)

    # Print stats
    print_reid_distance_stats(detections, sample_every=args.sample_every)

    if args.stats_only:
        return

    # Write video
    output = args.output
    if output is None:
        output = str(Path(video_path).with_suffix("")) + "_reid_viz.mp4"

    write_reid_distance_video(video_path, detections, output, sample_every=args.sample_every)


if __name__ == "__main__":
    _cli()
