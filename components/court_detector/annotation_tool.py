"""
Interactive tool to annotate 33 court keypoints (indices 0..32) on sampled video frames.

Videos are read from ``court_detector/annotated/videos``. Frames are sampled every
``--interval`` seconds on the *source* timeline (native FPS from the file), then
loaded via :class:`video_reader.VideoReader` using ``nearest_logical_for_physical``.

By default, new frames are prefilled with :class:`court_detector.court_detector.CourtDetector`
predictions (best confidence per class). Use ``--no-model`` to start from empty keypoints.

On each frame, the NBA court texture (``nba.png``) is alpha-blended using the homography
from :func:`court_detector.homography.find_homography_ransac` fitted to the current
annotations (same normalized convention as ``extract_homographies_from_video_v2``).

If ``--video`` is omitted, the tool walks all videos in ``annotated/videos`` in sorted
order and opens only those that are not yet fully annotated (see ``is_annotation_complete``).

Controls (main window must be focused):
- Left click: place active keypoint (image coordinates).
- Right click: clear keypoint nearest to cursor (within ~12 px in display space).
- [ / ] : previous / next keypoint index (0..32).
- 0-9 : activate keypoint 0-9; for 10-32 use keys or [/].
- c : clear active keypoint; x : clear all keypoints on this frame.
- s : save JSON to ``--output`` (autosave also after moving to another frame).
- n / p : next / previous *scheduled* sample (from the every-N-seconds list).
- arrow keys : shift current frame by ±1 on the *source* timeline (←/↑ earlier, →/↓ later).
- q / ESC : quit.

Run from repo ``components/`` root, e.g.:
    PYTHONPATH=. python court_detector/annotation_tool.py
    PYTHONPATH=. python court_detector/annotation_tool.py --video nba_long_01.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Allow running as script without installing package
_COMPONENTS_ROOT = Path(__file__).resolve().parent.parent
if str(_COMPONENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMPONENTS_ROOT))

from video_reader import VideoReader  # noqa: E402

from common.classes import CourtType  # noqa: E402
from config import load_default_config  # noqa: E402
from court_detector.court_constants import CourtConstants  # noqa: E402
from court_detector.court_detector import CourtDetector, project_homography  # noqa: E402
from court_detector.homography import find_homography_ransac  # noqa: E402

NUM_KEYPOINTS = 33
WINDOW_NAME = "court keypoints (0-32)"
NEAR_PX_DISPLAY = 14
COURT_OVERLAY_ALPHA = 0.5

# Arrow keys from cv2.waitKeyEx() (GTK/Qt); ←/↑ step one physical frame earlier, →/↓ later.
_KEY_ARROW_LEFT = 65361
_KEY_ARROW_UP = 65362
_KEY_ARROW_RIGHT = 65363
_KEY_ARROW_DOWN = 65364
# Some builds / platforms use these (waitKeyEx high word).
_WIN_ARROW_LEFT = 2424832
_WIN_ARROW_UP = 2555904
_WIN_ARROW_RIGHT = 2621440
_WIN_ARROW_DOWN = 2690048
_KEY_BRACKET_LEFT_CODES = {ord("["), ord("{"), 219}
_KEY_BRACKET_RIGHT_CODES = {ord("]"), ord("}"), 221}


def _clamp_physical_index(src_frame_count: int, p: int) -> int:
    if src_frame_count <= 0:
        return max(0, p)
    return max(0, min(int(p), src_frame_count - 1))


def _arrow_frame_delta(key: int) -> int | None:
    if key in (
        _KEY_ARROW_LEFT,
        _KEY_ARROW_UP,
        _WIN_ARROW_LEFT,
        _WIN_ARROW_UP,
    ):
        return -1
    if key in (
        _KEY_ARROW_RIGHT,
        _KEY_ARROW_DOWN,
        _WIN_ARROW_RIGHT,
        _WIN_ARROW_DOWN,
    ):
        return 1
    return None


def _is_prev_kp_key(k_raw: int, key_low: int) -> bool:
    return k_raw in _KEY_BRACKET_LEFT_CODES or key_low in _KEY_BRACKET_LEFT_CODES


def _is_next_kp_key(k_raw: int, key_low: int) -> bool:
    return k_raw in _KEY_BRACKET_RIGHT_CODES or key_low in _KEY_BRACKET_RIGHT_CODES


def _hsv_color(i: int) -> tuple[int, int, int]:
    h = int(180 * (i / max(NUM_KEYPOINTS, 1))) % 180
    return cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()


def physical_frames_every_interval(src_fps: float, total_physical: int, interval_sec: float) -> list[int]:
    if total_physical <= 0:
        return []
    fps = src_fps if src_fps > 1e-6 else 25.0
    step = max(1, int(round(interval_sec * fps)))
    return list(range(0, total_physical, step))


def default_keypoints_json_path(video_path: Path) -> Path:
    return Path(__file__).resolve().parent / "annotated" / "keypoints" / f"{video_path.stem}.json"


def is_annotation_complete(json_path: Path, physical_indices: list[int]) -> bool:
    """True if JSON exists and lists every sampled `physical_frame` we expect."""
    if not physical_indices or not json_path.is_file():
        return False
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    have = {int(fd["physical_frame"]) for fd in data.get("frames", [])}
    return set(physical_indices) <= have


def list_videos_dir(videos_dir: Path) -> list[Path]:
    return sorted(
        p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    )


def iter_videos_needing_annotation(videos_dir: Path, interval_sec: float) -> list[tuple[Path, Path, list[int]]]:
    """
    (video_path, output_json, physical_indices) for each video that is not fully annotated.
    """
    out: list[tuple[Path, Path, list[int]]] = []
    for vp in list_videos_dir(videos_dir):
        vr = VideoReader(str(vp), target_fps=30)
        try:
            phys = physical_frames_every_interval(vr.src_fps, vr.src_frame_count, interval_sec)
        finally:
            vr.release()
        if not phys:
            continue
        outp = default_keypoints_json_path(vp)
        if is_annotation_complete(outp, phys):
            continue
        out.append((vp, outp, phys))
    return out


def collect_annotation_totals(videos_dir: Path) -> tuple[int, int, int]:
    """
    Return:
      - total number of videos in `videos_dir`
      - number of videos that have at least one annotated frame in JSON
      - total number of annotated frames across all videos (unique physical_frame per JSON)
    """
    total_videos = 0
    annotated_videos = 0
    total_annotated_frames = 0
    for vp in list_videos_dir(videos_dir):
        total_videos += 1
        jp = default_keypoints_json_path(vp)
        if not jp.is_file():
            continue
        try:
            with open(jp, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        frames = data.get("frames", [])
        phys: set[int] = set()
        for fd in frames:
            if not isinstance(fd, dict):
                continue
            try:
                phys.add(int(fd.get("physical_frame")))
            except (TypeError, ValueError):
                continue
        if phys:
            annotated_videos += 1
            total_annotated_frames += len(phys)
    return total_videos, annotated_videos, total_annotated_frames


def fill_keypoints_from_detector(fa: FrameAnnotation, frame_bgr: np.ndarray, detector: CourtDetector) -> None:
    """Write best-detection-per-class into `fa.keypoints_xy` (pixel coords)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    centers, cls_ids, confs = detector.predict_keypoints(rgb)
    best: dict[int, tuple[float, float, float]] = {}
    for (x, y), c, cf in zip(centers, cls_ids, confs):
        ci = int(c)
        if not (0 <= ci < NUM_KEYPOINTS):
            continue
        cf = float(cf)
        if ci not in best or cf > best[ci][2]:
            best[ci] = (float(x), float(y), cf)
    for i in range(NUM_KEYPOINTS):
        if i in best:
            fa.keypoints_xy[i] = [best[i][0], best[i][1]]
        else:
            fa.keypoints_xy[i] = None


def homography_ransac_from_annotations(
    fa: FrameAnnotation,
    court_constants: CourtConstants,
    frame_w: int,
    frame_h: int,
) -> np.ndarray | None:
    """
    Homography mapping normalized image coords to normalized court coords
    (pixel / [W,H] -> court_meters / court_size), same pairing as v2 pipeline.
    Fitted with :func:`find_homography_ransac` (cosine-similarity RANSAC-style search).
    """
    if frame_w <= 0 or frame_h <= 0:
        return None
    fw, fh = float(frame_w), float(frame_h)
    court_size = np.array(court_constants.court_size, dtype=np.float64)
    src_pts: list[list[float]] = []
    dst_pts: list[list[float]] = []
    for cls_id, kp in enumerate(fa.keypoints_xy):
        if kp is None:
            continue
        if cls_id not in court_constants.cls_to_points:
            continue
        court_xy = np.array(court_constants.cls_to_points[cls_id][0], dtype=np.float64)
        src_pts.append([float(kp[0]) / fw, float(kp[1]) / fh])
        dst_pts.append((court_xy / court_size).tolist())
    n = len(src_pts)
    if n < 4:
        return None
    src_h = np.hstack([np.array(src_pts, dtype=np.float32), np.ones((n, 1), dtype=np.float32)])
    dst_h = np.hstack([np.array(dst_pts, dtype=np.float32), np.ones((n, 1), dtype=np.float32)])
    H = find_homography_ransac(src_h, dst_h)
    if H is None:
        return None
    return np.asarray(H, dtype=np.float64)


def blend_nba_court_overlay(
    frame_bgr: np.ndarray,
    H_frame_to_court: np.ndarray,
    court_bgr: np.ndarray,
    *,
    alpha: float = COURT_OVERLAY_ALPHA,
) -> np.ndarray:
    """Warp ``nba.png`` onto the frame using H^{-1} (court -> frame), as in ``visualizer.run_video``."""
    frame_h, frame_w = frame_bgr.shape[:2]
    ch, cw = court_bgr.shape[:2]
    H = np.asarray(H_frame_to_court, dtype=np.float64)
    H_inv = np.linalg.pinv(H)
    court_corners_norm = np.array(
        [
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5],
        ],
        dtype=np.float32,
    )
    frame_corners_norm = project_homography(court_corners_norm, H_inv)
    dst_pts = frame_corners_norm * np.array([[frame_w, frame_h]], dtype=np.float32)
    src_pts = np.array(
        [
            [0.0, 0.0],
            [float(cw), 0.0],
            [float(cw), float(ch)],
            [0.0, float(ch)],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts.astype(np.float32))
    warped_court = cv2.warpPerspective(court_bgr, M, (frame_w, frame_h))
    out = frame_bgr.astype(np.float64)
    out = alpha * warped_court.astype(np.float64) + (1.0 - alpha) * out
    return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class FrameAnnotation:
    physical_frame: int
    logical_frame: int
    width: int
    height: int
    keypoints_xy: list[list[float] | None] = field(default_factory=lambda: [None] * NUM_KEYPOINTS)
    """3×3 homography (frame-normalized → court-normalized), or None if <4 points / RANSAC failed."""
    homography: list[list[float]] | None = None
    """If True, :meth:`sync_homography` will recompute RANSAC homography from keypoints (not serialized)."""
    homography_dirty: bool = True

    def to_json_dict(self) -> dict:
        return {
            "physical_frame": self.physical_frame,
            "logical_frame": self.logical_frame,
            "width": self.width,
            "height": self.height,
            "keypoints_xy": self.keypoints_xy,
            "homography": self.homography,
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> FrameAnnotation:
        kps = d.get("keypoints_xy")
        if kps is None:
            kps = [None] * NUM_KEYPOINTS
        while len(kps) < NUM_KEYPOINTS:
            kps.append(None)
        kps = kps[:NUM_KEYPOINTS]
        h_raw = d.get("homography")
        homography: list[list[float]] | None = None
        homography_dirty = True
        if h_raw is not None and isinstance(h_raw, list) and len(h_raw) == 3:
            try:
                homography = [[float(h_raw[i][j]) for j in range(3)] for i in range(3)]
                homography_dirty = False
            except (TypeError, ValueError, IndexError):
                homography = None
                homography_dirty = True
        return cls(
            physical_frame=int(d["physical_frame"]),
            logical_frame=int(d.get("logical_frame", 0)),
            width=int(d["width"]),
            height=int(d["height"]),
            keypoints_xy=kps,
            homography=homography,
            homography_dirty=homography_dirty,
        )

    def homography_numpy(self) -> np.ndarray | None:
        if self.homography is None:
            return None
        return np.array(self.homography, dtype=np.float64)

    def sync_homography(self, court_constants: CourtConstants) -> None:
        """Recompute homography via ``find_homography_ransac`` from current keypoints and clear dirty flag."""
        H = homography_ransac_from_annotations(self, court_constants, self.width, self.height)
        if H is None:
            self.homography = None
        else:
            self.homography = H.tolist()
        self.homography_dirty = False

    def mark_keypoints_changed(self) -> None:
        self.homography_dirty = True


@dataclass
class SessionState:
    video_path: str
    video_stem: str
    src_fps: float
    interval_sec: float
    output_path: Path
    physical_indices: list[int]
    frames: dict[int, FrameAnnotation]  # physical_frame -> annotation
    idx_in_list: int = 0
    # Source (native) frame index being edited; arrows change this; n/p snap to scheduled samples.
    current_physical_frame: int = 0
    # Per scheduled slot index -> selected source frame (can be shifted by arrows).
    slot_selected_frames: dict[int, int] = field(default_factory=dict)
    src_frame_count: int = 0
    edited_frames: set[int] = field(default_factory=set)
    active_kp: int = 0
    display_scale: float = 1.0
    dirty: bool = False

    def current_physical(self) -> int:
        return self.current_physical_frame

    def get_or_create_frame(
        self,
        physical: int,
        logical: int,
        w: int,
        h: int,
        *,
        frame_bgr: np.ndarray | None,
        detector: CourtDetector | None,
        court_constants: CourtConstants | None,
    ) -> FrameAnnotation:
        if physical not in self.frames:
            fa = FrameAnnotation(
                physical_frame=physical,
                logical_frame=logical,
                width=w,
                height=h,
                keypoints_xy=[None] * NUM_KEYPOINTS,
                homography_dirty=True,
            )
            if detector is not None and frame_bgr is not None:
                fill_keypoints_from_detector(fa, frame_bgr, detector)
                self.dirty = True
            self.frames[physical] = fa
            if court_constants is not None:
                fa.sync_homography(court_constants)
        return self.frames[physical]

    def save(self) -> None:
        payload = {
            "video_file": self.video_stem,
            "video_path": self.video_path,
            "src_fps": self.src_fps,
            "interval_sec": self.interval_sec,
            "num_keypoints": NUM_KEYPOINTS,
            "physical_indices": self.physical_indices,
            "slot_selected_frames": [
                int(self.slot_selected_frames.get(i, p)) for i, p in enumerate(self.physical_indices)
            ],
            "frames": [self.frames[p].to_json_dict() for p in sorted(self.frames.keys()) if p in self.frames],
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.dirty = False
        print(f"Saved {len(self.frames)} annotated frame(s) → {self.output_path}")

    @classmethod
    def load_or_new(
        cls,
        video_path: Path,
        output_path: Path,
        interval_sec: float,
        vr: VideoReader,
        physical_indices: list[int],
    ) -> SessionState:
        src_fps = float(vr.src_fps or 25.0)
        stem = video_path.name
        if output_path.is_file():
            with open(output_path, encoding="utf-8") as f:
                data = json.load(f)
            frames: dict[int, FrameAnnotation] = {}
            for fd in data.get("frames", []):
                fa = FrameAnnotation.from_json_dict(fd)
                frames[fa.physical_frame] = fa
            loaded_phys = list(data.get("physical_indices", physical_indices)) or physical_indices
            loaded_sel_raw = data.get("slot_selected_frames")
            slot_selected_frames: dict[int, int] = {}
            if isinstance(loaded_sel_raw, list):
                for i, p in enumerate(loaded_phys):
                    try:
                        slot_selected_frames[i] = int(loaded_sel_raw[i])
                    except (TypeError, ValueError, IndexError):
                        slot_selected_frames[i] = int(p)
            else:
                slot_selected_frames = {i: int(p) for i, p in enumerate(loaded_phys)}
            print(f"Loaded existing annotations from {output_path} ({len(frames)} frames).")
            return cls(
                video_path=str(video_path.resolve()),
                video_stem=stem,
                src_fps=float(data.get("src_fps", src_fps)),
                interval_sec=float(data.get("interval_sec", interval_sec)),
                output_path=output_path,
                physical_indices=loaded_phys,
                slot_selected_frames=slot_selected_frames,
                frames=frames,
                edited_frames=set(frames.keys()),
            )
        return cls(
            video_path=str(video_path.resolve()),
            video_stem=stem,
            src_fps=src_fps,
            interval_sec=interval_sec,
            output_path=output_path,
            physical_indices=physical_indices,
            frames={},
        )


def read_frame_at_physical(vr: VideoReader, physical: int) -> tuple[int, np.ndarray] | None:
    logical = vr.nearest_logical_for_physical(physical)
    vr.set(cv2.CAP_PROP_POS_FRAMES, float(logical))
    ret, frame = vr.read()
    if not ret or frame is None:
        return None
    return logical, frame


def draw_overlay(
    display: np.ndarray,
    state: SessionState,
    fa: FrameAnnotation,
    *,
    help_lines: list[str],
    ransac_h_ok: bool | None = None,
) -> None:
    h, w = display.shape[:2]
    for i in range(NUM_KEYPOINTS):
        kp = fa.keypoints_xy[i]
        col = _hsv_color(i)
        if kp is None:
            continue
        x_disp = int(round(kp[0] * state.display_scale))
        y_disp = int(round(kp[1] * state.display_scale))
        cv2.circle(display, (x_disp, y_disp), 4, col, -1, cv2.LINE_AA)
        cv2.circle(display, (x_disp, y_disp), 6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            display,
            str(i),
            (x_disp + 6, y_disp - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            str(i),
            (x_disp + 6, y_disp - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            col,
            1,
            cv2.LINE_AA,
        )
    # active keypoint highlight
    ak = state.active_kp
    cv2.rectangle(display, (0, 0), (w, 42), (30, 30, 30), -1)
    placed = sum(1 for x in fa.keypoints_xy if x is not None)
    h_note = ""
    if ransac_h_ok is True:
        h_note = "  |  RANSAC H"
    elif ransac_h_ok is False:
        h_note = "  |  RANSAC: need ≥4 pts"
    slot_n = len(state.physical_indices)
    slot_txt = f"sample {state.idx_in_list + 1}/{slot_n}" if slot_n else "sample —"
    last_phys = max(state.src_frame_count - 1, 0)
    title = (
        f"{state.video_stem}  |  kp {ak}/32  |  {slot_txt}  "
        f"phys={fa.physical_frame}/{last_phys} log={fa.logical_frame}  |  placed {placed}/{NUM_KEYPOINTS}{h_note}"
    )
    cv2.putText(
        display,
        title,
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        _hsv_color(ak),
        2,
        cv2.LINE_AA,
    )
    y0 = h - 8 - 18 * len(help_lines)
    for j, line in enumerate(help_lines):
        cv2.putText(
            display,
            line,
            (8, y0 + j * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )


def on_mouse(event, x, y, flags, param):  # noqa: ARG001
    state: SessionState
    holder: list[FrameAnnotation]
    state, holder = param
    fa = holder[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        ix = x / state.display_scale
        iy = y / state.display_scale
        phys = state.current_physical()
        if phys not in state.frames:
            state.frames[phys] = fa
        fa.keypoints_xy[state.active_kp] = [float(ix), float(iy)]
        state.dirty = True
        state.edited_frames.add(phys)
        fa.mark_keypoints_changed()
    elif event == cv2.EVENT_RBUTTONDOWN:
        cleared = None
        best_d = NEAR_PX_DISPLAY**2
        for i, kp in enumerate(fa.keypoints_xy):
            if kp is None:
                continue
            xd = kp[0] * state.display_scale - x
            yd = kp[1] * state.display_scale - y
            d2 = xd * xd + yd * yd
            if d2 <= best_d:
                best_d = d2
                cleared = i
        if cleared is not None:
            phys = state.current_physical()
            if phys not in state.frames:
                state.frames[phys] = fa
            fa.keypoints_xy[cleared] = None
            state.dirty = True
            state.edited_frames.add(phys)
            fa.mark_keypoints_changed()


def run_session(
    video_path: Path,
    output_path: Path,
    interval_sec: float,
    *,
    use_model: bool = True,
    physical_list_override: list[int] | None = None,
    detector: CourtDetector | None = None,
) -> None:
    vr = VideoReader(str(video_path), target_fps=30)
    total_p = vr.src_frame_count
    physical_list = physical_list_override or physical_frames_every_interval(vr.src_fps, total_p, interval_sec)
    if not physical_list:
        print("No frames to sample.", file=sys.stderr)
        vr.release()
        return

    model_for_frames: CourtDetector | None = None
    if use_model:
        if detector is None:
            print("Loading court keypoint model for prefill…")
            model_for_frames = CourtDetector(cfg=load_default_config())
        else:
            model_for_frames = detector

    state = SessionState.load_or_new(video_path, output_path, interval_sec, vr, physical_list)
    # Recompute physical list if fresh session already set above
    if not state.physical_indices:
        state.physical_indices = physical_list

    state.src_frame_count = max(int(total_p), 0)
    if not state.slot_selected_frames:
        state.slot_selected_frames = {i: p for i, p in enumerate(state.physical_indices)}
    else:
        for i, p in enumerate(state.physical_indices):
            state.slot_selected_frames.setdefault(i, p)
    if state.physical_indices:
        state.idx_in_list = max(0, min(state.idx_in_list, len(state.physical_indices) - 1))
        state.current_physical_frame = _clamp_physical_index(
            state.src_frame_count,
            state.slot_selected_frames.get(state.idx_in_list, state.physical_indices[state.idx_in_list]),
        )
    else:
        state.idx_in_list = 0
        state.current_physical_frame = 0

    help_lines = [
        "[ ] active kp  |  0-9 kp  |  L click place  |  R click remove nearest",
        "c clear kp  |  x clear frame  |  n/p sample  |  arrows ±1 frame  |  s save  |  q quit",
    ]

    cur = read_frame_at_physical(vr, state.current_physical())
    if cur is None:
        print("Failed to read first frame.", file=sys.stderr)
        vr.release()
        return
    logical, frame_bgr = cur
    h0, w0 = frame_bgr.shape[:2]
    max_side = max(h0, w0)
    state.display_scale = 1.0 if max_side <= 1280 else 1280.0 / max_side

    court_constants = CourtConstants(CourtType.NBA)
    court_img_path = Path(__file__).resolve().parent / "nba.png"
    court_bgr = cv2.imread(str(court_img_path))
    if court_bgr is None:
        print(f"Warning: cannot load court texture {court_img_path}; overlay disabled.", file=sys.stderr)
        court_bgr = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def drop_transient_current_if_needed() -> None:
        """
        Drop temporary frame visited by arrows if user did not edit it.
        This prevents saving navigation-only frames on n/p/q transitions.
        """
        phys = state.current_physical()
        if phys in state.physical_indices:
            return
        if phys in state.edited_frames:
            return
        state.frames.pop(phys, None)

    def commit_current_slot_selection() -> None:
        """
        Persist current slot's selected frame and remove the unshifted base frame
        for this slot when it was only an auto-prefill artifact.
        """
        if not state.physical_indices:
            return
        idx = state.idx_in_list
        if not (0 <= idx < len(state.physical_indices)):
            return
        base_phys = state.physical_indices[idx]
        chosen_phys = state.current_physical()
        prev_chosen = state.slot_selected_frames.get(idx, base_phys)
        state.slot_selected_frames[idx] = chosen_phys
        if prev_chosen != chosen_phys:
            state.dirty = True
        # If user shifted slot away from base sample, keep only shifted frame.
        if chosen_phys != base_phys and base_phys not in state.edited_frames:
            if state.frames.pop(base_phys, None) is not None:
                state.dirty = True

    def ensure_fa() -> FrameAnnotation:
        phys = state.current_physical()
        nonlocal logical, frame_bgr
        r = read_frame_at_physical(vr, phys)
        if r is None:
            return state.get_or_create_frame(
                phys,
                logical,
                w0,
                h0,
                frame_bgr=None,
                detector=None,
                court_constants=court_constants,
            )
        logical, frame_bgr = r
        new_frame = phys not in state.frames
        return state.get_or_create_frame(
            phys,
            logical,
            frame_bgr.shape[1],
            frame_bgr.shape[0],
            frame_bgr=frame_bgr if new_frame else None,
            detector=model_for_frames if new_frame else None,
            court_constants=court_constants,
        )

    fa_holder: list[FrameAnnotation] = [ensure_fa()]
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, (state, fa_holder))

    while True:
        fa = ensure_fa()
        fa_holder[0] = fa
        if fa.homography_dirty:
            fa.sync_homography(court_constants)
        H_ann = fa.homography_numpy()
        ransac_ok: bool | None
        if H_ann is not None and court_bgr is not None:
            frame_vis = blend_nba_court_overlay(frame_bgr, H_ann, court_bgr)
            ransac_ok = True
        elif H_ann is None:
            frame_vis = frame_bgr.copy()
            ransac_ok = False
        else:
            frame_vis = frame_bgr.copy()
            ransac_ok = None

        display = cv2.resize(
            frame_vis,
            (
                int(round(frame_vis.shape[1] * state.display_scale)),
                int(round(frame_vis.shape[0] * state.display_scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
        draw_overlay(
            display,
            state,
            fa,
            help_lines=help_lines,
            ransac_h_ok=ransac_ok,
        )
        cv2.imshow(WINDOW_NAME, display)
        k_raw = cv2.waitKeyEx(30)
        if k_raw != -1 and (delta := _arrow_frame_delta(k_raw)) is not None:
            # Arrow keys only nudge the currently viewed physical frame.
            # Do not autosave here: save happens on annotation transitions (n/p/q) or explicit "s".
            drop_transient_current_if_needed()
            state.current_physical_frame = _clamp_physical_index(
                state.src_frame_count, state.current_physical_frame + delta
            )
            if state.physical_indices:
                state.slot_selected_frames[state.idx_in_list] = state.current_physical_frame
            continue

        key = k_raw & 0xFF
        if key in (27, ord("q")):
            commit_current_slot_selection()
            if state.dirty:
                state.save()
            break
        if key == ord("s"):
            commit_current_slot_selection()
            state.save()
        elif key == ord("n"):
            commit_current_slot_selection()
            if state.dirty:
                state.save()
            state.idx_in_list = min(state.idx_in_list + 1, len(state.physical_indices) - 1)
            state.current_physical_frame = _clamp_physical_index(
                state.src_frame_count,
                state.slot_selected_frames.get(state.idx_in_list, state.physical_indices[state.idx_in_list]),
            )
        elif key == ord("p"):
            commit_current_slot_selection()
            if state.dirty:
                state.save()
            state.idx_in_list = max(state.idx_in_list - 1, 0)
            state.current_physical_frame = _clamp_physical_index(
                state.src_frame_count,
                state.slot_selected_frames.get(state.idx_in_list, state.physical_indices[state.idx_in_list]),
            )
        elif key == ord("c"):
            phys = state.current_physical()
            if phys not in state.frames:
                state.frames[phys] = fa
            fa.keypoints_xy[state.active_kp] = None
            state.dirty = True
            state.edited_frames.add(phys)
            fa.mark_keypoints_changed()
        elif key == ord("x"):
            phys = state.current_physical()
            if phys not in state.frames:
                state.frames[phys] = fa
            fa.keypoints_xy = [None] * NUM_KEYPOINTS
            state.dirty = True
            state.edited_frames.add(phys)
            fa.mark_keypoints_changed()
        elif _is_prev_kp_key(k_raw, key):
            state.active_kp = (state.active_kp - 1) % NUM_KEYPOINTS
        elif _is_next_kp_key(k_raw, key):
            state.active_kp = (state.active_kp + 1) % NUM_KEYPOINTS
        elif ord("0") <= key <= ord("9"):
            state.active_kp = key - ord("0")

    vr.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate 33 court keypoints on sampled video frames.")
    default_dir = Path(__file__).resolve().parent / "annotated" / "videos"
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=default_dir,
        help="Directory with input videos",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Video file name inside videos-dir (e.g. nba_long_01.mp4)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Seconds between sampled frames (source timeline)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: annotated/keypoints/<video_stem>.json); only applies with --video",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Do not prefill keypoints with the detector (start empty on new frames)",
    )
    args = parser.parse_args()

    videos_dir = args.videos_dir
    if not videos_dir.is_dir():
        print(f"Videos directory not found: {videos_dir}", file=sys.stderr)
        sys.exit(1)

    use_model = not args.no_model

    if args.video:
        video_path = videos_dir / args.video
        if not video_path.is_file():
            print(f"Video not found: {video_path}", file=sys.stderr)
            sys.exit(1)
        out = args.output or default_keypoints_json_path(video_path)
        run_session(video_path, out, args.interval, use_model=use_model)
        total_videos, annotated_videos, total_annotated_frames = collect_annotation_totals(videos_dir)
        print(
            "\nAnnotation totals: "
            f"videos={total_videos}, annotated_videos={annotated_videos}, "
            f"annotated_frames={total_annotated_frames}"
        )
        return

    # Auto-queue: only videos missing a complete annotation file
    queue = iter_videos_needing_annotation(videos_dir, args.interval)
    if not queue:
        print("All videos in annotated/videos already have full keypoint JSON for this interval.")
        total_videos, annotated_videos, total_annotated_frames = collect_annotation_totals(videos_dir)
        print(
            "Annotation totals: "
            f"videos={total_videos}, annotated_videos={annotated_videos}, "
            f"annotated_frames={total_annotated_frames}"
        )
        return
    print(f"Queue: {len(queue)} video(s) need annotation (interval={args.interval}s).")
    shared: CourtDetector | None = None
    if use_model:
        print("Loading court keypoint model once for all queued videos…")
        shared = CourtDetector(cfg=load_default_config())
    for i, (video_path, out, phys) in enumerate(queue):
        print(f"\n[{i + 1}/{len(queue)}] {video_path.name} → {out.name}")
        run_session(
            video_path,
            out,
            args.interval,
            use_model=use_model,
            physical_list_override=phys,
            detector=shared,
        )

    print("\nDone — all listed videos are fully annotated (or you quit mid-way).")
    total_videos, annotated_videos, total_annotated_frames = collect_annotation_totals(videos_dir)
    print(
        "Annotation totals: "
        f"videos={total_videos}, annotated_videos={annotated_videos}, "
        f"annotated_frames={total_annotated_frames}"
    )


if __name__ == "__main__":
    main()
