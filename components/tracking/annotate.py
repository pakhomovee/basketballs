"""
Interactive tracking benchmark annotation tool.

Workflow
--------
1. Run detector + court detector + tracker on a video (no number recognition).
2. Display frames with bounding boxes and track IDs.
3. Allow user to:
   - Delete bad detections
   - Swap two IDs on the current frame
   - Assign an ID to a player (propagated FORWARD only; conflicts resolved by
     giving the conflicting track a fresh ID)
4. Export YOLO MOT-format txt.

Usage
-----
    cd components
    python -m tracking.annotate <video_path> [--output annotations.txt] [--court-type nba]

Controls
--------
    Right / D      → next frame
    Left / A       → previous frame
    PgDown         → skip +30 frames
    PgUp           → skip -30 frames
    Click player   → select player (highlighted in green)
    X              → delete selected detection
    T              → delete the selected track from all frames
    S + click      → swap: select two players then press S to swap their IDs
    I              → assign ID to selected player (opens numeric prompt)
    E              → export annotations and quit
    Q / Esc        → quit without saving
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from common.classes import CourtType
from common.classes.player import Player, PlayersDetections
from visualization.court_2d import Court2DView, _COURT_METERS, _COURT_PADDING

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

COMPONENTS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = COMPONENTS_DIR.parent
MODELS_DIR = REPO_ROOT / "models"
COURT_MODEL_PATH = MODELS_DIR / "court_detection_model.pt"
DETECTOR_MODEL_PATH = MODELS_DIR / "best-4.pt"
REID_MODEL_PATH = MODELS_DIR / "reid_model.pth"

# ── colours for up to 20 track IDs ──────────────────────────────────────────
TRACK_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
]


def _color_for_id(player_id: int) -> tuple[int, int, int]:
    """BGR colour for a track ID."""
    r, g, b = TRACK_COLORS[player_id % len(TRACK_COLORS)]
    return (b, g, r)  # OpenCV uses BGR


# ── pipeline: detect → court → embed → track ────────────────────────────────

def run_pipeline(
    video_path: str,
    court_type: CourtType = CourtType.NBA,
    no_cache: bool = False,
) -> PlayersDetections:
    """Run detector + court detector + embedding + tracker. No number recognition."""
    from cache import load_detections_cache, save_detections_cache
    from court_detector.court_detector import CourtDetector
    from detector import Detector, get_video_players_detections
    from reidentification import extract_reid_embeddings
    from team_clustering.embedding import DEFAULT_SEG_MODEL, extract_player_embeddings
    from tracking import FlowTracker

    seg_model = DEFAULT_SEG_MODEL

    players_detections: PlayersDetections | None = None
    if not no_cache:
        players_detections = load_detections_cache(
            video_path, seg_model, use_detector_cache=True, use_embeddings_cache=True,
        )
        if players_detections is not None:
            log.info("Loaded detections from cache")
            # Re-run court detection if the cache pre-dates homography support
            has_court = any(
                p.court_position is not None
                for players in players_detections.values()
                for p in players
            )
            if not has_court:
                log.info("Cache has no court positions — running court detector...")
                _cd = CourtDetector()
                _cd.run(video_path, players_detections, court_type)
                if not no_cache:
                    save_detections_cache(video_path, players_detections, seg_model)

    if players_detections is None:
        detector = Detector()
        all_detections = detector.detect_video(video_path)
        players_detections = get_video_players_detections(all_detections)

        court_detector = CourtDetector()
        court_detector.run(video_path, players_detections, court_type)

        extract_player_embeddings(video_path, players_detections)
        if os.path.isfile(str(REID_MODEL_PATH)):
            from common.utils.utils import get_device
            extract_reid_embeddings(
                video_path, players_detections, str(REID_MODEL_PATH), device=get_device(),
            )

        if not no_cache:
            save_detections_cache(video_path, players_detections, seg_model)

    # Run tracker
    frame_width = _get_frame_width(video_path)
    tracker = FlowTracker(num_tracks=10, frame_width=frame_width)
    tracker.track(players_detections)

    return players_detections


def _get_frame_width(video_path: str) -> float | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return w if w > 0 else None


# ── ID operations ────────────────────────────────────────────────────────────

def _next_free_id(detections: PlayersDetections) -> int:
    """Return an ID not currently used by any player in any frame."""
    used = set()
    for players in detections.values():
        for p in players:
            if p.player_id >= 0:
                used.add(p.player_id)
    return max(used, default=-1) + 1


def _sorted_frames_from(detections: PlayersDetections, frame_id: int) -> list[int]:
    """Return sorted frame ids starting from frame_id, inclusive."""
    return [fid for fid in sorted(detections.keys()) if fid >= frame_id]


def delete_detection(
    detections: PlayersDetections, frame_id: int, det_idx: int,
) -> None:
    """Remove a detection from a single frame."""
    players = detections.get(frame_id, [])
    if 0 <= det_idx < len(players):
        players.pop(det_idx)


def delete_track(
    detections: PlayersDetections, track_id: int,
) -> int:
    """Remove every detection carrying track_id across all frames."""
    removed = 0
    for frame_id, players in detections.items():
        kept_players = [player for player in players if player.player_id != track_id]
        removed += len(players) - len(kept_players)
        detections[frame_id] = kept_players
    return removed


def swap_ids(
    detections: PlayersDetections, frame_id: int, idx_a: int, idx_b: int,
) -> None:
    """Swap two IDs on the current frame and propagate the swap forward."""
    players = detections[frame_id]
    id_a = players[idx_a].player_id
    id_b = players[idx_b].player_id

    if id_a == id_b:
        return

    temp_id = _next_free_id(detections)
    while temp_id in (id_a, id_b):
        temp_id += 1

    for fid in _sorted_frames_from(detections, frame_id):
        for player in detections[fid]:
            if player.player_id == id_a:
                player.player_id = temp_id

    for fid in _sorted_frames_from(detections, frame_id):
        for player in detections[fid]:
            if player.player_id == id_b:
                player.player_id = id_a

    for fid in _sorted_frames_from(detections, frame_id):
        for player in detections[fid]:
            if player.player_id == temp_id:
                player.player_id = id_b


def assign_id_forward(
    detections: PlayersDetections, frame_id: int, det_idx: int, new_id: int,
) -> None:
    """
    Assign *new_id* to the selected detection and propagate FORWARD along its
    track.  If another track already carries *new_id* in future frames, that
    track gets a fresh unused ID (so we never create ID collisions).

    Propagation follows the existing player_id chain: starting from the
    selected detection's current ID, every detection in frames > frame_id with
    that same old ID is reassigned.
    """
    players = detections[frame_id]
    old_id = players[det_idx].player_id

    if old_id == new_id:
        return

    sorted_frames = _sorted_frames_from(detections, frame_id)
    if not sorted_frames:
        return

    # 2. Resolve conflicts: if new_id is carried by *another* track in frames
    #    >= frame_id, give that other track a fresh ID.
    fresh_id = _next_free_id(detections)
    # Make sure fresh_id != new_id and != old_id
    while fresh_id == new_id or fresh_id == old_id:
        fresh_id += 1

    for fid in sorted_frames:
        for p in detections[fid]:
            if p.player_id == new_id:
                # This is the conflicting track—reassign it
                p.player_id = fresh_id

    # 3. Now propagate new_id forward along the old track
    for fid in sorted_frames:
        for p in detections[fid]:
            if p.player_id == old_id:
                p.player_id = new_id


# ── YOLO MOT export ─────────────────────────────────────────────────────────

def export_yolo_mot(
    detections: PlayersDetections,
    output_path: str,
    img_w: int,
    img_h: int,
) -> None:
    """
    Export to YOLO MOT txt format.

    Each line:
        frame_id track_id class_id cx_norm cy_norm w_norm h_norm

    class_id is always 0 (player).
    Coordinates are normalised to [0, 1].
    """
    lines: list[str] = []
    for frame_id in sorted(detections.keys()):
        for p in detections[frame_id]:
            if p.player_id < 0:
                continue
            x1, y1, x2, y2 = p.bbox
            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            lines.append(
                f"{frame_id} {p.player_id} 0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            )
    Path(output_path).write_text("\n".join(lines) + "\n")
    log.info("Exported %d detections to %s", len(lines), output_path)


# ── Interactive UI ───────────────────────────────────────────────────────────

class AnnotationUI:
    """OpenCV-based interactive annotation window."""

    WINDOW = "Tracking Annotation"

    def __init__(
        self,
        video_path: str,
        detections: PlayersDetections,
        court_type: CourtType = CourtType.NBA,
    ):
        self.video_path = video_path
        self.detections = detections

        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.frame_ids = sorted(detections.keys())
        self.frame_idx = 0  # index into self.frame_ids

        self.selected_idx: int | None = None  # index in current frame's player list
        self.swap_first: int | None = None  # first selection for swap

        self._frame_cache: dict[int, np.ndarray] = {}
        self._court_type = court_type
        self._court_view = Court2DView(court_type)

    # ── frame reading ────────────────────────────────────────────────────

    def _read_frame(self, frame_id: int) -> np.ndarray:
        if frame_id in self._frame_cache:
            return self._frame_cache[frame_id].copy()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        if not ret:
            return np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        # Keep a limited cache
        if len(self._frame_cache) > 120:
            # Remove oldest entries
            oldest = sorted(self._frame_cache.keys())[:60]
            for k in oldest:
                del self._frame_cache[k]
        self._frame_cache[frame_id] = frame.copy()
        return frame

    @property
    def current_frame_id(self) -> int:
        return self.frame_ids[self.frame_idx]

    @property
    def current_players(self) -> list[Player]:
        return self.detections.get(self.current_frame_id, [])

    # ── rendering ────────────────────────────────────────────────────────

    def _render_court_panel(self, target_height: int) -> np.ndarray:
        """2D top-down court view with players coloured by track ID."""
        court_img = self._court_view._base_court.copy()
        ch, cw = court_img.shape[:2]
        length_m, width_m = _COURT_METERS[self._court_type]
        pad = _COURT_PADDING

        def to_px(x_m: float, y_m: float) -> tuple[int, int]:
            y_m = -y_m  # Court2DView negates y
            xn = (x_m / length_m) + 0.5
            yn = (y_m / width_m) + 0.5
            px = int(round(pad + xn * (cw - 2 * pad)))
            py = int(round(pad + yn * (ch - 2 * pad)))
            return (px, py)

        for i, p in enumerate(self.current_players):
            if p.court_position is None:
                continue
            cx, cy = to_px(*p.court_position)
            pid = p.player_id
            color = _color_for_id(pid) if pid >= 0 else (100, 100, 100)
            radius = 16
            outline_w = 2
            if i == self.selected_idx:
                color = (0, 255, 0)
                radius = 20
                outline_w = 3
            elif i == self.swap_first:
                color = (0, 255, 255)
                radius = 20
                outline_w = 3
            cv2.circle(court_img, (cx, cy), radius, color, -1, cv2.LINE_AA)
            cv2.circle(court_img, (cx, cy), radius, (255, 255, 255), outline_w, cv2.LINE_AA)
            label = str(pid) if pid >= 0 else "?"
            cv2.putText(
                court_img, label, (cx - 7, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        # Scale to match camera frame height
        scale = target_height / ch
        new_w = int(round(cw * scale))
        return cv2.resize(court_img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)

    def _render(self) -> np.ndarray:
        frame_id = self.current_frame_id
        img = self._read_frame(frame_id)
        players = self.current_players

        for i, p in enumerate(players):
            x1, y1, x2, y2 = p.bbox
            pid = p.player_id
            color = _color_for_id(pid) if pid >= 0 else (100, 100, 100)
            thickness = 2

            # Highlight selected
            if i == self.selected_idx:
                color = (0, 255, 0)
                thickness = 3
            if i == self.swap_first:
                color = (0, 255, 255)
                thickness = 3

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            label = f"ID {pid}" if pid >= 0 else "?"
            cv2.putText(
                img, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

        # HUD
        hud = (
            f"Frame {frame_id}/{self.frame_ids[-1]}  "
            f"({self.frame_idx + 1}/{len(self.frame_ids)})  "
            f"Players: {len(players)}"
        )
        cv2.putText(img, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        help_lines = [
            "Click=select  X=delete det  T=delete track  S=swap  I=assign  E=export  Q=quit",
        ]
        for i, line in enumerate(help_lines):
            cv2.putText(
                img, line, (10, self.img_h - 15 - 25 * i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
            )

        court_panel = self._render_court_panel(img.shape[0])
        return np.hstack([img, court_panel])

    # ── mouse callback ───────────────────────────────────────────────────

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Find which bbox was clicked
        players = self.current_players
        best_idx = None
        best_area = float("inf")
        for i, p in enumerate(players):
            x1, y1, x2, y2 = p.bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best_area = area
                    best_idx = i
        self.selected_idx = best_idx
        cv2.imshow(self.WINDOW, self._render())

    # ── numeric input via OpenCV ─────────────────────────────────────────

    def _prompt_number(self, prompt: str) -> int | None:
        """Prompt user for a number using the OpenCV window."""
        buf = ""
        while True:
            overlay = self._render()
            text = f"{prompt}: {buf}_"
            cv2.putText(
                overlay, text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
            )
            cv2.imshow(self.WINDOW, overlay)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # Esc
                return None
            if key == 13:  # Enter
                try:
                    return int(buf)
                except ValueError:
                    return None
            if key == 8 and buf:  # Backspace
                buf = buf[:-1]
            elif chr(key).isdigit():
                buf += chr(key)

    # ── main loop ────────────────────────────────────────────────────────

    def run(self, output_path: str) -> None:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, min(self.img_w, 1600), min(self.img_h, 900))
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)

        while True:
            cv2.imshow(self.WINDOW, self._render())
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):  # Q / Esc
                break

            # Navigation
            elif key in (ord("d"), 83, 3):  # D / Right arrow
                self._navigate(1)
            elif key in (ord("a"), 81, 2):  # A / Left arrow
                self._navigate(-1)
            elif key == 86:  # PgDown (macOS: fn+Down)
                self._navigate(30)
            elif key == 85:  # PgUp
                self._navigate(-30)

            # Delete selected
            elif key == ord("x") and self.selected_idx is not None:
                delete_detection(self.detections, self.current_frame_id, self.selected_idx)
                self.selected_idx = None
                self.swap_first = None

            # Delete selected track in all frames
            elif key == ord("t") and self.selected_idx is not None:
                track_id = self.current_players[self.selected_idx].player_id
                removed = delete_track(self.detections, track_id)
                log.info("Deleted track %d from %d detections", track_id, removed)
                self.selected_idx = None
                self.swap_first = None

            # Swap mode
            elif key == ord("s"):
                if self.swap_first is None and self.selected_idx is not None:
                    # First selection for swap
                    self.swap_first = self.selected_idx
                    self.selected_idx = None
                elif self.swap_first is not None and self.selected_idx is not None:
                    swap_ids(
                        self.detections, self.current_frame_id,
                        self.swap_first, self.selected_idx,
                    )
                    self.swap_first = None
                    self.selected_idx = None

            # Assign ID (forward propagation)
            elif key == ord("i") and self.selected_idx is not None:
                new_id = self._prompt_number("Assign ID")
                if new_id is not None and new_id >= 0:
                    assign_id_forward(
                        self.detections, self.current_frame_id,
                        self.selected_idx, new_id,
                    )
                    self.selected_idx = None

            # Export
            elif key == ord("e"):
                export_yolo_mot(
                    self.detections, output_path, self.img_w, self.img_h,
                )
                log.info("Exported! Press Q to quit or continue editing.")

        self.cap.release()
        cv2.destroyAllWindows()

    def _navigate(self, delta: int) -> None:
        self.frame_idx = max(0, min(len(self.frame_ids) - 1, self.frame_idx + delta))
        self.selected_idx = None
        self.swap_first = None


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive tracking benchmark annotation tool",
    )
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output YOLO MOT txt path (default: <video_stem>_tracking.txt)",
    )
    parser.add_argument(
        "--court-type", choices=["nba", "fiba"], default="nba",
        help="Court type for homography",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable detection cache",
    )
    args = parser.parse_args()

    court_type = CourtType.NBA if args.court_type == "nba" else CourtType.FIBA

    if args.output is None:
        stem = Path(args.video_path).stem
        args.output = str(Path(args.video_path).parent / f"{stem}_tracking.txt")

    log.info("Running pipeline on %s ...", args.video_path)
    detections = run_pipeline(args.video_path, court_type=court_type, no_cache=args.no_cache)
    log.info(
        "Pipeline done: %d frames, %d total detections",
        len(detections),
        sum(len(v) for v in detections.values()),
    )

    ui = AnnotationUI(args.video_path, detections, court_type=court_type)
    ui.run(args.output)


if __name__ == "__main__":
    main()
