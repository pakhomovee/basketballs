import cv2
import numpy as np

from common.logger import get_logger
from common.classes.pass_event import PassEvent
from visualization.skeleton import draw_skeleton

# Logs panel dimensions
_LOGS_PANEL_HEIGHT = 120
_LOGS_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LOGS_FONT_SCALE = 0.45
_LOGS_LINE_HEIGHT = 18
_LOGS_PADDING = 8
_LOGS_MAX_LINES = 5

# Default colors (BGR)
_DEFAULT_LEVEL_COLORS = {
    "info": (200, 200, 200),
    "warn": (0, 200, 255),  # orange
    "error": (0, 0, 255),  # red
    "debug": (150, 150, 150),
}
_DEFAULT_SOURCE_COLOR = (255, 200, 100)  # cyan
_DEFAULT_MESSAGE_COLOR = (220, 220, 220)
_DEFAULT_EMPTY_COLOR = (120, 120, 120)

# BGR colors for team IDs in side-by-side bboxes
_TEAM_COLORS = {
    0: (0, 165, 255),  # orange
    1: (255, 100, 0),  # blue
}
_TEAM_UNKNOWN_COLOR = (0, 255, 0)  # green
# BGR color for the jersey number box (top-right of player bbox)
_NUMBER_BOX_COLOR = (0, 0, 255)  # bright red
_NUMBER_BOX_PAD = 4

# Pass overlay (top of top / real-camera view)
_PASS_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PASS_FONT_SCALE = 0.62
_PASS_THICKNESS = 1
# How long the pass banner stays visible (frames before/after pass frame)
_PASS_FRAMES_BEFORE = 12
_PASS_FRAMES_AFTER = 55
_PASS_LINE_HEIGHT = 26
_PASS_BAR_PADDING = 10
_PASS_MAX_STACK = 6
# Semi-transparent dark tint (no solid black bar): blend original ROI with this color
_PASS_TINT_BGR = np.array([32, 36, 42], dtype=np.float32)
_PASS_TINT_ALPHA = 0.42  # how much of the tint vs original (0 = no tint, 1 = solid tint)
_PASS_TEXT_COLOR = (245, 245, 250)
_PASS_TEXT_DIM = (200, 200, 210)  # slightly dimmer for older stacked passes


def _render_logs_panel(
    width: int,
    frame_id: int,
    *,
    level_colors: dict[str, tuple[int, int, int]] | None = None,
    source_color: tuple[int, int, int] | None = None,
    message_color: tuple[int, int, int] | None = None,
    empty_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Render a logs panel with black background and per-part text colors."""
    panel = np.zeros((_LOGS_PANEL_HEIGHT, width, 3), dtype=np.uint8)  # black
    logger = get_logger()
    segments_list = logger.get_log_segments(
        frame_id,
        level_colors=level_colors or _DEFAULT_LEVEL_COLORS,
        source_color=source_color or _DEFAULT_SOURCE_COLOR,
        message_color=message_color or _DEFAULT_MESSAGE_COLOR,
    )
    empty_col = empty_color or _DEFAULT_EMPTY_COLOR

    if not segments_list:
        cv2.putText(
            panel,
            f"Frame {frame_id} — no logs",
            (_LOGS_PADDING, _LOGS_LINE_HEIGHT + _LOGS_PADDING),
            _LOGS_FONT,
            _LOGS_FONT_SCALE,
            empty_col,
            1,
            cv2.LINE_AA,
        )
        return panel

    max_chars = (width - 2 * _LOGS_PADDING) // 8
    for i, segments in enumerate(segments_list[:_LOGS_MAX_LINES]):
        y = _LOGS_PADDING + (i + 1) * _LOGS_LINE_HEIGHT
        x = _LOGS_PADDING
        for text, color in segments:
            if not text:
                continue
            if len(text) > max_chars:
                text = text[: max_chars - 3] + "..."
            (tw, th), _ = cv2.getTextSize(text, _LOGS_FONT, _LOGS_FONT_SCALE, 1)
            if x + tw > width - _LOGS_PADDING:
                text = text[: max(0, len(text) - 3)] + "..."
                (tw, th), _ = cv2.getTextSize(text, _LOGS_FONT, _LOGS_FONT_SCALE, 1)
            cv2.putText(panel, text, (x, y), _LOGS_FONT, _LOGS_FONT_SCALE, color, 1, cv2.LINE_AA)
            x += tw
    return panel


def _draw_pass_overlay(
    frame_top,
    frame_id: int,
    passes: list[PassEvent] | None,
) -> None:
    """
    Draw pass lines at the top of the real-camera frame when near a pass event.

    Overlapping passes stack vertically: older passes higher, the newest pass
    always on the bottom line of the overlay (closest to the court). The
    background is a light semi-transparent tint — not a solid black bar.
    """
    if not passes:
        return
    h, w = frame_top.shape[:2]
    active: list[PassEvent] = []
    for pe in passes:
        if pe.frame - _PASS_FRAMES_BEFORE <= frame_id <= pe.frame + _PASS_FRAMES_AFTER:
            active.append(pe)
    if not active:
        return
    # Oldest at top of stack, newest at bottom (so when a new pass appears, older ones shift up)
    active.sort(key=lambda p: p.frame)
    active = active[-_PASS_MAX_STACK:]

    lines: list[tuple[str, tuple[int, int, int]]] = []
    n = len(active)
    for i, pe in enumerate(active):
        text = f"Pass {pe.from_player_id} -> {pe.to_player_id}  |  team {pe.team_id}  |  f{pe.frame}"
        text = text[: min(len(text), 100)]
        # Newest line (last in list) is brightest; older lines slightly dimmer
        color = _PASS_TEXT_COLOR if i == n - 1 else _PASS_TEXT_DIM
        lines.append((text, color))

    bar_h = min(_PASS_BAR_PADDING * 2 + len(lines) * _PASS_LINE_HEIGHT, max(_PASS_LINE_HEIGHT * 2, h // 4))
    bar_h = max(bar_h, _PASS_BAR_PADDING + len(lines) * _PASS_LINE_HEIGHT)

    # Semi-transparent tint over top strip (keeps video visible underneath)
    roi = frame_top[0:bar_h, :].astype(np.float32)
    tint = np.broadcast_to(_PASS_TINT_BGR, roi.shape)
    blended = roi * (1.0 - _PASS_TINT_ALPHA) + tint * _PASS_TINT_ALPHA
    frame_top[0:bar_h, :] = np.clip(blended, 0, 255).astype(np.uint8)

    # Thin separator at bottom of overlay
    cv2.line(frame_top, (0, bar_h - 1), (w, bar_h - 1), (70, 70, 80), 1, cv2.LINE_AA)

    for i, (line, color) in enumerate(lines):
        y = _PASS_BAR_PADDING + (i + 1) * _PASS_LINE_HEIGHT - 4
        cv2.putText(
            frame_top,
            line,
            (_PASS_BAR_PADDING, y),
            _PASS_FONT,
            _PASS_FONT_SCALE,
            color,
            _PASS_THICKNESS,
            cv2.LINE_AA,
        )


def make_side_by_side_video(
    top_video_path: str,
    bottom_video_path: str,
    output_path: str,
    *,
    detections=None,  # dict[int, list[Player]] | None
    ball_detections=None,  # dict[int, list[Ball]] | None
    passes: list[PassEvent] | None = None,
    show_logs: bool = True,
    log_level_colors: dict[str, tuple[int, int, int]] | None = None,
    log_source_color: tuple[int, int, int] | None = None,
    log_message_color: tuple[int, int, int] | None = None,
    log_empty_color: tuple[int, int, int] | None = None,
) -> None:
    """
    Create video with top (real) and bottom (2D projection) views.
    Optionally draws player bboxes/IDs on the top video if `detections` is provided.
    Optionally draws ball center as a circle if `ball_detections` is provided.
    Optionally adds a logs panel at the bottom showing logs for each frame.

    top_video_path    — path to first (top) video.
    bottom_video_path — path to second (bottom) video.
    output_path       — path to output video.
    detections        — (optional) players dict to draw on top video.
    ball_detections   — (optional) ball dict (frame_id -> list[Ball]) to draw center circles.
    passes            — (optional) list of PassEvent; shows a short banner on the top view near each pass.
    show_logs         — if True, add logs panel for current frame (default True).
    log_level_colors  — BGR colors per level: {"info": (b,g,r), "warn": ..., "error": ..., "debug": ...}.
    log_source_color  — BGR color for [source] tag.
    log_message_color — BGR color for message text.
    log_empty_color   — BGR color for "no logs" text.
    """
    print("Writing side by side (top / bottom)" + (" + logs" if show_logs else ""))

    cap_top = cv2.VideoCapture(top_video_path)
    cap_bottom = cv2.VideoCapture(bottom_video_path)

    if not cap_top.isOpened():
        raise RuntimeError(f"Cannot open top video: {top_video_path}")
    if not cap_bottom.isOpened():
        raise RuntimeError(f"Cannot open bottom video: {bottom_video_path}")

    fps = cap_top.get(cv2.CAP_PROP_FPS) or 25.0
    top_w = int(cap_top.get(cv2.CAP_PROP_FRAME_WIDTH))
    top_h = int(cap_top.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bottom_w = int(cap_bottom.get(cv2.CAP_PROP_FRAME_WIDTH))
    bottom_h = int(cap_bottom.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w = top_w
    scale_bottom = target_w / float(bottom_w) if bottom_w > 0 else 1.0
    bottom_h_resized = int(bottom_h * scale_bottom)
    target_bottom_w = target_w
    target_bottom_h = bottom_h_resized

    out_w = target_w
    out_h = top_h + target_bottom_h
    if show_logs:
        out_h += _LOGS_PANEL_HEIGHT

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    try:
        frame_id = 0
        while True:
            ret_t, frame_top = cap_top.read()
            ret_b, frame_bottom = cap_bottom.read()
            if not ret_t or not ret_b:
                break

            if frame_top.shape[:2] != (top_h, top_w):
                frame_top = cv2.resize(frame_top, (target_w, top_h))

            # Draw player IDs and bboxes if detections provided; color by team_id
            if detections is not None and frame_id in detections:
                for player in detections[frame_id]:
                    if player.bbox:
                        x1, y1, x2, y2 = map(int, player.bbox)
                        color = _TEAM_COLORS.get(player.team_id, _TEAM_UNKNOWN_COLOR)
                        cv2.rectangle(frame_top, (x1, y1), (x2, y2), color, 2)
                        # Draw skeleton if available
                        if getattr(player, "skeleton", None) is not None:
                            draw_skeleton(frame_top, player.skeleton)
                        label = str(player.player_id)
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame_top, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
                        cv2.putText(frame_top, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        if player.number is not None and player.number.num is not None:
                            num_label = str(player.number.num)
                            (nw, nh), _ = cv2.getTextSize(num_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            # Number box at top-right of player bbox: right edge = x2, bottom = y1
                            nx1 = int(x2) - nw - 2 * _NUMBER_BOX_PAD
                            ny1 = int(y1) - nh - 2 * _NUMBER_BOX_PAD
                            nx2 = int(x2)
                            ny2 = int(y1)
                            cv2.rectangle(frame_top, (nx1, ny1), (nx2, ny2), _NUMBER_BOX_COLOR, -1)
                            cv2.rectangle(frame_top, (nx1, ny1), (nx2, ny2), (0, 0, 0), 1)
                            # putText (x,y) is bottom-left of baseline; place text inside box
                            text_x = nx1 + _NUMBER_BOX_PAD
                            text_y = ny2 - _NUMBER_BOX_PAD  # baseline
                            cv2.putText(
                                frame_top,
                                num_label,
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 0),
                                2,
                            )
                        # Ball possession: orange = pre-segment (raw), red = post-segment
                        cx = (x1 + x2) // 2
                        top_y = max(0, y1 - 40)
                        if getattr(player, "is_possession_raw", False):
                            cv2.arrowedLine(
                                frame_top,
                                (cx - 10, top_y),
                                (cx - 10, y1),
                                (0, 165, 255),
                                3,
                                tipLength=0.3,
                            )
                        if player.is_possession:
                            cv2.arrowedLine(
                                frame_top,
                                (cx + 10, top_y),
                                (cx + 10, y1),
                                (0, 0, 255),
                                3,
                                tipLength=0.3,
                            )

            # Draw ball center circle if provided
            if ball_detections is not None:
                balls_in_frame = ball_detections.get(frame_id, [])
                if balls_in_frame:
                    # Use the most confident ball for visualization
                    best_ball = max(balls_in_frame, key=lambda b: b.confidence or 0.0)
                    if best_ball.bbox and len(best_ball.bbox) >= 4:
                        x1, y1, x2, y2 = best_ball.bbox[:4]
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        cv2.circle(frame_top, (cx, cy), 6, (0, 165, 255), -1)

            _draw_pass_overlay(frame_top, frame_id, passes)

            frame_bottom = cv2.resize(frame_bottom, (target_bottom_w, target_bottom_h))

            combined = np.concatenate([frame_top, frame_bottom], axis=0)

            if show_logs:
                logs_panel = _render_logs_panel(
                    out_w,
                    frame_id,
                    level_colors=log_level_colors,
                    source_color=log_source_color,
                    message_color=log_message_color,
                    empty_color=log_empty_color,
                )
                combined = np.concatenate([combined, logs_panel], axis=0)

            writer.write(combined)
            frame_id += 1
    finally:
        cap_top.release()
        cap_bottom.release()
        writer.release()
        print(f"Side by side video saved to {output_path}")
