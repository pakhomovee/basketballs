import cv2
import numpy as np

from common.logger import get_logger

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


def make_side_by_side_video(
    top_video_path: str,
    bottom_video_path: str,
    output_path: str,
    *,
    show_logs: bool = True,
    log_level_colors: dict[str, tuple[int, int, int]] | None = None,
    log_source_color: tuple[int, int, int] | None = None,
    log_message_color: tuple[int, int, int] | None = None,
    log_empty_color: tuple[int, int, int] | None = None,
) -> None:
    """
    Create video with top (real) and bottom (2D projection) views.
    Optionally adds a logs panel at the bottom showing logs for each frame.

    top_video_path    — path to first (top) video.
    bottom_video_path — path to second (bottom) video.
    output_path       — path to output video.
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
