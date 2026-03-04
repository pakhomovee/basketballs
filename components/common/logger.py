"""
Global Logger component for frame-scoped logs.

Register logs by frame_id; retrieve logs for the current frame.
Used by the side-by-side viewer to display logs per frame.
"""

from dataclasses import dataclass
from typing import Literal

LogLevel = Literal["info", "warn", "error", "debug"]


@dataclass
class LogEntry:
    """Single log entry for a frame."""

    message: str
    level: LogLevel = "info"
    source: str | None = None  # optional component name, e.g. "court_detector"


class Logger:
    """
    Global logger that registers logs by frame_id.

    Use the singleton instance via :func:`get_logger`.
    """

    def __init__(self) -> None:
        self._logs: dict[int, list[LogEntry]] = {}

    def log(
        self,
        frame_id: int,
        message: str,
        level: LogLevel = "info",
        source: str | None = None,
    ) -> None:
        """Register a log entry for the given frame."""
        if frame_id not in self._logs:
            self._logs[frame_id] = []
        self._logs[frame_id].append(LogEntry(message=message, level=level, source=source))

    def get_logs(self, frame_id: int) -> list[LogEntry]:
        """Return all log entries for the given frame, in order."""
        return self._logs.get(frame_id, [])

    def get_log_strings(self, frame_id: int) -> list[str]:
        """Return log messages for the frame as strings (level + message)."""
        entries = self.get_logs(frame_id)
        lines = []
        for e in entries:
            prefix = f"[{e.level.upper()}]" if e.level != "info" else ""
            src = f" [{e.source}]" if e.source else ""
            lines.append(f"{prefix}{src} {e.message}".strip())
        return lines

    def get_log_segments(
        self,
        frame_id: int,
        *,
        level_colors: dict[str, tuple[int, int, int]] | None = None,
        source_color: tuple[int, int, int] | None = None,
        message_color: tuple[int, int, int] | None = None,
    ) -> list[list[tuple[str, tuple[int, int, int]]]]:
        """
        Return log entries as segments with per-part colors for rendering.
        Each line is a list of (text, bgr_color) tuples.
        """
        defaults = {
            "info": (200, 200, 200),
            "warn": (0, 200, 255),   # orange in BGR
            "error": (0, 0, 255),    # red
            "debug": (150, 150, 150),
        }
        lc = level_colors or defaults
        sc = source_color or (255, 200, 100)  # cyan-ish
        mc = message_color or (220, 220, 220)
        entries = self.get_logs(frame_id)
        result = []
        for e in entries:
            segments: list[tuple[str, tuple[int, int, int]]] = []
            if e.level != "info":
                segments.append((f"[{e.level.upper()}] ", lc.get(e.level, defaults["info"])))
            if e.source:
                segments.append((f"[{e.source}] ", sc))
            segments.append((e.message, mc))
            result.append(segments)
        return result

    def clear(self) -> None:
        """Clear all registered logs."""
        self._logs.clear()

    def has_logs(self) -> bool:
        """Return True if any logs have been registered."""
        return bool(self._logs)


# Singleton accessor
_logger: Logger | None = None


def get_logger() -> Logger:
    """Return the global Logger singleton."""
    global _logger
    if _logger is None:
        _logger = Logger()
    return _logger
