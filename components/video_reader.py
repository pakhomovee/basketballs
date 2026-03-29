"""Unified video reader that maps source frames to a target FPS on the fly."""

from __future__ import annotations

import math
import cv2
import numpy as np


class VideoReader:
    """
    Stream video frames mapped to *target_fps* without file re-encoding.
    Mirrors cv2.VideoCapture interface, but internally reads source frames
    """

    def __init__(self, video_path: str, target_fps: int = 30) -> None:
        self._path = video_path
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self._src_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._src_total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if target_fps and self._src_fps > 0:
            self._target_fps = float(target_fps)
        else:
            self._target_fps = self._src_fps if self._src_fps > 0 else 25.0

        self._fps_ratio = self._src_fps / self._target_fps if self._src_fps > 0 else 1.0  # physical per logical
        self._total_frames = math.ceil(self._src_total / self._fps_ratio) if self._fps_ratio > 0 else self._src_total

        # Streaming state
        self._logical_idx = 0
        self._physical_idx = -1  # last physical frame index we've actually read
        self._last_frame: np.ndarray | None = None

    @property
    def video_path(self) -> str:
        return self._path

    @property
    def fps(self) -> float:
        """Effective (target) frame rate."""
        return self._target_fps

    @property
    def src_fps(self) -> float:
        return self._src_fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def total_frames(self) -> int:
        """Total number of logical frames at *target_fps*."""
        return self._total_frames

    def physical_frame_for(self, logical: int) -> int:
        """Source frame index corresponding to logical frame *logical*."""
        return min(round(logical * self._fps_ratio), max(self._src_total - 1, 0))

    @property
    def src_frame_count(self) -> int:
        """Number of frames in the source file (native timeline)."""
        return self._src_total

    def nearest_logical_for_physical(self, physical_index: int) -> int:
        """
        Logical frame index whose source frame is closest to *physical_index*
        (same mapping as :meth:`physical_frame_for` / iteration order).

        Use to remap annotations stored in native/source frame indices to the
        project's target FPS timeline.
        """
        T = self._total_frames
        if T <= 0:
            return 0
        p_max = self.physical_frame_for(T - 1)
        p = max(0, min(int(physical_index), p_max))

        lo, hi = 0, T - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.physical_frame_for(mid) <= p:
                lo = mid
            else:
                hi = mid - 1

        candidates = [lo]
        if lo + 1 < T:
            candidates.append(lo + 1)
        return min(candidates, key=lambda x: (abs(self.physical_frame_for(x) - p), x))

    def read(self) -> tuple[bool, np.ndarray | None]:
        """cv2.VideoCapture-compatible ``read()``."""
        result = self.next_frame()
        if result is None:
            return False, None
        return True, result[1]

    def get(self, prop_id: int) -> float:
        """cv2.VideoCapture-compatible ``get()``."""
        if prop_id == cv2.CAP_PROP_FPS:
            return self._target_fps
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._total_frames)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self._logical_idx)
        return self._cap.get(prop_id)

    def set(self, prop_id: int, value: float) -> bool:
        """cv2.VideoCapture-compatible ``set()``."""
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            logical = int(value)
            if logical == 0:
                self.reset()
                return True
            physical = self.physical_frame_for(logical)
            ok = self._cap.set(cv2.CAP_PROP_POS_FRAMES, physical)
            self._logical_idx = logical
            self._physical_idx = physical - 1
            self._last_frame = None
            return ok
        return self._cap.set(prop_id, value)

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def next_frame(self) -> tuple[int, np.ndarray] | None:
        """Return ``(logical_frame_id, bgr_frame)`` or ``None`` at end."""
        if self._logical_idx >= self._total_frames:
            return None

        target_physical = self.physical_frame_for(self._logical_idx)

        # Advance the underlying capture to the desired physical frame.
        while self._physical_idx < target_physical:
            ret, frame = self._cap.read()
            if not ret:
                return None
            self._physical_idx += 1
            self._last_frame = frame

        if self._last_frame is None:
            return None

        result = (self._logical_idx, self._last_frame)
        self._logical_idx += 1
        return result

    def reset(self) -> None:
        """Seek back to the beginning."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._logical_idx = 0
        self._physical_idx = -1
        self._last_frame = None

    def release(self) -> None:
        self._cap.release()

    # ------------------------------------------------------------------
    # Protocols
    # ------------------------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, np.ndarray]:
        result = self.next_frame()
        if result is None:
            raise StopIteration
        return result

    def __len__(self) -> int:
        return self._total_frames

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
