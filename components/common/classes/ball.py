from dataclasses import dataclass, field


@dataclass
class Ball:
    bbox: list[int] = field(default_factory=list)  # [x1, y1, x2, y2]
    confidence: float | None = None


# frame_id -> list of Balls detections for that frame
BallDetections = dict[int, list[Ball]]
