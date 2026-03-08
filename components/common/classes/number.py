from dataclasses import dataclass, field


@dataclass
class Number:
    bbox: list[int] = field(default_factory=list)  # [x1, y1, x2, y2]
    confidence: float | None = None
    num: int | None = None


# frame_id -> list of Numbers detections for that frame
NumberDetections = dict[int, list[Number]]
