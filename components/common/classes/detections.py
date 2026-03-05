class Detection:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, class_id: int, confidence: float):
        # Coords in pixels
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.class_id = class_id
        self.confidence = confidence

    def __repr__(self):
        return f"Detection(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, confidence={self.confidence})"

    def get_bbox(self):
        return [self.x1, self.y1, self.x2, self.y2]


class FrameDetections:
    def __init__(self, frame_id: int, detections: list[Detection]):
        self.frame_id = frame_id
        self.detections = detections


VideoDetections = list[FrameDetections]
