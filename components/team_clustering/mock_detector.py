from common.classes.player import FrameDetections
from team_clustering.ground_truth import load_ground_truth, load_ground_truth_absolute


class MockDetector:
    """
    A stand-in detector that returns pre-annotated ground-truth bounding boxes
    as :class:`Player` objects.

    Use this to develop and test downstream pipeline stages (e.g. team
    clustering) without running a real detection model.

    Args:
        gt_path:    Path to the ground-truth annotation file.
        normalized: If ``True`` coordinates are normalized (x_center, y_center, w, h);
                    if ``False`` (default) they are absolute (x, y, w, h) — MOT format.
    """

    def __init__(self, gt_path: str, normalized: bool = False):
        self.gt_path = gt_path
        self.normalized = normalized

    def detect(self, video_path: str) -> FrameDetections:
        """
        Return per-frame detections loaded from ground-truth annotations.

        Args:
            video_path: Path to the video (needed to resolve normalized coords).

        Returns:
            ``{frame_id: [Player(...), ...]}``
        """
        if self.normalized:
            return load_ground_truth(self.gt_path, video_path)
        return load_ground_truth_absolute(self.gt_path)
