import cv2
from ultralytics import YOLO
import torch
from common.utils.utils import get_dir

class YoloTracker:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"            
        print(f"YoloTracker initialized on device: {self.device}")

    def track_video(self, video_path, output_path=None, conf=0.5, iou=0.5, persist=True):
        print(f"Tracking video: {video_path}, saving to: {output_path}")
        results = self.model.track(
            source=video_path,
            conf=conf,
            iou=iou,
            persist=persist,
            device=self.device,
            stream=True,
            save=True,
            exist_ok=True,
            verbose=True,
        )
        for _ in results:
            pass
        
        return results

if __name__ == "__main__":
    tracker = YoloTracker(get_dir("models/yolo11n.pt"))
    # Track a video
    results = tracker.track_video(get_dir("common/data/train/sample_item/img1.mp4"), output_path="output.mp4")
