import cv2
import numpy as np
from ultralytics import YOLO


class PlayerEmbedder:
    """Extracts player masks (via YOLO segmentation) and color-histogram embeddings."""

    def __init__(self, model_name="yolov8n-seg.pt", device="cpu"):
        self.model = YOLO(model_name)
        self.model.to(device)
        self.device = device

    def extract_embedding(self, crop, mask):
        """Compute a Hue+Saturation histogram from the masked region of *crop*."""
        if mask is None:
            h, w = crop.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (int(w * 0.25), int(h * 0.25)), (int(w * 0.75), int(h * 0.75)), 1, -1)
        else:
            mask = (mask > 0.5).astype(np.uint8)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], mask, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], mask, [8], [0, 256])
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        return np.concatenate([hist_h, hist_s])

    def get_player_masks_batch(self, crops):
        """
        Run YOLO segmentation on a batch of crops.
        Returns one mask per crop (the largest person detection), or None.
        """
        if not crops:
            return []

        results = self.model(crops, verbose=False, classes=[0], device=self.device)
        masks = []
        for i, result in enumerate(results):
            if not result.masks or not result.masks.xy:
                masks.append(None)
                continue

            best_mask = None
            max_area = 0
            crop_h, crop_w = crops[i].shape[:2]

            for poly in result.masks.xy:
                poly_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
                poly_int = np.array(poly).reshape((-1, 1, 2)).astype(int)
                cv2.fillPoly(poly_mask, [poly_int], 1)
                area = np.sum(poly_mask)
                if area > max_area:
                    max_area = area
                    best_mask = poly_mask

            masks.append(best_mask)
        return masks
