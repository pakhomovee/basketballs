"""Extract ReID embeddings for all player detections in a video."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .dataset import preprocess_reid_image
from .model import ReIDModel


class ReIDFeatureExtractor:
    """Extracts L2-normalised ReID features from player crops using a trained model."""

    def __init__(self, weights_path: str, device: str = "cuda"):
        self.device = device
        state = torch.load(weights_path, map_location=device, weights_only=True)
        num_classes = state["classifier.weight"].shape[0]
        self.model = ReIDModel(num_classes=num_classes, pretrained=False)
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def extract_batch(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        """Extract features for a list of BGR crops. Returns list of 2048-d numpy vectors."""
        tensors = [preprocess_reid_image(crop) for crop in crops]
        batch = torch.stack(tensors).to(self.device)
        features = self.model.extract_features(batch)
        return [f.cpu().numpy() for f in features]


def extract_reid_embeddings(
    video_path: str,
    detections: dict,
    weights_path: str,
    device: str = "cuda",
    batch_size: int = 64,
) -> None:
    """Replace player.embedding with ReID features for all detections, in-place."""
    from tqdm.auto import tqdm

    extractor = ReIDFeatureExtractor(weights_path, device)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    batch_crops: list[np.ndarray] = []
    batch_players: list = []

    for frame_id in tqdm(range(total_frames), desc="Extracting ReID embeddings"):
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        for player in detections.get(frame_id, []):
            if not player.bbox or len(player.bbox) < 4:
                continue
            x1, y1, x2, y2 = map(int, player.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            crop = frame[y1:y2, x1:x2]
            batch_crops.append(crop)
            batch_players.append(player)

            if len(batch_crops) >= batch_size:
                features = extractor.extract_batch(batch_crops)
                for player, feature in zip(batch_players, features):
                    player.reid_embedding = feature
                batch_crops = []
                batch_players = []

    if batch_crops:
        features = extractor.extract_batch(batch_crops)
        for player, feature in zip(batch_players, features):
            player.reid_embedding = feature

    cap.release()
