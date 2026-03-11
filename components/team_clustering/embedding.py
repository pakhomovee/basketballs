"""Player embedding extraction via YOLO segmentation masks."""

from __future__ import annotations

import cv2
import numpy as np
from tqdm.auto import tqdm
from ultralytics import YOLO

from common.classes.player import PlayersDetections
from common.utils.utils import get_device


def extract_player_embeddings(
    video_path: str,
    detections: PlayersDetections,
    seg_model: str = "yolov8n-seg.pt",
    batch_size: int = 32,
) -> None:
    """
    Extract mask-based embeddings for all players and set player.embedding in place.

    Uses YOLO segmentation for player masks, then Hue+Sat histograms on masked region.
    Run once before tracking; both tracker and team_clustering will use these embeddings.

    Reads video sequentially (no seeking) to match original behavior and avoid
    codec-related frame misalignment.
    """
    embedder = PlayerEmbedder(seg_model, get_device())
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    batch_crops, batch_players = [], []

    for frame_id in tqdm(range(total_frames), desc="Extracting embeddings"):
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
                _process_batch(embedder, batch_crops, batch_players)
                batch_crops, batch_players = [], []

    if batch_crops:
        _process_batch(embedder, batch_crops, batch_players)
    cap.release()


def _process_batch(embedder: PlayerEmbedder, crops: list, players: list) -> None:
    masks = embedder.get_player_masks_batch(crops)
    for crop, mask, player in zip(crops, masks, players):
        player.embedding = embedder.extract_embedding(crop, mask).astype(np.float32)


def extract_embeddings_from_image(
    frame: "np.ndarray",
    players: list,
    embedder: PlayerEmbedder | None = None,
    seg_model: str = "yolov8n-seg.pt",
    batch_size: int = 32,
) -> None:
    """
    Extract embeddings for players in a single image. Sets player.embedding in place.

    Args:
        frame: BGR image (H, W, 3).
        players: List of Player objects with bbox set.
        embedder: Optional pre-instantiated PlayerEmbedder. If None, creates one.
        seg_model: YOLO seg model name (used only if embedder is None).
        batch_size: Batch size for segmentation.
    """
    from common.utils.utils import get_device

    if embedder is None:
        embedder = PlayerEmbedder(seg_model, get_device())

    h, w = frame.shape[:2]
    batch_crops, batch_players = [], []

    for player in players:
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
            _process_batch(embedder, batch_crops, batch_players)
            batch_crops, batch_players = [], []

    if batch_crops:
        _process_batch(embedder, batch_crops, batch_players)


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
