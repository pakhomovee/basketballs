"""Player embedding extraction via full-frame YOLO segmentation masks."""

from __future__ import annotations

import cv2
import numpy as np
from tqdm.auto import tqdm
from ultralytics import YOLO

from common.classes.player import PlayersDetections
from common.utils.utils import get_device
from team_clustering.shared import DEFAULT_SEG_MODEL, clip_bbox

PLAYER_CLASS_ID = 1

HUE_BINS = 12
SATURATION_BINS = 8
VALUE_BINS = 8
LAB_BINS = 12
MIN_SATURATION_FOR_HUE = 40

TORSO_CENTER_X = 0.5
TORSO_CENTER_Y = 0.35
TORSO_SIGMA_X = 0.18
TORSO_SIGMA_Y = 0.18
SKIN_WEIGHT = 0.15

EMBEDDING_DIM = HUE_BINS + SATURATION_BINS + VALUE_BINS + 2 * LAB_BINS + 6


def collect_player_crops(
    frame: np.ndarray,
    players: list,
) -> list[tuple[object, tuple[int, int, int, int], np.ndarray]]:
    """Return ``(player, clipped_bbox, crop)`` tuples for valid player boxes in a frame."""
    frame_h, frame_w = frame.shape[:2]
    entries = []
    for player in players:
        if not player.bbox or len(player.bbox) < 4:
            continue

        clipped_bbox = clip_bbox(player.bbox, frame_w, frame_h)
        if clipped_bbox is None:
            continue

        x1, y1, x2, y2 = clipped_bbox
        entries.append((player, clipped_bbox, frame[y1:y2, x1:x2]))
    return entries


def _bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _normalized_histogram(
    values: np.ndarray,
    bins: int,
    value_range: tuple[float, float],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    if values.size == 0:
        return np.zeros(bins, dtype=np.float32)

    hist, _ = np.histogram(values, bins=bins, range=value_range, weights=weights)
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _weighted_mean_and_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    total_weight = float(weights.sum())
    if values.size == 0 or total_weight <= 0:
        return 0.0, 0.0

    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, variance**0.5


def extract_player_embeddings(
    video_path: str,
    detections: PlayersDetections,
    seg_model: str = str(DEFAULT_SEG_MODEL),
) -> None:
    """Populate player embeddings using full-frame segmentation masks."""
    embedder = PlayerEmbedder(seg_model, device=get_device())
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_id in tqdm(range(total_frames), desc="Extracting embeddings"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_entries = collect_player_crops(frame, detections.get(frame_id, []))
        if not frame_entries:
            continue

        frame_boxes = [box for _, box, _ in frame_entries]
        frame_masks = embedder.get_player_masks_for_frame(frame, frame_boxes)
        for (player, _, crop), mask in zip(frame_entries, frame_masks):
            player.embedding = embedder.extract_embedding(crop, mask).astype(np.float32)

    cap.release()


class PlayerEmbedder:
    """Extract player masks and jersey-focused color embeddings."""

    def __init__(self, model_path: str = str(DEFAULT_SEG_MODEL), device: str = "cpu"):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        self.player_class_id = PLAYER_CLASS_ID

    def _fallback_mask(self, crop: np.ndarray) -> np.ndarray:
        height, width = crop.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (int(width * 0.25), int(height * 0.15)), (int(width * 0.75), int(height * 0.65)), 1, -1)
        return mask

    def refine_embedding_mask(self, crop: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        if mask is None:
            return self._fallback_mask(crop)

        binary_mask = (mask > 0.5).astype(np.uint8)
        if not np.any(binary_mask):
            return self._fallback_mask(crop)

        ys, xs = np.nonzero(binary_mask)
        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1
        width = x2 - x1
        height = y2 - y1

        torso_x1 = x1 + int(width * 0.2)
        torso_x2 = x1 + int(width * 0.8)
        torso_y1 = y1 + int(height * 0.15)
        torso_y2 = y1 + int(height * 0.6)

        refined_mask = np.zeros_like(binary_mask)
        refined_mask[torso_y1:torso_y2, torso_x1:torso_x2] = 1
        refined_mask &= binary_mask

        if refined_mask.sum() < max(25, int(binary_mask.sum() * 0.1)):
            return binary_mask
        return refined_mask

    def _refine_embedding_mask(self, crop: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        return self.refine_embedding_mask(crop, mask)

    def _compute_embedding_weights(self, crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
        height, width = mask.shape
        y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)
        if height > 1:
            y_coords /= height - 1
        if width > 1:
            x_coords /= width - 1

        spatial_weights = np.exp(
            -(
                ((x_coords - TORSO_CENTER_X) ** 2) / (2 * TORSO_SIGMA_X**2)
                + ((y_coords - TORSO_CENTER_Y) ** 2) / (2 * TORSO_SIGMA_Y**2)
            )
        ).astype(np.float32)
        spatial_weights *= mask.astype(np.float32)

        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
        spatial_weights[skin_mask & (mask > 0)] *= SKIN_WEIGHT
        return spatial_weights

    def _weighted_histograms(
        self,
        hue: np.ndarray,
        saturation: np.ndarray,
        value: np.ndarray,
        lab_a: np.ndarray,
        lab_b: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        saturated = saturation >= MIN_SATURATION_FOR_HUE
        hue_hist = _normalized_histogram(
            hue[saturated],
            bins=HUE_BINS,
            value_range=(0, 180),
            weights=(weights[saturated] * saturation[saturated]) if np.any(saturated) else None,
        )
        saturation_hist = _normalized_histogram(saturation, SATURATION_BINS, (0, 256), weights)
        value_hist = _normalized_histogram(value, VALUE_BINS, (0, 256), weights)
        lab_a_hist = _normalized_histogram(lab_a, LAB_BINS, (0, 256), weights)
        lab_b_hist = _normalized_histogram(lab_b, LAB_BINS, (0, 256), weights)
        return hue_hist, saturation_hist, value_hist, lab_a_hist, lab_b_hist

    def _summary_stats(
        self,
        saturation: np.ndarray,
        value: np.ndarray,
        lightness: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        saturation_mean, saturation_std = _weighted_mean_and_std(saturation, weights)
        value_mean, value_std = _weighted_mean_and_std(value, weights)
        lightness_mean, lightness_std = _weighted_mean_and_std(lightness, weights)
        return np.array(
            [
                saturation_mean / 255.0,
                saturation_std / 255.0,
                value_mean / 255.0,
                value_std / 255.0,
                lightness_mean / 255.0,
                lightness_std / 255.0,
            ],
            dtype=np.float32,
        )

    def _empty_embedding(self) -> np.ndarray:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    def extract_embedding(self, crop: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        """Compute a jersey-focused color embedding for one player crop."""
        refined_mask = self.refine_embedding_mask(crop, mask)
        weights = self._compute_embedding_weights(crop, refined_mask)
        valid_pixels = weights > 0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

        hsv_pixels = hsv[valid_pixels]
        lab_pixels = lab[valid_pixels]
        pixel_weights = weights[valid_pixels].astype(np.float32)
        if hsv_pixels.size == 0 or lab_pixels.size == 0 or float(pixel_weights.sum()) <= 0:
            return self._empty_embedding()

        hue = hsv_pixels[:, 0].astype(np.float32)
        saturation = hsv_pixels[:, 1].astype(np.float32)
        value = hsv_pixels[:, 2].astype(np.float32)
        lightness = lab_pixels[:, 0].astype(np.float32)
        lab_a = lab_pixels[:, 1].astype(np.float32)
        lab_b = lab_pixels[:, 2].astype(np.float32)

        hue_hist, saturation_hist, value_hist, lab_a_hist, lab_b_hist = self._weighted_histograms(
            hue,
            saturation,
            value,
            lab_a,
            lab_b,
            pixel_weights,
        )
        summary_stats = self._summary_stats(saturation, value, lightness, pixel_weights)

        embedding = np.concatenate(
            [hue_hist, saturation_hist, value_hist, lab_a_hist, lab_b_hist, summary_stats]
        ).astype(np.float32)
        return _l2_normalize(embedding)

    def _mask_bbox(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def _build_frame_masks(
        self, result, frame_shape: tuple[int, int]
    ) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
        if not result.masks or not result.masks.xy:
            return []

        frame_h, frame_w = frame_shape
        masks: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
        for polygon in result.masks.xy:
            full_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            polygon_int = np.array(polygon).reshape((-1, 1, 2)).astype(int)
            cv2.fillPoly(full_mask, [polygon_int], 1)
            bbox = self._mask_bbox(full_mask)
            if bbox is not None:
                masks.append((full_mask, bbox))
        return masks

    def _match_mask_to_box(
        self,
        player_box: tuple[int, int, int, int],
        frame_masks: list[tuple[np.ndarray, tuple[int, int, int, int]]],
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = player_box
        best_mask = None
        best_score = 0.0

        for full_mask, mask_bbox in frame_masks:
            score = _bbox_iou(player_box, mask_bbox)
            if score <= best_score:
                continue

            cropped_mask = full_mask[y1:y2, x1:x2]
            if np.any(cropped_mask):
                best_score = score
                best_mask = cropped_mask

        return best_mask

    def get_player_masks_for_frame(
        self,
        frame: np.ndarray,
        player_boxes: list[tuple[int, int, int, int]],
    ) -> list[np.ndarray | None]:
        """Run segmentation on the full frame and crop one mask per player box."""
        if not player_boxes:
            return []

        result = self.model(frame, verbose=False, classes=[self.player_class_id], device=self.device)[0]
        frame_masks = self._build_frame_masks(result, frame.shape[:2])
        return [self._match_mask_to_box(player_box, frame_masks) for player_box in player_boxes]
