import collections

import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from common.utils.utils import get_device
from common.classes.player import Player, FrameDetections
from team_clustering.embedding import PlayerEmbedder

try:
    import umap
except ImportError:
    umap = None


class TeamClustering:
    """
    Clusters detected players into teams based on jersey colour histograms.

    After :meth:`run` completes, every :class:`Player` in the input detections
    is enriched with a ``team_id`` attribute.

    Usage::

        tc = TeamClustering(seg_model="yolov8x-seg.pt")
        tc.run(video_path, detections, k_frames=30)

        # Each Player now has .team_id set
        # tc.sample_detections holds (crop, mask) tuples for visualization
    """

    def __init__(self, seg_model="yolov8n-seg.pt", n_clusters=2):
        self.device = get_device()
        self.embedder = PlayerEmbedder(seg_model, self.device)
        self.n_clusters = n_clusters

        self.sample_detections: list[tuple] = []

    def run(
        self,
        video_path: str,
        detections: FrameDetections,
        k_frames: int = 1,
        batch_size: int = 32,
    ):
        """
        Run the full pipeline: extract embeddings, cluster, and enrich
        every :class:`Player` with ``team_id``.

        Args:
            video_path:  Path to the video file.
            detections:  Per-frame player detections.
            k_frames:    Process every k-th frame (1 = every frame).
            batch_size:  Crops per YOLO segmentation batch.
        """
        tracks = self._extract_tracks(video_path, detections, k_frames, batch_size)
        clusters = self._cluster(tracks)

        for players in detections.values():
            for player in players:
                if player.player_id in clusters:
                    player.team_id = clusters[player.player_id]

    def _extract_tracks(self, video_path, detections, k_frames, batch_size):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tracks = collections.defaultdict(list)
        self.sample_detections = []
        batch_crops, batch_ids = [], []
        frame_id = 1

        for _ in tqdm(range(total_frames), desc="Extracting features"):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id == 1 or (k_frames > 1 and (frame_id - 1) % k_frames == 0):
                for player in detections.get(frame_id, []):
                    x1, y1, x2, y2 = map(int, player.bbox)
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 - x1 < 5 or y2 - y1 < 5:
                        continue

                    batch_crops.append(frame[y1:y2, x1:x2])
                    batch_ids.append(player.player_id)

                    if len(batch_crops) >= batch_size:
                        self._process_batch(batch_crops, batch_ids, tracks)
                        batch_crops, batch_ids = [], []

            frame_id += 1

        if batch_crops:
            self._process_batch(batch_crops, batch_ids, tracks)

        cap.release()
        return tracks

    def _process_batch(self, crops, ids, tracks):
        masks = self.embedder.get_player_masks_batch(crops)
        for crop, mask, pid in zip(crops, masks, ids):
            tracks[pid].append(self.embedder.extract_embedding(crop, mask))
            if len(self.sample_detections) < 16 and mask is not None:
                self.sample_detections.append((crop, mask))

    def _cluster(self, tracks):
        obj_ids, features = [], []
        for pid, feats in tracks.items():
            if not feats:
                continue
            obj_ids.append(pid)
            features.append(np.mean(feats, axis=0))

        features = np.array(features)
        if len(features) < self.n_clusters:
            print(f"Not enough players ({len(features)}) to form {self.n_clusters} clusters.")
            return {}

        embedding = features
        if umap and len(features) > 15 and len(np.unique(features, axis=0)) >= 2:
            try:
                embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(features)
            except Exception as e:
                print(f"UMAP failed ({e}), falling back to raw features.")
        elif not umap:
            print("UMAP not available, skipping dimensionality reduction.")

        labels = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10).fit_predict(embedding)
        clusters = {pid: int(label) for pid, label in zip(obj_ids, labels)}
        print(f"Clustered {len(obj_ids)} players into {self.n_clusters} teams.")
        return clusters
