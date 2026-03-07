import collections

import numpy as np
from sklearn.cluster import KMeans

from common.classes.player import PlayersDetections

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

        tc = TeamClustering()
        tc.run(detections, k_frames=30)

        # Each Player now has .team_id set
    """

    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.sample_detections: list[tuple] = []

    def run(
        self,
        detections: PlayersDetections,
        k_frames: int = 1,
    ):
        """
        Run clustering using precomputed player.embedding and enrich with team_id.

        Args:
            detections:  Per-frame player detections (with player.embedding set).
            k_frames:    Sample every k-th frame (1 = every frame).
        """
        tracks = self._collect_embeddings(detections, k_frames)
        clusters = self._cluster(tracks)

        for players in detections.values():
            for player in players:
                if player.player_id in clusters:
                    player.team_id = clusters[player.player_id]

    def _collect_embeddings(self, detections, k_frames):
        """Collect precomputed embeddings from players (sampled by k_frames)."""
        tracks = collections.defaultdict(list)
        self.sample_detections = []

        for frame_id in sorted(detections.keys()):
            for player in detections[frame_id]:
                if player.player_id < 0:
                    continue
                emb = getattr(player, "embedding", None)
                if emb is not None:
                    tracks[player.player_id].append(emb)
        return tracks

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
        #if umap and len(features) > 15 and len(np.unique(features, axis=0)) >= 2:
        #    try:
        #        embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(features)
        #    except Exception as e:
        #        print(f"UMAP failed ({e}), falling back to raw features.")
        #elif not umap:
        #    print("UMAP not available, skipping dimensionality reduction.")

        labels = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10).fit_predict(embedding)
        clusters = {pid: int(label) for pid, label in zip(obj_ids, labels)}
        print(f"Clustered {len(obj_ids)} players into {self.n_clusters} teams.")
        return clusters
