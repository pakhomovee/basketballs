from __future__ import annotations

import collections
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.cluster import KMeans

from common.classes.player import PlayersDetections

if TYPE_CHECKING:
    from config import AppConfig


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _pool_track_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    return _normalize(np.median(np.asarray(embeddings, dtype=np.float32), axis=0))


def _cluster_track_embeddings(
    track_embeddings: dict[Any, list[np.ndarray]],
    n_clusters: int,
) -> dict[Any, int]:
    """Shared clustering logic: robust pooled embedding per track, then KMeans.

    Args:
        track_embeddings: Map from track key (e.g. player_id or track_idx) to
            list of embeddings for that track.
        n_clusters: Number of clusters (teams).

    Returns:
        Map from track key to team label (0..n_clusters-1).
        Empty dict if not enough tracks.
    """
    keys, features = [], []
    for key, feats in track_embeddings.items():
        if not feats:
            continue
        keys.append(key)
        features.append(_pool_track_embeddings(feats))

    features_arr = np.array(features)
    if len(features_arr) < n_clusters:
        return {}

    labels = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    ).fit_predict(features_arr)
    return {k: int(lbl) for k, lbl in zip(keys, labels)}


class TeamClustering:
    """
    Clusters detected players into teams based on jersey colour histograms.

    Supports two input formats:

    1. **Detection dict** (requires player.player_id set)::

        tc = TeamClustering()
        tc.run(detections)
        # Each Player now has .team_id set

    2. **Track indices + det list** (for use inside tracking loops)::

        labels = tc.cluster_from_tracks(tracks, det_list)
        # labels: dict[det_idx, team_label]
    """

    def __init__(self, cfg: "AppConfig | None" = None):
        if cfg is None:
            from config import load_default_config

            cfg = load_default_config()
        self.n_clusters = cfg.team_clustering.n_clusters

    def run(
        self,
        detections: PlayersDetections,
    ) -> None:
        """
        Run clustering using precomputed player.embedding and enrich with team_id.

        Args:
            detections: Per-frame player detections (with player.embedding set).
        """
        track_embeddings = self._collect_embeddings_from_detections(detections)
        clusters = _cluster_track_embeddings(track_embeddings, self.n_clusters)

        if not clusters:
            print(f"Not enough players to form {self.n_clusters} clusters.")
            return

        for players in detections.values():
            for player in players:
                if player.player_id in clusters:
                    player.team_id = clusters[player.player_id]
        print(f"Clustered {len(clusters)} players into {self.n_clusters} teams.")

    def cluster_from_tracks(
        self,
        tracks: list[list[int]],
        det_list: list[tuple[int, object]],
    ) -> dict[int, int]:
        """Cluster tracks into teams. Returns detection index → team label.

        Args:
            tracks: List of tracks, each track is a list of detection indices.
            det_list: List of (frame_id, player) for each detection index.

        Returns:
            dict[det_idx, team_label] for detections that belong to a track
            with embeddings. Empty if not enough tracks.
        """
        track_embeddings: dict[int, list[np.ndarray]] = {}
        track_det_indices: dict[int, list[int]] = {}
        for track_idx, det_indices in enumerate(tracks):
            vecs: list[np.ndarray] = []
            for det_idx in det_indices:
                emb = det_list[det_idx][1].embedding
                if emb is not None:
                    vecs.append(emb)
            if vecs:
                track_embeddings[track_idx] = vecs
                track_det_indices[track_idx] = det_indices

        clusters = _cluster_track_embeddings(track_embeddings, self.n_clusters)
        result: dict[int, int] = {}
        for track_idx, label in clusters.items():
            for det_idx in track_det_indices[track_idx]:
                result[det_idx] = label
        return result

    def cluster_from_track_segments(
        self,
        tracks: list[list[int]],
        det_list: list[tuple[int, object]],
        segment_len: int,
    ) -> dict[int, int]:
        """Cluster track *segments* into teams. Returns detection index → team label.

        Splits each track into non-overlapping segments of ``segment_len`` frames.
        Each segment is clustered separately, so an ID switch mid-track produces
        different team labels before vs after the switch.

        Args:
            tracks: List of tracks, each track is a list of detection indices.
            det_list: List of (frame_id, player) for each detection index.
            segment_len: Frame length of each segment (e.g. lookback).

        Returns:
            dict[det_idx, team_label] for detections in segments with embeddings.
            Empty if not enough segments.
        """
        segment_embeddings: dict[tuple[int, int], list[np.ndarray]] = {}
        segment_det_indices: dict[tuple[int, int], list[int]] = {}

        for track_idx, det_indices in enumerate(tracks):
            if not det_indices:
                continue
            ordered = sorted((det_list[d][0], d) for d in det_indices)
            frame_start = ordered[0][0]

            for frame_id, det_idx in ordered:
                seg_idx = (frame_id - frame_start) // segment_len
                key = (track_idx, seg_idx)

                emb = det_list[det_idx][1].embedding
                if emb is not None:
                    segment_embeddings.setdefault(key, []).append(emb)
                    segment_det_indices.setdefault(key, []).append(det_idx)

        clusters = _cluster_track_embeddings(segment_embeddings, self.n_clusters)
        result: dict[int, int] = {}
        for key, label in clusters.items():
            for det_idx in segment_det_indices[key]:
                result[det_idx] = label
        return result

    def _collect_embeddings_from_detections(
        self,
        detections: PlayersDetections,
    ) -> dict[int, list[np.ndarray]]:
        """Collect embeddings grouped by player_id (from detections dict)."""
        tracks: dict[int, list[np.ndarray]] = collections.defaultdict(list)

        for frame_id in sorted(detections.keys()):
            for player in detections[frame_id]:
                if player.player_id < 0:
                    continue
                emb = getattr(player, "embedding", None)
                if emb is not None:
                    tracks[player.player_id].append(emb)
        return dict(tracks)
