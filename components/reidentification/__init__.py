"""Player re-identification module."""

from .dataset import QueryGalleryDataset, SynergyReIDDataset
from .evaluate import evaluate
from .extract import ReIDFeatureExtractor, extract_reid_embeddings
from .model import ReIDModel
from .trainer import train

__all__ = [
    "ReIDModel",
    "SynergyReIDDataset",
    "QueryGalleryDataset",
    "train",
    "evaluate",
    "ReIDFeatureExtractor",
    "extract_reid_embeddings",
]
