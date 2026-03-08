"""Player re-identification module — ResNet-50 + BNNeck (Bag of Tricks).

Public API
----------
- ReIDModel       : backbone + BNNeck + classifier
- SynergyReIDDataset : dataset loader for SynergyReID format
- train           : full training loop
- evaluate        : mAP + CMC evaluation
"""

from .model import ReIDModel

# from .dataset import SynergyReIDDataset, QueryGalleryDataset
# from .trainer import train
# from .evaluate import evaluate
from .extract import ReIDFeatureExtractor, extract_reid_embeddings

__all__ = [
    "ReIDModel",
    #    "SynergyReIDDataset",
    #    "QueryGalleryDataset",
    #    "train",
    #    "evaluate",
    "ReIDFeatureExtractor",
    "extract_reid_embeddings",
]
