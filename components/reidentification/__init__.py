"""Player re-identification module."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ReIDModel",
    "SynergyReIDDataset",
    "QueryGalleryDataset",
    "ArcFaceLoss",
    "TripletLoss",
    "train",
    "evaluate",
    "ReIDFeatureExtractor",
    "extract_reid_embeddings",
]

_SYMBOL_TO_MODULE = {
    "ReIDModel": ".model",
    "SynergyReIDDataset": ".dataset",
    "QueryGalleryDataset": ".dataset",
    "ArcFaceLoss": ".losses",
    "TripletLoss": ".losses",
    "train": ".trainer",
    "evaluate": ".evaluate",
    "ReIDFeatureExtractor": ".extract",
    "extract_reid_embeddings": ".extract",
}


def __getattr__(name: str):
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
