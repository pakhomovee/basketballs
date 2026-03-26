"""Dataset path resolution and automatic download utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from common.utils.utils import download_and_extract

if TYPE_CHECKING:
    from config import BenchmarkDataset


def _resolve_dataset_path(path: str) -> Path:
    """Resolve a dataset path relative to the repo root if not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    # components/common/utils/datasets.py → parents[3] = repo root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / p


def ensure_dataset(dataset: "BenchmarkDataset") -> Path:
    """Ensure the dataset exists at the configured path, downloading if a URL is set.

    The zip archive is extracted into the parent of the configured path, so the
    archive should contain exactly the directory (or file) named in *path*.
    """
    local_path = _resolve_dataset_path(dataset.path)
    if not local_path.exists():
        if not dataset.url:
            raise FileNotFoundError(f"Dataset not found at {local_path} and no download URL is configured.")
        extract_dir = local_path.parent
        extract_dir.mkdir(parents=True, exist_ok=True)
        download_and_extract(dataset.url, str(extract_dir))
    return local_path
