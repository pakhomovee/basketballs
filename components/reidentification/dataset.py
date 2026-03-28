"""SynergyReID dataset loader with PK sampling for triplet training."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

_FILENAME_RE = re.compile(r"^(\d+)_(\d+)_(\d+)\.jpeg$")

INPUT_SIZE = (256, 128)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_IMAGENET_MEAN_TENSOR = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_IMAGENET_STD_TENSOR = torch.tensor(IMAGENET_STD).view(3, 1, 1)

Sample = tuple[str, int, int]


def _parse_filename(name: str) -> tuple[int, int, int] | None:
    m = _FILENAME_RE.match(name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _scan_samples(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for path in sorted(root.iterdir()):
        parsed = _parse_filename(path.name)
        if parsed is None:
            continue
        pid, seq, _frame = parsed
        samples.append((path.name, pid, seq))
    return samples


def _to_float_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert a (H, W, C) uint8 BGR numpy array to a (C, H, W) float32 tensor in [0, 1]."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INPUT_SIZE[1], INPUT_SIZE[0]))
    return torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)


def load_reid_image(path: str | Path) -> torch.Tensor:
    """Load, resize, and normalize a player-crop image (used for eval/inference)."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return preprocess_reid_image(image)


def _load_reid_image_raw(path: str | Path) -> torch.Tensor:
    """Load and resize a player-crop image as a [0, 1] float tensor — no normalization."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return _to_float_tensor(image)


def preprocess_reid_image(image: np.ndarray) -> torch.Tensor:
    tensor = _to_float_tensor(image)
    return (tensor - _IMAGENET_MEAN_TENSOR) / _IMAGENET_STD_TENSOR


class _BaseReIDImageDataset(Dataset):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.samples: list[Sample] = _scan_samples(self.root)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, fname: str) -> torch.Tensor:
        return load_reid_image(self.root / fname)


def build_train_transform(flip_prob: float = 0.5, erase_prob: float = 0.5) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a transform pipeline for training."""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.Pad(10),
            transforms.RandomCrop((INPUT_SIZE[0], INPUT_SIZE[1])),
            transforms.RandomErasing(p=erase_prob, value="random"),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class SynergyReIDDataset(_BaseReIDImageDataset):
    """Loads player crops from a flat directory of ``{pid}_{seq}_{frame}.jpeg`` files."""

    def __init__(
        self,
        root: str | Path,
        transform=None,
    ):
        super().__init__(root)
        self.transform = transform

        self.pid_to_label: dict[int, int] = {}
        self.label_to_indices: dict[int, list[int]] = defaultdict(list)

        raw_pids = sorted({pid for _, pid, _ in self.samples})
        for i, pid in enumerate(raw_pids):
            self.pid_to_label[pid] = i

        for idx, (_, pid, _) in enumerate(self.samples):
            label = self.pid_to_label[pid]
            self.label_to_indices[label].append(idx)

        self.num_pids = len(self.pid_to_label)

    def __getitem__(self, index: int):
        fname, pid, seq = self.samples[index]
        img = _load_reid_image_raw(self.root / fname)

        if self.transform is not None:
            img = self.transform(img)

        label = self.pid_to_label[pid]
        return img, label, seq


class QueryGalleryDataset(_BaseReIDImageDataset):
    """Loads query or gallery images. PIDs may be unknown (gallery in challenge set)."""

    def __init__(self, root: str | Path):
        super().__init__(root)

    def __getitem__(self, index: int):
        fname, pid, seq = self.samples[index]
        img = self._load_image(fname)
        return img, pid, seq


class PKSampler(Sampler):
    """Samples P random identities, then K random images for each identity."""

    def __init__(self, dataset: SynergyReIDDataset, p: int = 16, k: int = 4):
        self.label_to_indices = dataset.label_to_indices
        self.labels = list(self.label_to_indices.keys())
        self.p = p
        self.k = k
        self.batch_size = p * k

    def _sample_k_indices(self, label: int, rng: np.random.Generator) -> list[int]:
        indices = self.label_to_indices[label]
        replace = len(indices) < self.k
        return rng.choice(indices, size=self.k, replace=replace).tolist()

    def __iter__(self):
        rng = np.random.default_rng()
        labels = self.labels.copy()
        rng.shuffle(labels)

        batch: list[int] = []
        for label in labels:
            batch.extend(self._sample_k_indices(label, rng))

            if len(batch) >= self.batch_size:
                yield from batch[: self.batch_size]
                batch = batch[self.batch_size :]

    def __len__(self) -> int:
        return (len(self.labels) // self.p) * self.batch_size
