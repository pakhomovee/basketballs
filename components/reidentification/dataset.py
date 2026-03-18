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

# Standard ReID input size (height, width) — people are taller than wide
INPUT_SIZE = (256, 128)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGENET_MEAN_ARRAY = np.array(IMAGENET_MEAN, dtype=np.float32)
IMAGENET_STD_ARRAY = np.array(IMAGENET_STD, dtype=np.float32)

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


def load_reid_image(path: str | Path) -> torch.Tensor:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return preprocess_reid_image(image)


def preprocess_reid_image(image: np.ndarray) -> torch.Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INPUT_SIZE[1], INPUT_SIZE[0]))
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN_ARRAY) / IMAGENET_STD_ARRAY
    return torch.from_numpy(image).permute(2, 0, 1).float()


class _BaseReIDImageDataset(Dataset):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.samples: list[Sample] = _scan_samples(self.root)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, fname: str) -> torch.Tensor:
        return load_reid_image(self.root / fname)


def build_train_transform(flip_prob: float = 0.5, erase_prob: float = 0.5) -> Callable[[torch.Tensor], torch.Tensor]:
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.RandomErasing(p=erase_prob, value="random"),
        ]
    )


class SynergyReIDDataset(_BaseReIDImageDataset):
    """Loads player crops from a flat directory of ``{pid}_{seq}_{frame}.jpeg`` files.

    Returns (image_tensor, pid_label, cam_id) where pid_label is a contiguous
    class index (0 … N-1) and cam_id is the sequence number (used at eval time
    to exclude same-camera matches).
    """

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
        img = self._load_image(fname)

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


# ---------------------------------------------------------------------------
# PK Sampler: sample P identities × K instances per batch
# ---------------------------------------------------------------------------


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
