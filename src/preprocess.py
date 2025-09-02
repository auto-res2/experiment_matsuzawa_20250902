"""src/preprocess.py
Data loading and preprocessing utilities for the continual-learning streams.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torchvision.transforms as T
from torchvision.datasets import CIFAR100
import torch.utils.data as tud

# Root of JSON stream definitions (../streams/ as per repo spec)
STREAM_DIR = Path(__file__).resolve().parent.parent / "streams"

# ---------------------------------------------------------------------------
#  VISION TRANSFORMS
# ---------------------------------------------------------------------------

def _vision_tf(train: bool = True):
    if train:
        return T.Compose(
            [
                T.Resize(224),
                T.RandomResizedCrop(224, scale=(0.6, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandAugment(2, 9),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


# ---------------------------------------------------------------------------
#  GENERIC STREAM DATASET LOADER
# ---------------------------------------------------------------------------

def get_stream_dataset(name: str, task_idx: int, *, train: bool):
    """Return (subset, class_ids) for <name>.json definition in *streams/*."""
    split_file = STREAM_DIR / f"{name}.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Stream definition file {split_file} not found.")

    try:
        classes = json.loads(split_file.read_text())[str(task_idx)]
    except (json.JSONDecodeError, KeyError) as err:
        raise RuntimeError(f"Malformed stream definition in {split_file}: {err}") from err

    base = CIFAR100("data", train=train, download=True)
    idx = [i for i, (_, y) in enumerate(base) if y in classes]
    subset = tud.Subset(base, idx)
    subset.dataset.transform = _vision_tf(train)
    return subset, classes


__all__ = ["get_stream_dataset"]