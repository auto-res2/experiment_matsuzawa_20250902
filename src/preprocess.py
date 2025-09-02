"""src/preprocess.py
Data loading, augmentation and reproducibility utilities.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data.dataset import Subset
from torchvision.datasets import CIFAR100

# -----------------------------------------------------------------------------
#  Reproducibility utils
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Fix all random seeds for deterministic behaviour."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
#  Split CIFAR-100 loader (20 × 5 classes)
# -----------------------------------------------------------------------------

def get_split_cifar100(
    *,
    root: str | Path,
    task_json: str | Path,
    task_id: int,
    train: bool = True,
) -> Tuple[Subset, List[int]]:
    """Return a ``torch.utils.data.Subset`` with the classes of *task_id*.

    The JSON file is expected to map string task indices to lists of class ids.
    """

    task_json = Path(task_json)
    if not task_json.is_file():
        raise FileNotFoundError(
            f"Task-split json '{task_json}' not found. Did you forget to generate the stream?"
        )

    with task_json.open() as f:
        split = json.load(f)
    class_ids = split[str(task_id)]  # e.g. 5 class ints

    base = CIFAR100(root, train=train, download=True)
    idx = [i for i, (_, y) in enumerate(base) if y in class_ids]
    subset = Subset(base, idx)

    transform_train = T.Compose(
        [
            T.Resize(224),
            T.RandomResizedCrop(224, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandAugment(2, 9),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    transform_test = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    subset.dataset.transform = transform_train if train else transform_test
    return subset, class_ids

__all__ = ["set_seed", "get_split_cifar100"]
