"""src/preprocess.py
Data loading & utility helpers (deterministic seeding, Split CIFAR-100 loading).
"""
from __future__ import annotations

import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100

# -----------------------------------------------------------------------------
#  Reproducibility helpers -----------------------------------------------------
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy & PyTorch (incl. CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
#  Split CIFAR-100 loader -------------------------------------------------------
# -----------------------------------------------------------------------------

def _subset_dataset(ds: CIFAR100, class_ids: List[int]) -> CIFAR100:
    """Return *view* of ``ds`` that only keeps samples whose labels ∈ class_ids."""
    idx = [i for i, (_, y) in enumerate(ds) if y in class_ids]
    # in-place slice is fine for demo (avoids expensive copies)
    ds.data = ds.data[idx]
    ds.targets = [ds.targets[i] for i in idx]
    return ds


class _RandomTensorDataset(Dataset):
    """Synthetic dataset used when CIFAR-100 is unavailable (e.g., offline)."""

    def __init__(self, class_ids: List[int], samples_per_class: int, transform: T.Compose | None = None):
        self.labels: List[int] = []
        for cid in class_ids:
            self.labels.extend([cid] * samples_per_class)
        self.transform = transform

    def __len__(self) -> int:  # noqa: D401 – concise
        return len(self.labels)

    def __getitem__(self, idx: int):  # noqa: D401 – concise
        img = torch.randn(3, 128, 128)  # size compatible with downstream transforms
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


def get_split_cifar100(task_id: int, *, classes_per_task: int = 5, seed: int = 0
                       ) -> Tuple[DataLoader, DataLoader, List[int]]:
    """Return train/test DataLoaders for *one* incremental task.

    The CIFAR-100 class order is re-shuffled *per seed* – sufficient for a demo
    script.  A production set-up would pre-compute and store the order.
    """
    set_seed(seed)

    # -------- determine class slice ---------------------------------------
    all_classes = list(range(100))
    random.shuffle(all_classes)
    class_slice = all_classes[task_id * classes_per_task:(task_id + 1) * classes_per_task]

    # -------- transforms ---------------------------------------------------
    transform_train = T.Compose([
        T.RandomResizedCrop(112, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    transform_test = T.Compose([
        T.Resize(128),
        T.CenterCrop(112),
        T.ToTensor(),
        T.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

    # -------- dataset download --------------------------------------------
    dataset_loaded = False
    try:
        train_ds: CIFAR100 | Dataset = CIFAR100(root="data", train=True, download=True,
                           transform=transform_train)
        test_ds: CIFAR100 | Dataset = CIFAR100(root="data", train=False, download=True,
                          transform=transform_test)
        train_ds = _subset_dataset(train_ds, class_slice)
        test_ds = _subset_dataset(test_ds, class_slice)
        dataset_loaded = True
    except (RuntimeError, OSError) as e:
        # Most likely running in an environment without internet access. Use a synthetic fallback.
        print("[Warning] CIFAR-100 unavailable – using synthetic random data.  Reason:", str(e))

    if not dataset_loaded:
        # --------------- Synthetic fallback --------------------------------
        train_ds = _RandomTensorDataset(class_slice, samples_per_class=250, transform=transform_train)
        test_ds = _RandomTensorDataset(class_slice, samples_per_class=50, transform=transform_test)

    # ----------------- Dataloaders -----------------------------------------
    num_workers = min(os.cpu_count() or 1, 4)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, class_slice
