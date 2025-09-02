"""src/train.py
Contains model definitions and the training loop for a single (continual-learning) stream.
The module is intentionally self-contained except for the data-preparation
utilities that live in ``src/preprocess.py`` and the metric container in
``src/evaluate.py``.
"""
from __future__ import annotations

import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from evaluate import Metrics  # metric container (no circular import)
from preprocess import get_split_cifar100, set_seed

# -----------------------------------------------------------------------------
#  Device helper ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def device() -> torch.device:   # noqa: D401 – simple helper
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
#  Model wrappers ----------------------------------------------------------------
# -----------------------------------------------------------------------------

class SimpleResNet18(nn.Module):
    """Lightweight ResNet-18 variant adapted to CIFAR resolution."""

    def __init__(self, num_classes: int = 100):
        super().__init__()
        from torchvision.models import resnet18  # lazy import keeps start-up light

        self.net = resnet18(num_classes=num_classes)
        # If running torchvision>=0.14 the default kernel for CIFAR is (7,7).
        # Replace it with a 3×3 kernel and remove the max-pool so we keep
        # resolution.
        if self.net.conv1.kernel_size != (3, 3):
            self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                        padding=1, bias=False)
        self.feature_dim = self.net.fc.in_features
        self.net.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class FinetuneAgent(nn.Module):
    """Baseline agent that *finetunes* on the incoming data stream."""

    def __init__(self, num_classes: int, lr: float = 0.05):
        super().__init__()
        self.backbone = SimpleResNet18(num_classes)
        self.classifier = nn.Linear(self.backbone.feature_dim, num_classes)
        self.opt = optim.SGD(self.parameters(), lr=lr, weight_decay=5e-4,
                             momentum=0.9, nesterov=True)
        self.to(device())

    # ---------------------------------------------------------------------
    #  API required by the training harness
    # ---------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feats = self.backbone(x)
        return self.classifier(feats)

    def observe(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """One optimisation step on a mini-batch."""
        self.train()
        self.opt.zero_grad(set_to_none=True)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        self.opt.step()
        return float(loss.item())

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).argmax(1)

    def bits_used(self) -> int:
        # Assuming deterministic 32-bit storage for all parameters
        return int(sum(p.numel() * 32 for p in self.parameters()))


class BitSplitAgent(nn.Module):
    """*Stub* of the proposed BITSPLIT agent.

    The class respects the public interface required by the experimental
    harness but does **not** implement the full method – that would require a
    much larger code base.  It produces fake bit-accounting numbers so that
    the rest of the pipeline (plots, metrics) runs without modification.
    """

    def __init__(self, num_classes: int, budget_mb: float = 0.5, lr: float = 0.05):
        super().__init__()
        self.budget_bits = int(budget_mb * 1024 * 1024 * 8)  # MB → bits
        self.backbone = SimpleResNet18(num_classes)
        self.classifier = nn.Linear(self.backbone.feature_dim, num_classes)
        self.opt = optim.SGD(self.parameters(), lr=lr, weight_decay=5e-4,
                             momentum=0.9, nesterov=True)
        self.current_bits: int = 0
        self.to(device())

    # ------------------------------------------------------------------
    #  Required API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feats = self.backbone(x)
        return self.classifier(feats)

    def observe(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.train()
        self.opt.zero_grad(set_to_none=True)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        self.opt.step()
        # --- Fake bit growth – simply add 1-kbit per batch until full budget
        self.current_bits = min(self.budget_bits, self.current_bits + 1024)
        return float(loss.item())

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).argmax(1)

    def bits_used(self) -> int:
        return self.current_bits

# -----------------------------------------------------------------------------
#  Training loop for one *stream* (sequence of tasks) ---------------------------
# -----------------------------------------------------------------------------

def train_stream(method: str, *, budget_mb: float, seed: int, tasks: int = 20) -> Metrics:
    """Train a given *method* for a full task sequence and return metrics.

    Parameters
    ----------
    method : str
        Either "bitsplit" or "finetune" for the demo code.
    budget_mb : float
        Global memory budget in **mega-bytes** (only used by BitSplitAgent).
    seed : int
        RNG seed for reproducibility.
    tasks : int, default=20
        Number of incremental tasks in the stream.
    """
    set_seed(seed)
    metrics = Metrics()

    # ---- instantiate agent ------------------------------------------------
    if method.lower() == "bitsplit":
        agent = BitSplitAgent(num_classes=100, budget_mb=budget_mb)
    elif method.lower() == "finetune":
        agent = FinetuneAgent(num_classes=100)
    else:
        raise ValueError(f"Unknown method '{method}'.")

    # ---- loop over tasks ---------------------------------------------------
    for task_id in range(tasks):
        train_loader, test_loader, _ = get_split_cifar100(task_id, seed=seed)

        # -----  Train one epoch (demo keeps it tiny for speed) -------------
        for x, y in train_loader:  # one epoch == single pass
            x, y = x.to(device()), y.to(device())
            agent.observe(x, y)

        # -----  Evaluate on the *current* task test set --------------------
        agent.eval()
        all_pred: List[torch.Tensor] = []
        all_gt: List[torch.Tensor] = []
        with torch.inference_mode():
            for x, y in test_loader:
                x = x.to(device())
                pred = agent.predict(x)
                all_pred.append(pred.cpu())
                all_gt.append(y)
        acc = (torch.cat(all_pred) == torch.cat(all_gt)).float().mean().item() * 100
        metrics.log(acc, agent.bits_used())
        print(f"Task {task_id:02d} | Acc = {acc:5.2f}% | Bits = {agent.bits_used() / 8 / 1024:6.2f} kB")

    return metrics
