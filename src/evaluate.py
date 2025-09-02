"""src/evaluate.py
Evaluation helpers: accuracy metrics, forgetting/accuracy aggregation, and
common publication-quality plotting utilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import torch
import torch.utils.data as tud

sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
#  BASIC METRICS
# ---------------------------------------------------------------------------

def accuracy(loader: tud.DataLoader, model: torch.nn.Module, *, device: str = "cuda") -> float:
    """Compute *top-1* accuracy of *model* on *loader* (no grad)."""
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out, _ = model(xb)
            preds.append(out.argmax(1).cpu())
            gts.append(yb.cpu())
    preds = torch.cat(preds)
    gts = torch.cat(gts)
    return accuracy_score(gts, preds)


# ---------------------------------------------------------------------------
#  CONTINUAL-LEARNING AGGREGATES
# ---------------------------------------------------------------------------

def avg_accuracy(acc_mat: np.ndarray) -> float:
    """Average over final-row accuracies (ACC metric)."""
    return acc_mat[-1].mean() * 100.0


def avg_forgetting(acc_mat: np.ndarray) -> float:
    """Average forgetting as defined in Lopez-Paz & Ranzato (2017)."""
    T = acc_mat.shape[0]
    fgt: List[float] = []
    for t in range(T - 1):
        fgt.append(max(0, acc_mat[t, t] - acc_mat[-1, t]))
    return float(np.mean(fgt) * 100.0)


# ---------------------------------------------------------------------------
#  PLOTTING UTILITIES
# ---------------------------------------------------------------------------

def lineplot(
    xs: List[int] | np.ndarray,
    ys_dict: Dict[str, List[float] | np.ndarray],
    ylabel: str,
    title: str,
    fname: Path | str,
) -> None:
    """Matplotlib line plot with automatic annotation & tight-layout save."""
    plt.figure(figsize=(7, 4))
    for name, ys in ys_dict.items():
        plt.plot(xs, ys, label=name, marker="o", markersize=3)
    plt.xlabel("Task")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    # annotate final points
    for name, ys in ys_dict.items():
        plt.text(xs[-1], ys[-1], f"{ys[-1]:.1f}", fontsize=6)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


__all__ = [
    "accuracy",
    "avg_accuracy",
    "avg_forgetting",
    "lineplot",
]