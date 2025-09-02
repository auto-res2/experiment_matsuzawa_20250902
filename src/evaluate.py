"""src/evaluate.py
Metric containers and plotting helpers – separated from the training logic so
that they can be re-used for multiple experimental set-ups.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# -----------------------------------------------------------------------------
#  Metric container ------------------------------------------------------------
# -----------------------------------------------------------------------------

@dataclass
class Metrics:
    """Simple container that tracks accuracy & used bits after each task."""

    acc_task: List[float] = field(default_factory=list)
    bits_task: List[int] = field(default_factory=list)

    # ---------------------
    def log(self, acc: float, bits: int) -> None:
        self.acc_task.append(float(acc))
        self.bits_task.append(int(bits))

    # ---------------------
    def aacc(self) -> float:
        """Average accuracy across tasks (AACC)."""
        return float(np.mean(self.acc_task)) if self.acc_task else 0.0

    def facc(self) -> float:
        """Final accuracy after last task (FACC)."""
        return float(self.acc_task[-1]) if self.acc_task else 0.0

    def bpa(self) -> float:
        """Bits-per-percent-accuracy (BPA).  Lower is better."""
        eps = 1e-9
        # convert last stored *bits* to mega-bytes and divide by average acc
        return (self.bits_task[-1] / 8 / 1024 / 1024) / (self.aacc() + eps)

# -----------------------------------------------------------------------------
#  Plot helpers ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def save_line_plot(x: List, y: List, *, xlabel: str, ylabel: str, title: str, filename: str) -> None:
    """Save a labelled line-plot as *figures/<filename>.pdf*.

    For cleanliness we keep plotting related imports *inside* the function.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o", label=title)
    for xi, yi in zip(x, y):
        plt.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save figure in the requested directory
    # ------------------------------------------------------------------
    img_dir = Path(".research/iteration3/images")  # updated directory as requested
    img_dir.mkdir(parents=True, exist_ok=True)
    path = img_dir / f"{filename}.pdf"
    try:
        plt.savefig(path, bbox_inches="tight")
        print(f"[Figure saved] {path}")
    finally:
        plt.close()
