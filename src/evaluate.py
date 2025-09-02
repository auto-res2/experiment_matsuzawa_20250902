"""src/evaluate.py
Evaluation utilities: metrics computation and plotting helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="paper", font_scale=1.4)

# -----------------------------------------------------------------------------
#  Metric helpers
# -----------------------------------------------------------------------------

def calc_avg_acc(acc_matrix: np.ndarray) -> float:
    """Average accuracy after final task."""
    return float(acc_matrix[-1].mean() * 100)


def calc_forgetting(acc_matrix: np.ndarray) -> float:
    """Average forgetting across tasks (Chaudhry et al. 2018)."""
    T = acc_matrix.shape[0]
    fgt = []
    for t in range(T - 1):
        prev_max = acc_matrix[t, t]
        later = acc_matrix[-1, t]
        fgt.append(max(0, prev_max - later))
    return float(np.mean(fgt) * 100)

# -----------------------------------------------------------------------------
#  Plotting helpers
# -----------------------------------------------------------------------------

def save_lineplot(
    xs: List[int],
    ys_dict: Dict[str, List[float]],
    ylabel: str,
    title: str,
    fname: str,
):
    """Save a publication-quality line plot *fname* in PDF format.

    All images are stored under `.research/iteration6/images` as required.
    """
    # Ensure the output directory exists
    out_dir = Path(".research/iteration6/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / fname

    plt.figure(figsize=(6, 4))
    for name, ys in ys_dict.items():
        plt.plot(xs, ys, marker="o", label=name)
        for x, y in zip(xs, ys):
            plt.text(x, y, f"{y:.1f}", fontsize=7, ha="center", va="bottom")
    plt.xlabel("Task")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(full_path, bbox_inches="tight")
        print(f"Saved figure → {full_path}")
    except Exception as e:
        print(f"[WARN] Could not save figure {full_path}: {e}")
    plt.close()

__all__ = ["calc_avg_acc", "calc_forgetting", "save_lineplot"]
