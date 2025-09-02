"""src/main.py – entry point (``python -m src.main``).
This script orchestrates the (demo) continual-learning experiments by combining
training utilities from ``src.train`` with metric / plotting helpers from
``src.evaluate``.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Dict, List

import numpy as np

from .evaluate import Metrics, save_line_plot
from .train import train_stream

# -----------------------------------------------------------------------------
#  Demo Experiment (≈10 min CPU) ----------------------------------------------
# -----------------------------------------------------------------------------

def exp1_demo() -> None:
    """Budget-scaling experiment with 2 methods × 2 budgets × 2 seeds.

    A *minimal* variant of the paper's Experiment-1 so the whole project runs on
    commodity hardware in a reasonable time.  We use only 3 tasks from Split
    CIFAR-100 and a single epoch per task.
    """
    print("\n================ EXPERIMENT 1  – Budget-Scaling (DEMO) ================")
    print("Purpose: Demonstrate that BITSPLIT converts memory into accuracy vs. Finetune.\n")

    budgets = [0.05, 0.5]      # MB
    methods = ["bitsplit", "finetune"]
    seeds = [0, 1]

    # results[method][budget] = List[Metrics]
    results: Dict[str, Dict[float, List[Metrics]]] = {m: {b: [] for b in budgets} for m in methods}

    for b in budgets:
        for m in methods:
            for s in seeds:
                print(f"\n>>> Running method={m}, budget={b} MB, seed={s}")
                metrics = train_stream(m, budget_mb=b, seed=s, tasks=3)  # 3 tasks only
                results[m][b].append(metrics)

    # -------------- Aggregate metrics -------------------------------------
    summary_rows = []
    for m in methods:
        for b in budgets:
            aacc = np.mean([mtr.aacc() for mtr in results[m][b]])
            facc = np.mean([mtr.facc() for mtr in results[m][b]])
            bpa = np.mean([mtr.bpa() for mtr in results[m][b]])
            row = {"method": m, "budget_MB": b, "AACC": aacc, "FACC": facc, "BPA": bpa}
            summary_rows.append(row)
            print(json.dumps(row, indent=2))

    # -------------- Plot ---------------------------------------------------
    for metric_name in ["AACC", "BPA"]:
        for m in methods:
            y = [row[metric_name] for row in summary_rows if row["method"] == m]
            save_line_plot(budgets, y,
                           xlabel="Budget (MB)", ylabel=metric_name,
                           title=f"{metric_name} vs Budget – {m}",
                           filename=f"{metric_name.lower()}_{m}")

# -----------------------------------------------------------------------------
#  CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run continual-learning experiments (demo)")
    parser.add_argument("--experiment", type=str, default="exp1_demo",
                        choices=["exp1_demo"],
                        help="Which experiment to run (only exp1_demo is implemented in the demo code).")
    return parser.parse_args()


def main() -> None:  # entry-point
    args = _parse_args()
    start = datetime.now()

    if args.experiment == "exp1_demo":
        exp1_demo()
    else:
        raise NotImplementedError(args.experiment)

    elapsed = datetime.now() - start
    print(f"\n>>> Total runtime: {elapsed}\n")


if __name__ == "__main__":
    main()
