"""src/main.py
Entry point orchestrating the continual-learning experiments.
Run with:
    python -m src.main exp1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as tud
from sklearn.metrics import accuracy_score
from torch.cuda import amp

# Local imports
from src.preprocess import get_split_cifar100, set_seed
from src.train import (
    FDSketch,
    ReservoirBuffer,
    ViTLoRAClassifier,
    train_one_task,
)
from src.evaluate import calc_avg_acc, calc_forgetting, save_lineplot

# -----------------------------------------------------------------------------
#  Helper – ensure stream split exists
# -----------------------------------------------------------------------------

def _ensure_split_json() -> Path:
    """Create a default sequential split file if it doesn't already exist."""
    split_path = Path("streams/split_cifar100.json")
    if split_path.is_file():
        return split_path

    print(f"[INFO] Required split file '{split_path}' not found – generating default split.")
    split_path.parent.mkdir(parents=True, exist_ok=True)

    # Deterministic sequential 20×5 class split
    split = {str(i): list(range(i * 5, (i + 1) * 5)) for i in range(20)}

    with split_path.open("w") as f:
        json.dump(split, f)

    print(f"[INFO] Default split file generated at '{split_path}'.")
    return split_path

# -----------------------------------------------------------------------------
#  Experiment 1 – Split CIFAR-100
# -----------------------------------------------------------------------------

def run_exp1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==== Experiment-1: Head-to-Head Memory–Accuracy Benchmark ====")
    print("Dataset : Split CIFAR-100  (20 × 5 classes)")
    print("Backbone: ViT-B/16 + LoRA  |  Methods: FSR, ER, Finetune")

    # Ensure the JSON split file is present
    _ensure_split_json()

    results: Dict[str, Dict[str, List[float]]] = {
        m: {"acc": [], "fgt": []} for m in ["FSR", "ER", "Finetune"]
    }

    for seed in [21, 42, 84]:
        set_seed(seed)
        for method in results.keys():
            model = ViTLoRAClassifier(num_classes=100, fsr=(method == "FSR")).to(device)
            opt = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=5e-4,
                weight_decay=0.05,
            )
            scaler = amp.GradScaler()
            buffer = ReservoirBuffer(25 * 100) if method == "ER" else None
            sketches: Dict[int, FDSketch] | None = {} if method == "FSR" else None
            acc_matrix = np.zeros((20, 20))
            seen_classes: List[int] = []

            # -----------------------------
            # Task loop
            # -----------------------------
            for task_id in range(20):
                subset_train, cls_ids = get_split_cifar100(
                    root="data",
                    task_json="streams/split_cifar100.json",
                    task_id=task_id,
                    train=True,
                )
                seen_classes += cls_ids
                loader = tud.DataLoader(
                    subset_train, batch_size=64, shuffle=True, num_workers=4
                )

                train_one_task(
                    model=model,
                    task_loader=loader,
                    replay_method=method,
                    optimizer=opt,
                    scaler=scaler,
                    buffer=buffer,
                    sketches=sketches,
                    seen_classes=seen_classes,
                    device=str(device),
                )

                # -------- evaluation on all seen tasks --------
                model.eval()
                with torch.no_grad():
                    for j in range(task_id + 1):
                        subset_test, _ = get_split_cifar100(
                            root="data",
                            task_json="streams/split_cifar100.json",
                            task_id=j,
                            train=False,
                        )
                        test_loader = tud.DataLoader(
                            subset_test, batch_size=128, shuffle=False, num_workers=4
                        )
                        preds, gts = [], []
                        for xb, yb in test_loader:
                            xb = xb.to(device, non_blocking=True)
                            logits, _ = model(xb)
                            preds.append(logits.argmax(1).cpu())
                            gts.append(yb)
                        preds_np = torch.cat(preds).numpy()
                        gts_np = torch.cat(gts).numpy()
                        acc_matrix[task_id, j] = accuracy_score(gts_np, preds_np)

            acc = calc_avg_acc(acc_matrix)
            fgt = calc_forgetting(acc_matrix)
            results[method]["acc"].append(acc)
            results[method]["fgt"].append(fgt)
            print(f"Seed {seed:2d} | {method:8s}  ACC = {acc:5.2f}  FGT = {fgt:5.2f}")

    # ------------------------------------------------------------------
    # Aggregate across seeds & visualise
    # ------------------------------------------------------------------
    for m, rec in results.items():
        acc_mean, acc_std = np.mean(rec["acc"]), np.std(rec["acc"])
        fgt_mean, fgt_std = np.mean(rec["fgt"]), np.std(rec["fgt"])
        print(
            f"{m:9s} | ACC  {acc_mean:5.2f} ± {acc_std:4.2f} | FGT  {fgt_mean:5.2f} ± {fgt_std:4.2f}"
        )

    xs = list(range(1, 21))
    for metric, ylabel in [
        ("acc", "Average Accuracy (%)"),
        ("fgt", "Average Forgetting (%)"),
    ]:
        ys_dict = {m: results[m][metric] for m in results}
        fname = f"{metric}_cifar100.pdf"
        save_lineplot(xs, ys_dict, ylabel, f"{metric.upper()} vs Task", fname)

    print("Figures saved in .research/iteration2/images.")

# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="FSR Continual-Learning experiments")
    parser.add_argument("exp", choices=["exp1"], help="Experiment id to run")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.exp == "exp1":
        run_exp1()
    else:
        raise NotImplementedError("Only 'exp1' is available in this refactored version.")


if __name__ == "__main__":
    main()
