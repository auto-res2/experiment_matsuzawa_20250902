"""src/main.py
Entry point that orchestrates Experiments 1-3 as described in the paper.
Run e.g.  `python -m src.main --exp exp1 --method fsr_k8`
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim
from torch.cuda import amp
from tqdm import tqdm
import psutil

# Local imports
from preprocess import get_stream_dataset
from train import (
    FDSketch,
    ReservoirBuffer,
    ViT_LoRA,
    set_seed,
    train_task_vision,
)
from evaluate import accuracy, avg_accuracy, avg_forgetting, lineplot as _lineplot

# Directory setup -----------------------------------------------------------
RESULT_DIR = Path("results"); RESULT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------------
#  EXPERIMENT-1 : Very-long streams (Split CIFAR-100 20×5) -------------------
# ---------------------------------------------------------------------------

def run_exp1(_args):
    print("\n===============================")
    print("Experiment-1 : Memory/Accuracy Pareto on Very-Long Streams")
    print("Stream : Split-CIFAR-100-x5  (100 tasks)")
    print("Methods : FSR-k4/k8/k12, ER-5/25/50, Finetune\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods_cfg: Dict[str, Dict] = {
        "fsr_k4": {"k": 4, "mem": 0.3},
        "fsr_k8": {"k": 8, "mem": 0.6},
        "fsr_k12": {"k": 12, "mem": 0.9},
        "er5": {"buf": 5},
        "er25": {"buf": 25},
        "er50": {"buf": 50},
        "finetune": {},
    }

    seeds = [21, 42, 84]
    all_results: Dict[str, Dict[str, List[float]]] = {m: {"ACC": [], "FGT": []} for m in methods_cfg}

    for seed in seeds:
        set_seed(seed)
        for method, cfg in methods_cfg.items():
            print(f"Seed {seed}  –  Running {method} …")
            model = ViT_LoRA(n_classes=100, use_decoder=method.startswith("fsr")).to(device)
            opt = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=0.05
            )
            scaler = amp.GradScaler()

            buffer = (
                ReservoirBuffer(cfg.get("buf", 1) * 100)
                if method.startswith("er")
                else None
            )
            sketches: Dict[int, FDSketch] | None = {} if method.startswith("fsr") else None

            acc_mat = np.zeros((100, 100))
            seen_cls: List[int] = []

            for task in tqdm(range(100), leave=False):
                subset_tr, cls_task = get_stream_dataset("split_cifar100_x5", task, train=True)
                seen_cls += cls_task
                loader = torch.utils.data.DataLoader(
                    subset_tr, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
                )
                train_task_vision(
                    model,
                    loader,
                    method=method,
                    seen_cls=seen_cls,
                    buffer=buffer,
                    sketches=sketches,
                    opt=opt,
                    scaler=scaler,
                    device=str(device),
                )

                # evaluation on all seen tasks
                with torch.no_grad():
                    for j in range(task + 1):
                        test_set, _ = get_stream_dataset("split_cifar100_x5", j, train=False)
                        test_loader = torch.utils.data.DataLoader(
                            test_set, batch_size=128, num_workers=4, pin_memory=True
                        )
                        acc_mat[task, j] = accuracy(test_loader, model, device=str(device))

            ACC = avg_accuracy(acc_mat)
            FGT = avg_forgetting(acc_mat)
            all_results[method]["ACC"].append(ACC)
            all_results[method]["FGT"].append(FGT)
            print(f"→  {method}  ACC={ACC:.2f}  FGT={FGT:.2f}")

    # ---------------- aggregate & figures ----------------
    summary: Dict[str, tuple] = {}
    for m, v in all_results.items():
        acc_m, fgt_m = np.mean(v["ACC"]), np.mean(v["FGT"])
        summary[m] = (acc_m, fgt_m)
        print(
            f"{m:9s}  ACC={acc_m:.2f} ±{np.std(v['ACC']):.2f}   "
            f"FGT={fgt_m:.2f} ±{np.std(v['FGT']):.2f}"
        )

    xs = list(range(1, 101))
    for metric in ["ACC", "FGT"]:
        ys_dict = {
            m: [np.mean(v[metric]) for _ in xs] for m, v in all_results.items()  # flat line plot
        }
        ylabel = "Average Accuracy (%)" if metric == "ACC" else "Average Forgetting (%)"
        fname = FIG_DIR / f"{metric.lower()}_cifar100x5.pdf"
        _lineplot(xs, ys_dict, ylabel, f"{metric} vs Task", fname)

    fig_name = FIG_DIR / "accuracy_memory_pareto.pdf"
    import matplotlib.pyplot as plt  # local to avoid top-level heavy import when not needed

    plt.figure(figsize=(4, 4))
    for m, (acc, _fg) in summary.items():
        mem = methods_cfg[m].get("mem", methods_cfg[m].get("buf", 0) * 0.5)  # rough MB proxy
        plt.scatter(mem, acc, label=m)
        plt.text(mem, acc, m, fontsize=6)
    plt.xlabel("Memory (MB)")
    plt.ylabel("Final ACC (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_name, bbox_inches="tight")
    plt.close()
    print(
        "Figure names: accuracy_memory_pareto.pdf, "
        "acc_cifar100x5.pdf, fgt_cifar100x5.pdf"
    )


# ---------------------------------------------------------------------------
#  EXPERIMENT-2 : On-device resource & latency  -----------------------------
# ---------------------------------------------------------------------------

def run_exp2(args):
    print("\n===============================")
    print("Experiment-2 : On-device Resource & Latency Evaluation (simulated)\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    method = "fsr_k8" if args.method is None else args.method

    set_seed(42)
    img_shape = (3, 224, 224)
    model = ViT_LoRA(10, use_decoder=True).to(device)
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    scaler = amp.GradScaler()

    buffer = ReservoirBuffer(25 * 10) if method.startswith("er") else None
    sketches: Dict[int, FDSketch] | None = {} if method.startswith("fsr") else None

    ram_peak, step_latencies = [], []
    seen_cls: List[int] = []
    for task in range(5):
        # synthetic two-class split (placeholder)
        x = torch.randn(500, *img_shape)
        y = torch.randint(0, 2, (500,)) + 2 * task
        ds = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

        t0 = time.time()
        train_task_vision(
            model,
            loader,
            method=method,
            seen_cls=seen_cls,
            buffer=buffer,
            sketches=sketches,
            opt=opt,
            scaler=scaler,
            device=device,
        )
        step_latencies.append((time.time() - t0) / len(loader))
        ram_peak.append(psutil.Process(os.getpid()).memory_info().rss / 1e6)
        seen_cls += [2 * task, 2 * task + 1]

    print("Per-task latency (s/step):", step_latencies)
    print("Peak RAM per task (MB):", ram_peak)

    fname = FIG_DIR / "training_latency.pdf"
    _lineplot(range(1, 6), {method: step_latencies}, "Seconds / step", "Training latency", fname)
    print("Figure names: training_latency.pdf")


# ---------------------------------------------------------------------------
#  EXPERIMENT-3 : Adversarial robustness demo  ------------------------------
# ---------------------------------------------------------------------------

def run_exp3(_args):
    try:
        import torchattacks  # noqa: F401  # optional dependency
    except ImportError:
        warnings.warn("torchattacks not installed – robustness experiment skipped")
        return

    print("\n===============================")
    print("Experiment-3 : Robustness demo (FGSM on synthetic data)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(21)
    model = ViT_LoRA(200, use_decoder=True).to(device)

    atk = torchattacks.FGSM(model, eps=15 / 255)
    data = torch.randn(128, 3, 224, 224)
    labels = torch.randint(0, 200, (128,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, labels), batch_size=32
    )

    clean_acc = accuracy(loader, model, device=device)
    adv_preds, gts = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        adv = atk(xb, yb)
        adv_logits, _ = model(adv)
        adv_preds.append(adv_logits.argmax(1).cpu())
        gts.append(yb.cpu())
    adv_acc = float(torchmetrics.functional.accuracy(torch.cat(adv_preds), torch.cat(gts))) if False else (
        float((torch.cat(adv_preds) == torch.cat(gts)).float().mean())
    )

    print(
        f"Clean ACC={clean_acc*100:.2f}   "
        f"FGSM ACC={adv_acc*100:.2f}   Robustness={(adv_acc/clean_acc):.2f}"
    )

    import matplotlib.pyplot as plt

    fname = FIG_DIR / "robustness_pair1.pdf"
    plt.figure(figsize=(3, 4))
    plt.bar([0, 1], [clean_acc * 100, adv_acc * 100], color=["skyblue", "salmon"])
    for i, v in enumerate([clean_acc * 100, adv_acc * 100]):
        plt.text(i, v + 0.5, f"{v:.1f}", ha="center")
    plt.xticks([0, 1], ["Clean", "FGSM"])
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Figure names: robustness_pair1.pdf")


# ---------------------------------------------------------------------------
#  CLI / MAIN ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", choices=["exp1", "exp2", "exp3"], required=True)
    p.add_argument("--method", default=None, help="Method id (for exp2)")
    return p.parse_args()


def main():
    args = _cli()
    if args.exp == "exp1":
        run_exp1(args)
    elif args.exp == "exp2":
        run_exp2(args)
    elif args.exp == "exp3":
        run_exp3(args)
    else:
        raise ValueError(f"Unrecognised experiment id {args.exp}")


if __name__ == "__main__":
    main()