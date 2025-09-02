"""src/train.py
Training–related modules: model definition, memory buffers, sketching, and the
single-task training loop.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401
import torch.optim as optim
from torch.cuda import amp

import timm

# -----------------------------------------------------------------------------
#  Frequent-Directions Sketch
# -----------------------------------------------------------------------------


class FDSketch(nn.Module):
    """Lightweight Frequent-Directions sketch storing a per-class low-rank basis."""

    def __init__(self, d: int, k: int, device: str = "cuda", dtype: torch.dtype = torch.float16):
        super().__init__()
        self.k = k
        self.d = d
        self.register_buffer("S", torch.zeros(d, k, dtype=dtype, device=device))
        self.register_buffer("mu", torch.zeros(d, dtype=dtype, device=device))
        self.n = 0  # number of observed samples

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """Update the sketch with a new feature vector *x* (shape: ``(d,)``)."""
        x = x.to(self.S.dtype)
        self.n += 1
        self.mu = self.mu + (x - self.mu) / self.n  # running mean

        # ------- Frequent-Directions update -------
        u = x.view(-1, 1)  # (d,1)
        S_hat = torch.cat([self.S, u], dim=1)  # (d, k+1)
        try:
            u_svd, s, _ = torch.linalg.svd(S_hat, full_matrices=False)
        except RuntimeError:
            # fallback for older torch versions
            u_svd, s, _ = torch.svd(S_hat)
        shrink = torch.clamp_min(s ** 2 - s[-1] ** 2, 0).sqrt()
        self.S = (u_svd[:, : self.k] * shrink[: self.k])

    @property
    def basis(self) -> torch.Tensor:
        """Return an orthonormal basis ``(d,k)`` for the sketch sub-space."""
        q, _ = torch.linalg.qr(self.S, mode="reduced")
        return q

# -----------------------------------------------------------------------------
#  Reservoir buffer (ER baseline)
# -----------------------------------------------------------------------------


class ReservoirBuffer:
    """Standard reservoir replay buffer used by Experience Replay (ER)."""

    def __init__(self, max_samples: int, img_shape: Tuple[int, int, int] = (3, 224, 224)):
        self.max = max_samples
        self.x = torch.zeros((max_samples, *img_shape), dtype=torch.uint8)
        self.y = torch.zeros(max_samples, dtype=torch.long)
        self.n_seen = 0

    def add_batch(self, xb: torch.Tensor, yb: torch.Tensor):
        """Add a mini-batch to the buffer using reservoir sampling."""
        for xi, yi in zip(xb.cpu(), yb.cpu()):
            j = self.n_seen
            if j < self.max:
                self.x[j] = (xi * 255).to(torch.uint8)
                self.y[j] = yi
            else:
                m = random.randint(0, self.n_seen)
                if m < self.max:
                    self.x[m] = (xi * 255).to(torch.uint8)
                    self.y[m] = yi
            self.n_seen += 1

    def sample(self, n: int, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.n_seen > 0, "Replay buffer is empty."
        idx = torch.randint(0, min(self.n_seen, self.max), (n,))
        return (
            self.x[idx].float().div(255).to(device, non_blocking=True),
            self.y[idx].to(device, non_blocking=True),
        )

# -----------------------------------------------------------------------------
#  Vision model wrapper (ViT-LoRA + optional FSR decoder)
# -----------------------------------------------------------------------------


class ViTLoRAClassifier(nn.Module):
    """ViT-B/16 backbone (frozen) + lightweight LoRA adapters + classifier head.

    When ``fsr=True`` an additional decoder is instantiated for Feature-Sketch
    Replay (FSR).
    """

    def __init__(self, num_classes: int = 100, fsr: bool = False):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224.augreg_in21k", pretrained=True, num_classes=0
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # LoRA – here approximated with full linear adapters for simplicity.
        self.lora_adapters = nn.ModuleList([nn.Linear(768, 768, bias=False) for _ in range(12)])
        for l in self.lora_adapters:
            nn.init.kaiming_uniform_(l.weight, a=math.sqrt(5))

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )
        self.fsr = fsr
        if fsr:
            # Lightweight decoder mapping low-rank features back to embedding space
            self.decoder = nn.Sequential(
                nn.Linear(768, 1024),
                nn.GELU(),
                nn.Linear(1024, 768),
            )

    def forward(self, x: torch.Tensor | None = None, *, feats: torch.Tensor | None = None):
        """Forward pass.

        If *feats* is provided the backbone is skipped (used for latent replay).
        Returns a tuple ``(logits, features)``.
        """
        if feats is None:
            feats = self.backbone.forward_features(x)
            for adp in self.lora_adapters:  # simple residual LoRA adapters
                feats = feats + adp(feats)
        logits = self.classifier(feats)
        return logits, feats

# -----------------------------------------------------------------------------
#  Gradient-projection utility (ORTHOG-SUBSPACE)
# -----------------------------------------------------------------------------

def orthogonal_project_gradients(params: List[nn.Parameter], bases: torch.Tensor | None):
    """Project *params*' gradients onto the orthogonal complement of *bases*.

    *bases* should be a tensor of shape ``(d, k_total)`` containing stacked
    column-orthonormal basis vectors.  If *bases* is ``None`` the call is a no-op.
    """
    if bases is None:
        return
    U = bases  # (d, k)
    UT = U.t()
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.view(-1, 1)
            proj = U @ (UT @ g)
            g.sub_(proj)

# -----------------------------------------------------------------------------
#  Single-task training loop
# -----------------------------------------------------------------------------

def train_one_task(
    *,
    model: ViTLoRAClassifier,
    task_loader: torch.utils.data.DataLoader,
    replay_method: str,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
    buffer: ReservoirBuffer | None,
    sketches: Dict[int, FDSketch] | None,
    seen_classes: List[int],
    device: str = "cuda",
    epochs: int = 10,
):
    """Train *model* on a single task with the specified replay mechanism."""

    criterion = nn.CrossEntropyLoss()
    model.train()

    for _ in range(epochs):
        for xb, yb in task_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            # ------------------------------------------------------------------
            # Preserve a copy of the *current* mini-batch (before any replay)
            # ------------------------------------------------------------------
            xb_curr, yb_curr = xb.clone(), yb.clone()
            curr_bs = yb_curr.size(0)

            # ------------------------------------------------------------------
            # Ensure *yb* is in the correct integer format expected by CE loss
            # ------------------------------------------------------------------
            yb = yb.long().view(-1)

            # -----------------------------
            # Add replay samples
            # -----------------------------
            if replay_method == "ER" and buffer and buffer.n_seen > 0:
                xr, yr = buffer.sample(len(xb), device)
                xb = torch.cat([xb, xr], dim=0)
                yb = torch.cat([yb, yr.long()], dim=0)
            elif (
                replay_method == "FSR"
                and sketches is not None
                and len(sketches) > 0  # only when we already have some sketches
            ):
                available_classes = list(sketches.keys())
                cls_idx = torch.randint(0, len(available_classes), (len(xb),), device=device)
                cls_ids = [available_classes[i.item()] for i in cls_idx]
                mu = torch.stack([sketches[c].mu for c in cls_ids])  # (B,d)
                basis = torch.stack([sketches[c].basis for c in cls_ids])  # (B,d,k)
                z = torch.randn(len(xb), basis.size(-1), dtype=basis.dtype, device=device)
                f_tilde = mu + torch.bmm(basis, z.unsqueeze(-1)).squeeze(-1)
                with torch.no_grad():
                    pseudo_logits, _ = model(feats=model.decoder(f_tilde))
                pseudo_targets = torch.tensor(cls_ids, device=device, dtype=torch.long)
            # -----------------------------
            # Forward / backward
            # -----------------------------
            with amp.autocast():
                logits, feats = model(xb)
                loss = criterion(logits, yb.long())  # ensure target dtype is correct
                if (
                    replay_method == "FSR"
                    and sketches is not None
                    and len(sketches) > 0
                ):
                    loss_re = criterion(pseudo_logits, pseudo_targets)
                    loss = loss + 0.3 * loss_re
            scaler.scale(loss).backward()

            # Gradient projection for FSR
            if replay_method == "FSR" and sketches is not None and len(sketches) > 0:
                U = torch.cat([sketches[c].basis for c in sketches], dim=1)  # (d,k_tot)
                orthogonal_project_gradients(model.parameters(), U)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # -----------------------------
            # Update memory structures
            # -----------------------------
            if replay_method == "ER" and buffer is not None:
                # store **only** the current task samples, not the replay ones
                buffer.add_batch(xb_curr, yb_curr)
            elif replay_method == "FSR" and sketches is not None:
                with torch.no_grad():
                    real_feats = feats[:curr_bs]  # exclude potential replay rows
                    for feat, lbl in zip(real_feats, yb_curr):
                        lbl_int = int(lbl)
                        if lbl_int not in sketches:
                            sketches[lbl_int] = FDSketch(768, k=12)
                        sketches[lbl_int].update(feat.detach())

__all__ = [
    "FDSketch",
    "ReservoirBuffer",
    "ViTLoRAClassifier",
    "orthogonal_project_gradients",
    "train_one_task",
]
