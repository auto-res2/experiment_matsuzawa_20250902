"""src/train.py
Training related utilities: model definitions, memory buffers, sketching, and the core vision training loop.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # noqa: F401  # (kept for external use)
import torch.utils.data as tud  # noqa: F401
from torch.cuda import amp

# ----------------------------------------------------------------------------
#  GENERAL UTILITIES
# ----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility across python / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
#  FREQUENT-DIRECTIONS FEATURE SKETCH
# ----------------------------------------------------------------------------
class FDSketch(nn.Module):
    """Online Frequent-Directions sketch that stores low-rank class statistics.

    Parameters
    ----------
    d : int
        Feature dimensionality (e.g. 768 for ViT-B CLS token).
    k : int
        Target sketch rank.
    precision : {"fp16", "fp32", "int8"}
        Storage precision.  "int8" performs simple linear per-row quantisation
        but returns de-quantised fp tensors for downstream computation.
    device : str
        Target device for internal buffers.
    """

    def __init__(self, d: int, k: int, *, precision: str = "fp16", device: str = "cuda"):
        super().__init__()
        self.d, self.k = d, k
        dtype = torch.float16 if precision == "fp16" else torch.float32
        self.register_buffer("S", torch.zeros(d, k, dtype=dtype, device=device))
        self.register_buffer("mu", torch.zeros(d, dtype=dtype, device=device))
        self.n: int = 0  # number of samples observed so far
        self.precision = precision

        if precision == "int8":  # extra buffers for simple uint8 quantisation
            self.register_buffer("scale", torch.ones(d, 1, dtype=torch.float32, device=device))
            self.register_buffer("zero", torch.zeros(d, 1, dtype=torch.float32, device=device))

    # ------------------------------------------------------------------
    #  INTERNAL HELPERS
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _quantise(self, x: torch.Tensor) -> torch.Tensor:
        """Return *de-quantised* copy after storing quantisation params."""
        if self.precision != "int8":
            return x
        max_, min_ = x.max(dim=1, keepdim=True).values, x.min(dim=1, keepdim=True).values
        scale = (max_ - min_).clamp(min=1e-6) / 255
        zp = (-min_ / scale).clamp(0, 255)
        self.scale.copy_(scale)
        self.zero.copy_(zp)
        q = ((x / scale) + zp).round().clamp(0, 255).to(torch.uint8)
        # de-quantise back to fp for downstream SVD
        return q.float() * scale + (-zp * scale)

    # ------------------------------------------------------------------
    #  PUBLIC API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, feat: torch.Tensor) -> None:
        """Update sketch with a single *feature vector* (no batch)."""
        feat = feat.detach()
        self.n += 1
        self.mu += (feat - self.mu) / self.n  # running mean

        # Frequent-Directions core update (rank-k SVD shrinkage)
        u = feat.view(-1, 1)  # (d,1)
        S_hat = torch.cat([self.S, u], dim=1)  # (d, k+1)
        try:
            u_svd, s, _ = torch.linalg.svd(S_hat, full_matrices=False)
        except RuntimeError:  # numerical fallback
            s, u_svd = torch.linalg.eigvalsh(S_hat @ S_hat.T), None
        shrink = torch.clamp(s ** 2 - s[-1] ** 2, min=0).sqrt()
        self.S.copy_((u_svd[:, : self.k] * shrink[: self.k]))
        if self.precision == "int8":
            self.S.copy_(self._quantise(self.S))

    @property
    def basis(self) -> torch.Tensor:
        """Return orthonormal basis Q (d×k) via QR."""
        q, _ = torch.linalg.qr(self.S, mode="reduced")
        return q


# ----------------------------------------------------------------------------
#  RESERVOIR REPLAY BUFFER (baseline)
# ----------------------------------------------------------------------------
class ReservoirBuffer:
    """Reservoir sampling buffer for Experience Replay baselines."""

    def __init__(self, max_samples: int, img_shape: Tuple[int, int, int] = (3, 224, 224)) -> None:
        self.max = max_samples
        self.images = torch.zeros((max_samples, *img_shape), dtype=torch.uint8)
        self.labels = torch.zeros(max_samples, dtype=torch.long)
        self.n_seen: int = 0

    # ------------------------------------------------------------------
    def add_batch(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Add a batch of *normalised* (0-1) images/labels to the buffer."""
        for xi, yi in zip(x.cpu(), y.cpu()):
            j = self.n_seen
            if j < self.max:
                self.images[j] = (xi * 255).to(torch.uint8)
                self.labels[j] = yi
            else:
                m = random.randint(0, self.n_seen)
                if m < self.max:
                    self.images[m] = (xi * 255).to(torch.uint8)
                    self.labels[m] = yi
            self.n_seen += 1

    def sample(self, n: int, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, min(self.n_seen, self.max), (n,))
        return (
            self.images[idx].float().div(255).to(device),
            self.labels[idx].to(device),
        )

    # ------------------------------------------------------------------
    @property
    def memory_MB(self) -> float:
        """Rough memory usage in MB."""
        return self.images.element_size() * self.images.nelement() / 1e6


# ----------------------------------------------------------------------------
#  MODEL:  ViT-B/16  +  lightweight LoRA  +  optional decoder
# ----------------------------------------------------------------------------
try:
    import timm  # local import avoids unnecessary dependency for NLP runs
except ImportError as _err:  # pragma: no cover – handled at runtime
    timm = None  # pylint: disable=invalid-name

class ViT_LoRA(nn.Module):
    """Frozen ViT-B backbone with a minimal LoRA-style CLS adapter."""

    def __init__(self, n_classes: int, *, use_decoder: bool = False):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm is required for ViT_LoRA but not installed.")
        self.backbone = timm.create_model(
            "vit_base_patch16_224.augreg_in21k", pretrained=True, num_classes=0
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)  # freeze backbone

        # simple LoRA: an additional linear projection on CLS (rank = full)
        self.lora = nn.Linear(768, 768, bias=False)
        nn.init.kaiming_uniform_(self.lora.weight, a=math.sqrt(5))

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, n_classes),
        )

        self.use_decoder = use_decoder
        if use_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(768, 1024),
                nn.GELU(),
                nn.Linear(1024, 768),
            )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor | None = None,
        *,
        feats: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, features).  Exactly one of *x* or *feats* must be given."""
        if feats is None:
            if x is None:
                raise ValueError("Either x or feats must be provided to ViT_LoRA forward()")
            feats = self.backbone(x)  # CLS token
            feats = feats + self.lora(feats)
        logits = self.classifier(feats)
        return logits, feats


# ----------------------------------------------------------------------------
#  TRAINING HELPERS
# ----------------------------------------------------------------------------

def orth_proj_grads(params: List[nn.Parameter], U: torch.Tensor | None) -> None:
    """Project gradients onto the orthogonal complement of basis *U* (d×k)."""
    if U is None:
        return
    UT = U.t().contiguous()
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.data.view(-1, 1)
            component = U @ (UT @ g)
            g.sub_(component)


# -----------------------------------------------------------------------------
#  SINGLE-TASK VISION TRAINING LOOP
# -----------------------------------------------------------------------------

def train_task_vision(
    model: ViT_LoRA,
    loader: tud.DataLoader,
    *,
    method: str,
    seen_cls: List[int],
    buffer: ReservoirBuffer | None,
    sketches: Dict[int, FDSketch] | None,
    opt: optim.Optimizer,
    scaler: amp.GradScaler,
    device: str = "cuda",
    λ_align: float = 0.3,
) -> None:
    """Train *model* for one task/epoch using one of {finetune, er, fsr_k*}."""
    ce = nn.CrossEntropyLoss()
    model.train()

    for _epoch in range(10):  # fixed epoch budget
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            B = len(xb)

            # ----------------------------------------------------
            # 1) Add replay (ER or FSR)
            # ----------------------------------------------------
            replay_loss = torch.tensor(0.0, device=device)
            if method == "er" and buffer and buffer.n_seen:
                xr, yr = buffer.sample(B)
                xb = torch.cat([xb, xr], 0)
                yb = torch.cat([yb, yr], 0)
            elif method.startswith("fsr") and seen_cls:
                cls_sample = torch.tensor(random.choices(seen_cls, k=B), device=device)
                mu = torch.stack([sketches[c].mu for c in cls_sample.tolist()])
                basis = torch.stack([sketches[c].basis for c in cls_sample.tolist()])
                z = torch.randn(B, basis.shape[-1], device=device)
                f_rep = mu + torch.bmm(basis, z.unsqueeze(-1)).squeeze(-1)
                with torch.no_grad():
                    rep_logits, _ = model(feats=model.decoder(f_rep))
                replay_loss = ce(rep_logits, cls_sample)

            # ----------------------------------------------------
            # 2) Forward + losses
            # ----------------------------------------------------
            with amp.autocast():
                logits, feats = model(xb)
                loss_real = ce(logits, yb)

                # Alignment loss (FSR only)
                align_loss = torch.tensor(0.0, device=device)
                if method.startswith("fsr") and seen_cls:
                    for c in torch.unique(yb):
                        U = sketches[int(c.item())].basis  # (d,k)
                        proj = feats[yb == c] @ U @ U.t()
                        align_loss += F.mse_loss(feats[yb == c], proj)
                    align_loss /= len(torch.unique(yb))

                loss = loss_real + replay_loss + λ_align * align_loss

            scaler.scale(loss).backward()

            # 3) Orthogonal gradient projection (FSR)
            if method.startswith("fsr") and seen_cls:
                U_all = torch.cat([sketches[c].basis for c in seen_cls], 1)
                orth_proj_grads(model.parameters(), U_all)

            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            # ----------------------------------------------------
            # 4) Memory updates
            # ----------------------------------------------------
            if method == "er":
                buffer.add_batch(xb[:B], yb[:B])
            elif method.startswith("fsr"):
                with torch.no_grad():
                    feats_curr = feats[:B]
                    for f, y in zip(feats_curr, yb[:B]):
                        c = int(y.item())
                        if c not in sketches:
                            sketches[c] = FDSketch(768, k=int(method.split("_k")[-1]), device=device)
                        sketches[c].update(f)


__all__ = [
    "set_seed",
    "FDSketch",
    "ReservoirBuffer",
    "ViT_LoRA",
    "orth_proj_grads",
    "train_task_vision",
]