"""Evasion and poisoning attacks, plus the defences of Sec. 6.5.

Evasion  -- FGSM (Eq. 21) and PGD, used both for adversarial *training*
            (Eq. 22) and for the robustness *evaluation* reviewer R2-8 asks for.
Poisoning -- sign-flip, label-flip and Gaussian model poisoning by malicious
            hospitals.
Defences -- trimmed-mean aggregation (Eq. 24), norm clipping (Eq. 25) and the
            autoencoder update filter (Sec. 6.5.3).
"""
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# evasion attacks
# ---------------------------------------------------------------------------
def fgsm(model, x, y, epsilon: float):
    """Eq. 21: x* = x + eps * sign(grad_x J(f(x), y))."""
    if epsilon <= 0:
        return x
    was_training = model.training
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    adv = (x + epsilon * grad.sign()).detach()
    adv = adv.clamp(-1.0, 1.0)          # inputs are normalised to [-1, 1]
    if was_training:
        model.train()
    return adv


def pgd(model, x, y, epsilon: float, alpha: float = None, steps: int = 10):
    """L-inf PGD -- the stronger evaluation attack."""
    if epsilon <= 0:
        return x
    alpha = alpha if alpha is not None else max(epsilon / 4.0, 1e-3)
    was_training = model.training
    model.eval()
    x0 = x.clone().detach()
    adv = (x0 + torch.empty_like(x0).uniform_(-epsilon, epsilon)).clamp(-1, 1)
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.min(torch.max(adv, x0 - epsilon), x0 + epsilon).clamp(-1, 1)
    if was_training:
        model.train()
    return adv.detach()


# ---------------------------------------------------------------------------
# model poisoning
# ---------------------------------------------------------------------------
def poison_update(update: Dict[str, torch.Tensor], kind: str,
                  scale: float = 5.0, generator=None) -> Dict[str, torch.Tensor]:
    """Corrupt a client's update before it reaches the server."""
    if kind == "sign_flip":
        return {k: -scale * v for k, v in update.items()}
    if kind == "gaussian":
        out = {}
        for k, v in update.items():
            std = float(v.float().std()) * scale + 1e-8
            out[k] = torch.normal(0.0, std, size=v.shape, generator=generator,
                                  device=v.device, dtype=torch.float32)
        return out
    if kind == "label_flip":
        # label flipping is applied at the data level; the update itself is the
        # honest result of training on flipped labels, so pass it through.
        return update
    raise ValueError(f"unknown poison type {kind!r}")


def flip_labels(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """y -> (n_classes - 1 - y): the standard label-flip poisoning map."""
    return (n_classes - 1 - y).clamp(0, n_classes - 1)


# ---------------------------------------------------------------------------
# defences
# ---------------------------------------------------------------------------
def norm_clip(update: Dict[str, torch.Tensor], tau: float):
    """Eq. 25: theta <- theta * min(1, tau / ||theta||)."""
    total = float(torch.sqrt(sum((v.float() ** 2).sum() for v in update.values())))
    if total <= tau or tau <= 0:
        return update, total, 1.0
    s = tau / (total + 1e-12)
    return {k: v * s for k, v in update.items()}, total, s


def trimmed_mean(updates: List[Dict[str, torch.Tensor]],
                 weights: List[float] = None, trim: int = 1):
    """Eq. 24: drop the `trim` largest and smallest values coordinate-wise."""
    n = len(updates)
    if n == 0:
        raise ValueError("no updates to aggregate")
    if n <= 2 * trim:
        trim = max(0, (n - 1) // 2)
    keys = updates[0].keys()
    out = {}
    for k in keys:
        stack = torch.stack([u[k].float() for u in updates], dim=0)
        if trim > 0:
            sorted_, _ = torch.sort(stack, dim=0)
            stack = sorted_[trim:n - trim]
        out[k] = stack.mean(dim=0)
    return out


def weighted_average(updates: List[Dict[str, torch.Tensor]],
                     weights: List[float]) -> Dict[str, torch.Tensor]:
    dev = next(iter(updates[0].values())).device
    w = torch.tensor(weights, dtype=torch.float32, device=dev)
    w = w / w.sum()
    out = {}
    for k in updates[0].keys():
        stack = torch.stack([u[k].float() for u in updates], dim=0)
        shape = [len(updates)] + [1] * (stack.dim() - 1)
        out[k] = (stack * w.view(shape)).sum(dim=0)
    return out


def update_signature(update: Dict[str, torch.Tensor], n_bins: int = 16
                     ) -> np.ndarray:
    """Fixed-length descriptor of an update, fed to the anomaly autoencoder.

    Per-tensor L2 norms are summarised into `n_bins` quantiles, together with
    the global norm, mean, std, and sparsity -- enough to separate a poisoned
    update from a benign one without depending on the model architecture.
    """
    flat = torch.cat([v.reshape(-1).float() for v in update.values()])
    norms = torch.tensor([float(v.float().norm()) for v in update.values()])
    qs = torch.linspace(0, 1, n_bins)
    feat = torch.quantile(norms, qs) if norms.numel() > 1 else norms.repeat(n_bins)
    extra = torch.tensor([
        float(flat.norm()),
        float(flat.mean()),
        float(flat.std()),
        float((flat == 0).float().mean()),
        float(flat.abs().max()),
    ])
    sig = torch.cat([feat, extra]).numpy()
    return np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)


SIGNATURE_DIM = 16 + 5
