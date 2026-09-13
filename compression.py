"""Gradient compression with exact payload accounting.

Reviewer R2-6 asks for communication cost per round, total transmitted data,
compression ratio and percentage reduction. Everything reported here is
counted, not estimated: top-k sparsification keeps k = ratio * d coordinates,
and the transmitted payload is k values at `bits` precision plus k indices at
ceil(log2(d)) bits, which is what a real sparse encoder would put on the wire.
"""
import math
from typing import Dict, Tuple

import torch


def _flatten(update: Dict[str, torch.Tensor]):
    keys = list(update.keys())
    shapes = [update[k].shape for k in keys]
    flat = torch.cat([update[k].reshape(-1).float() for k in keys])
    return flat, keys, shapes


def _unflatten(flat: torch.Tensor, keys, shapes) -> Dict[str, torch.Tensor]:
    out, i = {}, 0
    for k, s in zip(keys, shapes):
        n = int(torch.tensor(s).prod()) if len(s) else 1
        out[k] = flat[i:i + n].reshape(s).clone()
        i += n
    return out


def dense_payload_bytes(num_params: int, bits: int = 32) -> int:
    """Baseline: every parameter sent at `bits` precision."""
    return int(math.ceil(num_params * bits / 8))


def sparse_payload_bytes(num_params: int, k: int, bits: int = 16) -> int:
    """Top-k payload: k values at `bits` plus k indices at ceil(log2(d)) bits."""
    if k <= 0:
        return 0
    index_bits = max(1, math.ceil(math.log2(max(num_params, 2))))
    return int(math.ceil(k * (bits + index_bits) / 8))


def topk_compress(update: Dict[str, torch.Tensor], ratio: float,
                  bits: int = 16) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Keep the `ratio` largest-magnitude coordinates, zero the rest.

    Returns the sparsified update and an accounting dict. Values are also
    round-tripped through a `bits`-wide uniform quantiser so the accuracy cost
    of the stated payload is actually paid by the model, not just reported.
    """
    flat, keys, shapes = _flatten(update)
    d = flat.numel()
    k = max(1, int(round(ratio * d)))

    if k >= d:
        kept = flat.clone()
        k = d
    else:
        _, idx = torch.topk(flat.abs(), k, sorted=False)
        kept = torch.zeros_like(flat)
        kept[idx] = flat[idx]

    kept = _quantize(kept, bits)

    dense_b = dense_payload_bytes(d, 32)
    sparse_b = sparse_payload_bytes(d, k, bits)
    stats = {
        "num_params": d,
        "k_kept": k,
        "dense_bytes": dense_b,
        "payload_bytes": sparse_b,
        "compression_ratio": dense_b / max(sparse_b, 1),
        "reduction_pct": 100.0 * (1.0 - sparse_b / max(dense_b, 1)),
    }
    return _unflatten(kept, keys, shapes), stats


def _quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Uniform symmetric quantisation to `bits`, dequantised back to float."""
    if bits >= 32:
        return x
    nz = x[x != 0]
    if nz.numel() == 0:
        return x
    scale = nz.abs().max()
    if scale <= 0:
        return x
    levels = 2 ** (bits - 1) - 1
    q = torch.round(x / scale * levels).clamp(-levels, levels)
    return q * scale / levels


def identity_compress(update: Dict[str, torch.Tensor], bits: int = 32):
    """No compression -- used by the ablation arm that disables it."""
    flat, _, _ = _flatten(update)
    d = flat.numel()
    b = dense_payload_bytes(d, bits)
    return update, {
        "num_params": d, "k_kept": d,
        "dense_bytes": dense_payload_bytes(d, 32),
        "payload_bytes": b,
        "compression_ratio": dense_payload_bytes(d, 32) / max(b, 1),
        "reduction_pct": 100.0 * (1.0 - b / max(dense_payload_bytes(d, 32), 1)),
    }


class CommTracker:
    """Accumulates uplink/downlink bytes so Table 8 can carry real numbers."""

    def __init__(self):
        self.uplink_bytes = 0
        self.downlink_bytes = 0
        self.dense_equivalent_bytes = 0
        self.per_round = []

    def record_round(self, uplink: int, downlink: int, dense_equiv: int):
        self.uplink_bytes += uplink
        self.downlink_bytes += downlink
        self.dense_equivalent_bytes += dense_equiv
        self.per_round.append({
            "uplink_bytes": uplink,
            "downlink_bytes": downlink,
            "total_bytes": uplink + downlink,
            "dense_equivalent_bytes": dense_equiv,
        })

    def summary(self) -> dict:
        rounds = max(len(self.per_round), 1)
        total = self.uplink_bytes + self.downlink_bytes
        return {
            "rounds": len(self.per_round),
            "total_uplink_MB": self.uplink_bytes / 1e6,
            "total_downlink_MB": self.downlink_bytes / 1e6,
            "total_MB": total / 1e6,
            "per_round_MB": total / rounds / 1e6,
            "uplink_per_round_MB": self.uplink_bytes / rounds / 1e6,
            "dense_equivalent_MB": self.dense_equivalent_bytes / 1e6,
            "compression_ratio": (self.dense_equivalent_bytes
                                  / max(self.uplink_bytes, 1)),
            "uplink_reduction_pct": 100.0 * (
                1.0 - self.uplink_bytes / max(self.dense_equivalent_bytes, 1)),
        }
