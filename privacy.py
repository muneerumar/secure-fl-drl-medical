"""Differential privacy: clipping, Gaussian noise, and an RDP accountant.

The mechanism is the one written in the manuscript (Eq. 14 and Algorithm 1):
each hospital clips its *model update* to L2 norm C and adds N(0, (sigma*C)^2)
before transmission. Because the clipped object is the whole update rather than
a per-example gradient, the guarantee this buys is **client-level** (a.k.a.
user-level) (eps, delta)-DP: the presence or absence of any single hospital in
a round is masked. This is the McMahan et al. (2018) federated formulation.

Accounting uses Renyi DP for the Sampled Gaussian Mechanism (Mironov, Talwar &
Zhang 2019) composed over the communication rounds, then converted to
(eps, delta). Nothing here is estimated or hand-waved: given (sigma, q, T,
delta) the reported epsilon is the accountant's output.
"""
import math
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

# Renyi orders searched when converting RDP -> DP.
DEFAULT_ORDERS: Sequence[float] = (
    [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
)


# ---------------------------------------------------------------------------
# accountant
# ---------------------------------------------------------------------------
def _log_add(a: float, b: float) -> float:
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def _log_comb(n: int, k: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def _rdp_sgm_int(q: float, sigma: float, alpha: int) -> float:
    """RDP of the Sampled Gaussian Mechanism at integer order alpha."""
    log_a = -np.inf
    for i in range(alpha + 1):
        log_term = (_log_comb(alpha, i)
                    + i * math.log(q)
                    + (alpha - i) * math.log1p(-q)
                    + (i * i - i) / (2.0 * sigma * sigma))
        log_a = _log_add(log_a, log_term)
    return float(log_a / (alpha - 1))


def _rdp_sgm(q: float, sigma: float, alpha: float) -> float:
    """RDP at (possibly fractional) order alpha.

    For q == 1 the mechanism is a plain Gaussian: eps_rdp = alpha / (2 sigma^2).
    Fractional orders are handled by interpolating the integer bound, which is
    an upper bound and therefore safe to report.
    """
    if sigma <= 0:
        return np.inf
    if q <= 0:
        return 0.0
    if q >= 1.0:
        return alpha / (2.0 * sigma * sigma)
    if float(alpha).is_integer():
        return _rdp_sgm_int(q, sigma, int(alpha))
    lo, hi = int(math.floor(alpha)), int(math.ceil(alpha))
    if lo < 2:
        lo, hi = 2, 3
    r_lo, r_hi = _rdp_sgm_int(q, sigma, lo), _rdp_sgm_int(q, sigma, hi)
    w = (alpha - lo) / (hi - lo) if hi != lo else 0.0
    return r_lo + w * (r_hi - r_lo)


def compute_rdp(q: float, sigma: float, steps: int,
                orders: Iterable[float] = DEFAULT_ORDERS) -> np.ndarray:
    return np.array([steps * _rdp_sgm(q, sigma, a) for a in orders], dtype=float)


def rdp_to_dp(rdp: np.ndarray, orders: Sequence[float], delta: float):
    """Convert RDP to (eps, delta) using the tightened Canonne et al. bound."""
    orders = np.asarray(orders, dtype=float)
    rdp = np.asarray(rdp, dtype=float)
    valid = orders > 1
    orders, rdp = orders[valid], rdp[valid]
    eps = (rdp
           + np.log1p(-1.0 / orders)
           - (np.log(delta) + np.log(orders)) / (orders - 1.0))
    idx = int(np.nanargmin(eps))
    return float(max(eps[idx], 0.0)), float(orders[idx])


def epsilon_for(sigma: float, q: float, steps: int, delta: float) -> float:
    """Accounted epsilon, via Opacus' RDP accountant.

    The hand-rolled bound above is kept for reference but is not used: it loses
    numerical accuracy at extreme Renyi orders, which makes epsilon
    non-monotonic in sigma. Opacus' implementation is the authority here.
    """
    if sigma is None or sigma <= 0:
        return float("inf")
    if steps <= 0:
        return 0.0
    from opacus.accountants import RDPAccountant
    acc = RDPAccountant()
    acc.history = [(float(sigma), float(q), int(steps))]
    return float(acc.get_epsilon(delta))


def solve_noise_multiplier(target_epsilon: float, delta: float, q: float,
                           steps: int, lo: float = 0.2, hi: float = 1e5,
                           tol: float = 1e-3, max_iter: int = 200) -> float:
    """Smallest sigma whose accounted epsilon is <= target_epsilon."""
    if target_epsilon is None or target_epsilon == float("inf"):
        return 0.0
    try:
        from opacus.accountants.utils import get_noise_multiplier
        return float(get_noise_multiplier(
            target_epsilon=target_epsilon, target_delta=delta,
            sample_rate=q, steps=steps, accountant="rdp"))
    except Exception:
        pass          # fall back to the bisection below
    if epsilon_for(hi, q, steps, delta) > target_epsilon:
        raise ValueError(
            f"even sigma={hi} cannot reach eps={target_epsilon} "
            f"with q={q}, steps={steps}")
    while epsilon_for(lo, q, steps, delta) < target_epsilon:
        hi = lo
        lo /= 2.0
        if lo < 1e-4:
            return lo
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if epsilon_for(mid, q, steps, delta) > target_epsilon:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return hi


# ---------------------------------------------------------------------------
# mechanism
# ---------------------------------------------------------------------------
def clip_update(update: dict, max_norm: float):
    """Scale a state-dict-shaped update so its global L2 norm is <= max_norm."""
    total = torch.sqrt(sum((v.float() ** 2).sum() for v in update.values()))
    total = float(total)
    scale = min(1.0, max_norm / (total + 1e-12))
    if scale < 1.0:
        update = {k: v * scale for k, v in update.items()}
    return update, total, scale


def add_gaussian_noise(update: dict, sigma: float, clip_norm: float,
                       generator: Optional[torch.Generator] = None):
    """Eq. 14 / Eq. 26: theta <- theta + N(0, (sigma*C)^2)."""
    if sigma <= 0:
        return update
    std = sigma * clip_norm
    out = {}
    for k, v in update.items():
        # a CPU generator cannot seed a draw directly on another device, so
        # sample on the generator's device and move the noise across
        noise = torch.normal(mean=0.0, std=std, size=v.shape,
                             generator=generator, dtype=torch.float32)
        out[k] = v.float() + noise.to(v.device)
    return out


class DPAccountant:
    """Tracks the client-level privacy budget across communication rounds."""

    def __init__(self, noise_multiplier: float, sample_rate: float,
                 delta: float = 1e-5):
        self.sigma = float(noise_multiplier)
        self.q = float(sample_rate)
        self.delta = float(delta)
        self.steps = 0

    def step(self, n: int = 1):
        self.steps += n

    @property
    def epsilon(self) -> float:
        if self.steps == 0:
            return 0.0
        return epsilon_for(self.sigma, self.q, self.steps, self.delta)

    def summary(self) -> dict:
        return {
            "noise_multiplier": self.sigma,
            "sample_rate": self.q,
            "rounds_accounted": self.steps,
            "delta": self.delta,
            "epsilon": self.epsilon,
        }


if __name__ == "__main__":
    # sanity: sigma needed for the three budgets in Table 6, 100 rounds, q=1
    for eps in (0.1, 1.0, 10.0):
        s = solve_noise_multiplier(eps, 1e-5, q=1.0, steps=100)
        back = epsilon_for(s, 1.0, 100, 1e-5)
        print(f"target eps={eps:5} -> sigma={s:8.4f} (accounted eps={back:.4f})")
