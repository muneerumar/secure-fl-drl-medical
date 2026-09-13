"""Client-side federated training -- Algorithm 1 of the manuscript.

Local CNN training (with optional FedProx proximal term and federated
adversarial training), then DP clipping + Gaussian noise, then gradient
compression, then transmission. Encryption is accounted for but not simulated
cryptographically: it is a constant-factor transport concern that does not
change any reported metric.
"""
import copy
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F

import attacks
import compression
import privacy


def state_delta(new: Dict[str, torch.Tensor],
                old: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (new[k].float() - old[k].float()) for k in new}


def apply_delta(base: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor],
                lr_scale: float = 1.0) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in base.items():
        if k in delta:
            out[k] = v.float() + lr_scale * delta[k].float()
        else:
            out[k] = v.clone()
    return out


class Client:
    """One participating hospital."""

    def __init__(self, cid: int, loader, model_fn, cfg, device,
                 n_classes: int, malicious: bool = False):
        self.cid = cid
        self.loader = loader
        self.model_fn = model_fn
        self.cfg = cfg
        self.device = device
        self.n_classes = n_classes
        self.malicious = malicious
        self.n_samples = len(loader.dataset)
        self.model = model_fn().to(device)
        # per-client compute/latency profile -> the C_t and L_t entries of the
        # DQN state (Eq. 15). Deterministic in the client id so runs replay.
        g = torch.Generator().manual_seed(1234 + cid)
        self.compute_capacity = float(torch.empty(1).uniform_(0.5, 1.0,
                                                              generator=g))
        self.base_latency_ms = float(torch.empty(1).uniform_(20.0, 120.0,
                                                             generator=g))

    # -- local training ----------------------------------------------------
    def train(self, global_state: Dict[str, torch.Tensor], round_idx: int,
              lr_scale: float = 1.0, use_fedprox: Optional[bool] = None,
              generator: Optional[torch.Generator] = None) -> dict:
        cfg = self.cfg
        use_fedprox = cfg.use_fedprox if use_fedprox is None else use_fedprox

        # fresh module each round: Opacus attaches per-sample-gradient hooks to
        # the module it wraps, and those cannot be re-attached to a module that
        # already carries them.
        self.model = self.model_fn().to(self.device)
        self.model.load_state_dict(global_state, strict=True)
        self.model.train()
        global_params = [p.detach().clone() for p in self.model.parameters()]

        # Eq. 13: eta_t = eta_0 / (1 + lambda * t), then the DQN's LAR factor
        lr = cfg.lr_fl / (1.0 + cfg.lr_decay_lambda * round_idx) * lr_scale
        if cfg.optimizer == "adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=lr,
                                   weight_decay=cfg.weight_decay)
        else:
            opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,
                                  weight_decay=cfg.weight_decay)

        # -- DP-SGD: wrap model/optimizer/loader with per-sample clipping ---
        loader = self.loader
        engine = None
        if cfg.use_dp and cfg.dp_level == "example":
            from opacus import PrivacyEngine
            engine = PrivacyEngine(accountant="rdp")
            self.model, opt, loader = engine.make_private(
                module=self.model, optimizer=opt, data_loader=self.loader,
                noise_multiplier=self.effective_sigma(round_idx),
                max_grad_norm=cfg.dp_clip_norm, poisson_sampling=True)

        t0 = time.perf_counter()
        total_loss, n_batches, n_seen = 0.0, 0, 0
        for _ in range(cfg.local_epochs):
            for x, y in loader:
                if x.numel() == 0:      # Poisson sampling can yield empty batches
                    continue
                x, y = x.to(self.device), y.to(self.device)
                if self.malicious and cfg.poison_type == "label_flip":
                    y = attacks.flip_labels(y, self.n_classes)

                # Eq. 22 / Sec. 6.5.2: replace a fraction of the batch with its
                # adversarial counterpart *in place*, so the batch size (and
                # therefore the per-sample gradient structure DP-SGD relies on)
                # is unchanged. Opacus' hooks are switched off while the attack
                # differentiates w.r.t. the input.
                if cfg.use_adv_training and cfg.adv_epsilon > 0:
                    m = max(1, int(cfg.adv_ratio * x.size(0)))
                    if engine is not None:
                        self.model.disable_hooks()
                    xa = attacks.fgsm(self.model, x[:m], y[:m], cfg.adv_epsilon)
                    if engine is not None:
                        self.model.enable_hooks()
                    self.model.train()
                    x = torch.cat([xa, x[m:]], dim=0)

                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(self.model(x), y)

                # Eq. 12: with no DP-SGD the proximal term goes into the
                # objective Adam actually minimises, which is what FedProx
                # specifies. Under DP-SGD it cannot: Opacus rebuilds .grad from
                # per-sample gradients and would discard a loss term that has no
                # per-sample structure, so that path applies it after the step.
                if use_fedprox and cfg.fedprox_mu > 0 and engine is None:
                    prox = sum(((p_ - g0) ** 2).sum()
                               for p_, g0 in zip(self.model.parameters(),
                                                 global_params))
                    loss = loss + 0.5 * cfg.fedprox_mu * prox

                loss.backward()
                opt.step()

                # Eq. 12: FedProx proximal term, applied as an explicit
                # proximal step rather than folded into the loss. Under DP-SGD
                # Opacus rebuilds .grad from the per-sample gradients, which
                # would silently discard a loss term that has no per-sample
                # structure. The term depends only on the (public) global
                # model, so applying it here costs no privacy budget.
                if use_fedprox and cfg.fedprox_mu > 0 and engine is not None:
                    with torch.no_grad():
                        for p_, g0 in zip(self.model.parameters(), global_params):
                            p_ -= lr * cfg.fedprox_mu * (p_ - g0)
                total_loss += float(loss.detach())
                n_batches += 1
                n_seen += x.size(0)
        train_time = time.perf_counter() - t0

        # unwrap Opacus' GradSampleModule so the state dict keys match the
        # global model again
        if engine is not None:
            self.model = self.model._module
        trained_state = {k: v.detach().clone()
                         for k, v in self.model.state_dict().items()}
        delta = state_delta(trained_state, global_state)

        info = {"clip_scale": 1.0, "raw_norm": None,
                "sigma": self.effective_sigma(round_idx) if cfg.use_dp else 0.0}

        # -- Algorithm 1: DP, then compression ----------------------------
        # With dp_level == "example" the noise has already been injected into
        # every gradient step above, so the update is released as-is. With
        # dp_level == "client" the whole update is clipped and noised here.
        if cfg.use_dp and cfg.dp_level == "client":
            delta, raw_norm, scale = privacy.clip_update(delta, cfg.dp_clip_norm)
            delta = privacy.add_gaussian_noise(delta, info["sigma"],
                                               cfg.dp_clip_norm,
                                               generator=generator)
            info.update({"clip_scale": scale, "raw_norm": raw_norm})
        else:
            info["raw_norm"] = float(torch.sqrt(
                sum((v.float() ** 2).sum() for v in delta.values())))

        if self.malicious and cfg.poison_type != "label_flip":
            delta = attacks.poison_update(delta, cfg.poison_type,
                                          cfg.poison_scale,
                                          generator=generator)

        if cfg.use_compression:
            delta, cstats = compression.topk_compress(
                delta, cfg.compression_ratio, cfg.quantization_bits)
        else:
            delta, cstats = compression.identity_compress(delta, 32)

        # simulated wall-clock latency of shipping this payload
        latency_ms = self.base_latency_ms + cstats["payload_bytes"] / 1e6 * 8.0

        return {
            "cid": self.cid,
            "delta": delta,
            "n_samples": self.n_samples,
            "loss": total_loss / max(n_batches, 1),
            "train_time_s": train_time,
            "compute_capacity": self.compute_capacity,
            "latency_ms": latency_ms,
            "comm": cstats,
            "malicious": self.malicious,
            **info,
        }

    def set_dp_sigma(self, sigma: float):
        """Server-assigned noise multiplier for this hospital.

        Solved per client so that every hospital ends the run at the *same*
        target epsilon despite holding different amounts of data (and therefore
        different subsampling rates).
        """
        self.dp_sigma = float(sigma)

    def effective_sigma(self, round_idx: int) -> float:
        """R1-2: is the noise scale fixed or adjusted across rounds?

        Default is a *fixed* multiplier for the whole run, which is what the
        accountant in privacy.py assumes. With dp_dynamic_noise the multiplier
        decays as sigma_t = sigma_0 / (1 + 0.01 t); the accountant then composes
        the per-round multipliers rather than a single constant.
        """
        sigma = getattr(self, "dp_sigma", None)
        if sigma is None:
            sigma = self.cfg.dp_noise_multiplier or 0.0
        if self.cfg.dp_dynamic_noise:
            sigma = sigma / (1.0 + 0.01 * round_idx)
        return sigma


@torch.no_grad()
def evaluate(model, loader, device, n_classes: int):
    """Accuracy / macro-P / macro-R / macro-F1 / AUC plus raw scores."""
    model.eval()
    logits_all, y_all = [], []
    t0 = time.perf_counter()
    n = 0
    for x, y in loader:
        x = x.to(device)
        logits_all.append(model(x).cpu())
        y_all.append(y)
        n += x.size(0)
    infer_s = time.perf_counter() - t0

    logits = torch.cat(logits_all)
    y = torch.cat(y_all).reshape(-1)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    return {
        "y_true": y.numpy(),
        "y_pred": pred.numpy(),
        "y_prob": probs.numpy(),
        "n": n,
        "inference_ms_per_sample": infer_s / max(n, 1) * 1e3,
    }


@torch.no_grad()
def _no_grad_noop():
    return None


def robust_evaluate(model, loader, device, epsilon: float, attack: str = "fgsm",
                    pgd_steps: int = 10, max_batches: Optional[int] = None):
    """Accuracy under evasion attack -- the numbers for R2-8."""
    model.eval()
    correct, total = 0, 0
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, y = x.to(device), y.to(device).reshape(-1)
        if epsilon > 0:
            if attack == "fgsm":
                x = attacks.fgsm(model, x, y, epsilon)
            elif attack == "pgd":
                x = attacks.pgd(model, x, y, epsilon, steps=pgd_steps)
            else:
                raise ValueError(f"unknown attack {attack!r}")
        with torch.no_grad():
            pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum())
        total += y.numel()
    return correct / max(total, 1)
