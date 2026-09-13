"""Server-side orchestration -- Algorithm 2 of the manuscript.

Runs one federation: per round the DQN picks (hospital selection, aggregation
strategy, learning-rate adjustment), the selected hospitals train locally under
DP + compression, the server filters and aggregates the updates, and the reward
of Eq. 17 is fed back to the agent.

Also provides the centralised-learning baseline used in Tables 4, 7 and 8.
"""
import copy
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

import attacks
import client as client_mod
import compression
import dqn as dqn_mod
import privacy
from client import Client
from models import UpdateAutoEncoder, build_model, count_parameters
from metrics import classification_metrics


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(pref)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AnomalyFilter:
    """Sec. 6.5.3 autoencoder over update signatures.

    R1-4d: it is *not* pre-trained on a separate clean corpus. It warms up on
    the first `ae_pretrain_rounds` rounds of live updates (treated as benign,
    the standard cold-start assumption) and only starts rejecting afterwards.
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.ae = UpdateAutoEncoder(attacks.SIGNATURE_DIM).to(device)
        self.opt = torch.optim.Adam(self.ae.parameters(), lr=1e-3)
        self.errors: List[float] = []
        self.rounds_seen = 0
        self.rejections = 0
        self.true_positives = 0
        self.false_positives = 0

    def _sig(self, update):
        s = attacks.update_signature(update)
        return torch.tensor(s, dtype=torch.float32, device=self.device)

    def fit_step(self, sigs: torch.Tensor, epochs: int = 20):
        self.ae.train()
        for _ in range(epochs):
            self.opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(self.ae(sigs), sigs)
            loss.backward()
            self.opt.step()

    def screen(self, results: List[dict]) -> List[dict]:
        """Return the subset of client results accepted for aggregation."""
        if not self.cfg.use_autoencoder_filter:
            return results
        sigs = torch.stack([self._sig(r["delta"]) for r in results])
        self.rounds_seen += 1

        if self.rounds_seen <= self.cfg.ae_pretrain_rounds:
            self.fit_step(sigs)
            with torch.no_grad():
                err = ((self.ae(sigs) - sigs) ** 2).mean(dim=1).cpu().numpy()
            self.errors.extend(err.tolist())
            return results

        self.ae.eval()
        with torch.no_grad():
            err = ((self.ae(sigs) - sigs) ** 2).mean(dim=1).cpu().numpy()
        mu, sd = float(np.mean(self.errors)), float(np.std(self.errors) + 1e-12)
        thresh = mu + self.cfg.ae_reject_sigma * sd

        accepted, benign_sigs = [], []
        for r, e in zip(results, err):
            if e > thresh:
                self.rejections += 1
                if r["malicious"]:
                    self.true_positives += 1
                else:
                    self.false_positives += 1
            else:
                accepted.append(r)
                benign_sigs.append(self._sig(r["delta"]))
                self.errors.append(float(e))

        if benign_sigs:
            self.fit_step(torch.stack(benign_sigs), epochs=5)
        self.errors = self.errors[-2000:]
        return accepted if accepted else results   # never starve the round

    def summary(self) -> dict:
        return {
            "rejections": self.rejections,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
        }


def select_clients(action_hs: str, clients: List[Client], history: dict,
                   fraction: float, rng: np.random.Generator) -> List[int]:
    """The HS factor of Eq. 16."""
    n = len(clients)
    if action_hs == "random":
        k = int(history.get("random_k") or max(2, n // 2))
        return sorted(rng.choice(n, size=min(k, n), replace=False).tolist())
    if action_hs == "all":
        return list(range(n))
    k = max(2, int(round(fraction * n))) if fraction < 1.0 else max(2, n // 2)
    if action_hs == "topk_accuracy":
        losses = history.get("client_loss", {})
        reverse = history.get("topk_direction") == "highest_loss"
        order = sorted(range(n), key=lambda c: losses.get(c, 0.0),
                       reverse=reverse)
        return sorted(order[:k])
    if action_hs == "diverse":
        # favour hospitals holding more distinct classes / more data
        div = history.get("client_diversity", {c: 1.0 for c in range(n)})
        w = np.array([div.get(c, 1.0) for c in range(n)], dtype=float)
        w = w / w.sum()
        return sorted(rng.choice(n, size=k, replace=False, p=w).tolist())
    raise ValueError(action_hs)


def aggregate(strategy: str, results: List[dict]) -> Dict[str, torch.Tensor]:
    deltas = [r["delta"] for r in results]
    weights = [float(r["n_samples"]) for r in results]
    if strategy in ("fedavg", "fedprox"):
        # FedProx changes the *local objective* (Eq. 12), not the aggregation;
        # the server still forms the sample-weighted mean (Eq. 10/FedAvg).
        return attacks.weighted_average(deltas, weights)
    if strategy == "trimmed_mean":
        return attacks.trimmed_mean(deltas, weights, trim=1)   # Eq. 24
    raise ValueError(f"unknown aggregation {strategy!r}")


def run_federated(cfg, fed: dict, verbose: bool = True,
                  log_path: Optional[str] = None) -> dict:
    """Run one full federation and return every metric the paper reports."""
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    meta = fed["meta"]
    n_classes = meta["n_classes"]
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    def model_fn():
        return build_model(meta)

    global_model = model_fn().to(device)
    num_params = count_parameters(global_model)

    # -- build hospitals ---------------------------------------------------
    mal_ids = set(rng.choice(cfg.num_clients, size=cfg.num_malicious,
                             replace=False).tolist()) if cfg.num_malicious else set()
    clients = [Client(c, fed["client_loaders"][c], model_fn, cfg, device,
                      n_classes, malicious=(c in mal_ids))
               for c in range(cfg.num_clients)]

    # -- resolve the DP noise multiplier from the target epsilon -----------
    # example-level: each hospital subsamples its own local set, so q and the
    # step count differ per hospital; sigma is solved per hospital so that all
    # of them finish the run at the same epsilon.
    dp_plan = None
    if cfg.use_dp:
        if cfg.dp_level == "example":
            dp_plan = {}
            for c in clients:
                q_i = min(1.0, cfg.batch_size / max(c.n_samples, 1))
                steps_i = (cfg.rounds * cfg.local_epochs
                           * int(np.ceil(c.n_samples / cfg.batch_size)))
                s_i = (cfg.dp_noise_multiplier
                       if cfg.dp_noise_multiplier is not None
                       else privacy.solve_noise_multiplier(
                           cfg.dp_target_epsilon, cfg.dp_delta, q=q_i,
                           steps=steps_i))
                c.set_dp_sigma(s_i)
                dp_plan[c.cid] = {
                    "n_samples": c.n_samples, "sample_rate": q_i,
                    "steps": steps_i, "noise_multiplier": s_i,
                    "epsilon": privacy.epsilon_for(s_i, q_i, steps_i,
                                                   cfg.dp_delta)}
            accountant = None
        else:
            q = 1.0 if cfg.client_fraction >= 1.0 else cfg.client_fraction
            if cfg.dp_noise_multiplier is None:
                cfg.dp_noise_multiplier = privacy.solve_noise_multiplier(
                    cfg.dp_target_epsilon, cfg.dp_delta, q=q, steps=cfg.rounds)
            for c in clients:
                c.set_dp_sigma(cfg.dp_noise_multiplier)
            accountant = privacy.DPAccountant(cfg.dp_noise_multiplier, q,
                                              cfg.dp_delta)
    else:
        accountant = None
        for c in clients:
            c.set_dp_sigma(0.0)

    diversity = {r["client"]: float(r["n_classes_present"])
                 for r in fed["report"]["clients"]}
    label_skew = float(np.std([r["n_samples"] for r in fed["report"]["clients"]])
                       / max(np.mean([r["n_samples"]
                                      for r in fed["report"]["clients"]]), 1e-9))

    agent = dqn_mod.DQNAgent(cfg, device="cpu") if cfg.use_dqn else None
    anomaly = AnomalyFilter(cfg, device)
    comm = compression.CommTracker()

    history = {"client_loss": {}, "client_diversity": diversity,
               "random_k": cfg.random_selection_k,
               "topk_direction": getattr(cfg, "topk_direction", "lowest_loss")}

    # Evaluate the initial global model so round 0 has a genuine predecessor
    # state and accuracy baseline. Without this the first round's transition is
    # discarded (99 stored for 100 rounds) and the first reward is computed
    # against an artificial accuracy of 0.
    _ev0 = client_mod.evaluate(global_model, fed["val_loader"], device, n_classes)
    _m0 = classification_metrics(_ev0["y_true"], _ev0["y_pred"], _ev0["y_prob"],
                                 n_classes)
    prev_acc = _m0["accuracy"]
    prev_state = dqn_mod.build_state(
        global_acc=_m0["accuracy"], global_acc_delta=0.0,
        global_loss=_m0["loss"], participation_rate=0.0,
        round_frac=0.0, label_skew=label_skew)
    prev_action, prev_reward = None, None
    curve, round_log = [], []
    best_acc, best_state = -1.0, None
    wall_times = []

    dense_up_per_client = compression.dense_payload_bytes(num_params, 32)
    t_run = time.perf_counter()

    for t in range(cfg.rounds):
        t_round = time.perf_counter()

        # ---- action (Eq. 16) --------------------------------------------
        if agent is not None and prev_state is not None:
            a_idx = agent.act(prev_state)
        elif agent is not None:
            a_idx = agent.act(dqn_mod.build_state())
        else:
            a_idx = None
        if a_idx is not None:
            act = agent.describe(a_idx)
        else:
            act = {"hospital_selection": ("random" if cfg.random_selection_k
                                          else "all"),
                   "aggregation": cfg.aggregation,
                   "lr_scale": 1.0}

        sel = select_clients(act["hospital_selection"], clients, history,
                             cfg.client_fraction, rng)

        # ---- local training ---------------------------------------------
        results = []
        for c in sel:
            res = clients[c].train(global_model.state_dict(), t,
                                   lr_scale=act["lr_scale"],
                                   # the local objective is governed by
                                   # cfg.use_fedprox alone. Previously it was
                                   # gated on the aggregation action, so any arm
                                   # that aggregated with FedAvg silently lost
                                   # the proximal term too -- which confounded
                                   # the no_dqn ablation with a FedProx removal.
                                   use_fedprox=cfg.use_fedprox,
                                   generator=gen)
            results.append(res)
            history["client_loss"][c] = res["loss"]

        # ---- server-side defences (Eq. 25, Sec. 6.5.3) -------------------
        if cfg.use_norm_clipping:
            for r in results:
                r["delta"], _, _ = attacks.norm_clip(r["delta"],
                                                     cfg.norm_clip_tau)
        accepted = anomaly.screen(results)

        # ---- aggregation --------------------------------------------------
        agg_delta = aggregate(act["aggregation"], accepted)
        new_state = client_mod.apply_delta(global_model.state_dict(), agg_delta)
        global_model.load_state_dict(new_state, strict=True)
        if accountant is not None:
            accountant.step()

        # ---- communication accounting (R2-6) -----------------------------
        uplink = sum(r["comm"]["payload_bytes"] for r in results)
        downlink = len(sel) * dense_up_per_client
        comm.record_round(uplink, downlink, len(results) * dense_up_per_client)

        # ---- evaluation ---------------------------------------------------
        ev = client_mod.evaluate(global_model, fed["val_loader"], device,
                                 n_classes)
        m = classification_metrics(ev["y_true"], ev["y_pred"], ev["y_prob"],
                                   n_classes)
        acc = m["accuracy"]
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(global_model.state_dict())

        round_s = time.perf_counter() - t_round
        wall_times.append(round_s)

        # ---- reward (Eq. 17) ----------------------------------------------
        # Eq. 17's cost terms, normalised to [0, 1] so the three reward
        # components are commensurate. Training cost is expressed as the
        # fraction of the federation activated rather than as wall-clock
        # seconds: seconds depend on how busy the host machine is, which would
        # make the reward -- and hence the learned policy -- depend on
        # unrelated system load.
        train_cost = len(sel) / max(cfg.num_clients, 1)
        lat_all = [r["latency_ms"] for r in results]
        max_lat = max(c.base_latency_ms for c in clients) + 100.0
        latency = float(np.mean(lat_all)) / max_lat
        r_t = dqn_mod.reward(acc, prev_acc, train_cost, latency, cfg)

        state = dqn_mod.build_state(
            global_acc=acc,
            global_acc_delta=acc - prev_acc,
            global_loss=m["loss"],
            mean_client_loss=float(np.mean([r["loss"] for r in results])),
            std_client_loss=float(np.std([r["loss"] for r in results])),
            mean_update_norm=float(np.mean([r.get("raw_norm") or 0.0
                                            for r in results])),
            std_update_norm=float(np.std([r.get("raw_norm") or 0.0
                                          for r in results])),
            participation_rate=len(sel) / cfg.num_clients,
            mean_compute=float(np.mean([r["compute_capacity"] for r in results])),
            min_compute=float(np.min([r["compute_capacity"] for r in results])),
            mean_latency=latency,
            max_latency=float(np.max([r["latency_ms"] for r in results])) / 1000.0,
            round_frac=t / max(cfg.rounds - 1, 1),
            label_skew=label_skew,
        )

        # The transition must credit the action that actually produced this
        # round's reward, i.e. a_idx (chosen from prev_state at the top of the
        # round), not prev_action (the previous round's choice).
        if agent is not None and prev_state is not None and a_idx is not None:
            agent.observe(prev_state, a_idx, r_t, state,
                          done=(t == cfg.rounds - 1))
            agent.learn()

        prev_state, prev_action, prev_acc, prev_reward = state, a_idx, acc, r_t

        entry = {
            "round": t + 1, "accuracy": acc, "f1": m["macro_f1"],
            "loss": m["loss"], "reward": r_t, "action": act,
            "n_selected": len(sel), "n_accepted": len(accepted),
            "uplink_MB": uplink / 1e6, "round_time_s": round_s,
            "epsilon": accountant.epsilon if accountant else None,
        }
        curve.append(acc)
        round_log.append(entry)
        if verbose and ((t + 1) % 10 == 0 or t == 0):
            print(f"  [{cfg.tag}] round {t+1:3d}/{cfg.rounds}  acc={acc:.4f}  "
                  f"f1={m['macro_f1']:.4f}  R={r_t:+.4f}  "
                  f"{act['hospital_selection']}/{act['aggregation']}/"
                  f"lr x{act['lr_scale']}  {round_s:.1f}s", flush=True)

    total_time = time.perf_counter() - t_run

    # ---- final test evaluation (use the best validation checkpoint) ------
    if best_state is not None:
        global_model.load_state_dict(best_state)
    ev = client_mod.evaluate(global_model, fed["test_loader"], device, n_classes)
    test = classification_metrics(ev["y_true"], ev["y_pred"], ev["y_prob"],
                                  n_classes)

    out = {
        "config": cfg.to_dict(),
        "meta": meta,
        "test": test,
        "y_true": ev["y_true"].tolist(),
        "y_prob": ev["y_prob"].tolist(),
        "inference_ms_per_sample": ev["inference_ms_per_sample"],
        "accuracy_curve": curve,
        "rounds": round_log,
        "communication": comm.summary(),
        "num_params": num_params,
        "wall_clock": {
            "total_s": total_time,
            "mean_round_s": float(np.mean(wall_times)),
            "std_round_s": float(np.std(wall_times)),
            "median_round_s": float(np.median(wall_times)),
        },
        "privacy": (accountant.summary() if accountant
                    else ({"level": "example",
                           "delta": cfg.dp_delta,
                           "target_epsilon": cfg.dp_target_epsilon,
                           "per_client": dp_plan,
                           "epsilon": max(v["epsilon"] for v in dp_plan.values()),
                           "mean_noise_multiplier": float(np.mean(
                               [v["noise_multiplier"] for v in dp_plan.values()]))}
                          if dp_plan else {"epsilon": None})),
        "anomaly_filter": anomaly.summary(),
        "malicious_clients": sorted(mal_ids),
        "partition_report": fed["report"],
        "convergence_round": convergence_round(curve),
        "model": global_model,
        "device": str(device),
    }
    if agent is not None:
        out["dqn"] = {
            "architecture": dqn_mod.architecture_summary(cfg),
            "mean_loss": float(np.mean(agent.losses)) if agent.losses else None,
            "final_epsilon_greedy": agent.epsilon(),
            "action_counts": _action_counts(round_log),
        }
    if log_path:
        _save(out, log_path)
    return out


def _action_counts(round_log):
    from collections import Counter
    c = Counter((e["action"]["hospital_selection"], e["action"]["aggregation"],
                 e["action"]["lr_scale"]) for e in round_log)
    return {f"{a}|{b}|x{l}": n for (a, b, l), n in c.most_common()}


def convergence_round(curve: List[float], frac: float = 0.99) -> Optional[int]:
    """First round reaching `frac` of the run's best accuracy (Table 8)."""
    if not curve:
        return None
    target = frac * max(curve)
    for i, a in enumerate(curve):
        if a >= target:
            return i + 1
    return len(curve)


def _save(out: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {k: v for k, v in out.items() if k != "model"}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)


# ---------------------------------------------------------------------------
# centralised baseline
# ---------------------------------------------------------------------------
def run_centralized(cfg, fed: dict, epochs: Optional[int] = None,
                    verbose: bool = True) -> dict:
    """All data pooled on one server -- the CL row of Tables 4, 7, 8."""
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    meta = fed["meta"]
    model = build_model(meta).to(device)
    epochs = epochs or cfg.rounds
    loader = torch.utils.data.DataLoader(
        fed["train"], batch_size=cfg.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_fl)

    curve, best_acc, best_state, times = [], -1.0, None, []
    t0 = time.perf_counter()
    for ep in range(epochs):
        te = time.perf_counter()
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device).reshape(-1)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if cfg.use_adv_training and cfg.adv_epsilon > 0:
                m = max(1, int(cfg.adv_ratio * x.size(0)))
                xa = attacks.fgsm(model, x[:m], y[:m], cfg.adv_epsilon)
                model.train()
                loss = 0.5 * loss + 0.5 * F.cross_entropy(model(xa), y[:m])
            loss.backward()
            opt.step()
        times.append(time.perf_counter() - te)

        ev = client_mod.evaluate(model, fed["val_loader"], device,
                                 meta["n_classes"])
        m = classification_metrics(ev["y_true"], ev["y_pred"], ev["y_prob"],
                                   meta["n_classes"])
        curve.append(m["accuracy"])
        if m["accuracy"] > best_acc:
            best_acc, best_state = m["accuracy"], copy.deepcopy(model.state_dict())
        if verbose and ((ep + 1) % 10 == 0 or ep == 0):
            print(f"  [CL] epoch {ep+1:3d}/{epochs}  acc={m['accuracy']:.4f}",
                  flush=True)
    total = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    ev = client_mod.evaluate(model, fed["test_loader"], device, meta["n_classes"])
    test = classification_metrics(ev["y_true"], ev["y_pred"], ev["y_prob"],
                                  meta["n_classes"])

    n_params = count_parameters(model)
    # CL uploads the raw training set once; that is its communication cost.
    raw_bytes = int(np.prod(fed["train"].x.shape) * 4)
    return {
        "config": cfg.to_dict(), "meta": meta, "test": test,
        "y_true": ev["y_true"].tolist(), "y_prob": ev["y_prob"].tolist(),
        "inference_ms_per_sample": ev["inference_ms_per_sample"],
        "accuracy_curve": curve,
        "convergence_round": convergence_round(curve),
        "num_params": n_params,
        "communication": {
            "rounds": epochs,
            "total_MB": raw_bytes / 1e6,
            "per_round_MB": raw_bytes / 1e6 / max(epochs, 1),
            "note": "raw patient data uploaded once to the central server",
        },
        "wall_clock": {"total_s": total, "mean_round_s": float(np.mean(times)),
                       "std_round_s": float(np.std(times))},
        "privacy": {"epsilon": None, "note": "no formal guarantee; raw data shared"},
        "model": model, "device": str(device),
    }
