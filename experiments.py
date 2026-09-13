"""Experiment suite -- one entry point per reviewer request.

    main      Tables 4/7/8 + Figures 4/5   (CL, FedAvg, FedProx, FL-DRL)
    ablation  R2-4: remove one component at a time
    privacy   Table 6: accuracy vs accounted epsilon
    attacks   R2-8: evasion (FGSM/PGD) and model poisoning
    partition R2-2: IID vs Dirichlet non-IID

Every arm is run over `--seeds` independent seeds so that R2-9 (mean, std,
confidence intervals) is satisfied by construction.
"""
import argparse
import copy
import json
import os
import pickle
import time
from dataclasses import replace

import numpy as np
import torch

from config import Config
from data import build_federation
import client as client_mod
import metrics
import server

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _fed_cache(cfg, cache={}):
    key = (cfg.dataset, cfg.num_clients, cfg.partition, cfg.dirichlet_alpha,
           cfg.batch_size, cfg.seed)
    if key not in cache:
        cache[key] = build_federation(cfg)
    return cache[key]


def _slim(res: dict) -> dict:
    """Drop the live model object so the result is JSON/pickle friendly."""
    return {k: v for k, v in res.items() if k != "model"}


def _save(obj, name: str):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"  -> wrote {path}")
    return path


# ---------------------------------------------------------------------------
def run_arm(name: str, cfg: Config, fed, kind: str = "federated",
            evaluate_robustness: bool = True) -> dict:
    print(f"\n=== {name} (seed {cfg.seed}) ===", flush=True)
    t0 = time.perf_counter()
    if kind == "centralized":
        res = server.run_centralized(cfg, fed)
    else:
        res = server.run_federated(cfg, fed)
    res["arm"] = name
    res["elapsed_s"] = time.perf_counter() - t0

    if evaluate_robustness:
        device = server.pick_device(cfg.device)
        model = res["model"]
        rob = {}
        for eps in (0.0, 0.01, 0.03, 0.05, 0.1):
            rob[f"fgsm_eps{eps}"] = client_mod.robust_evaluate(
                model, fed["test_loader"], device, eps, "fgsm")
        for eps in (0.03, 0.05):
            rob[f"pgd_eps{eps}"] = client_mod.robust_evaluate(
                model, fed["test_loader"], device, eps, "pgd", pgd_steps=10)
        res["robustness"] = rob
        print(f"  robustness: clean={rob['fgsm_eps0.0']:.4f} "
              f"fgsm@0.03={rob['fgsm_eps0.03']:.4f} "
              f"pgd@0.03={rob['pgd_eps0.03']:.4f}", flush=True)

    t = res["test"]
    print(f"  test acc={t['accuracy']:.4f} f1={t['macro_f1']:.4f} "
          f"auc={t['auc']:.4f}  ({res['elapsed_s']/60:.1f} min)", flush=True)
    return res


# ---------------------------------------------------------------------------
def exp_main(base: Config, seeds, rounds):
    """CL / FedAvg / FedProx / FL-DRL -- Tables 4, 7, 8 and Figures 4, 5."""
    all_runs = {}
    for seed in seeds:
        cfg = replace(base, seed=seed, rounds=rounds)
        fed = _fed_cache(cfg)

        arms = {
            "CL": (replace(cfg, tag="CL"), "centralized"),
            "FedAvg": (replace(cfg, tag="FedAvg", use_dqn=False,
                               use_fedprox=False, aggregation="fedavg"),
                       "federated"),
            "FedProx": (replace(cfg, tag="FedProx", use_dqn=False,
                                use_fedprox=True, aggregation="fedprox"),
                        "federated"),
            "FL-DRL": (replace(cfg, tag="FL-DRL", use_dqn=True,
                               use_fedprox=True), "federated"),
        }
        for name, (c, kind) in arms.items():
            res = run_arm(name, c, fed, kind)
            all_runs.setdefault(name, []).append(_slim(res))

    _save(all_runs, "main.json")
    return all_runs


def exp_ablation(base: Config, seeds, rounds):
    """R2-4: contribution of each component of FL-DRL."""
    variants = {
        "full": {},
        "no_dqn": {"use_dqn": False},
        "no_fedprox": {"use_fedprox": False, "aggregation": "fedavg"},
        "no_dp": {"use_dp": False, "dp_noise_multiplier": 0.0},
        "no_compression": {"use_compression": False},
        "no_adv_training": {"use_adv_training": False},
        "no_anomaly_filter": {"use_autoencoder_filter": False},
    }
    out = {}
    for seed in seeds:
        cfg = replace(base, seed=seed, rounds=rounds)
        fed = _fed_cache(cfg)
        for name, over in variants.items():
            c = replace(cfg, tag=f"abl-{name}", **over)
            if "dp_noise_multiplier" not in over:
                c = replace(c, dp_noise_multiplier=None)
            res = run_arm(f"ablation/{name}", c, fed)
            out.setdefault(name, []).append(_slim(res))
    _save(out, "ablation.json")
    return out


def exp_privacy(base: Config, seeds, rounds, budgets=(0.1, 0.5, 1.0, 5.0, 10.0)):
    """Table 6: accuracy vs a properly accounted privacy budget."""
    out = {}
    for seed in seeds:
        cfg = replace(base, seed=seed, rounds=rounds)
        fed = _fed_cache(cfg)
        for eps in budgets:
            c = replace(cfg, tag=f"eps{eps}", dp_target_epsilon=eps,
                        dp_noise_multiplier=None, use_dp=True)
            res = run_arm(f"privacy/eps={eps}", c, fed,
                          evaluate_robustness=False)
            out.setdefault(str(eps), []).append(_slim(res))
        c = replace(cfg, tag="no_dp", use_dp=False, dp_noise_multiplier=0.0)
        res = run_arm("privacy/no-DP", c, fed, evaluate_robustness=False)
        out.setdefault("inf", []).append(_slim(res))
    _save(out, "privacy.json")
    return out


def exp_attacks(base: Config, seeds, rounds):
    """R2-8: model poisoning with and without the proposed defences."""
    settings = {
        "clean": {"num_malicious": 0},
        "poison2_defended": {"num_malicious": 2, "poison_type": "sign_flip"},
        "poison2_undefended": {"num_malicious": 2, "poison_type": "sign_flip",
                               "use_norm_clipping": False,
                               "use_autoencoder_filter": False,
                               "aggregation": "fedavg"},
        "poison3_defended": {"num_malicious": 3, "poison_type": "sign_flip"},
        "labelflip3_defended": {"num_malicious": 3, "poison_type": "label_flip"},
        "labelflip3_undefended": {"num_malicious": 3, "poison_type": "label_flip",
                                  "use_norm_clipping": False,
                                  "use_autoencoder_filter": False},
        "no_adv_training": {"num_malicious": 0, "use_adv_training": False},
    }
    out = {}
    for seed in seeds:
        cfg = replace(base, seed=seed, rounds=rounds)
        fed = _fed_cache(cfg)
        for name, over in settings.items():
            c = replace(cfg, tag=f"atk-{name}", dp_noise_multiplier=None, **over)
            res = run_arm(f"attacks/{name}", c, fed)
            out.setdefault(name, []).append(_slim(res))
    _save(out, "attacks.json")
    return out


def exp_partition(base: Config, seeds, rounds):
    """R2-2 / R2-12: sensitivity to how the data is split across hospitals."""
    settings = {
        "iid": {"partition": "iid"},
        "dirichlet_1.0": {"partition": "dirichlet", "dirichlet_alpha": 1.0},
        "dirichlet_0.5": {"partition": "dirichlet", "dirichlet_alpha": 0.5},
        "dirichlet_0.1": {"partition": "dirichlet", "dirichlet_alpha": 0.1},
    }
    out = {}
    for seed in seeds:
        for name, over in settings.items():
            cfg = replace(base, seed=seed, rounds=rounds,
                          tag=f"part-{name}", dp_noise_multiplier=None, **over)
            fed = _fed_cache(cfg)
            res = run_arm(f"partition/{name}", cfg, fed,
                          evaluate_robustness=False)
            r = _slim(res)
            r["partition_summary"] = fed["report"]
            out.setdefault(name, []).append(r)
    _save(out, "partition.json")
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["main", "ablation", "privacy",
                                           "attacks", "partition", "all"])
    ap.add_argument("--dataset", default="pneumoniamnist")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    base = Config(dataset=args.dataset, num_clients=args.clients,
                  rounds=args.rounds, device=args.device,
                  local_epochs=args.local_epochs, dirichlet_alpha=args.alpha)

    print(f"device={server.pick_device(args.device)}  dataset={args.dataset}  "
          f"clients={args.clients}  rounds={args.rounds}  seeds={args.seeds}")

    table = {"main": exp_main, "ablation": exp_ablation, "privacy": exp_privacy,
             "attacks": exp_attacks, "partition": exp_partition}
    todo = list(table) if args.experiment == "all" else [args.experiment]
    for name in todo:
        t0 = time.perf_counter()
        table[name](base, args.seeds, args.rounds)
        print(f"[{name}] done in {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
