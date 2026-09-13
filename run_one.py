"""Run a single (experiment, arm, seed) and write one JSON file.

Kept deliberately small so the whole suite can be driven as independent
processes: runs are resumable (an existing output file is skipped) and a crash
in one arm cannot lose the others.
"""
import argparse
import json
import os
import time
from dataclasses import replace

import hashlib
import numpy as np
import torch

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---- arm definitions -------------------------------------------------------
def arm_overrides(experiment: str, arm: str) -> tuple:
    """Return (config overrides, kind) for an arm. kind: federated|centralized."""
    E = experiment
    if E == "main":
        return {
            "CL": ({}, "centralized"),
            "FedAvg": ({"use_dqn": False, "use_fedprox": False,
                        "aggregation": "fedavg"}, "federated"),
            "FedProx": ({"use_dqn": False, "use_fedprox": True,
                         "aggregation": "fedprox"}, "federated"),
            "FL-DRL": ({"use_dqn": True, "use_fedprox": True}, "federated"),
            # matched-rate control: FL-DRL selects 6.57 of 10 hospitals per
            # round on average, so this samples 7 uniformly at random
            "FL-DRL-small": ({"use_dqn": True, "use_fedprox": True,
                              "dqn_action_mode": "selection_only"},
                             "federated"),
            "FL-DRL-hard": ({"use_dqn": True, "use_fedprox": True,
                             "dqn_action_mode": "selection_only",
                             "topk_direction": "highest_loss"},
                            "federated"),
            "RandomSub": ({"use_dqn": False, "use_fedprox": True,
                           "aggregation": "fedprox",
                           "random_selection_k": 7}, "federated"),
        }[arm]

    if E == "ablation":
        return {
            "full": ({}, "federated"),
            "no_dqn": ({"use_dqn": False}, "federated"),
            "no_fedprox": ({"use_fedprox": False, "aggregation": "fedavg"},
                           "federated"),
            "plus_dp": ({"use_dp": True, "local_epochs": 1}, "federated"),
            "no_compression": ({"use_compression": False}, "federated"),
            "no_adv_training": ({"use_adv_training": False}, "federated"),
            "no_anomaly_filter": ({"use_autoencoder_filter": False}, "federated"),
        }[arm]

    if E == "privacy":
        # Swept over the noise multiplier rather than a target epsilon: at this
        # step count a small target epsilon is simply unreachable, so fixing
        # sigma and *reporting* the accounted epsilon is the honest direction
        # to run the sweep in.
        if arm == "inf":
            return ({"use_dp": False, "local_epochs": 1}, "federated")
        return ({"use_dp": True, "dp_noise_multiplier": float(arm),
                 "dp_target_epsilon": None, "local_epochs": 1}, "federated")

    if E == "privopt":
        # Privacy-optimised configuration: one local epoch per round instead of
        # five, which cuts the composed step count 5x and so buys a far smaller
        # accounted epsilon at the same noise multiplier.
        if arm == "inf":
            return ({"use_dp": False, "local_epochs": 1}, "federated")
        return ({"use_dp": True, "dp_noise_multiplier": float(arm),
                 "dp_target_epsilon": None, "local_epochs": 1}, "federated")

    if E == "reward":
        # Reward-weight calibration for Eq. 17, run on a held-out seed.
        # Arm name encodes "beta_gamma"; alpha is fixed at 1.0 throughout.
        b, g = arm.split("_")
        return ({"reward_alpha": 1.0, "reward_beta": float(b),
                 "reward_gamma": float(g), "use_dqn": True}, "federated")

    if E == "attacks":
        return {
            "clean": ({"num_malicious": 0}, "federated"),
            "poison2_defended": ({"num_malicious": 2,
                                  "poison_type": "sign_flip"}, "federated"),
            "poison2_undefended": ({"num_malicious": 2,
                                    "poison_type": "sign_flip",
                                    "use_norm_clipping": False,
                                    "use_autoencoder_filter": False,
                                    "use_dqn": False,
                                    "aggregation": "fedavg"}, "federated"),
            "poison3_defended": ({"num_malicious": 3,
                                  "poison_type": "sign_flip"}, "federated"),
            "labelflip3_defended": ({"num_malicious": 3,
                                     "poison_type": "label_flip"}, "federated"),
            "labelflip3_undefended": ({"num_malicious": 3,
                                       "poison_type": "label_flip",
                                       "use_norm_clipping": False,
                                       "use_autoencoder_filter": False},
                                      "federated"),
            "no_adv_training": ({"num_malicious": 0,
                                 "use_adv_training": False}, "federated"),
        }[arm]

    if E == "partition":
        return {
            "iid": ({"partition": "iid"}, "federated"),
            "dirichlet_1.0": ({"partition": "dirichlet",
                               "dirichlet_alpha": 1.0}, "federated"),
            "dirichlet_0.5": ({"partition": "dirichlet",
                               "dirichlet_alpha": 0.5}, "federated"),
            "dirichlet_0.1": ({"partition": "dirichlet",
                               "dirichlet_alpha": 0.1}, "federated"),
        }[arm]

    raise ValueError(f"unknown experiment {E!r}")


ARMS = {
    "main": ["CL", "FedAvg", "FedProx", "FL-DRL", "RandomSub"],
    "ablation": ["full", "no_dqn", "no_fedprox", "plus_dp", "no_compression",
                 "no_adv_training", "no_anomaly_filter"],
    "privacy": ["0.6", "0.8", "1.0", "1.5", "2.0", "inf"],
    "privopt": ["0.6", "1.0", "1.5", "2.0", "inf"],
    "attacks": ["clean", "poison2_defended", "poison2_undefended",
                "poison3_defended", "labelflip3_defended",
                "labelflip3_undefended", "no_adv_training"],
    "partition": ["iid", "dirichlet_1.0", "dirichlet_0.5", "dirichlet_0.1"],
    "reward": [f"{b}_{g}" for b in ("0.01", "0.05", "0.2")
               for g in ("0.01", "0.05", "0.2")],
}


SOURCE_FILES = ["config.py", "server.py", "client.py", "dqn.py", "models.py",
                "data.py", "privacy.py", "compression.py", "attacks.py",
                "metrics.py", "run_one.py"]


def source_hash() -> str:
    """Hash of every file that can change a result.

    Stored in each result so a run can be attributed to the code that produced
    it, and so cached results from superseded code are rejected rather than
    silently reused.
    """
    h = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    for name in SOURCE_FILES:
        path = os.path.join(here, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:16]


def out_path(experiment: str, arm: str, seed: int,
             dataset: str = "bloodmnist") -> str:
    safe = arm.replace("/", "_")
    return os.path.join(RESULTS, dataset, experiment, f"{safe}__seed{seed}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--dataset", default="bloodmnist")
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    path = out_path(args.experiment, args.arm, args.seed, args.dataset)
    src = source_hash()
    if os.path.exists(path) and not args.force:
        try:
            with open(path) as f:
                prev = json.load(f)
        except Exception:
            prev = {}
        if prev.get("source_hash") == src:
            print(f"SKIP (exists, same source) {path}")
            return
        print(f"STALE (source changed {prev.get('source_hash')} -> {src}), "
              f"re-running {path}")

    # imported here so --help stays fast and threads are set first
    from config import Config
    from data import build_federation
    import client as client_mod
    import server

    over, kind = arm_overrides(args.experiment, args.arm)
    kwargs = dict(dataset=args.dataset, num_clients=args.clients,
                  rounds=args.rounds, local_epochs=args.local_epochs,
                  dirichlet_alpha=args.alpha, seed=args.seed, device="cpu",
                  tag=f"{args.experiment}/{args.arm}")
    kwargs.update(over)          # arm overrides win over the CLI defaults
    cfg = Config(**kwargs)

    fed = build_federation(cfg)
    t0 = time.perf_counter()
    if kind == "centralized":
        res = server.run_centralized(cfg, fed, verbose=True)
    else:
        res = server.run_federated(cfg, fed, verbose=True)
    res["arm"] = args.arm
    res["experiment"] = args.experiment
    res["source_hash"] = src
    res["package_versions"] = {
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    res["elapsed_s"] = time.perf_counter() - t0

    # robustness sweep (R2-8) -- skipped for the privacy sweep to save compute
    if args.experiment not in ("privacy", "reward"):
        device = server.pick_device("cpu")
        model = res["model"]
        rob = {}
        for eps in (0.0, 0.01, 0.03, 0.05, 0.1):
            rob[f"fgsm_{eps}"] = client_mod.robust_evaluate(
                model, fed["test_loader"], device, eps, "fgsm")
        for eps in (0.03, 0.05):
            rob[f"pgd_{eps}"] = client_mod.robust_evaluate(
                model, fed["test_loader"], device, eps, "pgd", pgd_steps=10)
        res["robustness"] = rob

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {k: v for k, v in res.items() if k != "model"}
    with open(path, "w") as f:
        json.dump(payload, f, default=float)

    t = res["test"]
    print(f"DONE {args.experiment}/{args.arm} seed={args.seed} "
          f"acc={t['accuracy']:.4f} f1={t['macro_f1']:.4f} auc={t['auc']:.4f} "
          f"({res['elapsed_s']/60:.1f} min) -> {path}", flush=True)


if __name__ == "__main__":
    main()
