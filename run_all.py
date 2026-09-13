"""Drive the whole suite as parallel, resumable subprocesses.

    python run_all.py --workers 4 --rounds 100 \
        --experiments main ablation privacy attacks partition

Each (experiment, arm, seed) is one subprocess writing one JSON file. Re-running
skips finished jobs, so the suite can be stopped and resumed freely.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import json

from run_one import ARMS, out_path, source_hash

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "bin", "python")


def build_jobs(experiments, seeds_map, rounds, clients, dataset, local_epochs,
               alpha, threads):
    SRC = source_hash()
    jobs = []
    for exp in experiments:
        for arm in ARMS[exp]:
            for seed in seeds_map[exp]:
                path = out_path(exp, arm, seed, dataset)
                if os.path.exists(path):
                    try:
                        with open(path) as f:
                            prev = json.load(f)
                    except Exception:
                        prev = {}
                    # only reuse a result produced by the current source
                    if prev.get("source_hash") == SRC:
                        continue
                jobs.append([
                    PY, os.path.join(HERE, "run_one.py"),
                    "--experiment", exp, "--arm", arm, "--seed", str(seed),
                    "--rounds", str(rounds), "--clients", str(clients),
                    "--dataset", dataset, "--local-epochs", str(local_epochs),
                    "--alpha", str(alpha), "--threads", str(threads),
                ])
    return jobs


def run(job, log_dir):
    exp, arm, seed = job[3], job[5], job[7]
    ds = job[job.index("--dataset") + 1]
    log = os.path.join(log_dir,
                       f"{ds}__{exp}__{arm.replace('/', '_')}__s{seed}.log")
    t0 = time.perf_counter()
    with open(log, "w") as f:
        p = subprocess.run(job, stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
    dt = (time.perf_counter() - t0) / 60
    status = "ok" if p.returncode == 0 else f"FAIL({p.returncode})"
    print(f"[{status}] {exp}/{arm} seed={seed}  {dt:.1f} min  (log: {log})",
          flush=True)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+",
                    default=["main", "ablation", "privacy", "attacks",
                             "partition"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--dataset", default="bloodmnist")
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--main-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--other-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds_map = {e: (args.main_seeds if e == "main" else args.other_seeds)
                 for e in args.experiments}
    jobs = build_jobs(args.experiments, seeds_map, args.rounds, args.clients,
                      args.dataset, args.local_epochs, args.alpha, args.threads)

    log_dir = os.path.join(HERE, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print(f"{len(jobs)} job(s) to run on {args.workers} workers "
          f"({args.threads} torch threads each)")
    if args.dry_run:
        for j in jobs:
            print("  ", j[3], j[5], "seed", j[7])
        return

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        codes = list(ex.map(lambda j: run(j, log_dir), jobs))
    bad = sum(1 for c in codes if c != 0)
    print(f"\nfinished {len(jobs)} job(s) in {(time.perf_counter()-t0)/60:.1f} "
          f"min; {bad} failure(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
