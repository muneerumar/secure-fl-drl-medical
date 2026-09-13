"""Choose the Eq. 17 reward weights from the held-out grid.

Selection rule, fixed in advance: highest best-validation accuracy on the
held-out seed. Efficiency (uplink, wall-clock) is reported alongside but is not
part of the selection criterion, so the chosen point cannot be accused of
having been picked to flatter the efficiency story.
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(HERE, "results", "bloodmnist", "reward")


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(GRID, "*.json"))):
        d = json.load(open(f))
        cfg = d["config"]
        rows.append({
            "arm": d["arm"],
            "beta": cfg["reward_beta"],
            "gamma": cfg["reward_gamma"],
            "val_best": max(d["accuracy_curve"]),
            "val_final": d["accuracy_curve"][-1],
            "test_acc": d["test"]["accuracy"],
            "test_f1": d["test"]["macro_f1"],
            "auc": d["test"]["auc"],
            "uplink_MB": d["communication"]["uplink_per_round_MB"],
            "wall_s": d["wall_clock"]["mean_round_s"],
            "convergence": d["convergence_round"],
        })
    return rows


def main():
    rows = load()
    if not rows:
        print("no grid results yet")
        return
    rows.sort(key=lambda r: -r["val_best"])

    print(f"{'beta':>6} {'gamma':>6} {'val_best':>9} {'test_acc':>9} "
          f"{'macroF1':>8} {'AUC':>7} {'uplink_MB':>10} {'wall_s':>7} {'conv':>5}")
    print("-" * 78)
    for r in rows:
        print(f"{r['beta']:>6} {r['gamma']:>6} {r['val_best']:>9.4f} "
              f"{r['test_acc']:>9.4f} {r['test_f1']:>8.4f} {r['auc']:>7.4f} "
              f"{r['uplink_MB']:>10.3f} {r['wall_s']:>7.1f} "
              f"{r['convergence']:>5}")

    best = rows[0]
    print(f"\nselected by best validation accuracy: "
          f"beta = {best['beta']}, gamma = {best['gamma']} "
          f"(val {best['val_best']:.4f}, test {best['test_acc']:.4f})")
    print(f"  uplink {best['uplink_MB']:.3f} MB/round, "
          f"wall-clock {best['wall_s']:.1f} s/round")

    if len(rows) > 1:
        worst = rows[-1]
        print(f"  spread across the grid: validation accuracy "
              f"{worst['val_best']:.4f} to {best['val_best']:.4f}, "
              f"uplink {min(r['uplink_MB'] for r in rows):.3f} to "
              f"{max(r['uplink_MB'] for r in rows):.3f} MB/round")

    out = os.path.join(HERE, "reward_selection.json")
    with open(out, "w") as f:
        json.dump({"selected": best, "grid": rows}, f, indent=2)
    print(f"\nwrote {out}")
    print(f"\nTo adopt: set reward_beta = {best['beta']} and "
          f"reward_gamma = {best['gamma']} in config.py, then re-run the "
          f"DQN-dependent arms.")


if __name__ == "__main__":
    main()
