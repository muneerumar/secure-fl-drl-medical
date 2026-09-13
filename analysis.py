"""Turn the raw per-run JSON into the tables and figures the paper needs.

    python analysis.py

Writes tables/*.csv (and a paper_tables.md summarising them) plus figures/*.png.
Every table carries mean +- std over seeds, which is what reviewer R2-9 asked
for; nothing is reported from a single run.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TABLES = os.path.join(HERE, "tables")
FIGURES = os.path.join(HERE, "figures")


DATASET = os.environ.get("FLDRL_DATASET", "bloodmnist")


def load(experiment: str, dataset: str = None) -> dict:
    """arm -> list of run dicts (one per seed)."""
    dataset = dataset or DATASET
    out = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(RESULTS, dataset, experiment,
                                              "*.json"))):
        with open(path) as f:
            d = json.load(f)
        out[d["arm"]].append(d)
    return dict(out)


def ms(values, fmt="{:.4f}") -> str:
    v = np.asarray(values, dtype=float)
    if len(v) == 1:
        return fmt.format(v[0])
    return f"{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1))}"


def _get(run, *path, default=None):
    cur = run
    for p in path:
        if cur is None:
            return default
        cur = cur.get(p) if isinstance(cur, dict) else None
    return default if cur is None else cur


def write_csv(name: str, header, rows):
    os.makedirs(TABLES, exist_ok=True)
    path = os.path.join(TABLES, name)
    with open(path, "w") as f:
        f.write(",".join(str(h) for h in header) + "\n")
        for r in rows:
            f.write(",".join(str(c) for c in r) + "\n")
    print(f"  -> {path}")
    return path


def md_table(header, rows) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
def table_performance(runs, order=None) -> tuple:
    """Accuracy / precision / recall / F1 / AUC -- replaces Table 7."""
    header = ["Model", "Accuracy", "Macro precision", "Macro recall",
              "Macro F1", "AUC", "Seeds"]
    rows = []
    for arm in (order or sorted(runs)):
        if arm not in runs:
            continue
        rs = runs[arm]
        rows.append([
            arm,
            ms([r["test"]["accuracy"] for r in rs]),
            ms([r["test"]["macro_precision"] for r in rs]),
            ms([r["test"]["macro_recall"] for r in rs]),
            ms([r["test"]["macro_f1"] for r in rs]),
            ms([r["test"]["auc"] for r in rs]),
            len(rs),
        ])
    return header, rows


def table_efficiency(runs, order=None) -> tuple:
    """Convergence / communication / inference / wall-clock -- Table 8 + R1-6."""
    header = ["Model", "Convergence round", "Uplink per round (MB)",
              "Total payload (MB, up+down)", "Compression ratio",
              "Uplink reduction (%)",
              "Inference (ms/sample)", "Wall-clock per round (s)"]
    rows = []
    for arm in (order or sorted(runs)):
        if arm not in runs:
            continue
        rs = runs[arm]
        comm = [r.get("communication", {}) for r in rs]
        has_up = all("uplink_per_round_MB" in c for c in comm)
        rows.append([
            arm,
            ms([r["convergence_round"] for r in rs], "{:.1f}"),
            ms([c["uplink_per_round_MB"] for c in comm], "{:.3f}") if has_up
            else "n/a (raw data upload)",
            ms([c.get("total_MB", float("nan")) for c in comm], "{:.1f}"),
            ms([c["compression_ratio"] for c in comm], "{:.2f}") if has_up
            else "n/a",
            ms([c["uplink_reduction_pct"] for c in comm], "{:.1f}") if has_up
            else "n/a",
            ms([r["inference_ms_per_sample"] for r in rs], "{:.4f}"),
            ms([_get(r, "wall_clock", "mean_round_s", default=float("nan"))
                for r in rs], "{:.1f}"),
        ])
    return header, rows


def table_privacy(runs) -> tuple:
    """Accuracy vs *accounted* epsilon -- replaces Table 6."""
    header = ["Noise multiplier σ", "Accounted ε (worst client)",
              "Accounted ε (median client)", "Accuracy", "Macro F1", "AUC"]
    rows = []
    def key(a):
        return float("inf") if a == "inf" else float(a)
    for arm in sorted(runs, key=key):
        rs = runs[arm]
        if arm == "inf":
            eps_w = eps_m = "∞ (no DP)"
            sigma = "0 (disabled)"
        else:
            sigma = arm
            per = [_get(r, "privacy", "per_client", default={}) for r in rs]
            worst = [max(v["epsilon"] for v in p.values()) for p in per if p]
            med = [float(np.median([v["epsilon"] for v in p.values()]))
                   for p in per if p]
            eps_w = ms(worst, "{:.1f}") if worst else "n/a"
            eps_m = ms(med, "{:.1f}") if med else "n/a"
        rows.append([
            sigma, eps_w, eps_m,
            ms([r["test"]["accuracy"] for r in rs]),
            ms([r["test"]["macro_f1"] for r in rs]),
            ms([r["test"]["auc"] for r in rs]),
        ])
    return header, rows


def table_ablation(runs) -> tuple:
    """R2-4: what each component contributes."""
    header = ["Variant", "Accuracy", "Macro F1", "AUC", "Δ Accuracy vs full",
              "Uplink/round (MB)"]
    base = None
    if "full" in runs:
        base = float(np.mean([r["test"]["accuracy"] for r in runs["full"]]))
    order = ["full", "no_dqn", "no_fedprox", "no_dp", "no_compression",
             "no_adv_training", "no_anomaly_filter"]
    rows = []
    for arm in [a for a in order if a in runs] + \
               [a for a in sorted(runs) if a not in order]:
        rs = runs[arm]
        acc = float(np.mean([r["test"]["accuracy"] for r in rs]))
        rows.append([
            arm,
            ms([r["test"]["accuracy"] for r in rs]),
            ms([r["test"]["macro_f1"] for r in rs]),
            ms([r["test"]["auc"] for r in rs]),
            "—" if base is None or arm == "full" else f"{acc - base:+.4f}",
            ms([_get(r, "communication", "uplink_per_round_MB",
                     default=float("nan")) for r in rs], "{:.3f}"),
        ])
    return header, rows


def table_robustness(runs, order=None) -> tuple:
    """R2-8: accuracy under evasion attack."""
    header = ["Model", "Clean", "FGSM ε=0.01", "FGSM ε=0.03", "FGSM ε=0.05",
              "FGSM ε=0.1", "PGD ε=0.03", "PGD ε=0.05"]
    keys = ["fgsm_0.0", "fgsm_0.01", "fgsm_0.03", "fgsm_0.05", "fgsm_0.1",
            "pgd_0.03", "pgd_0.05"]
    rows = []
    for arm in (order or sorted(runs)):
        if arm not in runs:
            continue
        rs = [r for r in runs[arm] if "robustness" in r]
        if not rs:
            continue
        rows.append([arm] + [ms([r["robustness"][k] for r in rs]) for k in keys])
    return header, rows


def table_poisoning(runs) -> tuple:
    """R2-8: model poisoning, with and without the proposed defences."""
    header = ["Setting", "Accuracy", "Macro F1", "AUC",
              "Updates rejected", "True positives", "False positives"]
    rows = []
    for arm in sorted(runs):
        rs = runs[arm]
        rows.append([
            arm,
            ms([r["test"]["accuracy"] for r in rs]),
            ms([r["test"]["macro_f1"] for r in rs]),
            ms([r["test"]["auc"] for r in rs]),
            ms([_get(r, "anomaly_filter", "rejections", default=0)
                for r in rs], "{:.1f}"),
            ms([_get(r, "anomaly_filter", "true_positives", default=0)
                for r in rs], "{:.1f}"),
            ms([_get(r, "anomaly_filter", "false_positives", default=0)
                for r in rs], "{:.1f}"),
        ])
    return header, rows


def table_partition(runs) -> tuple:
    header = ["Partition", "Clients missing a class", "Min / max samples",
              "Accuracy", "Macro F1", "AUC"]
    rows = []
    for arm in sorted(runs):
        rs = runs[arm]
        rep = [r.get("partition_report", {}) for r in rs]
        rows.append([
            arm,
            ms([p.get("clients_missing_a_class", 0) for p in rep], "{:.1f}"),
            f"{int(np.mean([p.get('min_samples', 0) for p in rep]))} / "
            f"{int(np.mean([p.get('max_samples', 0) for p in rep]))}",
            ms([r["test"]["accuracy"] for r in rs]),
            ms([r["test"]["macro_f1"] for r in rs]),
            ms([r["test"]["auc"] for r in rs]),
        ])
    return header, rows


# ---------------------------------------------------------------------------
def significance(main) -> dict:
    """R1-8 (DeLong vs CL) and R2-10 (paired t-tests across seeds)."""
    out = {}
    if "CL" not in main:
        return out
    n_classes = main["CL"][0]["meta"]["n_classes"]

    # DeLong on the seed-0 test scores, which every arm shares
    for arm in main:
        if arm == "CL":
            continue
        try:
            a = main[arm][0]
            c = main["CL"][0]
            y = np.asarray(c["y_true"])
            pa = np.asarray(a["y_prob"])
            pc = np.asarray(c["y_prob"])
            if n_classes == 2:
                out[f"delong_{arm}_vs_CL"] = metrics.delong_test(
                    y, pa[:, 1], pc[:, 1])
            else:
                out[f"delong_{arm}_vs_CL"] = metrics.delong_multiclass(
                    y, pa, pc, n_classes)
        except Exception as e:
            out[f"delong_{arm}_vs_CL"] = {"error": str(e)}

    # paired t-tests across seeds
    def accs(arm, seeds=None):
        by_seed = {r["config"]["seed"]: r["test"]["accuracy"] for r in main[arm]}
        if seeds is None:
            return by_seed
        return [by_seed[s] for s in seeds]
    pairs = [("FL-DRL", "RandomSub"), ("FL-DRL", "CL"), ("FL-DRL", "FedAvg"),
             ("FL-DRL", "FedProx"), ("RandomSub", "FedProx"),
             ("FedProx", "FedAvg")]
    for a, b in pairs:
        if a not in main or b not in main:
            continue
        shared = sorted(set(accs(a)) & set(accs(b)))   # join on seed id
        if len(shared) > 1:
            out[f"ttest_{a}_vs_{b}_accuracy"] = metrics.paired_ttest(
                accs(a, shared), accs(b, shared))
            out[f"ttest_{a}_vs_{b}_accuracy"]["seeds"] = shared
    for arm in main:
        out[f"summary_{arm}_accuracy"] = metrics.summarize_runs(
            list(accs(arm).values()))
        out[f"summary_{arm}_auc"] = metrics.summarize_runs(
            [r["test"]["auc"] for r in main[arm]])
    return out


# ---------------------------------------------------------------------------
def figures(main, privacy_runs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib unavailable: {e})")
        return
    os.makedirs(FIGURES, exist_ok=True)

    # Figure 4: accuracy vs communication round, mean +- std band
    if main:
        plt.figure(figsize=(7, 4.5))
        for arm in ["CL", "FedAvg", "FedProx", "FL-DRL"]:
            if arm not in main:
                continue
            curves = [r["accuracy_curve"] for r in main[arm]]
            n = min(len(c) for c in curves)
            arr = np.array([c[:n] for c in curves])
            m, s = arr.mean(axis=0), arr.std(axis=0)
            x = np.arange(1, n + 1)
            plt.plot(x, m, label=arm, linewidth=1.8)
            if len(curves) > 1:
                plt.fill_between(x, m - s, m + s, alpha=0.15)
        plt.xlabel("Communication round (CL: epoch)")
        plt.ylabel("Validation accuracy")
        plt.title("Convergence behaviour (mean ± s.d. over seeds)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        p = os.path.join(FIGURES, "fig4_convergence.png")
        plt.savefig(p, dpi=200)
        plt.close()
        print(f"  -> {p}")

    # Figure 5: ROC curves
    if main:
        plt.figure(figsize=(5.5, 5))
        n_classes = main[list(main)[0]][0]["meta"]["n_classes"]
        for arm in ["CL", "FedAvg", "FedProx", "FL-DRL"]:
            if arm not in main:
                continue
            r = main[arm][0]
            fpr, tpr = metrics.roc_points(np.asarray(r["y_true"]),
                                          np.asarray(r["y_prob"]), n_classes)
            plt.plot(fpr, tpr, label=f"{arm} (AUC={r['test']['auc']:.3f})",
                     linewidth=1.8)
        plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC comparison (test set, seed 0)")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        p = os.path.join(FIGURES, "fig5_roc.png")
        plt.savefig(p, dpi=200)
        plt.close()
        print(f"  -> {p}")

    # Figure 3: privacy-utility trade-off with real epsilon on the x axis
    if privacy_runs:
        xs, ys, es = [], [], []
        for arm, rs in privacy_runs.items():
            if arm == "inf":
                continue
            per = [_get(r, "privacy", "per_client", default={}) for r in rs]
            worst = [max(v["epsilon"] for v in p.values()) for p in per if p]
            if not worst:
                continue
            xs.append(float(np.mean(worst)))
            a = [r["test"]["accuracy"] for r in rs]
            ys.append(float(np.mean(a)))
            es.append(float(np.std(a)))
        if xs:
            o = np.argsort(xs)
            xs = np.array(xs)[o]; ys = np.array(ys)[o]; es = np.array(es)[o]
            plt.figure(figsize=(6.5, 4.5))
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, linewidth=1.8)
            if "inf" in privacy_runs:
                base = float(np.mean([r["test"]["accuracy"]
                                      for r in privacy_runs["inf"]]))
                plt.axhline(base, color="grey", linestyle="--",
                            label=f"no DP ({base:.3f})")
                plt.legend()
            plt.xscale("log")
            plt.xlabel("Accounted privacy budget ε (worst-case hospital, δ=1e-5)")
            plt.ylabel("Test accuracy")
            plt.title("Privacy–utility trade-off")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            p = os.path.join(FIGURES, "fig3_privacy_utility.png")
            plt.savefig(p, dpi=200)
            plt.close()
            print(f"  -> {p}")


# ---------------------------------------------------------------------------
def main_():
    os.makedirs(TABLES, exist_ok=True)
    doc = ["# Regenerated tables\n",
           "All values are mean ± sample standard deviation over independent "
           "seeds.\n"]

    main = load("main")
    order = ["CL", "FedAvg", "FedProx", "RandomSub", "FL-DRL"]

    if main:
        h, r = table_performance(main, order)
        write_csv("table7_performance.csv", h, r)
        doc += ["\n## Table 7 — classification performance\n", md_table(h, r)]

        h, r = table_efficiency(main, order)
        write_csv("table8_efficiency.csv", h, r)
        doc += ["\n\n## Table 8 — convergence, communication, inference\n",
                md_table(h, r)]

        h, r = table_robustness(main, order)
        if r:
            write_csv("table_robustness_main.csv", h, r)
            doc += ["\n\n## Robustness under evasion attack (new)\n",
                    md_table(h, r)]

    priv = load("privacy")
    if priv:
        h, r = table_privacy(priv)
        write_csv("table6_privacy.csv", h, r)
        doc += ["\n\n## Table 6 — accuracy vs accounted privacy budget\n",
                md_table(h, r)]

    abl = load("ablation")
    if abl:
        h, r = table_ablation(abl)
        write_csv("table9_ablation.csv", h, r)
        doc += ["\n\n## Table 9 — ablation study (new, R2-4)\n", md_table(h, r)]

    atk = load("attacks")
    if atk:
        h, r = table_poisoning(atk)
        write_csv("table10_poisoning.csv", h, r)
        doc += ["\n\n## Table 10 — model poisoning (new, R2-8)\n",
                md_table(h, r)]
        h, r = table_robustness(atk)
        if r:
            write_csv("table11_evasion.csv", h, r)
            doc += ["\n\n## Table 11 — evasion robustness by setting\n",
                    md_table(h, r)]

    part = load("partition")
    if part:
        h, r = table_partition(part)
        write_csv("table12_partition.csv", h, r)
        doc += ["\n\n## Table 12 — sensitivity to the hospital partition\n",
                md_table(h, r)]

    sig = significance(main) if main else {}
    if sig:
        path = os.path.join(TABLES, "significance.json")
        with open(path, "w") as f:
            json.dump(sig, f, indent=2, default=float)
        print(f"  -> {path}")
        doc.append("\n\n## Statistical tests\n")
        for k, v in sig.items():
            if k.startswith("delong") and "p_value" in v:
                doc.append(f"- **{k}**: ΔAUC = {v['difference']:+.4f}, "
                           f"z = {v['z']:.3f}, p = {v['p_value']:.4g}, "
                           f"95% CI [{v['ci95_low']:+.4f}, {v['ci95_high']:+.4f}]")
            elif k.startswith("ttest"):
                doc.append(f"- **{k}**: Δ = {v['mean_difference']:+.4f}, "
                           f"t = {v['t']:.3f}, p = {v['p_value']:.4g} "
                           f"(n = {v['n_pairs']} seeds)")
            elif k.startswith("summary"):
                doc.append(f"- **{k}**: {v['mean']:.4f} ± {v['std']:.4f} "
                           f"(95% CI [{v['ci_low']:.4f}, {v['ci_high']:.4f}], "
                           f"n = {v['n']})")

    figures(main, priv)

    path = os.path.join(HERE, "paper_tables.md")
    with open(path, "w") as f:
        f.write("\n".join(doc) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main_()
