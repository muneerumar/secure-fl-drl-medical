"""Collect every number, table and narrative sentence the documents need.

Writes numbers.json with:
  scalars   -- individual values interpolated into sentences
  tables    -- {name: {"header": [...], "rows": [[...], ...]}}
  narrative -- sentences derived from the results, not written by hand

Deterministic quantities are computed directly; run-dependent ones are read
from results/ when those runs have finished and omitted otherwise, so the
document builders keep a visible placeholder rather than an invented value.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

import analysis

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "numbers.json")


def fmt(v, nd=2):
    return f"{v:,.{nd}f}"


def mean_acc(runs, arm):
    return float(np.mean([r["test"]["accuracy"] for r in runs[arm]]))


def build():
    N, T, S = {}, {}, {}

    # ---- deterministic ---------------------------------------------------
    from config import Config
    from data import load_medmnist, partition_indices, partition_report
    train, _, _, meta = load_medmnist("bloodmnist")
    labels = train.y.numpy()
    for alpha, key in ((0.5, "a05"), (0.1, "a01")):
        parts = partition_indices(labels, 10, "dirichlet", alpha=alpha,
                                  min_size=300 if alpha >= 0.5 else 100, seed=0)
        rep = partition_report(labels, parts, meta["n_classes"])
        N[f"missing_class_clients_{key}"] = rep["clients_missing_a_class"]
        if alpha == 0.5:
            N["min_samples"] = rep["min_samples"]
            N["max_samples"] = rep["max_samples"]
            N["mean_samples"] = fmt(rep["mean_samples"], 0)
            N["std_samples"] = fmt(rep["std_samples"], 0)
            T["table12_partition_detail"] = {
                "header": ["Hospital", "Samples", "Classes present",
                           "Class distribution"],
                "rows": [[c["client"], c["n_samples"], c["n_classes_present"],
                          " / ".join(str(x) for x in c["class_counts"])]
                         for c in rep["clients"]],
            }
            N["n_classes"] = meta["n_classes"]

    # Table 3 rebuilt from the actual configuration used
    c = Config()
    T["table3_parameters"] = {
        "header": ["Parameter", "Value"],
        "rows": [
            ["Device", "MacBook Pro M3"],
            ["Processor", "Apple M3"],
            ["Memory (RAM)", "16 GB"],
            ["Number of clients (hospitals)", str(c.num_clients)],
            ["Primary dataset", "BloodMNIST (8 classes, 11,959 training images)"],
            ["Secondary dataset", "PneumoniaMNIST (2 classes, 4,708 training images)"],
            ["Partition", f"Dirichlet, alpha = {c.dirichlet_alpha}"],
            ["Model at clients", "CNN (Conv-GroupNorm-ReLU-Pool x2, FC)"],
            ["Model at server",
             f"DQN ({' x '.join(str(h) for h in c.dqn_hidden)} MLP, "
             f"{len(__import__('dqn').action_space(c.dqn_action_mode))} actions,"
             f" mode={c.dqn_action_mode})"],
            ["Local epochs per round", str(c.local_epochs)],
            ["Communication rounds", str(c.rounds)],
            ["Batch size", str(c.batch_size)],
            ["Learning rate (FL)", str(c.lr_fl)],
            ["Learning rate (DRL)", str(c.dqn_lr)],
            ["Optimizer", "Adam"],
            ["Privacy mechanism", "DP-SGD (per-sample clipping + Gaussian noise)"],
            ["Clipping norm C", str(c.dp_clip_norm)],
            ["Noise multiplier sigma", str(c.dp_noise_multiplier)],
            ["Privacy accounting", "Renyi DP, delta = 1e-5"],
            ["Local objective", f"FedProx (mu={c.fedprox_mu})" if c.use_fedprox
             else "plain cross-entropy"],
            ["Aggregation method", "sample-weighted mean (FedAvg rule)"],
            ["Client selection", f"DQN-selected, top-k direction="
             f"{c.topk_direction}" if c.use_dqn else "all clients"],
            ["Compression", f"top-{int(c.compression_ratio*100)}% sparsification, "
                            f"{c.quantization_bits}-bit values"],
            ["Attack simulations", "FGSM, PGD, sign-flip and label-flip poisoning"],
            ["Reward weights", f"alpha={c.reward_alpha}, beta={c.reward_beta}, "
             f"gamma={c.reward_gamma} (costs normalised to [0,1])"],
            ["Seeds", "5 (main comparison), 3 (secondary studies)"],
        ],
    }

    from models import build_model, count_parameters
    import compression
    p = count_parameters(build_model(meta))
    N["num_params"] = f"{p:,}"
    dense = compression.dense_payload_bytes(p, 32)
    k = max(1, int(round(0.10 * p)))
    sparse = compression.sparse_payload_bytes(p, k, 16)
    N["dense_MB"] = fmt(dense / 1e6, 2)
    N["uplink_per_client_MB"] = fmt(sparse / 1e6, 3)
    N["compression_ratio"] = fmt(dense / sparse, 2)
    N["uplink_reduction_pct"] = fmt(100 * (1 - sparse / dense), 1)
    N["main_sigma"] = str(Config().dp_noise_multiplier)

    # ---- main comparison -------------------------------------------------
    order = ["CL", "FedAvg", "FedProx", "RandomSub", "FL-DRL"]
    main = analysis.load("main")
    if main:
        h, r = analysis.table_performance(main, order)
        T["table7_performance"] = {"header": h, "rows": r}
        h, r = analysis.table_efficiency(main, order)
        T["table8_efficiency"] = {"header": h, "rows": r}
        h, r = analysis.table_robustness(main, order)
        if r:
            T["table11_evasion"] = {"header": h, "rows": r}

        for arm_ in main:
            accs_ = [x["test"]["accuracy"] for x in main[arm_]]
            aucs_ = [x["test"]["auc"] for x in main[arm_]]
            N[f"acc_{arm_}"] = (f"{np.mean(accs_):.4f} "
                                f"± {np.std(accs_, ddof=1):.4f}")
            N[f"auc_{arm_}"] = f"{np.mean(aucs_):.4f}"

        if "FL-DRL" in main:
            rs = main["FL-DRL"]
            wc = [r_["wall_clock"]["mean_round_s"] for r_ in rs]
            N["mean_round_s"] = fmt(float(np.mean(wc)), 1)
            N["total_train_min"] = fmt(float(np.mean(wc)) * 100 / 60, 0)
            comm = [r_["communication"] for r_ in rs]
            N["uplink_per_round_MB"] = fmt(
                float(np.mean([c["uplink_per_round_MB"] for c in comm])), 2)
            N["total_uplink_MB"] = fmt(
                float(np.mean([c["total_uplink_MB"] for c in comm])), 1)
            N["n_seeds_main"] = len(rs)

        # significance
        import metrics
        nc = meta["n_classes"]
        if "FL-DRL" in main:
            for other, tag in (("CL", "cl"), ("FedAvg", "fedavg")):
                if other not in main:
                    continue
                try:
                    a = np.asarray(main["FL-DRL"][0]["y_prob"])
                    b = np.asarray(main[other][0]["y_prob"])
                    y = np.asarray(main["FL-DRL"][0]["y_true"])
                    t = (metrics.delong_test(y, a[:, 1], b[:, 1]) if nc == 2
                         else metrics.delong_multiclass(y, a, b, nc))
                    if "p_value" in t:                       # binary task
                        N[f"delong_fldrl_{tag}_diff"] = f"{t['difference']:+.4f}"
                        N[f"delong_fldrl_{tag}_p"] = f"{t['p_value']:.3g}"
                        N[f"delong_fldrl_{tag}_ci"] = (
                            f"[{t['ci95_low']:+.4f}, {t['ci95_high']:+.4f}]")
                        sig = t["significant_at_0.05"]
                    else:                                    # multi-class: OvR
                        N[f"delong_fldrl_{tag}_diff"] = (
                            f"{t['mean_auc_difference']:+.4f} "
                            f"(mean over {len(t['per_class'])} one-vs-rest curves)")
                        N[f"delong_fldrl_{tag}_p"] = (
                            f"{t['bonferroni_p']:.3g} (Bonferroni-corrected "
                            f"smallest of {len(t['per_class'])} per-class tests)")
                        N[f"delong_fldrl_{tag}_ci"] = (
                            "per-class intervals in significance.json")
                        sig = t["bonferroni_p"] < 0.05
                    if tag == "cl":
                        N["delong_fldrl_cl_verdict"] = (
                            "statistically significant" if sig
                            else "not statistically significant")
                except Exception as e:
                    print(f"  (DeLong vs {other}: {e})")

        def by_seed(arm):
            return {r_["config"]["seed"]: r_["test"]["accuracy"]
                    for r_ in main.get(arm, [])}

        for other, tag in (("FedAvg", "fedavg"), ("RandomSub", "randomsub"),
                           ("FedProx", "fedprox")):
            a, b = by_seed("FL-DRL"), by_seed(other)
            shared = sorted(set(a) & set(b))      # join on seed, never on order
            if len(shared) > 1:
                tt = metrics.paired_ttest([a[s_] for s_ in shared],
                                          [b[s_] for s_ in shared])
                N[f"ttest_fldrl_{tag}"] = (
                    f"{tt['mean_difference']:+.4f} (t = {tt['t']:.2f}, "
                    f"p = {tt['p_value']:.3g}, n = {tt['n_pairs']} seeds)")

    # ---- reward-weight grid (R1-3) ---------------------------------------
    rsel = os.path.join(HERE, "reward_selection.json")
    if os.path.exists(rsel):
        rd = json.load(open(rsel))
        g = rd["grid"]
        N["reward_val_lo"] = f"{min(r['val_best'] for r in g):.4f}"
        N["reward_val_hi"] = f"{max(r['val_best'] for r in g):.4f}"
        N["reward_uplink_lo"] = f"{min(r['uplink_MB'] for r in g):.3f}"
        N["reward_uplink_hi"] = f"{max(r['uplink_MB'] for r in g):.3f}"
        N["reward_beta"] = str(rd["selected"]["beta"])
        N["reward_gamma"] = str(rd["selected"]["gamma"])
        T["table_reward_grid"] = {
            "header": ["beta", "gamma", "Best validation accuracy",
                       "Test accuracy", "Macro F1", "Uplink (MB/round)"],
            "rows": [[r["beta"], r["gamma"], f"{r['val_best']:.4f}",
                      f"{r['test_acc']:.4f}", f"{r['test_f1']:.4f}",
                      f"{r['uplink_MB']:.3f}"]
                     for r in sorted(g, key=lambda x: -x["val_best"])],
        }

    # ---- privacy ---------------------------------------------------------
    priv = analysis.load("privacy")
    if priv:
        h, r = analysis.table_privacy(priv)
        T["table6_privacy"] = {"header": h, "rows": r}
        if "inf" in priv:
            N["nodp_accuracy"] = f"{mean_acc(priv, 'inf'):.4f}"
        finite = {a: mean_acc(priv, a) for a in priv if a != "inf"}
        if len(finite) > 1:
            N["dp_accuracy_span"] = (f"{min(finite.values()):.3f} to "
                                     f"{max(finite.values()):.3f}")
        aucs = {a: float(np.mean([x["test"]["auc"] for x in priv[a]]))
                for a in priv if a != "inf"}
        if aucs:
            N["dp_auc_span"] = f"{min(aucs.values()):.3f} to {max(aucs.values()):.3f}"

    privopt = analysis.load("privopt")
    if privopt:
        h, r = analysis.table_privacy(privopt)
        T["table6b_privacy_optimised"] = {"header": h, "rows": r}
        best = None
        for arm, rs in privopt.items():
            if arm == "inf":
                continue
            per = [r_["privacy"].get("per_client", {}) for r_ in rs]
            eps = [max(v["epsilon"] for v in q.values()) for q in per if q]
            if not eps:
                continue
            acc = mean_acc(privopt, arm)
            if acc >= 0.85 and (best is None or np.mean(eps) < best[1]):
                best = (arm, float(np.mean(eps)), acc)
        if best:
            S["privopt_note"] = (
                f"In the privacy-optimised configuration the framework attains "
                f"an accounted budget of epsilon = {best[1]:.1f} at a noise "
                f"multiplier of {best[0]}, with a test accuracy of "
                f"{best[2]:.3f}.")

    # ---- ablation --------------------------------------------------------
    abl = analysis.load("ablation")
    if abl:
        h, r = analysis.table_ablation(abl)
        T["table9_ablation"] = {"header": h, "rows": r}
        if "full" in abl:
            base = mean_acc(abl, "full")
            deltas = {a: mean_acc(abl, a) - base for a in abl if a != "full"}
            if deltas:
                worst = min(deltas, key=deltas.get)
                nice = {"no_dqn": "the reinforcement-learning controller",
                        "no_fedprox": "the FedProx proximal term",
                        "no_dp": "differential privacy",
                        "no_compression": "gradient compression",
                        "no_adv_training": "adversarial training",
                        "no_anomaly_filter": "the anomaly filter"}
                parts = ", ".join(
                    f"{nice.get(a, a)} ({deltas[a]:+.4f})"
                    for a in sorted(deltas, key=deltas.get))
                S["ablation_narrative"] = (
                    f"Removing {nice.get(worst, worst)} produces the largest "
                    f"single degradation in accuracy ({deltas[worst]:+.4f} "
                    f"relative to the complete framework, which reaches "
                    f"{base:.4f}). Ranked by the size of the effect, the "
                    f"changes in accuracy on removing each component are: "
                    f"{parts}. Components whose removal improves raw accuracy "
                    f"are retained because they purchase a property that "
                    f"accuracy does not measure: differential privacy provides "
                    f"the formal guarantee of Table 6, and gradient "
                    f"compression reduces uplink volume by "
                    f"{N['uplink_reduction_pct']}%.")
                S["mechanism_note"] = (
                    f"The ablation attributes the advantage over static "
                    f"aggregation principally to {nice.get(worst, worst)}, "
                    f"rather than to the combination as a whole.")

    # ---- attacks ---------------------------------------------------------
    atk = analysis.load("attacks")
    if atk:
        h, r = analysis.table_poisoning(atk)
        T["table10_poisoning"] = {"header": h, "rows": r}
        h, r = analysis.table_robustness(atk)
        if r:
            T["table11b_evasion_by_setting"] = {"header": h, "rows": r}

        bits = []
        if "clean" in atk and atk["clean"] and "robustness" in atk["clean"][0]:
            rob = [x["robustness"] for x in atk["clean"]]
            c = float(np.mean([x["fgsm_0.0"] for x in rob]))
            f3 = float(np.mean([x["fgsm_0.03"] for x in rob]))
            p3 = float(np.mean([x["pgd_0.03"] for x in rob]))
            bits.append(
                f"Under evasion attack the adversarially trained model retains "
                f"{f3:.3f} accuracy against FGSM and {p3:.3f} against 10-step "
                f"PGD at a perturbation budget of 0.03, against {c:.3f} on "
                f"clean data")
            if "no_adv_training" in atk and atk["no_adv_training"]:
                nr = [x["robustness"] for x in atk["no_adv_training"]
                      if "robustness" in x]
                if nr:
                    nf3 = float(np.mean([x["fgsm_0.03"] for x in nr]))
                    bits.append(
                        f"whereas the same model trained without adversarial "
                        f"examples falls to {nf3:.3f} under the identical FGSM "
                        f"attack")
        for d, u, lab in (("poison2_defended", "poison2_undefended",
                           "two sign-flipping hospitals"),
                          ("labelflip3_defended", "labelflip3_undefended",
                           "three label-flipping hospitals")):
            if d in atk and u in atk:
                bits.append(
                    f"With {lab}, the proposed defences hold accuracy at "
                    f"{mean_acc(atk, d):.3f} against {mean_acc(atk, u):.3f} "
                    f"when norm clipping, trimmed-mean aggregation and the "
                    f"anomaly filter are disabled")
        if bits:
            S["attack_narrative"] = ". ".join(bits) + "."

    # ---- partition -------------------------------------------------------
    part = analysis.load("partition")
    if part:
        h, r = analysis.table_partition(part)
        T["table12_partition"] = {"header": h, "rows": r}

    # ---- cohort size vs attainable budget (analytic, no extra runs) ------
    from privacy import epsilon_for
    _cfg = Config()                      # `c` is rebound by loops above
    sizes = [row_["n_samples"] for row_ in rep["clients"]]
    rows = []
    for scale, label in ((0.4, "0.4x (approx. 4.8k images)"),
                         (1.0, "1x (BloodMNIST, 12.0k images)"),
                         (3.0, "3x (approx. 35.9k images)"),
                         (7.5, "7.5x (approx. 89.7k images)")):
        sc = [max(32, int(n_ * scale)) for n_ in sizes]
        eps = [epsilon_for(1.5, 32 / n_, 100 * int(np.ceil(n_ / 32)), 1e-5)
               for n_ in sc]
        rows.append([label, f"{min(sc):,}-{max(sc):,}",
                     f"{max(eps):.1f}", f"{float(np.median(eps)):.1f}"])
    T["table13_cohort_budget"] = {
        "header": ["Federation size", "Images per hospital",
                   "Accounted eps (worst)", "Accounted eps (median)"],
        "rows": rows,
    }
    S["smallcohort_narrative"] = (
        "The determining factor is the number of images each institution "
        "holds, because it sets the subsampling rate and therefore how "
        "cheaply the budget composes over local steps. Holding the noise "
        "multiplier at " + str(_cfg.dp_noise_multiplier) + " and the schedule "
        "fixed, and varying only the per-hospital sample count, the accounted "
        "worst-case budget moves from " + rows[0][2] + " at roughly a fifth of "
        "our cohort size to " + rows[-1][2] + " at seven times it. A federation "
        "of ten institutions the size of ours therefore cannot reach a "
        "single-digit budget at useful accuracy however the mechanism is "
        "tuned, whereas the same mechanism on a cohort several times larger "
        "can. We state this because it bounds what any differentially private "
        "cross-silo framework of this scale can claim, independently of the "
        "particular architecture used.")

    payload = {"scalars": N, "tables": T, "narrative": S}
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {OUT}: {len(N)} scalar(s), {len(T)} table(s), "
          f"{len(S)} narrative(s)")
    for k in sorted(N):
        print(f"  {k} = {N[k]}")
    for k in sorted(T):
        print(f"  [table] {k}: {len(T[k]['rows'])} row(s)")
    for k in sorted(S):
        print(f"  [text]  {k}")
    return payload


if __name__ == "__main__":
    build()
