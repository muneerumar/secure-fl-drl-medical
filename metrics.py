"""Classification metrics, ROC/AUC, and DeLong's test.

DeLong's test gives reviewer R1-8 the statistical significance of the AUC gap
between FL-DRL and the centralised baseline on the *same* test set (correlated
ROC curves), which is exactly the comparison that comment asks about.
"""
from typing import Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: np.ndarray, n_classes: int) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_prob = np.asarray(y_prob)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred,
                                                 average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred,
                                                    average="weighted",
                                                    zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted",
                                              zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(n_classes))).tolist(),
    }
    # cross-entropy of the reported probabilities
    p = np.clip(y_prob[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    out["loss"] = float(-np.mean(np.log(p)))

    try:
        if n_classes == 2:
            out["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            out["auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr",
                                             average="macro"))
    except ValueError:
        out["auc"] = float("nan")
    return out


def roc_points(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int):
    """FPR/TPR for Figure 5 (macro-averaged in the multi-class case)."""
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob)
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        return fpr, tpr
    grid = np.linspace(0, 1, 200)
    tprs = []
    for c in range(n_classes):
        if (y_true == c).sum() == 0:
            continue
        f, t, _ = roc_curve((y_true == c).astype(int), y_prob[:, c])
        tprs.append(np.interp(grid, f, t))
    return grid, np.mean(tprs, axis=0)


# ---------------------------------------------------------------------------
# DeLong's test for two correlated ROC curves
# ---------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    r = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and xs[j + 1] == xs[i]:
            j += 1
        r[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = r
    return out


def _fast_delong(scores: np.ndarray, n_pos: int):
    """Structural components of the AUC covariance (Sun & Xu 2014).

    `scores` is (k, n) with the n_pos positives first.
    """
    m, n = n_pos, scores.shape[1] - n_pos
    pos, neg = scores[:, :m], scores[:, m:]
    k = scores.shape[0]

    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(scores[r]) for r in range(k)])

    auc = (tz[:, :m].sum(axis=1) / (m * n)
           - (m + 1.0) / (2.0 * n))
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s01 = np.cov(v01)
    s10 = np.cov(v10)
    if k == 1:
        s01, s10 = np.array([[float(s01)]]), np.array([[float(s10)]])
    cov = s01 / m + s10 / n
    return auc, cov


def delong_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray
                ) -> dict:
    """Two-sided DeLong test that AUC_a == AUC_b on the same samples.

    y_true must be binary (0/1); prob_* are the positive-class scores.
    """
    y_true = np.asarray(y_true).reshape(-1)
    order = np.argsort(-y_true, kind="stable")     # positives first
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    if n_pos == 0 or n_pos == len(y_true):
        raise ValueError("need both classes present for DeLong's test")

    scores = np.vstack([np.asarray(prob_a).reshape(-1)[order],
                        np.asarray(prob_b).reshape(-1)[order]])
    auc, cov = _fast_delong(scores, n_pos)

    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = auc[0] - auc[1]
    if var <= 0:
        z, p = 0.0, 1.0
    else:
        z = diff / np.sqrt(var)
        p = 2.0 * stats.norm.sf(abs(z))
    ci = 1.96 * np.sqrt(var) if var > 0 else 0.0
    return {
        "auc_a": float(auc[0]), "auc_b": float(auc[1]),
        "difference": float(diff),
        "std_error": float(np.sqrt(var)) if var > 0 else 0.0,
        "z": float(z), "p_value": float(p),
        "ci95_low": float(diff - ci), "ci95_high": float(diff + ci),
        "significant_at_0.05": bool(p < 0.05),
    }


def delong_multiclass(y_true, prob_a, prob_b, n_classes: int) -> dict:
    """One-vs-rest DeLong per class, then a Bonferroni-corrected summary."""
    y_true = np.asarray(y_true).reshape(-1)
    per_class = {}
    for c in range(n_classes):
        yb = (y_true == c).astype(int)
        if yb.sum() in (0, len(yb)):
            continue
        per_class[c] = delong_test(yb, np.asarray(prob_a)[:, c],
                                   np.asarray(prob_b)[:, c])
    if not per_class:
        return {"per_class": {}, "min_p": None}
    ps = [v["p_value"] for v in per_class.values()]
    return {
        "per_class": per_class,
        "min_p": float(min(ps)),
        "bonferroni_p": float(min(1.0, min(ps) * len(ps))),
        "mean_auc_difference": float(np.mean([v["difference"]
                                              for v in per_class.values()])),
    }


# ---------------------------------------------------------------------------
# aggregation across seeds (R2-9)
# ---------------------------------------------------------------------------
def summarize_runs(values, confidence: float = 0.95) -> dict:
    """mean, std, and a t-based CI over independent seeds."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean, sd = float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        h = stats.t.ppf(0.5 + confidence / 2, n - 1) * sd / np.sqrt(n)
    else:
        h = 0.0
    return {"mean": mean, "std": sd, "n": n,
            "ci_low": mean - h, "ci_high": mean + h,
            "values": v.tolist()}


def paired_ttest(a, b) -> dict:
    """Paired t-test across seeds -- for 'is FL-DRL really better than FL?'."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_rel(a, b)
    d = a - b
    return {"mean_difference": float(d.mean()),
            "t": float(t), "p_value": float(p),
            "significant_at_0.05": bool(p < 0.05), "n_pairs": int(len(a))}
