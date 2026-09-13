# Secure Federated Deep Reinforcement Learning for Collaborative Medical AI

Reference implementation and complete experimental record for the paper
*Secure Federated Deep Reinforcement Learning for Collaborative Medical AI over
Heterogeneous Healthcare Data*.

The repository contains the simulator, the analysis code, and every result file
behind the tables and figures in the manuscript. All 97 runs in
`results/bloodmnist/` were produced by the source in this repository and carry a
hash of that source, so any result can be traced to the exact code that
generated it (see [Provenance](#provenance)).

## What is implemented

A federated learning simulation of ten hospitals training a shared CNN on a
non-IID partition of a MedMNIST benchmark, with four mechanisms layered on top:

| Component | Where | Summary |
|---|---|---|
| Federated averaging / FedProx | `server.py`, `client.py` | weighted aggregation; proximal term against client drift |
| Server-side DQN controller | `dqn.py`, `server.py` | selects the participating subset each round from a state of per-client loss, staleness, capacity and latency |
| Differential privacy | `privacy.py`, `client.py` | example-level DP-SGD via Opacus with Rényi DP accounting |
| Gradient compression | `compression.py` | top-k sparsification with exact byte accounting |
| Robustness | `attacks.py`, `server.py` | FGSM/PGD adversarial training; sign-flip and label-flip poisoning; trimmed-mean aggregation, norm clipping and an autoencoder update filter |

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
sh fetch_data.sh
```

`fetch_data.sh` places the MedMNIST archives in `./medmnist_data`. Training also
downloads them automatically on first use, but `make_numbers.py` needs them
present as well, because it recomputes the partition statistics from the raw
labels — so it is simplest to fetch them once up front.

Reproduce the full suite (97 runs):

```bash
.venv/bin/python run_all.py --dataset bloodmnist \
  --experiments main ablation attacks privacy partition \
  --rounds 100 --main-seeds 0 1 2 3 4 --other-seeds 0 1 2 \
  --workers 4 --threads 2
```

Runs are resumable: `run_all.py` skips any result whose file already exists and
whose source hash matches the current code. Because this repository ships the
completed results, the command above will report everything as already done. To
re-run from scratch, move `results/bloodmnist/` aside first.

Regenerate the derived numbers:

```bash
.venv/bin/python make_numbers.py     # writes results/numbers.json
.venv/bin/python count_valid.py      # verifies all 97 runs against the source hash
```

A single run:

```bash
.venv/bin/python run_one.py --experiment main --arm FL-DRL --seed 0 --rounds 100
```

## Repository layout

```
config.py         every hyperparameter, as one dataclass
data.py           MedMNIST loading and Dirichlet non-IID partition across hospitals
models.py         client CNN, server Q-network, update autoencoder
client.py         local training: FedProx, DP-SGD, adversarial training, compression
server.py         round loop: selection, aggregation, defences, reward, DQN transitions
dqn.py            state, action space, reward, replay buffer, target network, epsilon-greedy
privacy.py        DP clipping and noise; Renyi DP accounting
compression.py    top-k sparsification with byte accounting
attacks.py        FGSM, PGD, sign-flip and label-flip poisoning, trimmed mean, norm clipping
metrics.py        accuracy, macro P/R/F1, AUC, DeLong test, paired t-tests
run_one.py        one job; also defines the source hash
run_all.py        parallel driver over the job grid
experiments.py    the experiment definitions (one per reviewer request)
select_reward.py  reward-weight sensitivity sweep
analysis.py       tables and figures
make_numbers.py   collects every number quoted in the manuscript
count_valid.py    counts results matching the current source hash
results/          all completed runs, plus numbers.json and reward_selection.json
```

## Experiments and what they answer

| Directory | Runs | Arms |
|---|---|---|
| `results/bloodmnist/main/` | 25 | `CL`, `FedAvg`, `FedProx`, `FL-DRL`, `RandomSub` × 5 seeds |
| `results/bloodmnist/ablation/` | 21 | `full`, `no_dqn`, `no_fedprox`, `no_compression`, `no_adv_training`, `no_anomaly_filter`, `plus_dp` × 3 seeds |
| `results/bloodmnist/attacks/` | 21 | `clean`, sign-flip and label-flip poisoning with and without defences, `no_adv_training` × 3 seeds |
| `results/bloodmnist/privacy/` | 18 | DP-SGD noise multiplier σ ∈ {0.6, 0.8, 1.0, 1.5, 2.0} and a non-private arm (`inf`) × 3 seeds |
| `results/bloodmnist/partition/` | 12 | `iid` and Dirichlet α ∈ {0.1, 0.5, 1.0} × 3 seeds |
| `results/bloodmnist/reward/` | 9 | reward-weight grid β × γ ∈ {0.01, 0.05, 0.2}² on a held-out seed |

Each result JSON carries the full config, per-round history, test-set labels and
predicted probabilities (so ROC and DeLong tests can be recomputed without
re-training), communication byte counts, wall-clock timings, the privacy
accounting, and the source hash.

## Headline results

BloodMNIST, 8 classes, 11,959 training images, 10 hospitals, Dirichlet α = 0.5
with a floor of 300 images per hospital, 100 communication rounds, 5 local
epochs, batch 32, Adam at 1e-3, FedProx μ = 0.01, top-10% compression.
Differential privacy is disabled in the main comparison so all methods are
compared on equal footing, and is studied separately in `privacy/`.

Accuracy is mean ± SD over 5 seeds:

| Method | Accuracy | Macro AUC |
|---|---|---|
| Centralized | 0.9246 ± 0.0034 | 0.9933 |
| FedAvg | 0.8900 ± 0.0075 | 0.9862 |
| FedProx | 0.9103 ± 0.0023 | 0.9909 |
| Random subsampling | 0.9122 ± 0.0030 | 0.9910 |
| FL-DRL (proposed) | 0.9094 ± 0.0022 | 0.9909 |

Paired t-tests over the 5 shared seeds:

| Comparison | Difference | t | p |
|---|---|---|---|
| FL-DRL vs FedAvg | +0.0194 | 6.18 | 0.0035 |
| FL-DRL vs FedProx | −0.0009 | −0.78 | 0.478 |
| FL-DRL vs Random subsampling | −0.0028 | −1.81 | 0.145 |

DeLong tests on the ROC curves (Bonferroni-corrected over 8 one-vs-rest curves):
FL-DRL vs FedAvg +0.0034, p = 0.0095; FL-DRL vs centralized −0.0026,
p = 0.00035.

Communication: 422,056 parameters, 1.69 MB dense per client per round, reduced
to 0.185 MB by top-10% sparsification — a **89.1%** reduction in uplink volume,
1.23 MB per round across the federation and 123.3 MB over a full run.

Cost of differential privacy. The sweep is parameterised by the DP-SGD noise
multiplier σ; ε is then *computed* per hospital by the Opacus RDP accountant at
δ = 1e-5, rather than fixed in advance. Because each hospital holds a different
number of images it has a different sampling rate and therefore a different ε,
so the worst case across the ten hospitals is the one that binds:

| σ | worst-case ε | accuracy (seed 0) |
|---|---|---|
| none | ∞ | 0.8717 |
| 0.6 | 63.4 | 0.6823 |
| 0.8 | 30.7 | 0.6691 |
| 1.0 | 19.1 | 0.6597 |
| 1.5 | 9.5 | 0.6074 |
| 2.0 | 6.3 | 0.5735 |

Across all three seeds accuracy falls from 0.8739 without DP to a span of
0.583–0.706 (macro AUC 0.883–0.940). Single-digit ε is reachable only at
σ ≥ 1.5, and costs roughly 27 accuracy points at this dataset scale.

All of these figures are regenerated by `make_numbers.py` into
`results/numbers.json`; none are transcribed by hand.

## Provenance

`run_one.py` hashes the eleven files that can change a result — `config.py`,
`server.py`, `client.py`, `dqn.py`, `models.py`, `data.py`, `privacy.py`,
`compression.py`, `attacks.py`, `metrics.py`, `run_one.py` — and stores the
digest in every result file. A cached result whose hash does not match the
current source is rejected rather than silently reused, so results from
superseded code cannot leak into a table.

The source hash for this release is `35b330e1384a74ca`. Verify with:

```bash
.venv/bin/python count_valid.py
```

which should report 97/97.

The nine runs in `results/bloodmnist/reward/` predate the hashing mechanism and
carry a null hash; they are a sensitivity sweep and feed no headline number.

## Environment

Results were produced on an Apple M3 with 16 GB of unified memory, running on
CPU, with Python 3.13.13 and the versions pinned in `requirements.txt` (torch
2.14.0, numpy 2.5.2, scipy 1.18.1, scikit-learn 1.9.0, opacus 1.6.0, medmnist
3.0.2). Mean wall-clock time is 45.2 s per communication round; a full 100-round
run takes roughly 75 minutes.

Exact reproduction of floating-point results requires the same package versions.
The provenance hash covers the source, not the environment, so re-running under
different versions may shift the last digits.

## Data

MedMNIST v2 (Yang et al., *Scientific Data* 10, 41, 2023), BloodMNIST subset,
downloaded automatically from Zenodo on first use. The archives are not
committed; `fetch_data.sh` downloads them and prints checksums for verification.
MedMNIST is released under CC BY 4.0. No patient-level or identifiable data is
used or redistributed here.

## License

Released under the MIT License; see `LICENSE`.
