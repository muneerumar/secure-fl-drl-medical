"""Server-side DQN controller -- Sec. 6.1 and 6.2 of the manuscript.

State   (Eq. 15)  S_t = {theta_t, P_t, H_t, C_t, L_t}
Action  (Eq. 16)  A_t = (HS, AS, LAR)
Reward  (Eq. 17)  R_t = alpha*AImp - beta*TCost - gamma*CL
Target  (Eq. 19)  Q(S,A) = R + gamma * max_A' Q(S', A'; theta^-)
Loss    (Eq. 20)  L = E[(Q_target - Q(S,A))^2]

Trained with a replay buffer, a periodically synced target network and
epsilon-greedy exploration, exactly as listed in Sec. 6.2.
"""
import itertools
import random
from collections import deque, namedtuple
from typing import List

import numpy as np
import torch
import torch.nn as nn

from models import QNetwork

Transition = namedtuple("Transition", "s a r s2 done")

# ---- action space (Eq. 16) ------------------------------------------------
HOSPITAL_SELECTION = ["all", "topk_accuracy", "diverse"]        # HS
AGGREGATION_STRATEGY = ["fedavg", "fedprox", "trimmed_mean"]    # AS
LR_ADJUSTMENT = [0.5, 1.0, 2.0]                                 # LAR

ACTIONS: List[tuple] = list(itertools.product(
    HOSPITAL_SELECTION, AGGREGATION_STRATEGY, LR_ADJUSTMENT))
N_ACTIONS = len(ACTIONS)          # 3 * 3 * 3 = 27


def action_space(mode: str = "full") -> List[tuple]:
    """The controller's action set.

    "full"            -- the full 27-action product of Eq. 16.
    "selection_only"  -- hospital selection alone (3 actions), with the
                         aggregation rule fixed to FedProx and no learning-rate
                         scaling. With one MDP transition per communication
                         round, the number of available transitions equals the
                         number of rounds, so a 27-action space is far larger
                         than the data can identify; this variant matches the
                         action space to that budget.
    """
    if mode == "full":
        return list(ACTIONS)
    if mode == "selection_only":
        return [(hs, "fedprox", 1.0) for hs in HOSPITAL_SELECTION]
    raise ValueError(f"unknown action mode {mode!r}")


def describe_action(a: int, mode: str = "full") -> dict:
    hs, ag, lar = action_space(mode)[a]
    return {"hospital_selection": hs, "aggregation": ag, "lr_scale": lar}


# ---- state (Eq. 15) -------------------------------------------------------
STATE_FIELDS = [
    "global_acc",           # theta_t quality
    "global_acc_delta",     # H_t trend
    "global_loss",
    "mean_client_loss",     # H_t
    "std_client_loss",
    "mean_update_norm",     # theta_t drift
    "std_update_norm",
    "participation_rate",   # P_t
    "mean_compute",         # C_t
    "min_compute",
    "mean_latency",         # L_t
    "max_latency",
    "round_frac",           # t / T
    "label_skew",           # MedMNIST modality/prevalence heterogeneity (6.3)
]
STATE_DIM = len(STATE_FIELDS)


def build_state(**kw) -> np.ndarray:
    return np.array([float(kw.get(f, 0.0)) for f in STATE_FIELDS],
                    dtype=np.float32)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s = torch.tensor(np.stack([b.s for b in batch]), dtype=torch.float64)
        a = torch.tensor([b.a for b in batch], dtype=torch.int64)
        r = torch.tensor([b.r for b in batch], dtype=torch.float32)
        s2 = torch.tensor(np.stack([b.s2 for b in batch]), dtype=torch.float64)
        d = torch.tensor([float(b.done) for b in batch], dtype=torch.float32)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


class RunningNorm:
    """Online mean/variance for state whitening (Welford).

    The state of Eq. 15 mixes quantities whose natural scales differ by orders
    of magnitude (accuracy ~1e0, accuracy delta ~1e-3, update norms ~1e1). Fed
    raw into the Q-network, the large-scale features dominate the first layer
    and the informative small-scale ones are effectively invisible.
    """

    def __init__(self, dim: int, var_floor: float = 1e-6):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        # Welford's M2 accumulates sum of squared deviations and MUST start at
        # zero. Starting it at one injects a spurious unit variance that, for a
        # feature whose true s.d. is ~6e-3, inflates the denominator by ~17x and
        # leaves the "standardised" feature with s.d. ~0.06 instead of 1.
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.var_floor = var_floor
        self.frozen = False

    def update(self, x: np.ndarray):
        if self.frozen:
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def freeze(self):
        self.frozen = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.n < 2:
            return np.clip(x, -10.0, 10.0).astype(np.float32)
        var = np.maximum(self.m2 / (self.n - 1), self.var_floor)
        return np.clip((x - self.mean) / np.sqrt(var),
                       -10.0, 10.0).astype(np.float32)


class DQNAgent:
    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.mode = getattr(cfg, "dqn_action_mode", "full")
        self.actions = action_space(self.mode)
        self.n_actions = len(self.actions)
        self.q = QNetwork(STATE_DIM, self.n_actions,
                          tuple(cfg.dqn_hidden)).to(device)
        self.target = QNetwork(STATE_DIM, self.n_actions,
                               tuple(cfg.dqn_hidden)).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.dqn_lr)
        self.buffer = ReplayBuffer(cfg.dqn_buffer_size)
        self.norm = RunningNorm(STATE_DIM)
        self.steps = 0
        self.losses = []

    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.steps / max(c.dqn_eps_decay_rounds, 1))
        return c.dqn_eps_start + frac * (c.dqn_eps_end - c.dqn_eps_start)

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy:
            self.norm.update(np.asarray(state, dtype=np.float64))
        if not greedy and (self.steps < self.cfg.dqn_warmup
                           or random.random() < self.epsilon()):
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(self.norm(state), dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            return int(self.q(s).argmax(dim=1).item())

    def describe(self, a: int) -> dict:
        hs, ag, lar = self.actions[a]
        return {"hospital_selection": hs, "aggregation": ag, "lr_scale": lar}

    def observe(self, s, a, r, s2, done=False):
        # store RAW states. Normalising at insertion time freezes each
        # transition into whatever statistics existed then; those statistics
        # keep changing, so replayed transitions and live decisions would be
        # expressed in different coordinate systems.
        self.buffer.push(np.asarray(s, dtype=np.float64), a, r,
                         np.asarray(s2, dtype=np.float64), done)
        self.steps += 1

    def learn(self):
        c = self.cfg
        if len(self.buffer) < max(c.dqn_batch_size, c.dqn_warmup):
            return None
        s, a, r, s2, d = self.buffer.sample(c.dqn_batch_size)
        # apply the *current* statistics to both sides of the transition
        s = torch.tensor(np.stack([self.norm(x) for x in s.numpy()]),
                         dtype=torch.float32)
        s2 = torch.tensor(np.stack([self.norm(x) for x in s2.numpy()]),
                          dtype=torch.float32)
        s, a, r, s2, d = (t.to(self.device) for t in (s, a, r, s2, d))

        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(s2).max(dim=1).values
            target = r + c.dqn_gamma * q_next * (1.0 - d)          # Eq. 19
        loss = nn.functional.mse_loss(q_sa, target)                 # Eq. 20

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        if self.steps % c.dqn_target_update == 0:
            self.target.load_state_dict(self.q.state_dict())

        self.losses.append(float(loss.detach()))
        return float(loss.detach())


def reward(acc_now: float, acc_prev: float, train_cost: float,
           latency: float, cfg) -> float:
    """Eq. 17.

    train_cost and latency must arrive already normalised to [0, 1]; the caller
    is responsible for that. Passing raw units here makes the cost terms swamp
    the accuracy-improvement term, which reduces the policy to "minimise cost"
    and leaves it blind to accuracy.
    """
    a_imp = acc_now - acc_prev
    return (cfg.reward_alpha * a_imp
            - cfg.reward_beta * train_cost
            - cfg.reward_gamma * latency)


def architecture_summary(cfg) -> dict:
    """Everything reviewer R2-3 asks to see written down."""
    return {
        "state_dim": STATE_DIM,
        "state_fields": STATE_FIELDS,
        "action_mode": getattr(cfg, "dqn_action_mode", "full"),
        "n_actions": len(action_space(getattr(cfg, "dqn_action_mode", "full"))),
        "action_factors": {
            "hospital_selection": HOSPITAL_SELECTION,
            "aggregation_strategy": AGGREGATION_STRATEGY,
            "lr_adjustment": LR_ADJUSTMENT,
        },
        "hidden_layers": list(cfg.dqn_hidden),
        "activation": "ReLU",
        "optimizer": "Adam",
        "learning_rate": cfg.dqn_lr,
        "discount_gamma": cfg.dqn_gamma,
        "replay_buffer_size": cfg.dqn_buffer_size,
        "replay_batch_size": cfg.dqn_batch_size,
        "target_update_every_rounds": cfg.dqn_target_update,
        "epsilon_schedule": {
            "start": cfg.dqn_eps_start, "end": cfg.dqn_eps_end,
            "linear_decay_rounds": cfg.dqn_eps_decay_rounds,
            "warmup_random_rounds": cfg.dqn_warmup,
        },
        "reward_weights": {"alpha": cfg.reward_alpha,
                           "beta": cfg.reward_beta,
                           "gamma": cfg.reward_gamma},
        "loss": "MSE (Eq. 20)",
    }
