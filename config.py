"""Configuration for the FL-DRL simulation.

Values follow Table 3 (List of Simulation Parameters) of the manuscript.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Config:
    # ---- dataset / federation -------------------------------------------
    dataset: str = "bloodmnist"          # MedMNIST subset (primary)
    num_clients: int = 10                # Table 3: 10-20 clients
    partition: str = "dirichlet"         # "iid" | "dirichlet" | "shard"
    dirichlet_alpha: float = 0.5         # lower = more non-IID
    min_samples_per_client: int = 300

    # ---- local training (Table 3) ---------------------------------------
    local_epochs: int = 5
    batch_size: int = 32
    lr_fl: float = 1e-3                  # client Adam learning rate
    optimizer: str = "adam"              # "adam" | "sgd" (ablation R1-4b)
    weight_decay: float = 0.0
    lr_decay_lambda: float = 0.01        # Eq. 13: eta_t = eta_0 / (1 + lambda*t)

    # ---- federation ------------------------------------------------------
    rounds: int = 100                    # Table 3: 100 communication rounds
    aggregation: str = "fedavg"          # "fedavg" | "fedprox" | "trimmed_mean"
    fedprox_mu: float = 0.01             # Eq. 12 proximal coefficient
    use_fedprox: bool = True
    client_fraction: float = 1.0         # fraction sampled when DQN not selecting
    # Uniform-random client selection baseline: when set (and the DQN is off),
    # exactly this many hospitals are sampled uniformly at random each round.
    # Used to test whether the learned policy beats random subsampling at a
    # matched participation rate.
    random_selection_k: Optional[int] = None

    # ---- differential privacy (Eq. 14, Alg. 1) ---------------------------
    # Disabled for the main comparison so that all four methods are
    # compared on equal footing; the privacy experiments enable it and
    # report the accounted budget (Sec. 7.2).
    use_dp: bool = False
    # "example" = per-sample gradient clipping + noise inside local training
    #             (DP-SGD; what Algorithm 1 literally describes, and what gains
    #             privacy amplification by subsampling).
    # "client"  = clip and noise the whole model update (client-level DP).
    dp_level: str = "example"
    dp_clip_norm: float = 1.0            # C: L2 clipping bound
    # Operating point: sigma is fixed and the resulting epsilon is *reported*.
    # Solving for a small target epsilon instead is not viable at this step
    # count (see paper_tables.md) -- it forces sigma so high the model
    # collapses to the majority class.
    dp_target_epsilon: Optional[float] = None
    dp_noise_multiplier: Optional[float] = 1.5
    dp_delta: float = 1e-5
    dp_dynamic_noise: bool = False       # R1-2: fixed vs decaying noise scale

    # ---- gradient compression (Alg. 1) -----------------------------------
    use_compression: bool = True
    compression_ratio: float = 0.10      # top-k fraction of coordinates kept
    quantization_bits: int = 16          # payload element width after top-k

    # ---- DQN controller (Sec. 6.1-6.2) -----------------------------------
    use_dqn: bool = True
    dqn_lr: float = 1e-4                 # Table 3: DRL learning rate
    dqn_gamma: float = 0.90              # Eq. 18-19 discount factor
    dqn_hidden: List[int] = field(default_factory=lambda: [128, 128])
    dqn_buffer_size: int = 5000
    dqn_batch_size: int = 32
    dqn_target_update: int = 10          # rounds between target-net syncs
    dqn_eps_start: float = 1.0
    dqn_eps_end: float = 0.05
    dqn_eps_decay_rounds: int = 60
    dqn_warmup: int = 20                 # rounds of random policy before learning
    # "full" = 27 actions (Eq. 16); "selection_only" = 3 actions, matching the
    # action space to the ~100 MDP transitions a 100-round federation provides.
    dqn_action_mode: str = "selection_only"
    # Direction of the "top-k by accuracy" hospital-selection action. The
    # manuscript specifies adaptive participation but not which end of the
    # distribution to favour. "lowest_loss" keeps the hospitals the global
    # model already fits; "highest_loss" keeps the ones it fits worst.
    topk_direction: str = "highest_loss"
    # Eq. 17 reward weights: R = alpha*AImp - beta*TCost - gamma*CL
    reward_alpha: float = 1.0
    # Selected on held-out seed 100 by best validation accuracy over
    # beta,gamma in {0.01, 0.05, 0.2}^2 (see reward_selection.json).
    reward_beta: float = 0.01
    reward_gamma: float = 0.01

    # ---- adversarial training / robustness (Sec. 6.5) --------------------
    use_adv_training: bool = True
    adv_epsilon: float = 0.03            # FGSM/PGD perturbation budget
    adv_ratio: float = 0.5               # fraction of each batch made adversarial
    use_norm_clipping: bool = True
    norm_clip_tau: float = 5.0           # Eq. 25 threshold
    use_autoencoder_filter: bool = True  # Sec. 6.5.3 anomaly detector
    ae_pretrain_rounds: int = 5          # R1-4d: warm-up on clean updates
    ae_reject_sigma: float = 3.0

    # ---- poisoning simulation (R2-8) -------------------------------------
    num_malicious: int = 0
    poison_type: str = "sign_flip"       # "sign_flip" | "label_flip" | "gaussian"
    poison_scale: float = 5.0

    # ---- runtime ---------------------------------------------------------
    seed: int = 0
    device: str = "auto"                 # "auto" | "cpu" | "mps" | "cuda"
    eval_every: int = 1
    num_workers: int = 0
    tag: str = "run"

    def to_dict(self):
        return asdict(self)
