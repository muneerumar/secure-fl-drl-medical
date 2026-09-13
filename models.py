"""Networks used by the framework.

MedCNN   -- client-side classifier, Sec. 5.3 (Conv -> BN -> ReLU -> Pool,
            twice, then Flatten -> FC -> FC -> output).
QNetwork -- server-side DQN, Sec. 6.2.
UpdateAutoEncoder -- anomaly detector over client updates, Sec. 6.5.3.
"""
import torch
import torch.nn as nn


class MedCNN(nn.Module):
    """Compact CNN for 28x28 MedMNIST images."""

    def __init__(self, n_channels: int = 1, n_classes: int = 2, width: int = 32,
                 groups: int = 8):
        super().__init__()
        # GroupNorm rather than BatchNorm: per-sample gradients (DP-SGD) and
        # federated averaging both break on BatchNorm's batch statistics.
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, width, 3, padding=1, bias=False),
            nn.GroupNorm(groups, width),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 28 -> 14

            nn.Conv2d(width, width * 2, 3, padding=1, bias=False),
            nn.GroupNorm(groups, width * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class QNetwork(nn.Module):
    """MLP Q-function over the FL state (Sec. 6.1.1) and action set (6.1.2)."""

    def __init__(self, state_dim: int, n_actions: int, hidden=(128, 128)):
        super().__init__()
        layers, d = [], state_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        return self.net(s)


class UpdateAutoEncoder(nn.Module):
    """Small AE over a fixed-length signature of a client update.

    Sec. 6.5.3: updates whose reconstruction error is anomalous relative to the
    running distribution of benign updates are rejected before aggregation.
    """

    def __init__(self, dim: int, hidden: int = 32, latent: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def build_model(meta, width: int = 32) -> MedCNN:
    return MedCNN(n_channels=meta["n_channels"],
                  n_classes=meta["n_classes"], width=width)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_bytes(model: nn.Module, bits: int = 32) -> int:
    return int(count_parameters(model) * bits / 8)
