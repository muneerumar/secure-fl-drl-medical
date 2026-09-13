"""MedMNIST loading and federated partitioning.

Answers reviewer comments R2-2 (partition strategy, samples/client, class
distribution) and R1-1 (clients whose local set is missing whole classes).
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import medmnist
from medmnist import INFO

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medmnist_data")


class ArrayDataset(Dataset):
    """Tensor-backed dataset; keeps everything in RAM (28x28 is tiny)."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        # images: (N, H, W) or (N, H, W, C), uint8
        if images.ndim == 3:
            images = images[..., None]
        x = images.astype(np.float32) / 255.0
        x = np.transpose(x, (0, 3, 1, 2))          # NCHW
        self.x = torch.from_numpy(np.ascontiguousarray(x))
        # normalise to mean .5/std .5 like the standard MedMNIST recipe
        self.x = (self.x - 0.5) / 0.5
        self.y = torch.from_numpy(labels.astype(np.int64).reshape(-1))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def load_medmnist(name: str):
    """Return (train, val, test) ArrayDatasets plus metadata."""
    name = name.lower()
    if name not in INFO:
        raise ValueError(f"unknown MedMNIST subset {name!r}; options: {sorted(INFO)}")
    info = INFO[name]
    cls = getattr(medmnist, info["python_class"])
    os.makedirs(DATA_ROOT, exist_ok=True)
    splits = {}
    for split in ("train", "val", "test"):
        ds = cls(split=split, download=True, root=DATA_ROOT)
        splits[split] = ArrayDataset(ds.imgs, ds.labels)
    meta = {
        "name": name,
        "n_channels": info["n_channels"],
        "n_classes": len(info["label"]),
        "task": info["task"],
        "label_names": info["label"],
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
    }
    return splits["train"], splits["val"], splits["test"], meta


# ---------------------------------------------------------------------------
# partitioning
# ---------------------------------------------------------------------------
def partition_indices(labels: np.ndarray, num_clients: int, scheme: str,
                      alpha: float = 0.5, min_size: int = 20,
                      seed: int = 0) -> list:
    """Split sample indices across clients.

    iid        -- uniform random split (homogeneous hospitals)
    dirichlet  -- per-class Dirichlet(alpha) split; the standard non-IID
                  benchmark. Small alpha => hospitals specialise, and some
                  hold zero examples of some classes (see R1-1).
    shard      -- sort by label, cut into 2*num_clients shards, 2 per client
                  (McMahan et al. pathological non-IID split).
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    classes = np.unique(labels)

    if scheme == "iid":
        perm = rng.permutation(n)
        return [np.sort(p) for p in np.array_split(perm, num_clients)]

    if scheme == "shard":
        order = np.argsort(labels, kind="stable")
        shards = np.array_split(order, num_clients * 2)
        shard_ids = rng.permutation(len(shards))
        out = []
        for c in range(num_clients):
            picked = np.concatenate([shards[shard_ids[2 * c]],
                                     shards[shard_ids[2 * c + 1]]])
            out.append(np.sort(picked))
        return out

    if scheme != "dirichlet":
        raise ValueError(f"unknown partition scheme {scheme!r}")

    # Dirichlet, retried until every client clears min_size
    for _ in range(200):
        buckets = [[] for _ in range(num_clients)]
        for c in classes:
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            props = rng.dirichlet(np.repeat(alpha, num_clients))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for b, part in zip(buckets, np.split(idx, cuts)):
                b.extend(part.tolist())
        sizes = [len(b) for b in buckets]
        if min(sizes) >= min_size:
            return [np.sort(np.array(b, dtype=np.int64)) for b in buckets]
    raise RuntimeError(
        f"could not partition with alpha={alpha} and min_size={min_size}; "
        "raise alpha or lower min_samples_per_client")


def partition_report(labels: np.ndarray, parts: list, n_classes: int) -> dict:
    """Per-client sample counts and class histogram -> Table for R2-2."""
    rows = []
    for cid, idx in enumerate(parts):
        y = labels[idx]
        hist = np.bincount(y, minlength=n_classes).tolist()
        rows.append({
            "client": cid,
            "n_samples": int(len(idx)),
            "class_counts": hist,
            "n_classes_present": int((np.array(hist) > 0).sum()),
            "missing_classes": [c for c in range(n_classes) if hist[c] == 0],
        })
    sizes = np.array([r["n_samples"] for r in rows])
    return {
        "clients": rows,
        "total": int(sizes.sum()),
        "min_samples": int(sizes.min()),
        "max_samples": int(sizes.max()),
        "mean_samples": float(sizes.mean()),
        "std_samples": float(sizes.std()),
        "clients_missing_a_class": int(sum(1 for r in rows if r["missing_classes"])),
    }


def make_client_loaders(train_ds, parts, batch_size, seed=0, num_workers=0):
    loaders = []
    for cid, idx in enumerate(parts):
        sub = torch.utils.data.Subset(train_ds, idx.tolist())
        g = torch.Generator().manual_seed(seed * 1000 + cid)
        loaders.append(DataLoader(sub, batch_size=batch_size, shuffle=True,
                                  generator=g, num_workers=num_workers,
                                  drop_last=False))
    return loaders


def build_federation(cfg):
    """One call: datasets, partition, loaders, and the R2-2 report."""
    train, val, test, meta = load_medmnist(cfg.dataset)
    labels = train.y.numpy()
    parts = partition_indices(labels, cfg.num_clients, cfg.partition,
                              alpha=cfg.dirichlet_alpha,
                              min_size=cfg.min_samples_per_client,
                              seed=cfg.seed)
    report = partition_report(labels, parts, meta["n_classes"])
    loaders = make_client_loaders(train, parts, cfg.batch_size, seed=cfg.seed,
                                  num_workers=cfg.num_workers)
    val_loader = DataLoader(val, batch_size=256, shuffle=False)
    test_loader = DataLoader(test, batch_size=256, shuffle=False)
    return {
        "train": train, "val": val, "test": test, "meta": meta,
        "parts": parts, "report": report, "client_loaders": loaders,
        "val_loader": val_loader, "test_loader": test_loader,
    }


if __name__ == "__main__":
    from config import Config
    cfg = Config()
    fed = build_federation(cfg)
    print(json.dumps(fed["meta"], indent=2))
    r = fed["report"]
    print(f"clients={len(r['clients'])} total={r['total']} "
          f"min={r['min_samples']} max={r['max_samples']} "
          f"mean={r['mean_samples']:.1f}+-{r['std_samples']:.1f} "
          f"missing-a-class={r['clients_missing_a_class']}")
    for row in r["clients"]:
        print(f"  client {row['client']:2d}: n={row['n_samples']:5d} "
              f"classes={row['class_counts']}")
