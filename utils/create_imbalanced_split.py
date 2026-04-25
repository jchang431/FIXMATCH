import os
import numpy as np
from torchvision import datasets


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def _split_train_val(targets, val_ratio, seed):
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in range(10):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n_val = int(len(idx) * val_ratio)
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.array(train_idx), np.array(val_idx)


def make_lt_by_difficulty(
    root="./data",
    out_path="splits/cifar10_imb_difficulty_seed42.npz",
    val_ratio=0.1,
    seed=42,
):
    """harder classes (cat, dog, bird) get more labeled samples."""
    rng = np.random.RandomState(seed)
    dataset = datasets.CIFAR10(root=root, train=True, download=False, transform=None)
    targets = np.array(dataset.targets)

    train_idx, val_idx = _split_train_val(targets, val_ratio, seed)
    train_targets = targets[train_idx]

    # ranking from supervised 10% per-class accuracy (jychang's analysis)
    counts = {
        3: 700,  # cat
        5: 700,  # dog
        2: 700,  # bird
        4: 400,  # deer
        7: 400,  # horse
        0: 400,  # airplane
        6: 400,  # frog
        8: 267,  # ship
        9: 267,  # truck
        1: 266,  # automobile
    }

    labeled, unlabeled = [], []
    for c in range(10):
        local = np.where(train_targets == c)[0]
        rng.shuffle(local)
        n = counts[c]
        labeled.extend(train_idx[local[:n]])
        unlabeled.extend(train_idx[local[n:]])

    labeled = np.array(labeled)
    unlabeled = np.array(unlabeled)
    rng.shuffle(labeled)
    rng.shuffle(unlabeled)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, labeled=labeled, unlabeled=unlabeled, val=val_idx)

    print(f"Saved: {out_path}")
    print(f"Labeled: {len(labeled)}, Unlabeled: {len(unlabeled)}, Val: {len(val_idx)}")
    for c in range(10):
        print(f"  class {c} ({CLASS_NAMES[c]}): {counts[c]}")
    print()


def make_lt_standard(
    root="./data",
    out_path="splits/cifar10_imb_lt_if10_seed42.npz",
    val_ratio=0.1,
    seed=42,
    imb_factor=10,
    head_count=1100,
):
    """standard CIFAR-10-LT exponential decay."""
    rng = np.random.RandomState(seed)
    dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
    targets = np.array(dataset.targets)

    train_idx, val_idx = _split_train_val(targets, val_ratio, seed)
    train_targets = targets[train_idx]

    counts = [int(head_count * (imb_factor ** (-c / 9))) for c in range(10)]

    labeled, unlabeled = [], []
    for c in range(10):
        local = np.where(train_targets == c)[0]
        rng.shuffle(local)
        n = min(counts[c], len(local))
        labeled.extend(train_idx[local[:n]])
        unlabeled.extend(train_idx[local[n:]])

    labeled = np.array(labeled)
    unlabeled = np.array(unlabeled)
    rng.shuffle(labeled)
    rng.shuffle(unlabeled)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, labeled=labeled, unlabeled=unlabeled, val=val_idx)

    print(f"Saved: {out_path}")
    print(f"Labeled: {len(labeled)}, Unlabeled: {len(unlabeled)}, Val: {len(val_idx)}")
    for c in range(10):
        print(f"  class {c} ({CLASS_NAMES[c]}): {counts[c]}")
    print()

def make_lt_by_difficulty_1pct(
    root="./data",
    out_path="splits/cifar10_imb_difficulty_1pct_seed42.npz",
    val_ratio=0.1,
    seed=42,
):
    rng = np.random.RandomState(seed)
    dataset = datasets.CIFAR10(root=root, train=True, download=False, transform=None)
    targets = np.array(dataset.targets)

    train_idx, val_idx = _split_train_val(targets, val_ratio, seed)
    train_targets = targets[train_idx]

    counts = {
        3: 70,  # cat
        5: 70,  # dog
        2: 70,  # bird
        4: 40,  # deer
        7: 40,  # horse
        0: 40,  # airplane
        6: 40,  # frog
        8: 27,  # ship
        9: 27,  # truck
        1: 26,  # automobile
    }

    labeled, unlabeled = [], []
    for c in range(10):
        local = np.where(train_targets == c)[0]
        rng.shuffle(local)
        n = min(counts[c], len(local))
        labeled.extend(train_idx[local[:n]])
        unlabeled.extend(train_idx[local[n:]])

    labeled = np.array(labeled)
    unlabeled = np.array(unlabeled)
    rng.shuffle(labeled)
    rng.shuffle(unlabeled)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, labeled=labeled, unlabeled=unlabeled, val=val_idx)

    print(f"Saved: {out_path}")
    print(f"Labeled: {len(labeled)}, Unlabeled: {len(unlabeled)}, Val: {len(val_idx)}")
    for c in range(10):
        print(f"  class {c} ({CLASS_NAMES[c]}): {counts[c]}")
    print()

if __name__ == "__main__":
    # 10% imbalance splits
    make_lt_by_difficulty()
    make_lt_standard()

    # 1% imbalance splits
    make_lt_by_difficulty_1pct()
    make_lt_standard(
        out_path="splits/cifar10_imb_lt_if10_1pct_seed42.npz",
        head_count=110,
    )