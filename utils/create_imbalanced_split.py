import os
import numpy as np
from torchvision import datasets


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def split_train_val(targets, val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in range(10):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n_val = int(len(idx) * val_ratio)
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.array(train_idx), np.array(val_idx)


def make_split(root, out_path, counts, val_ratio=0.1, seed=42, download=False):
    rng = np.random.RandomState(seed)
    dataset = datasets.CIFAR10(root=root, train=True, download=download, transform=None)
    targets = np.array(dataset.targets)

    train_idx, val_idx = split_train_val(targets, val_ratio, seed)
    train_targets = targets[train_idx]

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
    np.savez(
        out_path,
        labeled=labeled, unlabeled=unlabeled, val=val_idx,
        labeled_idx=labeled, unlabeled_idx=unlabeled, val_idx=val_idx,
        class_counts=np.array([counts[c] for c in range(10)]),
    )

    print(f"Saved: {out_path}")
    print(f"  labeled={len(labeled)}, unlabeled={len(unlabeled)}, val={len(val_idx)}")
    for c in range(10):
        print(f"  class {c} ({CLASS_NAMES[c]}): {counts[c]}")
    print()


def make_lt_by_difficulty(root="./data", out_path="splits/cifar10_imb_difficulty_seed42.npz",
                          val_ratio=0.1, seed=42):
    # difficulty ranking from supervised 10% per-class accuracy
    counts = {
        0: 400,  # airplane
        1: 266,  # automobile (easy)
        2: 700,  # bird (hard)
        3: 700,  # cat (hard)
        4: 400,  # deer
        5: 700,  # dog (hard)
        6: 400,  # frog
        7: 400,  # horse
        8: 267,  # ship (easy)
        9: 267,  # truck (easy)
    }
    make_split(root, out_path, counts, val_ratio, seed)


def make_lt_by_difficulty_1pct(root="./data", out_path="splits/cifar10_imb_difficulty_1pct_seed42.npz",
                                val_ratio=0.1, seed=42):
    counts = {
        0: 40, 1: 26, 2: 70, 3: 70, 4: 40,
        5: 70, 6: 40, 7: 40, 8: 27, 9: 27,
    }
    make_split(root, out_path, counts, val_ratio, seed)


def make_catdog_focused_1pct(root="./data", out_path="splits/cifar10_imb_catdog_focus_1pct_seed42.npz",
                              val_ratio=0.1, seed=42):
    # cat/dog get more labels - they were the most confused pair at 1%
    counts = {
        0: 35, 1: 25, 2: 55, 3: 90, 4: 35,
        5: 90, 6: 35, 7: 35, 8: 25, 9: 25,
    }
    make_split(root, out_path, counts, val_ratio, seed)


def make_lt_standard(root="./data", out_path="splits/cifar10_imb_lt_if10_seed42.npz",
                     val_ratio=0.1, seed=42, imb_factor=10, head_count=1100):
    # CIFAR-10-LT: exponential decay by class index
    counts = {c: int(head_count * (imb_factor ** (-c / 9))) for c in range(10)}
    make_split(root, out_path, counts, val_ratio, seed)


if __name__ == "__main__":
    # 10% splits
    make_lt_by_difficulty()
    make_lt_standard(out_path="splits/cifar10_imb_lt_if10_seed42.npz", head_count=1100)

    # 1% splits
    make_lt_by_difficulty_1pct()
    make_lt_standard(out_path="splits/cifar10_imb_lt_if10_1pct_seed42.npz", head_count=110)

    # severe LT (IF=50)
    make_lt_standard(
        out_path="splits/cifar10_imb_lt_if50_seed42.npz",
        imb_factor=50, head_count=1600,
    )

    # 25% LT
    make_lt_standard(
        out_path="splits/cifar10_imb_lt_if10_25pct_seed42.npz",
        head_count=2750,
    )

    make_catdog_focused_1pct()