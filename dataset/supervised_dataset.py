import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


def get_labeled_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_labeled_transform_RA():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandAugment(num_ops=2, magnitude=10),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def split_train_val_indices(labels, val_ratio=0.1, seed=42, num_classes=10):
    rng = np.random.RandomState(seed)
    labels = np.array(labels)

    train_idx, val_idx = [], []

    for c in range(num_classes):
        cls_idx = np.where(labels == c)[0]
        rng.shuffle(cls_idx)

        n_val = int(len(cls_idx) * val_ratio)
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return np.array(train_idx), np.array(val_idx)


def get_labeled_subset_indices(labels, train_idx, labeled_ratio, seed=42, num_classes=10):
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    train_idx = np.array(train_idx)

    selected = []

    for c in range(num_classes):
        cls_idx = train_idx[labels[train_idx] == c]
        rng.shuffle(cls_idx)

        n_select = int(len(cls_idx) * labeled_ratio)
        if labeled_ratio > 0:
            n_select = max(n_select, 1)

        selected.extend(cls_idx[:n_select].tolist())

    rng.shuffle(selected)
    return np.array(selected)


class LabeledDataset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.data = base_dataset.data[indices]
        self.targets = np.array(base_dataset.targets)[indices].tolist()
        self.transform = transform

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        target = self.targets[index]
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


def build_supervised_datasets(
    root="./data",
    labeled_ratio=0.1,
    val_ratio=0.1,
    seed=42,
    download=True,
    use_randaugment=False,
):
    full_train = datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=None,
    )

    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=get_eval_transform(),
    )

    train_idx, val_idx = split_train_val_indices(
        labels=full_train.targets,
        val_ratio=val_ratio,
        seed=seed,
        num_classes=10,
    )

    labeled_train_idx = get_labeled_subset_indices(
        labels=full_train.targets,
        train_idx=train_idx,
        labeled_ratio=labeled_ratio,
        seed=seed,
        num_classes=10,
    )

    train_transform = (
        get_labeled_transform_RA() if use_randaugment else get_labeled_transform()
    )

    train_dataset = LabeledDataset(
        base_dataset=full_train,
        indices=labeled_train_idx,
        transform=train_transform,
    )

    val_dataset = LabeledDataset(
        base_dataset=full_train,
        indices=val_idx,
        transform=get_eval_transform(),
    )

    return train_dataset, val_dataset, test_dataset
