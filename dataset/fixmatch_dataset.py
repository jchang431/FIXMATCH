import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)

import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)

# Transform # code credit: https://github.com/google-research/fixmatch for Augmentation Descriptioon

class TransformFixMatch:
    def __init__(self, mean=CIFAR10_MEAN, std=CIFAR10_STD):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        ])

        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandAugment(num_ops=2, magnitude=10),
        ])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.cutout = transforms.RandomErasing(
            p=1.0,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value="random",
        )

    def __call__(self, img):
        weak = self.weak(img)
        strong = self.strong(img)

        weak = self.normalize(weak)
        strong = self.normalize(strong)
        strong = self.cutout(strong)

        return weak, strong


def get_labeled_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


# split train and val, 10%, 25%, 50%, 100% labeled 

def split_train_val_indices(num_samples, val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    num_val = int(num_samples * val_ratio)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    return train_indices, val_indices


def split_labeled_unlabeled(labels, labeled_ratio, num_classes=10, seed=42):
    labels = np.array(labels)
    rng = np.random.RandomState(seed)

    labeled_idx = []
    unlabeled_idx = []

    for c in range(num_classes):
        cls_idx = np.where(labels == c)[0]
        rng.shuffle(cls_idx)

        num_labeled = int(len(cls_idx) * labeled_ratio)
        if labeled_ratio > 0:
            num_labeled = max(num_labeled, 1)

        labeled_idx.extend(cls_idx[:num_labeled])
        unlabeled_idx.extend(cls_idx[num_labeled:])

    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)

    rng.shuffle(labeled_idx)
    rng.shuffle(unlabeled_idx)

    # 100% labeled case, maintain the FixMatch Pipeline
    if len(unlabeled_idx) == 0:
        unlabeled_idx = labeled_idx.copy()

    return labeled_idx, unlabeled_idx


# dataset

class LabeledDataset(Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.data = base_dataset.data[indices]
        self.targets = np.array(base_dataset.targets)[indices].tolist()
        self.transform = transform if transform is not None else get_labeled_transform()

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        target = self.targets[index]

        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


class UnlabeledDataset(Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.data = base_dataset.data[indices]
        self.transform = transform if transform is not None else TransformFixMatch()

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        weak, strong = self.transform(img)
        return weak, strong

    def __len__(self):
        return len(self.data)


def build_fixmatch_datasets(
    root="./data",
    labeled_ratio=0.1,
    val_ratio=0.1,
    seed=42,
    download=True,
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

    train_indices, val_indices = split_train_val_indices(
        num_samples=len(full_train),
        val_ratio=val_ratio,
        seed=seed,
    )

    train_labels = np.array(full_train.targets)[train_indices]

    labeled_local_idx, unlabeled_local_idx = split_labeled_unlabeled(
        labels=train_labels,
        labeled_ratio=labeled_ratio,
        num_classes=10,
        seed=seed,
    )

    labeled_indices = train_indices[labeled_local_idx]
    unlabeled_indices = train_indices[unlabeled_local_idx]

    train_labeled_dataset = LabeledDataset(
        base_dataset=full_train,
        indices=labeled_indices,
    )

    train_unlabeled_dataset = UnlabeledDataset(
        base_dataset=full_train,
        indices=unlabeled_indices,
    )

    val_dataset = LabeledDataset(
        base_dataset=full_train,
        indices=val_indices,
        transform=get_eval_transform(),
    )

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset

