from models.supervised import SupervisedModel
from utils.train_utils import Trainer
from utils.data_utils import AverageMeter

import time
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class SupervisedTrainer(Trainer):
    def __init__(self, config, checkpoint_dir=None, device=None):
        super().__init__(config, output_dir=checkpoint_dir, device=device)

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ])

        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ])

        if self.dataset.lower() != "cifar":
            raise ValueError("unsupported dataset, use cifar")

        full_train_for_indices = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=None,
        )

        full_train_train_tf = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )

        full_train_eval_tf = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=eval_transform,
        )

        self.testset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=eval_transform,
        )

        train_idx, val_idx = self._split_train_val(full_train_for_indices.targets, val_ratio=0.1)

        label_pct = self.config.data.label_pct
        labeled_train_idx = self._get_labeled_subset_indices(
            np.array(full_train_for_indices.targets),
            train_idx,
            label_pct,
        )

        self.trainset = Subset(full_train_train_tf, labeled_train_idx)
        self.valset = Subset(full_train_eval_tf, val_idx)

        self.train_loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_loader = DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.test_loader = DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.net = self._init_model()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.criterion = torch.nn.CrossEntropyLoss()

    def _split_train_val(self, targets, val_ratio=0.1):
        rng = np.random.default_rng(42)
        targets = np.array(targets)

        train_idx, val_idx = [], []

        for c in range(10):
            cls_idx = np.where(targets == c)[0]
            rng.shuffle(cls_idx)

            n_val = int(len(cls_idx) * val_ratio)
            val_idx.extend(cls_idx[:n_val].tolist())
            train_idx.extend(cls_idx[n_val:].tolist())

        return train_idx, val_idx

    def _get_labeled_subset_indices(self, targets, train_idx, pct):
        rng = np.random.default_rng(42)
        train_idx = np.array(train_idx)
        indices = []

        per_class_total = {}
        for c in range(10):
            cls_idx = train_idx[targets[train_idx] == c]
            per_class_total[c] = len(cls_idx)

        for c in range(10):
            cls_idx = train_idx[targets[train_idx] == c]
            n_select = max(1, int(len(cls_idx) * pct))
            chosen = rng.choice(cls_idx, n_select, replace=False)
            indices.extend(chosen.tolist())

        print(f"Using {int(pct*100)}% labeled train data: {len(indices)} samples")
        return indices

    @staticmethod
    def _build_model(cfg):
        return SupervisedModel(cfg)

    def _init_model(self):
        return self._build_model(self.config).to(self.device)

    def _init_optimizer(self):
        return torch.optim.Adam(
            self.net.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )

    def _init_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.train.n_epochs,
        )

    def train(self):
        start = time.time()
        best_val_acc = -1.0
        best_path = f"{self.output_dir}/best_supervised_{self.dataset.lower()}.pth"

        for epoch in range(self.config.train.n_epochs):
            self.net.train()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()

            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.net(x)
                loss = self.criterion(logits, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                preds = torch.argmax(logits, dim=1)
                acc = (preds == y).float().mean().item()

                loss_meter.update(loss.item(), x.size(0))
                acc_meter.update(acc, x.size(0))

            self.scheduler.step()

            val_loss, val_acc = self.evaluate(self.val_loader)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(self.net, best_path)

            print(
                f"Epoch [{epoch+1}/{self.config.train.n_epochs}] "
                f"Train Loss {loss_meter.avg:.4f} "
                f"Train Acc {acc_meter.avg:.4f} "
                f"Val Loss {val_loss:.4f} "
                f"Val Acc {val_acc:.4f}"
            )

        print(f"Training completed in {(time.time() - start):.3f}s")

        best_model = self.load_model(best_path, map_location=self.device)
        best_model.to(self.device)
        self.net = best_model

        test_loss, test_acc = self.evaluate(self.test_loader)
        print(f"Final Test Loss {test_loss:.4f}  Final Test Acc {test_acc:.4f}")

        return test_loss, test_acc

    def evaluate(self, loader):
        self.net.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.net(x)
                loss = self.criterion(logits, y)

                preds = torch.argmax(logits, dim=1)
                acc = (preds == y).float().mean().item()

                loss_meter.update(loss.item(), x.size(0))
                acc_meter.update(acc, x.size(0))

        return loss_meter.avg, acc_meter.avg
