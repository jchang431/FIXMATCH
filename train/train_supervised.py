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

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ])

        if self.dataset.lower() == "cifar":
            full_trainset = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=train_transform,
            )
            self.testset = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=True,
                transform=test_transform,
            )
        else:
            raise ValueError("unsupported dataset, use cifar")

        label_pct = self.config.data.label_pct
        self.trainset = self._get_subset(full_trainset, label_pct)

        self.train_loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
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

    def _get_subset(self, dataset, pct):
        rng = np.random.default_rng(42)
        targets = np.array(dataset.targets)

        indices = []
        n_classes = 10
        total_n = len(dataset)
        per_class = max(1, int(total_n * pct / n_classes))

        for c in range(n_classes):
            cls_idx = np.where(targets == c)[0]
            chosen = rng.choice(cls_idx, per_class, replace=False)
            indices.extend(chosen.tolist())

        print(f"Using {int(pct*100)}% labeled data: {len(indices)} samples")
        return Subset(dataset, indices)

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
            val_loss, val_acc = self.evaluate()

            print(
                f"Epoch [{epoch+1}/{self.config.train.n_epochs}] "
                f"Train Loss {loss_meter.avg:.4f} "
                f"Train Acc {acc_meter.avg:.4f} "
                f"Val Loss {val_loss:.4f} "
                f"Val Acc {val_acc:.4f}"
            )

        print(f"Completed in {(time.time() - start):.3f}s")

    def evaluate(self):
        self.net.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.net(x)
                loss = self.criterion(logits, y)

                preds = torch.argmax(logits, dim=1)
                acc = (preds == y).float().mean().item()

                loss_meter.update(loss.item(), x.size(0))
                acc_meter.update(acc, x.size(0))

        return loss_meter.avg, acc_meter.avg
