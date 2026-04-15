from models.supervised import SupervisedModel
from utils.train_utils import Trainer
from utils.data_utils import AverageMeter

import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SupervisedTrainer(Trainer):
    def __init__(self, config, checkpoint_dir=None, device=None):
        super().__init__(config, output_dir=checkpoint_dir, device=device)

        # overwrite SimCLR dataset with supervised dataset
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ])

        if self.dataset.lower() == "cifar":
            self.trainset = datasets.CIFAR10(
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

    @staticmethod
    def _build_model(cfg):
        return SupervisedModel(cfg)

    def _init_model(self):
        net = self._build_model(self.config).to(self.device)
        return net

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

            for i, (x, y) in enumerate(self.train_loader):
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
                f"Epoch: [{epoch+1}/{self.config.train.n_epochs}]\t"
                f"Train Loss {loss_meter.avg:.4f}\t"
                f"Train Acc {acc_meter.avg:.4f}\t"
                f"Val Loss {val_loss:.4f}\t"
                f"Val Acc {val_acc:.4f}"
            )

        print(f"Completed in {(time.time() - start):.3f}")
        self.save_model(
            self.net,
            f"{self.output_dir}/supervised_{self.dataset.lower()}.pth"
        )

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
