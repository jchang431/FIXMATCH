from models.supervised import SupervisedModel
from utils.data_utils import AverageMeter, set_seed, get_device
import time
import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class SupervisedTrainer:
    def __init__(self, config, checkpoint_dir=None, device=None):

        self.config = config
        self.num_workers = self.config.train.num_workers
        self.batch_size = self.config.train.batch_size
        self.lr = self.config.train.lr
        self.n_epochs = self.config.train.n_epochs
        self.dataset = self.config.data.dataset
    
        set_seed(seed=42)
    
        if device is None:
            self.device = get_device()
        else:
            self.device = device
    
        self.data_dir = "./data"
        os.makedirs(self.data_dir, exist_ok=True)
    
        if checkpoint_dir is None:
            self.output_dir = "./outputs/supervised"
        else:
            self.output_dir = checkpoint_dir
        os.makedirs(self.output_dir, exist_ok=True)


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

        # for indices only
        full_train_plain = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=None,
        )

        # train split with augmentation
        full_train_train_tf = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )

        # val split without augmentation
        full_train_eval_tf = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=eval_transform,
        )

        # held-out test set
        self.testset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=eval_transform,
        )

        train_idx, val_idx = self._split_train_val(
            full_train_plain.targets,
            val_ratio=0.1,
        )

        label_pct = self.config.data.label_pct
        labeled_train_idx = self._get_labeled_subset_indices(
            np.array(full_train_plain.targets),
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
        selected = []

        for c in range(10):
            cls_idx = train_idx[targets[train_idx] == c]
            n_select = max(1, int(len(cls_idx) * pct))
            chosen = rng.choice(cls_idx, n_select, replace=False)
            selected.extend(chosen.tolist())

        print(f"Using {int(pct * 100)}% labeled train data: {len(selected)} samples")
        return selected

    @staticmethod
    def _build_model(cfg):
        return SupervisedModel(cfg)

    def _init_model(self):
        return self._build_model(self.config).to(self.device)
    
    @staticmethod
    def save_model(model, model_path):
        model_s = torch.jit.script(model)
        model_s.save(model_path)
        print(f"Model saved to {model_path}")
    
    @staticmethod
    def load_model(model_path, map_location='cpu'):
        model = torch.jit.load(model_path, map_location=map_location)
        return model

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
        pct = self.config.data.label_pct

        print(f"\n{'=' * 45}")
        print(f"  {int(pct * 100)}% labeled")
        print(f"{'=' * 45}")

        best_val_acc = -1.0
        best_path = f"{self.output_dir}/best_supervised_{self.dataset.lower()}.pth"

        history = []
        start = time.time()

        for ep in range(1, self.config.train.n_epochs + 1):
            self.net.train()
            train_loss_meter = AverageMeter()

            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.net(x)
                loss = self.criterion(logits, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss_meter.update(loss.item(), x.size(0))

            self.scheduler.step()

            val_loss, val_acc = self.evaluate(self.val_loader)
            history.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(self.net, best_path)

            if ep % 10 == 0 or ep == self.config.train.n_epochs:
                elapsed = time.time() - start
                print(
                    f"  ep {ep:3d}/{self.config.train.n_epochs}  "
                    f"train_loss={train_loss_meter.avg:.3f}  "
                    f"val_loss={val_loss:.3f}  "
                    f"val_acc={val_acc:.4f}  "
                    f"({elapsed:.0f}s)"
                )

        best_model = self.load_model(best_path, map_location=self.device)
        best_model.to(self.device)
        self.net = best_model

        test_loss, test_acc = self.evaluate(self.test_loader)
        per_class = self.evaluate_per_class(self.test_loader)

        print(f"\n  ★ Final test loss: {test_loss:.4f}")
        print(f"  ★ Final test accuracy: {test_acc * 100:.2f}%")

        return history, per_class, test_acc, self.net

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

    def evaluate_per_class(self, loader):
        self.net.eval()

        correct = [0 for _ in range(10)]
        total = [0 for _ in range(10)]

        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.net(x)
                preds = torch.argmax(logits, dim=1)

                for label, pred in zip(y, preds):
                    label = label.item()
                    pred = pred.item()
                    total[label] += 1
                    if label == pred:
                        correct[label] += 1

        per_class = {}
        for i, cls_name in enumerate(class_names):
            if total[i] > 0:
                per_class[cls_name] = correct[i] / total[i]
            else:
                per_class[cls_name] = 0.0

        return per_class
