from models.supervised import SupervisedModel
from utils.data_utils import AverageMeter, set_seed, get_device
from dataset.supervised_dataset import build_supervised_datasets

import time
import os
import torch
from torch.utils.data import DataLoader


class SupervisedTrainer:
    def __init__(self, config, checkpoint_dir=None, device=None):
        self.config = config
        self.num_workers = self.config.train.num_workers
        self.batch_size = self.config.train.batch_size
        self.lr = self.config.train.lr
        self.n_epochs = self.config.train.n_epochs
        self.dataset = self.config.data.dataset

        set_seed(seed=getattr(self.config, "seed", 42))

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

        if self.dataset.lower() != "cifar":
            raise ValueError("unsupported dataset, use cifar")

        self.trainset, self.valset, self.testset = build_supervised_datasets(
            root=self.data_dir,
            labeled_ratio=self.config.data.label_pct,
            val_ratio=self.config.data.val_ratio,
            seed=getattr(self.config, "seed", 42),
            download=True,
            use_randaugment=getattr(self.config.data, "use_randaugment", False),
        )

        print(
            f"Using {int(self.config.data.label_pct * 100)}% labeled train data: "
            f"{len(self.trainset)} samples"
        )

        self.train_loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        self.net = self._init_model()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.criterion = torch.nn.CrossEntropyLoss()

    @staticmethod
    def _build_model(cfg):
        return SupervisedModel(cfg)

    def _init_model(self):
        return self._build_model(self.config).to(self.device)

    def save_model(self, model, model_path):
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path, map_location="cpu"):
        model = self._build_model(self.config).to(self.device)
        state_dict = torch.load(model_path, map_location=map_location)
        model.load_state_dict(state_dict)
        return model

    def _init_optimizer(self):
        
        cfg = self.config
        model = self.net
        opt_type = cfg.optimizer.type.lower()

        if opt_type == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=cfg.train.lr,
                momentum=cfg.optimizer.momentum,
                weight_decay=cfg.optimizer.weight_decay,
                nesterov=cfg.optimizer.nesterov,
            )

        elif opt_type == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.train.lr,
                weight_decay=cfg.optimizer.weight_decay,
                betas=tuple(cfg.optimizer.betas),
            )

        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer.type}")

        return optimizer
        
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
        best_path = f"{self.output_dir}/best_supervised_resnet18_{self.dataset.lower()}.pth"

        history = []
        start = time.time()

        for ep in range(1, self.config.train.n_epochs + 1):
            self.net.train()
            train_loss_meter = AverageMeter()

            for x, y in self.train_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                logits = self.net(x)
                loss = self.criterion(logits, y)

                self.optimizer.zero_grad(set_to_none=True)
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
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

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
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

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
            per_class[cls_name] = correct[i] / total[i] if total[i] > 0 else 0.0

        return per_class
