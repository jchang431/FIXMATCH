import torch
from torch.utils.data import DataLoader

from dataset.fixmatch_dataset import build_fixmatch_datasets
from models.losses.fixmatch_loss import fixmatch_loss
from models.fixmatch import FixMatchModel
from utils.ema import ModelEMA

def evaluate(model, loader, device):
    
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total

def evaluate_per_class(model, loader, device):
    
    model.eval()

    correct = [0 for _ in range(10)]
    total = [0 for _ in range(10)]

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

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

def train_fixmatch(cfg):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make train data sets(labled and unlabeled), val data sets and test_datasets
    train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset = build_fixmatch_datasets(
        root="./data",
        labeled_ratio=cfg.data.label_pct,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.seed,
        split_path=getattr(cfg.data, "split_path", None),
        ra_num_ops=getattr(cfg.fixmatch, "ra_num_ops", 2),
        ra_magnitude=getattr(cfg.fixmatch, "ra_magnitude", 10),
        use_cutout=getattr(cfg.fixmatch, "use_cutout", True),
    )
    # train, labeled
    labeled_loader = DataLoader(
        train_labeled_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        drop_last=True,
    )
    # train, unlabeled
    unlabeled_loader = DataLoader(
        train_unlabeled_dataset,
        batch_size=cfg.train.batch_size * cfg.train.mu,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        drop_last=True,
    )
    # val datasets
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )
    # test datasets
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    model = FixMatchModel(cfg).to(device)

    if cfg.ema.use:
        ema_model = ModelEMA(model, decay=cfg.ema.decay)
    else:
        ema_model = None

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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs
    )

    best_val_acc = 0.0
    history = []
    best_model_path = f"best_fixmatch_{int(cfg.data.label_pct * 100)}pct.pt"

    for epoch in range(cfg.train.epochs):
        model.train()

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        steps_per_epoch = max(len(labeled_loader), len(unlabeled_loader))

        epoch_loss = 0.0
        epoch_loss_x = 0.0
        epoch_loss_u = 0.0
        epoch_mask = 0.0

        for step in range(steps_per_epoch):
            try:
                x_l, y_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_l, y_l = next(labeled_iter)

            try:
                x_uw, x_us = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                x_uw, x_us = next(unlabeled_iter)

            x_l = x_l.to(device)
            y_l = y_l.to(device)
            x_uw = x_uw.to(device)
            x_us = x_us.to(device)

            out = fixmatch_loss(
                model,
                x_l,
                y_l,
                x_uw,
                x_us,
                threshold=cfg.fixmatch.threshold,
                lambda_u=cfg.fixmatch.lambda_u,
            )

            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ema_model is not None:
              ema_model.update(model)

            epoch_loss += out["loss"].item()
            epoch_loss_x += out["loss_x"].item()
            epoch_loss_u += out["loss_u"].item()
            epoch_mask += out["mask"].item()

        scheduler.step()

        epoch_loss /= steps_per_epoch
        epoch_loss_x /= steps_per_epoch
        epoch_loss_u /= steps_per_epoch
        epoch_mask /= steps_per_epoch

        eval_model = ema_model.ema if ema_model is not None else model
        val_acc = evaluate(eval_model, val_loader, device)
        history.append(val_acc)

        print(
            f"Epoch [{epoch+1}/{cfg.train.epochs}] "
            f"loss={epoch_loss:.4f}, "
            f"loss_x={epoch_loss_x:.4f}, "
            f"loss_u={epoch_loss_u:.4f}, "
            f"mask={epoch_mask:.4f}, "
            f"val_acc={val_acc:.4f}"
        )


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model = ema_model.ema if ema_model is not None else model
            torch.save(save_model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    test_acc = evaluate(model, test_loader, device)
    per_class = evaluate_per_class(model, test_loader, device)

    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Test acc: {test_acc:.4f}")

    print("\nPer-class accuracy:")
    for cls_name, acc in per_class.items():
        print(f"  {cls_name:12s}: {acc * 100:.2f}%")

    return history, per_class, test_acc, model
