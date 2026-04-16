import torch
from torch.utils.data import DataLoader

from dataset.fixmatch_dataset import build_fixmatch_datasets
from models.losses.fixmatch_loss import fixmatch_loss
from models.fixmatch import FixMatchModel

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


def train_fixmatch(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset = build_fixmatch_datasets(
        root="./data",
        labeled_ratio=cfg.data.labeled_ratio,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.seed,
    )

    labeled_loader = DataLoader(
        train_labeled_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        drop_last=True,
    )

    unlabeled_loader = DataLoader(
        train_unlabeled_dataset,
        batch_size=cfg.train.batch_size * cfg.train.mu,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    model = FixMatchModel(cfg).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.train.lr,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs
    )

    best_val_acc = 0.0

    for epoch in range(cfg.train.epochs):
        model.train()

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        steps_per_epoch = max(len(labeled_loader), len(unlabeled_loader))

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

        scheduler.step()

        val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch+1}/{cfg.train.epochs}] "
            f"loss={out['loss'].item():.4f}, "
            f"loss_x={out['loss_x'].item():.4f}, "
            f"loss_u={out['loss_u'].item():.4f}, "
            f"mask={out['mask'].item():.4f}, "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_fixmatch.pt")

    model.load_state_dict(torch.load("best_fixmatch.pt"))
    test_acc = evaluate(model, test_loader, device)
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Test acc: {test_acc:.4f}")
