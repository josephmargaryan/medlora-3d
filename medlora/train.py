# medlora/train.py
from __future__ import annotations
import contextlib
import numpy as np
import torch
from tqdm.auto import tqdm
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss


def build_losses_and_metrics():
    # Industry-standard choice for med-seg: Dice + CE with softmax + one-hot targets
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    return loss_fn


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, roi):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    inferer = SlidingWindowInferer(roi_size=roi, sw_batch_size=2, overlap=0.5)

    losses = []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = inferer(images, model)
        loss = loss_fn(logits, labels)
        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1, keepdim=True)
        num_classes = int(logits.shape[1])
        y_onehot = (
            torch.nn.functional.one_hot(
                labels.long().squeeze(1), num_classes=num_classes
            )
            .permute(0, 4, 1, 2, 3)
            .float()
        )
        p_onehot = (
            torch.nn.functional.one_hot(
                preds.long().squeeze(1), num_classes=num_classes
            )
            .permute(0, 4, 1, 2, 3)
            .float()
        )
        dice_metric(y_pred=p_onehot.to(device), y=y_onehot.to(device))

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    mean_dice = float(dice_metric.aggregate().item())
    return mean_loss, mean_dice


def train(
    model,
    loaders,
    device,
    roi,
    max_epochs: int = 100,
    lr: float = 1e-4,
    wd: float = 1e-4,
    early_stopping: bool = True,
    patience: int = 10,
    min_epochs: int = 10,
    tag: str = "run",
):
    train_loader, val_loader, train_eval_loader = loaders
    loss_fn = build_losses_and_metrics()
    inferer = SlidingWindowInferer(roi_size=roi, sw_batch_size=2, overlap=0.5)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd
    )

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_state, best_dice, best_counter = None, -1.0, 0
    tr_losses, va_losses, va_dices = [], [], []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = []

        pbar = tqdm(
            train_loader, desc=f"[{tag}] epoch {epoch}/{max_epochs}", leave=False
        )
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            ctx = (
                torch.amp.autocast("cuda", dtype=torch.float16)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with ctx:
                logits = model(images)
                loss = loss_fn(logits, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(running):.4f}")

        tr_losses.append(float(np.mean(running)) if running else float("nan"))

        vloss, vdice = evaluate(model, val_loader, loss_fn, device, roi)
        va_losses.append(vloss)
        va_dices.append(vdice)

        if vdice > best_dice:
            best_dice = vdice
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_counter = 0
        else:
            best_counter += 1

        if early_stopping and epoch >= min_epochs and best_counter >= patience:
            break

    # Train-eval (no aug) for generalization gap
    _, tdice_eval = evaluate(model, train_eval_loader, loss_fn, device, roi)

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": tr_losses,
        "val_losses": va_losses,
        "val_dices": va_dices,
        "best_val_dice": best_dice,
        "train_eval_dice": tdice_eval,
    }, best_state
