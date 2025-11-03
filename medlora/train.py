import time, torch
import numpy as np
from tqdm.auto import tqdm
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from .utils import EarlyStopper

def build_losses_and_metrics():
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    dice = DiceMetric(include_background=False, reduction="mean")
    return loss_fn, dice

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, roi):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    inferer = SlidingWindowInferer(roi_size=roi, sw_batch_size=2, overlap=0.5)
    losses, dices = [], []
    for batch in loader:
        images = batch["image"].to(device); labels = batch["label"].to(device)
        logits = inferer(images, model)
        loss = loss_fn(logits, labels)
        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1, keepdim=True)
        y_onehot = torch.nn.functional.one_hot(labels.long().squeeze(1), num_classes=int(logits.shape[1]))\
                   .permute(0,4,1,2,3).float()
        p_onehot = torch.nn.functional.one_hot(preds.long().squeeze(1), num_classes=int(logits.shape[1]))\
                   .permute(0,4,1,2,3).float()
        dice_metric(y_pred=p_onehot.to(device), y=y_onehot.to(device))
    dices.append(dice_metric.aggregate().item())
    return float(np.mean(losses)), float(np.mean(dices))

def train(model, loaders, device, roi, max_epochs=100, lr=1e-4, wd=1e-4,
          early_stopping=True, patience=10, min_epochs=30, tag="run"):
    train_loader, val_loader, train_eval_loader = loaders
    loss_fn, _ = build_losses_and_metrics()
    inferer = SlidingWindowInferer(roi_size=roi, sw_batch_size=2, overlap=0.5)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    stopper = EarlyStopper(patience=patience, min_epochs=min_epochs, delta=0.0) if early_stopping else None

    tr_losses, va_losses, va_dices = [], [], []
    best_state, best_dice = None, -1.0

    for epoch in range(1, max_epochs+1):
        model.train()
        running = []
        pbar = tqdm(train_loader, desc=f"[{tag}] epoch {epoch}/{max_epochs}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device); labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            running.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(running):.4f}")
        tr_losses.append(float(np.mean(running)))

        vloss, vdice = evaluate(model, val_loader, loss_fn, device, roi)
        va_losses.append(vloss); va_dices.append(vdice)

        improved = vdice > best_dice
        if improved:
            best_dice = vdice
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if stopper:
            if stopper.step(vdice, epoch): break

    # train-eval (no aug) for generalization gap
    tloss_eval, tdice_eval = evaluate(model, train_eval_loader, loss_fn, device, roi)

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "train_losses": tr_losses, "val_losses": va_losses, "val_dices": va_dices,
        "best_val_dice": best_dice, "train_eval_dice": tdice_eval
    }, best_state
