import os, json, random, time, yaml
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_yaml(d: dict, path: Path):
    with open(path, "w") as f: yaml.safe_dump(d, f, sort_keys=False)

def save_json(d: dict, path: Path):
    with open(path, "w") as f: json.dump(d, f, indent=2)

def plot_curves(train_losses, val_losses, val_dices, outdir: Path):
    plt.figure(); plt.plot(train_losses,label="train"); plt.plot(val_losses,label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(outdir/"loss_curve.png", dpi=120); plt.close()

    plt.figure(); plt.plot(val_dices,label="val_dice")
    plt.xlabel("epoch"); plt.ylabel("dice"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(outdir/"dice_curve.png", dpi=120); plt.close()

class EarlyStopper:
    def __init__(self, patience=10, min_epochs=30, delta=0.0):
        self.patience = patience; self.min_epochs = min_epochs; self.delta = delta
        self.best = -1e9; self.counter = 0; self.stop = False

    def step(self, value, epoch):
        improved = value > self.best + self.delta
        if improved:
            self.best = value; self.counter = 0
        else:
            if epoch >= self.min_epochs:
                self.counter += 1
                if self.counter >= self.patience: self.stop = True
        return improved
