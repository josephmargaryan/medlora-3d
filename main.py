# main.py
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from monai.apps import DecathlonDataset

from medlora.constants import DEFAULT_ROI, MSD_TASKS
from medlora.data import build_loaders
from medlora.lora import apply_lora_to_encoder
from medlora.models import build_swin_unetr, load_ct_ssl_encoder
from medlora.train import train
from medlora.utils import (
    ensure_dir,
    param_stats,
    plot_curves,
    save_json,
    save_yaml,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    # Experiment config
    p.add_argument("--model", choices=["swinv1", "swinv2"], required=True)
    p.add_argument("--dataset", choices=MSD_TASKS, required=True)
    p.add_argument("--method", choices=["fft", "lora"], required=True)
    p.add_argument(
        "--train-fraction", type=int, choices=[5, 20, 80, 100], required=True
    )
    p.add_argument("--seed", type=int, default=0)

    # Training schedule
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument(
        "--early-stopping", type=lambda x: str(x).lower() == "true", default=True
    )
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-epochs", type=int, default=10)

    # Loader & paths
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-dir", type=Path, default=Path("/content/data"))
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--splits-dir", type=Path, default=Path("splits"))

    # Optim / method-specific knobs
    p.add_argument("--lr-fft", type=float, default=1e-4)
    p.add_argument("--wd-fft", type=float, default=1e-4)
    p.add_argument("--lr-lora", type=float, default=5e-4)
    p.add_argument("--wd-lora", type=float, default=0.0)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)

    # Prediction export
    p.add_argument(
        "--save-preds", type=lambda x: str(x).lower() == "true", default=False
    )
    p.add_argument("--pred-splits", choices=["val", "test", "both"], default="test")
    p.add_argument("--save-nii", type=lambda x: str(x).lower() == "true", default=False)

    return p.parse_args()


def run():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- IO & dirs ---
    args.data_dir = args.data_dir.expanduser()
    ensure_dir(args.data_dir)
    ensure_dir(args.runs_dir)
    ensure_dir(args.splits_dir)

    exp_dir = (
        args.runs_dir
        / args.dataset
        / args.model
        / args.method
        / f"frac{args.train_fraction}"
        / f"seed{args.seed}"
    )
    ensure_dir(exp_dir)

    # --- loaders & splits (patient-level, fixed val split) ---
    train_loader, val_loader, train_eval_loader, split_dict = build_loaders(
        args.data_dir,
        args.dataset,
        args.train_fraction,
        args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    split_dir = args.splits_dir / args.dataset
    ensure_dir(split_dir)
    with open(split_dir / f"frac{args.train_fraction}_seed{args.seed}.json", "w") as f:
        json.dump(split_dict, f, indent=2)

    # --- channels / classes from MSD metadata ---
    props = DecathlonDataset(
        root_dir=str(args.data_dir),
        task=args.dataset,
        section="training",
        transform=None,
        download=False,
        val_frac=0.0,
        cache_rate=0.0,
        num_workers=0,
    ).get_properties()
    in_ch = len(props.get("modality", {}) or {}) or 1
    out_ch = len(props.get("labels", {}) or {}) or 2

    # --- model & CT-SSL encoder init ---
    use_v2 = args.model == "swinv2"
    model = build_swin_unetr(in_ch=in_ch, out_ch=out_ch, use_v2=use_v2).to(device)
    loaded, total = load_ct_ssl_encoder(model)
    print(f"[SSL] loaded {loaded}/{total} encoder tensors.")

    # --- method wiring ---
    if args.method == "fft":
        for p in model.parameters():
            p.requires_grad = True
        lr, wd = args.lr_fft, args.wd_fft
        tag = (
            f"{args.dataset}|{args.model}|fft|frac{args.train_fraction}|seed{args.seed}"
        )
    else:  # LoRA strict-PEFT: encoder Linear layers only + seg head
        model = apply_lora_to_encoder(
            model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
        )
        lr, wd = args.lr_lora, args.wd_lora
        tag = f"{args.dataset}|{args.model}|lora(r={args.lora_r},a={args.lora_alpha})|frac{args.train_fraction}|seed{args.seed}"

    # parameter breakdown (for auditing)
    stats = param_stats(model)
    print(
        "Param stats | total: "
        f"{stats['total']/1e6:.3f}M | trainable: {stats['trainable']/1e6:.3f}M | "
        f"lora: {stats['lora']/1e6:.3f}M | head: {stats['head']/1e6:.3f}M | "
        f"decoder_base: {stats['decoder_base']/1e6:.3f}M | encoder_base: {stats['encoder_base']/1e6:.3f}M"
    )

    # --- train ---
    start = time.time()
    logs, best_state = train(
        model=model,
        loaders=(train_loader, val_loader, train_eval_loader),
        device=device,
        roi=DEFAULT_ROI,
        max_epochs=args.epochs,
        lr=lr,
        wd=wd,
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_epochs=args.min_epochs,
        tag=tag,
    )
    wall = time.time() - start

    # --- save artifacts ---
    torch.save(best_state, exp_dir / "best.ckpt")
    save_yaml(vars(args), exp_dir / "config.yaml")
    final = {
        "best_val_dice": logs["best_val_dice"],
        "train_eval_dice": logs["train_eval_dice"],
        "generalization_gap": logs["train_eval_dice"] - logs["best_val_dice"],
        "ssl_loaded_encoder_tensors": int(loaded),
        "wall_clock_sec": wall,
        # parameter stats
        "params_total": stats["total"],
        "params_trainable_total": stats["trainable"],
        "params_trainable_lora": stats["lora"],
        "params_trainable_head": stats["head"],
        "params_trainable_decoder_base": stats["decoder_base"],
        "params_trainable_encoder_base": stats["encoder_base"],
    }
    save_json(final, exp_dir / "final_metrics.json")

    with open(exp_dir / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_dice"])
        for i, (tl, vl, vd) in enumerate(
            zip(logs["train_losses"], logs["val_losses"], logs["val_dices"]), start=1
        ):
            w.writerow([i, tl, vl, vd])
    plot_curves(logs["train_losses"], logs["val_losses"], logs["val_dices"], exp_dir)

    # --- optional predictions export (val/test) ---
    if args.save_preds:
        from medlora.predict import save_predictions

        save_predictions(
            model=model,
            data_dir=args.data_dir,
            task=args.dataset,
            split_dict=split_dict,
            outdir=exp_dir,
            device=device,
            which=args.pred_splits,
            save_nii=args.save_nii,
            roi_size=DEFAULT_ROI,
        )

    print("Done:", exp_dir)


if __name__ == "__main__":
    run()
