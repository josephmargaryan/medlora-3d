import argparse, time
from pathlib import Path
import torch, json
from medlora.constants import DEFAULT_ROI, MSD_TASKS
from medlora.utils import set_seed, ensure_dir, save_yaml, save_json, plot_curves, count_trainable
from medlora.data import build_loaders
from medlora.models import build_swin_unetr, load_ct_ssl_encoder
from medlora.lora import apply_lora_to_swin_encoder
from medlora.train import train

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["swinv1","swinv2"], required=True)
    p.add_argument("--dataset", choices=MSD_TASKS, required=True)
    p.add_argument("--method", choices=["fft","lora"], required=True)
    p.add_argument("--train-fraction", type=int, choices=[5,20,80,100], required=True)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--early-stopping", type=lambda x: str(x).lower()=="true", default=True)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-epochs", type=int, default=30)

    p.add_argument("--data-dir", type=Path, default=Path("/content/data"))
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--splits-dir", type=Path, default=Path("splits"))

    # method-specific defaults (overrideable)
    p.add_argument("--lr-fft", type=float, default=1e-4)
    p.add_argument("--wd-fft", type=float, default=1e-4)
    p.add_argument("--lr-lora", type=float, default=5e-4)
    p.add_argument("--wd-lora", type=float, default=0.0)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    return p.parse_args()

def run():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # IO
    exp_dir = args.runs_dir / args.dataset / args.model / args.method / f"frac{args.train_faction if hasattr(args,'train_faction') else args.train_fraction}" / f"seed{args.seed}"
    ensure_dir(exp_dir)
    # loaders + save split json
    train_loader, val_loader, train_eval_loader, split_dict = build_loaders(args.data_dir, args.dataset, args.train_fraction, args.seed)
    split_dir = args.splits_dir / args.dataset; ensure_dir(split_dir)
    with open(split_dir / f"frac{args.train_fraction}_seed{args.seed}.json", "w") as f:
        json.dump(split_dict, f, indent=2)

    # model
    in_ch = train_loader.dataset[0]["image"].shape[0] if hasattr(train_loader.dataset[0]["image"], "shape") else 1  # safe fallback
    out_ch = int(train_loader.dataset[0]["label"].max().item() + 1) if torch.is_tensor(train_loader.dataset[0]["label"]) else 2
    use_v2 = (args.model == "swinv2")
    model = build_swin_unetr(in_ch=in_ch, out_ch=out_ch, use_v2=use_v2).to(device)
    loaded, total = load_ct_ssl_encoder(model)
    print(f"[SSL] loaded {loaded}/{total} encoder tensors.")

    # methods
    if args.method == "fft":
        for p in model.parameters(): p.requires_grad = True
        lr, wd = args.lr_fft, args.wd_fft
        tag = f"{args.dataset}|{args.model}|fft|frac{args.train_fraction}|seed{args.seed}"
    else:
        model = apply_lora_to_swin_encoder(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        lr, wd = args.lr_lora, args.wd_lora
        tag = f"{args.dataset}|{args.model}|lora(r={args.lora_r},a={args.lora_alpha})|frac{args.train_fraction}|seed{args.seed}"

    n_params = count_trainable(model)
    print(f"Trainable params: {n_params/1e6:.3f}M")

    # train
    start = time.time()
    logs, best_state = train(
        model=model,
        loaders=(train_loader, val_loader, train_eval_loader),
        device=device, roi=DEFAULT_ROI, max_epochs=args.epochs,
        lr=lr, wd=wd, early_stopping=args.early_stopping,
        patience=args.patience, min_epochs=args.min_epochs, tag=tag
    )
    wall = time.time() - start

    # save artifacts
    torch.save(best_state, exp_dir/"best.ckpt")
    save_yaml(vars(args), exp_dir/"config.yaml")
    final = {
        "best_val_dice": logs["best_val_dice"],
        "train_eval_dice": logs["train_eval_dice"],
        "generalization_gap": logs["train_eval_dice"] - logs["best_val_dice"],
        "trainable_params": n_params,
        "ssl_loaded_encoder_tensors": int(loaded),
        "wall_clock_sec": wall,
    }
    save_json(final, exp_dir/"final_metrics.json")

    # curves
    # (write CSV and plots)
    import csv
    with open(exp_dir/"metrics.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["epoch","train_loss","val_loss","val_dice"])
        for i,(tl,vl,vd) in enumerate(zip(logs["train_losses"], logs["val_losses"], logs["val_dices"]), start=1):
            w.writerow([i, tl, vl, vd])
    plot_curves(logs["train_losses"], logs["val_losses"], logs["val_dices"], exp_dir)
    print("Done:", exp_dir)

if __name__ == "__main__":
    run()
