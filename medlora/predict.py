# medlora/predict.py
from __future__ import annotations
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Literal, Optional

from monai.apps import DecathlonDataset
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    EnsureTyped,
)
import nibabel as nib

from .data import make_transforms, is_ct
from .constants import DEFAULT_ROI


def _case_id_from_path(p: str) -> str:
    name = Path(p).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return Path(p).stem


def _to_numpy_image_and_mask(batch) -> tuple[np.ndarray, Optional[np.ndarray]]:
    img = batch["image"][0, 0].detach().cpu().numpy()  # (D,H,W)
    lbl = None
    if "label" in batch:
        lbl = batch["label"][0, 0].detach().cpu().numpy().astype(np.int16)
    return img, lbl


def _choose_slice_for_vis(img: np.ndarray, pred: np.ndarray) -> int:
    z_scores = (pred > 0).sum(axis=(1, 2))
    return int(z_scores.argmax()) if z_scores.max() > 0 else (img.shape[0] // 2)


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _save_vis_png(img: np.ndarray, pred: np.ndarray, out_png: Path, title: str):
    z = _choose_slice_for_vis(img, pred)
    g = _norm01(img[z])
    m = (pred[z] > 0).astype(np.float32)
    plt.figure(figsize=(5, 5))
    plt.imshow(g, cmap="gray")
    plt.imshow(np.ma.masked_where(m == 0, m), alpha=0.35)
    plt.axis("off")
    plt.title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def _save_3panel_png(
    img: np.ndarray, lbl: np.ndarray, pred: np.ndarray, out_png: Path, title: str
):
    z = _choose_slice_for_vis(img, pred)
    g = _norm01(img[z])
    pm = (pred[z] > 0).astype(np.float32)
    lm = (lbl[z] > 0).astype(np.float32)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(g, cmap="gray")
    axs[0].set_title("Image")
    axs[0].axis("off")
    axs[1].imshow(g, cmap="gray")
    axs[1].imshow(np.ma.masked_where(lm == 0, lm), alpha=0.35)
    axs[1].set_title("GT")
    axs[1].axis("off")
    axs[2].imshow(g, cmap="gray")
    axs[2].imshow(np.ma.masked_where(pm == 0, pm), alpha=0.35)
    axs[2].set_title("Pred")
    axs[2].axis("off")
    fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def _save_pred_nii(batch, pred: np.ndarray, out_nii: Path):
    meta = (
        batch["image"].meta
        if hasattr(batch["image"], "meta")
        else getattr(batch["image"], "meta", {})
    )
    affine = (
        meta.get("affine") if meta and "affine" in meta else np.eye(4, dtype=np.float32)
    )
    img = nib.Nifti1Image(pred.astype(np.uint8), affine)
    out_nii.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_nii))


def _make_image_only_eval_transforms(task: str, roi=DEFAULT_ROI):
    """Eval transforms for TEST split (no labels)."""
    ct = is_ct(task)
    base = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 2.0) if ct else (1.0, 1.0, 1.0),
            mode=("bilinear",),
        ),
    ]
    norm = (
        [
            ScaleIntensityRanged(
                keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            )
        ]
        if ct
        else [NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)]
    )
    return Compose(base + norm + [EnsureTyped(keys=["image"])])


@torch.no_grad()
def save_predictions(
    model: torch.nn.Module,
    data_dir: Path,
    task: str,
    split_dict: Dict[str, List[dict]],
    outdir: Path,
    device: torch.device,
    which: Literal["val", "test", "both"] = "test",
    save_nii: bool = False,
    roi_size: tuple[int, int, int] = DEFAULT_ROI,
):
    """
    Save per-case prediction images (and optional NIfTI) for val/test.

    - 'val': uses the exact val items in split_dict (has labels -> 3-panel PNG).
    - 'test': uses MSD official test split (no labels -> image+pred PNG).
    """
    model.eval()
    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=2, overlap=0.5)

    # VAL: label-aware eval transforms (reuse data.make_transforms with aug=False)
    if which in ("val", "both"):
        eval_tf_val = make_transforms(task, roi=roi_size, aug=False)
        val_items = split_dict.get("val", [])
        val_ds = Dataset(val_items, transform=eval_tf_val)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        val_dir = outdir / "preds" / "val"
        for batch in val_loader:
            images = batch["image"].to(device)
            logits = inferer(images, model)
            pred = torch.argmax(logits, dim=1, keepdim=True)[0, 0].cpu().numpy()
            img_np, lbl_np = _to_numpy_image_and_mask(batch)
            case_path = batch["image"].meta["filename_or_obj"][0]
            cid = _case_id_from_path(case_path)
            _save_3panel_png(
                img_np, lbl_np, pred, val_dir / f"{cid}.png", title=f"{task} | {cid}"
            )
            if save_nii:
                _save_pred_nii(batch, pred, val_dir / f"{cid}_pred.nii.gz")

    if which in ("test", "both"):
        test_tf = _make_image_only_eval_transforms(task, roi=roi_size)
        test_ds_raw = DecathlonDataset(
            root_dir=str(data_dir),
            task=task,
            section="test",
            transform=None,
            download=False,
            val_frac=0.0,
            cache_rate=0.0,
            num_workers=0,
        )
        test_ds = Dataset(list(test_ds_raw.data), transform=test_tf)
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        test_dir = outdir / "preds" / "test"
        for batch in test_loader:
            images = batch["image"].to(device)
            logits = inferer(images, model)
            pred = torch.argmax(logits, dim=1, keepdim=True)[0, 0].cpu().numpy()
            img_np, _ = _to_numpy_image_and_mask(batch)
            case_path = batch["image"].meta["filename_or_obj"][0]
            cid = _case_id_from_path(case_path)
            _save_vis_png(
                img_np, pred, test_dir / f"{cid}.png", title=f"{task} | {cid}"
            )
            if save_nii:
                _save_pred_nii(batch, pred, test_dir / f"{cid}_pred.nii.gz")
