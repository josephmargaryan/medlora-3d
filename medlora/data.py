import json
from pathlib import Path
import numpy as np
from monai.apps import DecathlonDataset
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
)
from .constants import CT_TASKS, DEFAULT_ROI, DEFAULT_VAL_FRAC


def is_ct(task):
    return task in CT_TASKS


def make_transforms(task, roi=DEFAULT_ROI, aug=True):
    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0) if is_ct(task) else (1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
    ]
    norm = (
        [
            ScaleIntensityRanged(
                keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            )
        ]
        if is_ct(task)
        else [NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)]
    )
    if aug:
        extra = [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi,
                pos=1,
                neg=1,
                num_samples=1,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=(0, 1, 2)),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            EnsureTyped(keys=["image", "label"]),
        ]
    else:
        extra = [EnsureTyped(keys=["image", "label"])]
    return Compose(base + norm + extra)


def load_decathlon_list(data_dir: Path, task: str, download=True):
    data_dir = Path(data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    ds = DecathlonDataset(
        root_dir=str(data_dir),
        task=task,
        section="training",
        transform=None,
        download=download,
        val_frac=0.0,
        cache_rate=0.0,
        num_workers=0,
    )
    return list(ds.data)


def train_val_split(indexes, val_frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(indexes))
    val_count = max(1, int(val_frac * len(indexes)))
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    return [indexes[i] for i in train_idx], [indexes[i] for i in val_idx]


def take_fraction(items, frac_percent, seed):
    if frac_percent >= 100:
        return items
    n = max(1, int(len(items) * (frac_percent / 100.0)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(items))[:n]
    return [items[i] for i in perm]


def build_loaders(
    data_dir: Path, task: str, fraction: int, seed: int, batch_size=2, num_workers=2
):
    all_items = load_decathlon_list(data_dir, task, download=True)
    # fixed val split
    train_pool, val_items = train_val_split(
        all_items, DEFAULT_VAL_FRAC, seed=0
    )  # seed=0 to keep "test" fixed across runs
    train_items = take_fraction(train_pool, fraction, seed=seed)

    train_tf = make_transforms(task, aug=True)
    eval_tf = make_transforms(task, aug=False)

    train_ds = Dataset(train_items, transform=train_tf)
    val_ds = Dataset(val_items, transform=eval_tf)
    # train-eval (no aug) to compute generalization gap
    train_eval_ds = Dataset(train_items, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Save split for reproducibility
    return (
        train_loader,
        val_loader,
        train_eval_loader,
        {"train": train_items, "val": val_items},
    )
