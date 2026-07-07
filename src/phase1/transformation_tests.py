from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1.teachers import TeacherModel
from src.phase1.train import evaluate, train_teacher_classifier
from src.common.utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    save_training_history_csv,
    set_seed,
    _load_flowers102,
    _load_tiny_imagenet,
    _make_loader,
    _make_subset,
    _split_indices,
)

SEED = 291652
DATA_ROOT = PROJECT_ROOT / "data"
EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "transform_test"
TABLE_DIR = EXPERIMENT_DIR / "tables"
FIGURE_DIR = EXPERIMENT_DIR / "figures"
TEMP_CKPT = EXPERIMENT_DIR / "_temp_checkpoint.pth"

ALL_TEACHERS = ("resnet50", "convnext_base", "vgg16_bn")
ALL_DATASETS = ("oxford-pets", "flowers-102", "tiny-imagenet-200")
NUM_WORKERS = 4

DEFAULTS: dict[str, Any] = {
    "num_epochs": 15,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "patience": 4,
}

DATASET_OVERRIDES: dict[str, dict[str, Any]] = {
    "flowers-102": {"batch_size": 64, "num_epochs": 15},
    "tiny-imagenet-200": {"num_epochs": 12},
}

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

SETUP_DESCRIPTIONS: dict[str, str] = {
    "baseline_rrc":
        "RandomResizedCrop(224, (0.08,1.0)) + HFlip  [current train_teachers default]",
    "mild_crop":
        "Resize(256) -> RandomCrop(224) + HFlip  [gentle crop for Oxford-Pets]",
    "rrc_tight":
        "RandomResizedCrop(224, (0.5,1.0)) + HFlip  [>=50% crop area, tighter scale]",
    "rrc_colorjitter":
        "baseline_rrc + ColorJitter(0.4, 0.4, 0.4, 0.1)  [colour-space augmentation]",
    "rrc_randaugment":
        "baseline_rrc + RandAugment(num_ops=2, magnitude=9)  [Cubuk et al. 2020]",
    "rrc_trivialaugment":
        "baseline_rrc + TrivialAugmentWide()  [Mueller & Hutter 2021]",
    "rrc_randomerasing":
        "baseline_rrc + RandomErasing(p=0.25, scale=(0.02,0.1))  [Zhong et al. 2020]",
    "no_aug":
        "Resize(256) -> CenterCrop(224)  [control condition, no random transforms]",
}


def build_setup_transforms() -> dict[str, transforms.Compose]:
    """Returns one train-transform Compose per setup name."""
    _rrc = transforms.RandomResizedCrop(224, scale=(0.08, 1.0))
    _hflip = transforms.RandomHorizontalFlip()
    _totensor = transforms.ToTensor()
    _norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    return {
        "baseline_rrc": transforms.Compose([_rrc, _hflip, _totensor, _norm]),
        "mild_crop": transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            _hflip,
            _totensor,
            _norm,
        ]),
        "rrc_tight": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            _hflip,
            _totensor,
            _norm,
        ]),
        "rrc_colorjitter": transforms.Compose([
            _rrc,
            _hflip,
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            _totensor,
            _norm,
        ]),
        "rrc_randaugment": transforms.Compose([
            _rrc,
            _hflip,
            transforms.RandAugment(num_ops=2, magnitude=9),
            _totensor,
            _norm,
        ]),
        "rrc_trivialaugment": transforms.Compose([
            _rrc,
            _hflip,
            transforms.TrivialAugmentWide(),
            _totensor,
            _norm,
        ]),
        "rrc_randomerasing": transforms.Compose([
            _rrc,
            _hflip,
            _totensor,
            _norm,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        ]),
        "no_aug": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            _totensor,
            _norm,
        ]),
    }



def _load_oxford_pets_with_transform(
    data_root: Path,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> tuple[Any, Any, Any, int]:
    root = Path(data_root)
    train_full = datasets.OxfordIIITPet(
        root=str(root),
        split="trainval",
        target_types="category",
        transform=train_transform,
        download=False,
    )
    val_full = datasets.OxfordIIITPet(
        root=str(root),
        split="trainval",
        target_types="category",
        transform=eval_transform,
        download=False,
    )
    test_dataset = datasets.OxfordIIITPet(
        root=str(root),
        split="test",
        target_types="category",
        transform=eval_transform,
        download=False,
    )
    
    train_indices, val_indices = _split_indices(len(train_full))
    
    return (
        _make_subset(train_full, train_indices),
        _make_subset(val_full, val_indices),
        test_dataset,
        len(train_full.classes),
    )


def build_loaders(
    dataset_name: str,
    data_root: Path,
    batch_size: int,
    num_workers: int,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    train_subset_frac: float = 1.0,
) -> tuple[Any, Any, Any, int]:
    if dataset_name == "oxford-pets":
        train_ds, val_ds, test_ds, n_cls = _load_oxford_pets_with_transform(
            data_root, train_transform, eval_transform,
        )
    elif dataset_name == "flowers-102":
        train_ds, val_ds, test_ds, n_cls = _load_flowers102(
            data_root, train_transform, eval_transform,
        )
    elif dataset_name == "tiny-imagenet-200":
        train_ds, val_ds, test_ds, n_cls = _load_tiny_imagenet(
            data_root, train_transform, eval_transform,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")

    if 0.0 < train_subset_frac < 1.0:
        n_keep = max(2, int(len(train_ds) * train_subset_frac))
        train_ds = torch.utils.data.Subset(train_ds, list(range(n_keep)))

    return (
        _make_loader(train_ds, batch_size, num_workers, shuffle=True),
        _make_loader(val_ds, batch_size, num_workers, shuffle=False),
        _make_loader(test_ds, batch_size, num_workers, shuffle=False),
        n_cls,
    )



def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _history_csv_path(setup_name: str, teacher_name: str, dataset_name: str) -> Path:
    return TABLE_DIR / f"training_history_{setup_name}_{teacher_name}_{dataset_name}.csv"


def _unnormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu().float() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"



def save_aug_sample_grids(
    setups: dict[str, transforms.Compose],
    data_root: Path,
    num_workers: int,
    n_images: int = 8,
) -> None:
    for setup_name, train_transform in setups.items():
        out_path = FIGURE_DIR / f"augmentation_samples_{setup_name}.png"
        set_seed(SEED)
        try:
            train_ds, _, _, _ = _load_oxford_pets_with_transform(
                data_root, train_transform, EVAL_TRANSFORM,
            )
            loader = _make_loader(
                train_ds, batch_size=n_images, num_workers=num_workers, shuffle=True,
            )
            images, _ = next(iter(loader))
        except Exception as exc:
            print(f"  {setup_name}: skipped grid ({exc})")
            continue

        n_show = min(n_images, images.shape[0])
        n_cols = min(n_show, 4)
        n_rows = math.ceil(n_show / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        for i, ax in enumerate(np.array(axes).reshape(-1)):
            if i < n_show:
                ax.imshow(_unnormalize(images[i]))
            ax.axis("off")
        fig.suptitle(f"{setup_name}\n{SETUP_DESCRIPTIONS[setup_name]}", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {setup_name}: {out_path.relative_to(PROJECT_ROOT)}")



def run_one(
    setup_name: str,
    teacher_name: str,
    dataset_name: str,
    train_transform: transforms.Compose,
    config: dict[str, Any],
    train_subset_frac: float,
    num_workers: int,
    device: str,
    done_keys: set[tuple[str, str, str]],
) -> dict[str, Any] | None:
    key = (setup_name, teacher_name, dataset_name)
    if key in done_keys:
        print(f"  [skip] already in results.csv")
        return None

    set_seed(SEED)
    train_loader, val_loader, test_loader, num_classes = build_loaders(
        dataset_name=dataset_name,
        data_root=DATA_ROOT,
        batch_size=config["batch_size"],
        num_workers=num_workers,
        train_transform=train_transform,
        eval_transform=EVAL_TRANSFORM,
        train_subset_frac=train_subset_frac,
    )

    model = TeacherModel(backbone_name=teacher_name, num_classes=num_classes)
    history = train_teacher_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config["num_epochs"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        device=device,
        checkpoint_path=str(TEMP_CKPT),
        patience=config["patience"],
    )

    ckpt = torch.load(TEMP_CKPT, map_location=device)
    model.classifier.load_state_dict(ckpt["classifier_state_dict"])
    model.to(device)
    _, test_acc = evaluate(model, test_loader, device)

    save_training_history_csv(
        history,
        save_path=_history_csv_path(setup_name, teacher_name, dataset_name),
        teacher_name=teacher_name,
        dataset_name=dataset_name,
    )

    del train_loader, val_loader, test_loader, model, ckpt
    _cleanup()
    return {
        "Setup": setup_name,
        "Teacher": teacher_name,
        "Dataset": dataset_name,
        "Mode": "frozen",
        "Val Acc (%)": round(float(history["best_val_acc"]), 4),
        "Test Acc (%)": round(float(test_acc), 4),
        "Epochs Run": len(history["train_acc"]),
        "Best Epoch": int(history["best_epoch"]),
    }



def plot_results_bars(df: pd.DataFrame, setup_names: list[str]) -> None:
    for teacher in df["Teacher"].unique():
        sub = df[df["Teacher"] == teacher]
        dsets = sorted(sub["Dataset"].unique())
        n_setups = len(setup_names)
        x = np.arange(len(dsets))
        width = 0.8 / n_setups
        offsets = np.linspace(-(0.4 - width / 2), 0.4 - width / 2, n_setups)

        fig, ax = plt.subplots(figsize=(max(8, 3 * len(dsets)), 5))
        for i, sname in enumerate(setup_names):
            heights = []
            for ds in dsets:
                vals = sub.loc[(sub["Dataset"] == ds) & (sub["Setup"] == sname), "Test Acc (%)"].values
                heights.append(float(vals[0]) if len(vals) and not np.isnan(vals[0]) else 0.0)
            ax.bar(x + offsets[i], heights, width=width, label=sname, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(dsets, rotation=15, ha="right")
        ax.set_ylabel("Test Acc (%)")
        ax.set_title(f"Augmentation sweep: {teacher} (frozen linear probe)")
        ax.legend(fontsize=7, ncol=2, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = FIGURE_DIR / f"results_{teacher}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out.relative_to(PROJECT_ROOT)}")


def plot_heatmap(df: pd.DataFrame, setup_names: list[str]) -> None:
    teachers = sorted(df["Teacher"].unique())
    dsets = sorted(df["Dataset"].unique())
    cols = [(t, d) for t in teachers for d in dsets]

    matrix = np.full((len(setup_names), len(cols)), float("nan"))
    for i, sname in enumerate(setup_names):
        for j, (t, d) in enumerate(cols):
            vals = df.loc[
                (df["Setup"] == sname) & (df["Teacher"] == t) & (df["Dataset"] == d),
                "Test Acc (%)",
            ].values
            if len(vals):
                matrix[i, j] = float(vals[0])

    fig, ax = plt.subplots(figsize=(max(12, 2 * len(cols)), max(4, 0.65 * len(setup_names))))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="Test Acc (%)")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{t}\n{d}" for t, d in cols], fontsize=7, rotation=30, ha="right")
    ax.set_yticks(range(len(setup_names)))
    ax.set_yticklabels(setup_names, fontsize=8)
    for i in range(len(setup_names)):
        for j in range(len(cols)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if val < 30 or val > 85 else "black")
                
    ax.set_title("Test accuracy (%) by augmentation setup, encoder, and dataset")
    fig.tight_layout()
    out = FIGURE_DIR / "results_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.relative_to(PROJECT_ROOT)}")



def parse_args() -> argparse.Namespace:
    all_setup_names = list(build_setup_transforms().keys())
    parser = argparse.ArgumentParser(
        description="Data-augmentation sweep for Phase 1 teacher encoders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--encoders", nargs="+", default=list(ALL_TEACHERS),
        choices=ALL_TEACHERS, metavar="ENC",
        help="Encoders to include in the sweep.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(ALL_DATASETS),
        choices=ALL_DATASETS, metavar="DS",
        help="Datasets to include in the sweep.",
    )
    parser.add_argument(
        "--setups", nargs="+", default=all_setup_names,
        choices=all_setup_names, metavar="SETUP",
        help="Transform setups to run (default: all).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override num_epochs for all runs.",
    )
    parser.add_argument(
        "--train-subset-frac", type=float, default=1.0,
        help="Fraction of training data to use (0 < f <= 1).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=NUM_WORKERS,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke test: 1 epoch, 2%% of training data.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = _resolve_device()

    if args.quick:
        args.epochs = args.epochs or 1
        args.train_subset_frac = min(args.train_subset_frac, 0.02)

    for d in (TABLE_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    all_setups = build_setup_transforms()
    selected_setups = {k: all_setups[k] for k in args.setups}

    pd.DataFrame([
        {"Setup": k, "Description": SETUP_DESCRIPTIONS[k]}
        for k in selected_setups
    ]).to_csv(TABLE_DIR / "transform_setups.csv", index=False)

    save_aug_sample_grids(selected_setups, DATA_ROOT, args.num_workers)

    results_path = TABLE_DIR / "results.csv"
    existing_df = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    done_keys: set[tuple[str, str, str]] = set(
        zip(existing_df.get("Setup", []), existing_df.get("Teacher", []), existing_df.get("Dataset", []))
    ) if not existing_df.empty else set()
    results: list[dict] = existing_df.to_dict("records") if not existing_df.empty else []

    total = len(args.setups) * len(args.encoders) * len(args.datasets)
    done = 0
    for setup_name, train_transform in selected_setups.items():
        for teacher_name in args.encoders:
            for dataset_name in args.datasets:
                done += 1
                cfg = {**DEFAULTS, **DATASET_OVERRIDES.get(dataset_name, {})}
                if args.epochs is not None:
                    cfg["num_epochs"] = args.epochs
                print(
                    f"\n[{done}/{total}] {setup_name} / {teacher_name} / {dataset_name}"
                    f"  epochs={cfg['num_epochs']}  device={device}"
                )
                row = run_one(
                    setup_name=setup_name,
                    teacher_name=teacher_name,
                    dataset_name=dataset_name,
                    train_transform=train_transform,
                    config=cfg,
                    train_subset_frac=args.train_subset_frac,
                    num_workers=args.num_workers,
                    device=device,
                    done_keys=done_keys,
                )
                if row is not None:
                    results.append(row)
                    done_keys.add((setup_name, teacher_name, dataset_name))
                    pd.DataFrame(results).to_csv(results_path, index=False)

    if not results:
        print("\nNo results to report.")
        return

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path.relative_to(PROJECT_ROOT)}")
    print(results_df.to_string(index=False))

    print("\n[Summary figures]")
    plot_results_bars(results_df, list(selected_setups.keys()))
    plot_heatmap(results_df, list(selected_setups.keys()))
    print("\nDone.")


if __name__ == "__main__":
    main()
