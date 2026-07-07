from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1.teachers import TeacherModel
from src.phase1.train import evaluate, train_teacher_classifier, train_teacher_finetune
from src.common.utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    compute_flops,
    count_parameters,
    count_total_parameters,
    get_dataloaders,
    plot_training_curves,
    save_training_history_csv,
    set_seed,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 291652
DATA_ROOT = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "teachers"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

TEACHERS = ("resnet50", "convnext_base", "vgg16_bn")
FINETUNE_TEACHERS = ("resnet50", "convnext_base")
DATASETS = ("oxford-pets", "flowers-102", "tiny-imagenet-200")
NUM_WORKERS = 8

DEFAULTS: dict[str, Any] = {
    "num_epochs": 15,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "patience": 4
}

RUN_OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {
    ("convnext_base", "oxford-pets"):         {"num_epochs": 20},
    ("convnext_base", "tiny-imagenet-200"): {"num_epochs": 20},
    ("convnext_base", "flowers-102"):       {"num_epochs": 30, "batch_size": 64},
    ("resnet50", "oxford-pets"):              {"num_epochs": 20},
    ("resnet50", "tiny-imagenet-200"):      {"num_epochs": 20},
    ("resnet50", "flowers-102"):            {"num_epochs": 30, "batch_size": 64},
    ("vgg16_bn", "oxford-pets"):             {"num_epochs": 20},
    ("vgg16_bn", "tiny-imagenet-200"):       {"num_epochs": 20},
    ("vgg16_bn", "flowers-102"):             {"num_epochs": 30, "batch_size": 64},
}

DEFAULTS_FT: dict[str, Any] = {
    "num_epochs": 10,
    "lr": 1e-3,
    "encoder_lr": 5e-5,
    "lr_decay": 0.8,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "label_smoothing": 0.1,
    "n_blocks": 1,
    "patience": 4
}

RUN_OVERRIDES_FT: dict[tuple[str, str], dict[str, Any]] = {
    ("convnext_base", "oxford-pets"):         {"num_epochs": 15},
    ("convnext_base", "tiny-imagenet-200"): {"num_epochs": 20},
    ("convnext_base", "flowers-102"):       {"num_epochs": 20, "batch_size": 64},
    ("resnet50", "oxford-pets"):              {"num_epochs": 20},
    ("resnet50", "tiny-imagenet-200"):      {"num_epochs": 20},
    ("resnet50", "flowers-102"):            {"num_epochs": 20, "batch_size": 64},
}

def parse_args() -> argparse.Namespace:
    """Parses training-stage and figure-rendering options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-frozen",
        action="store_true",
        help="Skip frozen-head training and use existing frozen checkpoints.",
    )
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Skip light fine-tuning for ResNet-50 and ConvNeXt-Tiny.",
    )
    parser.add_argument(
        "--render-feature-maps-only",
        action="store_true",
        help="Render feature maps from existing checkpoints without training.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit the figure-level title and overwrite the canonical figure files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("oxford-pets", "flowers-102", "tiny-imagenet-200"),
        help="Override the configured dataset list.",
    )
    return parser.parse_args()

def ensure_directories() -> None:
    for directory in (
        CHECKPOINT_DIR / "frozen",
        CHECKPOINT_DIR / "finetune",
        FIGURE_DIR,
        TABLE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def checkpoint_path(
    teacher_name: str,
    dataset_name: str,
    mode: str = "frozen",
) -> Path:
    return CHECKPOINT_DIR / mode / f"{teacher_name}_{dataset_name}.pth"


def history_csv_path(
    teacher_name: str,
    dataset_name: str,
    mode: str = "frozen",
) -> Path:
    return TABLE_DIR / f"training_history_{mode}_{teacher_name}_{dataset_name}.csv"


def run_config(
    defaults: dict[str, Any],
    overrides: dict[tuple[str, str], dict[str, Any]],
    teacher_name: str,
    dataset_name: str,
    num_epochs: int | None,
) -> dict[str, Any]:
    config = {**defaults, **overrides.get((teacher_name, dataset_name), {})}
    if num_epochs is not None:
        config["num_epochs"] = num_epochs
    return config


def cleanup_cuda() -> None:
    """Releases Python references and cached CUDA allocations between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def describe_model(
    model: TeacherModel,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> None:
    model.to(device)
    model.eval()
    images, _ = next(iter(loader))
    with torch.no_grad():
        feature_map, pooled = model.extract_features(images[:1].to(device))
    total_params = count_total_parameters(model)
    trainable_params = count_parameters(model)
    print(f"Backbone: {model.backbone_name}")
    print(f"Total params: {total_params / 1e6:.2f}M")
    print(
        f"Trainable params: {trainable_params:,} "
        f"({100.0 * trainable_params / total_params:.2f}%)"
    )
    print(f"Encoder output shape: {tuple(feature_map.shape)}")
    print(f"Feature dim: {model.feature_dim} | pooled shape: {tuple(pooled.shape)}")


def write_combined_histories(
    teachers: list[str],
    datasets: list[str],
    mode: str,
) -> None:
    frames = []
    for teacher_name in teachers:
        for dataset_name in datasets:
            csv_path = history_csv_path(teacher_name, dataset_name, mode)
            if csv_path.exists():
                frames.append(pd.read_csv(csv_path))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            TABLE_DIR / f"teacher_training_histories_{mode}.csv",
            index=False,
        )


def train_frozen_teachers(
    teachers: list[str],
    datasets: list[str],
    num_workers: int,
    device: str,
) -> None:
    for teacher_name in teachers:
        for dataset_name in datasets:
            config = run_config(
                DEFAULTS,
                RUN_OVERRIDES,
                teacher_name,
                dataset_name,
                num_epochs=None,
            )
            output_checkpoint = checkpoint_path(teacher_name, dataset_name)

            print("=" * 100)
            print(f"Teacher: {teacher_name} | Dataset: {dataset_name} | cfg: {config}")
            set_seed(SEED)
            train_loader, val_loader, _, num_classes = get_dataloaders(
                dataset_name=dataset_name,
                data_root=DATA_ROOT,
                batch_size=config["batch_size"],
                num_workers=num_workers,
            )
            model = TeacherModel(
                backbone_name=teacher_name,
                num_classes=num_classes,
            )
            describe_model(model, train_loader, device)
            history = train_teacher_classifier(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config["num_epochs"],
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                patience=config["patience"],
                device=device,
                checkpoint_path=str(output_checkpoint),
            )
            save_training_history_csv(
                history=history,
                save_path=history_csv_path(teacher_name, dataset_name),
                teacher_name=teacher_name,
                dataset_name=dataset_name,
            )
            write_combined_histories(teachers, datasets, mode="frozen")
            figure = plot_training_curves(
                history=history,
                title=f"{teacher_name} on {dataset_name} (frozen)",
                save_path=FIGURE_DIR
                / f"teacher_frozen_{teacher_name}_{dataset_name}.png",
            )
            plt.close(figure)
            del model, train_loader, val_loader
            cleanup_cuda()


def load_model_from_checkpoint(
    teacher_name: str,
    dataset_name: str,
    num_classes: int,
    mode: str,
    device: str,
) -> tuple[TeacherModel, dict[str, Any]]:
    """Restores a frozen or fine-tuned teacher checkpoint."""
    saved_checkpoint = checkpoint_path(teacher_name, dataset_name, mode)
    checkpoint = torch.load(saved_checkpoint, map_location=device)
    model = TeacherModel(
        backbone_name=teacher_name,
        num_classes=num_classes,
    ).to(device)
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    if mode == "finetune":
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        model.unfreeze_top(checkpoint.get("n_blocks_unfrozen", 1))
    return model, checkpoint


def evaluate_teachers(
    teachers: list[str],
    datasets: list[str],
    num_workers: int,
    device: str,
    mode: str,
) -> pd.DataFrame:
    results = []
    flop_cache: dict[tuple[str, int], float] = {}
    defaults = DEFAULTS if mode == "frozen" else DEFAULTS_FT
    overrides = RUN_OVERRIDES if mode == "frozen" else RUN_OVERRIDES_FT

    for teacher_name in teachers:
        for dataset_name in datasets:
            saved_checkpoint = checkpoint_path(teacher_name, dataset_name, mode)
            if not saved_checkpoint.exists():
                print(f"Skipping missing checkpoint: {saved_checkpoint}")
                continue

            config = run_config(
                defaults,
                overrides,
                teacher_name,
                dataset_name,
                num_epochs=None,
            )
            _, _, test_loader, num_classes = get_dataloaders(
                dataset_name=dataset_name,
                data_root=DATA_ROOT,
                batch_size=config["batch_size"],
                num_workers=num_workers,
            )
            model, checkpoint = load_model_from_checkpoint(
                teacher_name,
                dataset_name,
                num_classes,
                mode,
                device,
            )
            _, test_acc = evaluate(model, test_loader, device)
            total_params = count_total_parameters(model)
            trainable_params = count_parameters(model)
            flop_key = (teacher_name, num_classes)
            if flop_key not in flop_cache:
                try:
                    flop_cache[flop_key] = compute_flops(model)
                except Exception as exc:
                    print(
                        f"Could not compute FLOPs for "
                        f"{teacher_name}/{dataset_name}: {exc}"
                    )
                    flop_cache[flop_key] = np.nan
                gflops = flop_cache[flop_key]
            else:
                gflops = flop_cache[flop_key]

            result = {
                "Teacher": teacher_name,
                "Dataset": dataset_name,
                "Val Acc (%)": checkpoint.get("best_val_acc", np.nan),
                "Test Acc (%)": test_acc,
                "Total Params (M)": total_params / 1e6,
                "Trainable Params": trainable_params,
                "Trainable Params (%)": 100.0 * trainable_params / total_params,
                "GFLOPs": gflops,
            }
            if mode == "finetune":
                result = {
                    "Mode": "finetune",
                    **result,
                    "Blocks Unfrozen": checkpoint.get("n_blocks_unfrozen", 1),
                }
            results.append(result)
            del model, test_loader
            cleanup_cuda()

    results_frame = pd.DataFrame(results)
    output_name = (
        "teacher_results.csv"
        if mode == "frozen"
        else "teacher_results_finetune.csv"
    )
    results_frame.to_csv(TABLE_DIR / output_name, index=False)
    print(results_frame.to_string(index=False))
    return results_frame


def unnormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype).view(3, 1, 1)
    image = (image.cpu() * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def render_feature_maps(
    teachers: list[str],
    datasets: list[str],
    num_workers: int,
    device: str,
    no_title: bool = False,
) -> None:
    for dataset_name in datasets:
        _, _, test_loader, num_classes = get_dataloaders(
            dataset_name=dataset_name,
            data_root=DATA_ROOT,
            batch_size=1,
            num_workers=num_workers,
        )
        image, _ = next(iter(test_loader))
        figure, axes = plt.subplots(
            1,
            len(teachers),
            figsize=(5 * len(teachers), 4),
        )
        axes = np.atleast_1d(axes)

        for axis, teacher_name in zip(axes, teachers):
            saved_checkpoint = checkpoint_path(teacher_name, dataset_name)
            if not saved_checkpoint.exists():
                axis.axis("off")
                axis.set_title(f"{teacher_name}\nmissing checkpoint")
                continue

            model, _ = load_model_from_checkpoint(
                teacher_name,
                dataset_name,
                num_classes,
                mode="frozen",
                device=device,
            )
            model.eval()
            with torch.no_grad():
                feature_map, _ = model.extract_features(image.to(device))
            activation = feature_map.mean(dim=1, keepdim=True)
            activation = F.interpolate(
                activation,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze().float().cpu()
            activation = (activation - activation.min()) / (
                activation.max() - activation.min() + 1e-8
            )
            axis.imshow(unnormalize(image[0]))
            axis.imshow(activation.numpy(), cmap="magma", alpha=0.45)
            axis.set_title(teacher_name)
            axis.axis("off")
            del model
            cleanup_cuda()

        if not no_title:
            figure.suptitle(f"Feature map mean activation: {dataset_name}")
        figure.tight_layout()
        figure.savefig(
            FIGURE_DIR / f"feature_maps_{dataset_name}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)
        del test_loader
        cleanup_cuda()


def train_finetuned_teachers(
    teachers: list[str],
    datasets: list[str],
    num_workers: int,
    device: str,
) -> None:
    selected_teachers = [
        teacher_name
        for teacher_name in teachers
        if teacher_name in FINETUNE_TEACHERS
    ]
    unsupported = sorted(set(teachers) - set(selected_teachers))
    for teacher_name in unsupported:
        print(f"Skipping unsupported fine-tuning teacher: {teacher_name}")

    for teacher_name in selected_teachers:
        for dataset_name in datasets:
            config = run_config(
                DEFAULTS_FT,
                RUN_OVERRIDES_FT,
                teacher_name,
                dataset_name,
                num_epochs=None,
            )
            output_checkpoint = checkpoint_path(
                teacher_name,
                dataset_name,
                mode="finetune",
            )

            frozen_checkpoint = checkpoint_path(teacher_name, dataset_name)
            if not frozen_checkpoint.exists():
                print(
                    f"Skipping fine-tuning without frozen warm-up: "
                    f"{frozen_checkpoint}"
                )
                continue

            print("=" * 100)
            print(
                f"[FT] Teacher: {teacher_name} | "
                f"Dataset: {dataset_name} | cfg: {config}"
            )
            set_seed(SEED)
            train_loader, val_loader, _, num_classes = get_dataloaders(
                dataset_name=dataset_name,
                data_root=DATA_ROOT,
                batch_size=config["batch_size"],
                num_workers=num_workers,
            )
            model = TeacherModel(
                backbone_name=teacher_name,
                num_classes=num_classes,
            )
            checkpoint = torch.load(frozen_checkpoint, map_location="cpu")
            model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
            print(
                f"Loaded frozen head from {frozen_checkpoint.name} "
                f"(best val acc: {checkpoint.get('best_val_acc', np.nan):.2f}%)"
            )
            model.unfreeze_top(config["n_blocks"])
            describe_model(model, train_loader, device)
            history = train_teacher_finetune(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config["num_epochs"],
                lr=config["lr"],
                encoder_lr=config["encoder_lr"],
                lr_decay=config["lr_decay"],
                weight_decay=config["weight_decay"],
                label_smoothing=config["label_smoothing"],
                n_blocks=config["n_blocks"],
                patience=config["patience"],
                device=device,
                checkpoint_path=str(output_checkpoint),
            )
            save_training_history_csv(
                history=history,
                save_path=history_csv_path(
                    teacher_name,
                    dataset_name,
                    mode="finetune",
                ),
                teacher_name=teacher_name,
                dataset_name=dataset_name,
            )
            write_combined_histories(
                selected_teachers,
                datasets,
                mode="finetune",
            )
            figure = plot_training_curves(
                history=history,
                title=f"{teacher_name} on {dataset_name} (fine-tuned)",
                save_path=FIGURE_DIR
                / f"teacher_finetune_{teacher_name}_{dataset_name}.png",
            )
            plt.close(figure)
            del model, train_loader, val_loader
            cleanup_cuda()


def write_comparison_table() -> None:
    frozen_csv = TABLE_DIR / "teacher_results.csv"
    finetune_csv = TABLE_DIR / "teacher_results_finetune.csv"
    frames = []
    if frozen_csv.exists():
        frozen_frame = pd.read_csv(frozen_csv)
        frozen_frame.insert(0, "Mode", "frozen")
        frozen_frame["Blocks Unfrozen"] = 0
        frames.append(frozen_frame)
    if finetune_csv.exists():
        frames.append(pd.read_csv(finetune_csv))
    if not frames:
        print("No result CSVs found. Run training or evaluation first.")
        return

    comparison_frame = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["Teacher", "Dataset", "Mode"])
        .reset_index(drop=True)
    )
    comparison_frame.to_csv(
        TABLE_DIR / "teacher_results_comparison.csv",
        index=False,
    )
    print("Saved outputs/tables/teacher_results_comparison.csv")


def main() -> None:
    args = parse_args()
    ensure_directories()
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected_datasets = args.datasets or list(DATASETS)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {device}")

    if args.render_feature_maps_only:
        render_feature_maps(
            teachers=list(TEACHERS),
            datasets=selected_datasets,
            num_workers=NUM_WORKERS,
            device=device,
            no_title=args.no_title,
        )
        return

    if not args.skip_frozen:
        train_frozen_teachers(
            teachers=list(TEACHERS),
            datasets=selected_datasets,
            num_workers=NUM_WORKERS,
            device=device,
        )
        evaluate_teachers(
            teachers=list(TEACHERS),
            datasets=selected_datasets,
            num_workers=NUM_WORKERS,
            device=device,
            mode="frozen",
        )
        render_feature_maps(
            teachers=list(TEACHERS),
            datasets=selected_datasets,
            num_workers=NUM_WORKERS,
            device=device,
            no_title=args.no_title,
        )

    if not args.skip_finetune:
        finetune_teachers = list(FINETUNE_TEACHERS)
        train_finetuned_teachers(
            teachers=finetune_teachers,
            datasets=selected_datasets,
            num_workers=NUM_WORKERS,
            device=device,
        )

        evaluate_teachers(
            teachers=finetune_teachers,
            datasets=selected_datasets,
            num_workers=NUM_WORKERS,
            device=device,
            mode="finetune",
        )
        write_comparison_table()


if __name__ == "__main__":
    main()
