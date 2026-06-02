"""Utilities for data loading, reproducibility, profiling, plotting, and CSVs."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

_DEFAULT_SEED = 291652
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TinyImageNetValDataset(Dataset):
    """Tiny ImageNet validation split labeled by ``val_annotations.txt``.

    Args:
        val_root: Path to ``tiny-imagenet-200/val``.
        class_to_idx: Mapping from WordNet IDs to integer class labels.
        transform: Optional transform applied to each image.
    """

    def __init__(
        self,
        val_root: str | Path,
        class_to_idx: dict[str, int],
        transform: Optional[Any] = None,
    ) -> None:
        self.root = Path(val_root)
        self.images_dir = self.root / "images"
        self.transform = transform
        self.class_to_idx = dict(class_to_idx)
        self.classes = [
            name
            for name, _ in sorted(
                class_to_idx.items(),
                key=lambda item: item[1],
            )
        ]
        self.samples: list[tuple[Path, int]] = []

        annotations_path = self.root / "val_annotations.txt"
        if not annotations_path.exists():
            raise FileNotFoundError(
                f"Missing Tiny ImageNet annotations: {annotations_path}"
            )

        with annotations_path.open("r", encoding="utf-8") as annotations_file:
            for line in annotations_file:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                filename, class_name = parts[0], parts[1]
                image_path = self.images_dir / filename
                if class_name in self.class_to_idx and image_path.exists():
                    self.samples.append(
                        (image_path, self.class_to_idx[class_name])
                    )

        if not self.samples:
            raise RuntimeError(
                f"No labeled Tiny ImageNet validation images found in {self.root}"
            )
        self.targets = [target for _, target in self.samples]

    def __len__(self) -> int:
        """Returns the number of labeled validation images."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Loads an image and its integer target.

        Args:
            index: Dataset index.

        Returns:
            Transformed image and target label.
        """
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def set_seed(seed: int) -> None:
    """Sets Python, NumPy, and PyTorch random seeds.

    Args:
        seed: Seed value for all supported random generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Counts trainable parameters.

    Args:
        model: PyTorch module.

    Returns:
        Number of parameters with ``requires_grad=True``.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_total_parameters(model: nn.Module) -> int:
    """Counts all model parameters.

    Args:
        model: PyTorch module.

    Returns:
        Total parameter count.
    """
    return sum(parameter.numel() for parameter in model.parameters())


def compute_flops(
    model: nn.Module,
    input_size: tuple[int, int, int, int] = (1, 3, 224, 224),
) -> float:
    """Computes model complexity in GFLOPs using fvcore.

    Args:
        model: PyTorch module to analyze.
        input_size: Dummy input shape.

    Returns:
        Total GFLOPs for one forward pass.
    """
    from fvcore.nn import FlopCountAnalysis

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    was_training = model.training
    model.eval()
    dummy_input = torch.randn(input_size, device=device)
    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy_input).total()
    if was_training:
        model.train()
    return float(flops) / 1e9


def plot_training_curves(
    history: dict[str, Any],
    title: str,
    save_path: Optional[str | Path] = None,
) -> Any:
    """Plots train/validation loss and accuracy curves.

    Args:
        history: History returned by ``train_teacher_classifier``.
        title: Figure title.
        save_path: Optional output image path. Bare filenames are saved under
            ``results/figures``.

    Returns:
        Created matplotlib figure.
    """
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.get("train_loss", []), label="Train")
    axes[0].plot(epochs, history.get("val_loss", []), label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.get("train_acc", []), label="Train")
    axes[1].plot(epochs, history.get("val_acc", []), label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.suptitle(title)
    figure.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        if output_path.parent == Path("."):
            output_path = Path("outputs") / "figures" / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")

    return figure


def save_training_history_csv(
    history: dict[str, Any],
    save_path: str | Path,
    teacher_name: str,
    dataset_name: str,
) -> pd.DataFrame:
    """Saves per-epoch metrics used to draw a training-curve figure.

    Args:
        history: History returned by ``train_teacher_classifier``.
        save_path: Destination CSV path.
        teacher_name: Teacher backbone identifier.
        dataset_name: Dataset identifier.

    Returns:
        DataFrame written to disk, suitable for aggregate concatenation.

    Raises:
        ValueError: If the four metric series do not have equal lengths.
    """
    train_loss = history.get("train_loss", [])
    train_acc = history.get("train_acc", [])
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_acc", [])
    lengths = {len(train_loss), len(train_acc), len(val_loss), len(val_acc)}
    if len(lengths) != 1:
        raise ValueError("Training history metric lists must have equal lengths.")

    history_frame = pd.DataFrame(
        {
            "Teacher": teacher_name,
            "Dataset": dataset_name,
            "Epoch": range(1, len(train_loss) + 1),
            "Train Loss": train_loss,
            "Train Acc (%)": train_acc,
            "Val Loss": val_loss,
            "Val Acc (%)": val_acc,
        }
    )
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_frame.to_csv(output_path, index=False)
    return history_frame


def get_dataloaders(
    dataset_name: str,
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Builds train, validation, and test DataLoaders for a supported dataset.

    Args:
        dataset_name: Dataset identifier or common alias.
        data_root: Root directory containing dataset folders.
        batch_size: Batch size for all DataLoaders.
        num_workers: Number of DataLoader workers.

    Returns:
        Tuple ``(train_loader, val_loader, test_loader, num_classes)``.
    """
    canonical_name = _normalize_dataset_name(dataset_name)
    random_resized_crop_scale = (
        (0.7, 1.0)
        if canonical_name == "stanford-cars"
        else (0.08, 1.0)
    )
    train_transform, eval_transform = _build_transforms(
        random_resized_crop_scale=random_resized_crop_scale
    )
    root = Path(data_root)

    if canonical_name == "cifar-100":
        datasets_and_classes = _load_cifar100(
            root,
            train_transform,
            eval_transform,
        )
    elif canonical_name == "flowers-102":
        datasets_and_classes = _load_flowers102(
            root,
            train_transform,
            eval_transform,
        )
    elif canonical_name == "stanford-cars":
        datasets_and_classes = _load_stanford_cars(
            root,
            train_transform,
            eval_transform,
        )
    elif canonical_name == "tiny-imagenet-200":
        datasets_and_classes = _load_tiny_imagenet(
            root,
            train_transform,
            eval_transform,
        )
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    train_dataset, val_dataset, test_dataset, num_classes = datasets_and_classes
    train_loader = _make_loader(
        train_dataset,
        batch_size,
        num_workers,
        shuffle=True,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size,
        num_workers,
        shuffle=False,
    )
    test_loader = _make_loader(
        test_dataset,
        batch_size,
        num_workers,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader, num_classes


def _normalize_dataset_name(dataset_name: str) -> str:
    name = dataset_name.lower().replace("_", "-")
    aliases = {
        "cifar100": "cifar-100",
        "cifar-100": "cifar-100",
        "flowers102": "flowers-102",
        "flowers-102": "flowers-102",
        "stanfordcars": "stanford-cars",
        "stanford-cars": "stanford-cars",
        "tinyimagenet": "tiny-imagenet-200",
        "tiny-imagenet": "tiny-imagenet-200",
        "tiny-imagenet-200": "tiny-imagenet-200",
    }
    return aliases.get(name, name)


def _build_transforms(
    random_resized_crop_scale: tuple[float, float] = (0.08, 1.0),
) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=random_resized_crop_scale,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def _resolve_dataset_root(data_root: Path, folder_name: str) -> Path:
    candidate = data_root / folder_name
    return candidate if candidate.exists() else data_root


def _split_indices(
    length: int,
    val_fraction: float = 0.1,
    seed: int = _DEFAULT_SEED,
) -> tuple[list[int], list[int]]:
    if length < 2:
        raise ValueError("Need at least two samples for a train/validation split.")
    val_size = min(max(1, int(round(length * val_fraction))), length - 1)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    return indices[val_size:], indices[:val_size]


def _copy_subset_metadata(
    subset: Subset,
    dataset: Dataset,
    indices: list[int],
) -> Subset:
    for attribute in ("classes", "class_to_idx"):
        if hasattr(dataset, attribute):
            setattr(subset, attribute, getattr(dataset, attribute))
    targets = _get_targets(dataset)
    if targets is not None:
        setattr(subset, "targets", [targets[index] for index in indices])
    return subset


def _make_subset(dataset: Dataset, indices: list[int]) -> Subset:
    return _copy_subset_metadata(Subset(dataset, indices), dataset, indices)


def _get_targets(dataset: Dataset) -> list[int] | None:
    for attribute in ("targets", "labels", "_labels"):
        if hasattr(dataset, attribute):
            return [int(value) for value in getattr(dataset, attribute)]
    if hasattr(dataset, "samples"):
        return [int(sample[1]) for sample in getattr(dataset, "samples")]
    return None


def _seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(_DEFAULT_SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _load_cifar100(
    data_root: Path,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> tuple[Dataset, Dataset, Dataset, int]:
    root = _resolve_dataset_root(data_root, "cifar-100")
    train_full = datasets.CIFAR100(
        root=str(root),
        train=True,
        transform=train_transform,
        download=False,
    )
    val_full = datasets.CIFAR100(
        root=str(root),
        train=True,
        transform=eval_transform,
        download=False,
    )
    test_dataset = datasets.CIFAR100(
        root=str(root),
        train=False,
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


def _load_flowers102(
    data_root: Path,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> tuple[Dataset, Dataset, Dataset, int]:
    root = _resolve_dataset_root(data_root, "flowers-102")
    return (
        datasets.Flowers102(
            root=str(root),
            split="train",
            transform=train_transform,
            download=False,
        ),
        datasets.Flowers102(
            root=str(root),
            split="val",
            transform=eval_transform,
            download=False,
        ),
        datasets.Flowers102(
            root=str(root),
            split="test",
            transform=eval_transform,
            download=False,
        ),
        102,
    )


def _load_stanford_cars(
    data_root: Path,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> tuple[Dataset, Dataset, Dataset, int]:
    root = _resolve_dataset_root(data_root, "stanford-cars")
    train_dir = root / "train"
    test_dir = root / "test"

    if train_dir.exists() and test_dir.exists():
        train_full = datasets.ImageFolder(
            root=str(train_dir),
            transform=train_transform,
        )
        val_full = datasets.ImageFolder(
            root=str(train_dir),
            transform=eval_transform,
        )
        test_dataset = datasets.ImageFolder(
            root=str(test_dir),
            transform=eval_transform,
        )
        num_classes = len(train_full.classes)
    else:
        train_full = datasets.StanfordCars(
            root=str(root),
            split="train",
            transform=train_transform,
            download=False,
        )
        val_full = datasets.StanfordCars(
            root=str(root),
            split="train",
            transform=eval_transform,
            download=False,
        )
        test_dataset = datasets.StanfordCars(
            root=str(root),
            split="test",
            transform=eval_transform,
            download=False,
        )
        num_classes = len(getattr(train_full, "classes", [])) or 196

    train_indices, val_indices = _split_indices(len(train_full))
    return (
        _make_subset(train_full, train_indices),
        _make_subset(val_full, val_indices),
        test_dataset,
        num_classes,
    )


def _load_tiny_imagenet(
    data_root: Path,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> tuple[Dataset, Dataset, Dataset, int]:
    root = _resolve_dataset_root(data_root, "tiny-imagenet-200")
    train_root = root / "train"
    val_root = root / "val"
    train_full = datasets.ImageFolder(
        root=str(train_root),
        transform=train_transform,
    )
    val_full = datasets.ImageFolder(
        root=str(train_root),
        transform=eval_transform,
    )
    test_dataset = TinyImageNetValDataset(
        val_root=val_root,
        class_to_idx=train_full.class_to_idx,
        transform=eval_transform,
    )
    train_indices, val_indices = _split_indices(len(train_full))
    return (
        _make_subset(train_full, train_indices),
        _make_subset(val_full, val_indices),
        test_dataset,
        len(train_full.classes),
    )

