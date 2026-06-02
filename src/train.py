"""Training and evaluation loops for Phase 1 teacher classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.teachers import TeacherModel


def _resolve_device(device: str | torch.device) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def evaluate(
    model: TeacherModel,
    loader: DataLoader,
    device: str | torch.device,
) -> tuple[float, float]:
    """Evaluates a teacher classifier.

    Args:
        model: Teacher model with a classifier head.
        loader: DataLoader yielding ``(images, labels)`` batches.
        device: Torch device name or object.

    Returns:
        Tuple ``(average_loss, accuracy_percent)``.
    """
    device_obj = _resolve_device(device)
    criterion = nn.CrossEntropyLoss()
    model.to(device_obj)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device_obj, non_blocking=True)
            targets = targets.to(device_obj, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device_obj.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size

    average_loss = total_loss / max(total_samples, 1)
    accuracy = 100.0 * total_correct / max(total_samples, 1)
    return average_loss, accuracy


def train_teacher_classifier(
    model: TeacherModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    checkpoint_path: Optional[str] = None,
) -> dict[str, list[float] | float | int]:
    """Trains only the teacher model's linear classifier head.

    Args:
        model: Teacher model whose encoder is frozen.
        train_loader: DataLoader for training batches.
        val_loader: DataLoader for validation batches.
        num_epochs: Number of training epochs.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        device: Torch device name.
        checkpoint_path: Optional path for the best classifier checkpoint.

    Returns:
        History dictionary containing per-epoch train/validation metrics,
        ``best_val_acc``, and ``best_epoch``. Accuracies are percentages.
    """
    device_obj = _resolve_device(device)
    model.to(device_obj)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
    )

    history: dict[str, list[float] | float | int] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_acc": 0.0,
        "best_epoch": 0,
    }
    best_val_acc = -1.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=False,
        )
        for images, targets in progress:
            images = images.to(device_obj, non_blocking=True)
            targets = targets.to(device_obj, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device_obj.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size
            progress.set_postfix(
                loss=f"{total_loss / max(total_samples, 1):.4f}",
                acc=f"{100.0 * total_correct / max(total_samples, 1):.2f}%",
            )

        scheduler.step()
        train_loss = total_loss / max(total_samples, 1)
        train_acc = 100.0 * total_correct / max(total_samples, 1)
        val_loss, val_acc = evaluate(model, val_loader, device_obj)

        for key, value in (
            ("train_loss", train_loss),
            ("train_acc", train_acc),
            ("val_loss", val_loss),
            ("val_acc", val_acc),
        ):
            values = history[key]
            assert isinstance(values, list)
            values.append(value)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history["best_val_acc"] = best_val_acc
            history["best_epoch"] = epoch
            if checkpoint_path is not None:
                checkpoint_file = Path(checkpoint_path)
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "classifier_state_dict": model.classifier.state_dict(),
                        "backbone_name": model.backbone_name,
                        "num_classes": model.num_classes,
                        "feature_dim": model.feature_dim,
                        "best_val_acc": best_val_acc,
                        "epoch": epoch,
                    },
                    checkpoint_file,
                )

        print(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%"
        )

    return history


def train_teacher_finetune(
    model: TeacherModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    lr: float = 1e-3,
    encoder_lr: float = 5e-5,
    lr_decay: float = 0.8,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    n_blocks: int = 1,
    device: str = "cuda",
    checkpoint_path: Optional[str] = None,
) -> dict[str, list[float] | float | int]:
    """Trains the teacher with a partially unfrozen encoder (light fine-tuning).

    The head receives the full learning rate. Unfrozen encoder blocks receive
    ``encoder_lr``, scaled by ``lr_decay`` per block going deeper (LLRD). The
    encoder must have been partially unfrozen via ``model.unfreeze_top()``
    before calling this function. The encoder is kept in eval mode throughout so
    that BatchNorm running statistics are never updated.

    Args:
        model: Teacher with at least one encoder block unfrozen.
        train_loader: DataLoader for training batches.
        val_loader: DataLoader for validation batches.
        num_epochs: Number of fine-tuning epochs.
        lr: AdamW learning rate for the linear head.
        encoder_lr: Base learning rate for the topmost unfrozen encoder block.
        lr_decay: LR multiplier applied per block going deeper (LLRD).
        weight_decay: AdamW weight decay applied to all parameter groups.
        label_smoothing: Cross-entropy label smoothing (0 = standard CE).
        n_blocks: Number of encoder blocks that were unfrozen; stored in the
            checkpoint for downstream phases.
        device: Torch device name.
        checkpoint_path: Optional path for the best fine-tuned checkpoint.

    Returns:
        History dict with per-epoch train/val metrics, ``best_val_acc``, and
        ``best_epoch``. Accuracies are percentages.
    """
    device_obj = _resolve_device(device)
    model.to(device_obj)
    use_cuda = device_obj.type == "cuda"

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Build parameter groups: head at lr, unfrozen encoder blocks with LLRD.
    # Blocks (within the final encoder stage) are iterated from the top
    # outward, so the topmost unfrozen block gets encoder_lr and each deeper
    # block gets * lr_decay.
    param_groups: list[dict] = [
        {"params": list(model.classifier.parameters()), "lr": lr},
    ]
    final_stage = list(model.encoder.children())[-1]
    unfrozen_blocks = [
        block
        for block in final_stage.children()
        if any(p.requires_grad for p in block.parameters())
    ]
    for depth, block in enumerate(reversed(unfrozen_blocks)):
        block_params = [p for p in block.parameters() if p.requires_grad]
        if block_params:
            param_groups.append({
                "params": block_params,
                "lr": encoder_lr * (lr_decay ** depth),
            })

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    history: dict[str, list[float] | float | int] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_acc": 0.0,
        "best_epoch": 0,
    }
    best_val_acc = -1.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=False,
        )
        for images, targets in progress:
            images = images.to(device_obj, non_blocking=True)
            targets = targets.to(device_obj, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size
            progress.set_postfix(
                loss=f"{total_loss / max(total_samples, 1):.4f}",
                acc=f"{100.0 * total_correct / max(total_samples, 1):.2f}%",
            )

        scheduler.step()
        train_loss = total_loss / max(total_samples, 1)
        train_acc = 100.0 * total_correct / max(total_samples, 1)
        val_loss, val_acc = evaluate(model, val_loader, device_obj)

        for key, value in (
            ("train_loss", train_loss),
            ("train_acc", train_acc),
            ("val_loss", val_loss),
            ("val_acc", val_acc),
        ):
            values = history[key]
            assert isinstance(values, list)
            values.append(value)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history["best_val_acc"] = best_val_acc
            history["best_epoch"] = epoch
            if checkpoint_path is not None:
                checkpoint_file = Path(checkpoint_path)
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "classifier_state_dict": model.classifier.state_dict(),
                        "encoder_state_dict": model.encoder.state_dict(),
                        "backbone_name": model.backbone_name,
                        "num_classes": model.num_classes,
                        "feature_dim": model.feature_dim,
                        "best_val_acc": best_val_acc,
                        "epoch": epoch,
                        "finetuned": True,
                        "n_blocks_unfrozen": n_blocks,
                    },
                    checkpoint_file,
                )

        print(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%"
        )

    return history

