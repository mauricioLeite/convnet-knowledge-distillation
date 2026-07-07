from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.phase1.teachers import TeacherModel

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
    patience: int = 0,
) -> dict[str, list[float] | float | int]:
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
    epochs_no_improve = 0

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
            epochs_no_improve = 0
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
        else:
            epochs_no_improve += 1

        print(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%"
        )

        if patience > 0 and epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

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
    patience: int = 0,
) -> dict[str, list[float] | float | int]:
    device_obj = _resolve_device(device)
    model.to(device_obj)
    use_cuda = device_obj.type == "cuda"

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
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
    epochs_no_improve = 0

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
            epochs_no_improve = 0
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
        else:
            epochs_no_improve += 1

        print(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%"
        )

        if patience > 0 and epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    return history
