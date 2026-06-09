"""Single-stage distillation for the teacher-head student.

The student (``student_teacher_head.StudentTeacherHead``) ends in a
``[teacher_dim, 7, 7]`` conv map; ``project(x)`` global-average-pools it to a
post-GAP vector, and ``forward(x)`` runs the glued teacher head.

Training is end-to-end with a combinable loss::

    loss = mse_weight * MSE(student.project(x), teacher_post_gap(x))
         + ce_weight  * CE(student(x), y)
         + kd_weight  * KD_KL(student(x), teacher(x); T)

By default only the MSE (post-GAP feature distillation) term is active; setting
``ce_weight`` / ``kd_weight`` > 0 brings the glued head into the loss.
"""

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def teacher_encode(model: nn.Module, x: torch.Tensor, kind: str) -> torch.Tensor:
    """Distillation target: the representation the teacher's classifier consumes.

    Post-GAP ``[teacher_dim]`` for ResNet/ConvNeXt; for VGG the flattened
    ``avgpool(7)`` map (``512*7*7 = 25088``), which is what its classifier
    actually uses -- not a post-GAP vector.
    """
    model.eval()
    if kind == "resnet50":
        x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
        x = model.avgpool(x)
    elif kind == "convnext_base":
        x = model.features(x)
        x = model.avgpool(x)
        x = model.classifier[0](x)  # LayerNorm2d: post-GAP feature the student matches
    elif kind == "vgg16":
        x = model.features(x)
        x = model.avgpool(x)  # AdaptiveAvgPool2d(7) -> [512, 7, 7]; classifier consumes the flattened map
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return torch.flatten(x, 1)


@torch.no_grad()
def evaluate(teacher, student, loader, kind, device,
             mse_weight=1.0, ce_weight=0.0, kd_weight=0.0, T=4.0) -> tuple[float, float, float]:
    """Returns ``(val_loss, val_mse, val_acc)`` on the loader.

    ``val_loss`` is the *full* combined loss using the same weights as training
    (MSE + optional CE + optional KD); ``val_mse`` is the post-GAP feature MSE
    alone; ``val_acc`` uses the glued head.
    """
    student.eval()
    loss_sum = 0.0
    mse_sum = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            fmap = student.encoder(x)
            fs = torch.flatten(student.classifier[0](fmap), 1)
            ft = teacher_encode(teacher, x, kind)
            mse = F.mse_loss(fs.float(), ft.float())
            logits = student.classifier(fmap)
            loss = mse_weight * mse
            if ce_weight > 0:
                loss = loss + ce_weight * F.cross_entropy(logits, y)
            if kd_weight > 0:
                teacher_logits = teacher(x)
                loss = loss + kd_weight * F.kl_div(
                    F.log_softmax(logits.float() / T, dim=1),
                    F.softmax(teacher_logits.float() / T, dim=1),
                    reduction="batchmean",
                ) * (T * T)
        bs = x.size(0)
        loss_sum += loss.item() * bs
        mse_sum += mse.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        n += bs
    return loss_sum / n, mse_sum / n, correct / n


def train_student(
    teacher: nn.Module,
    student: nn.Module,
    train_loader,
    val_loader,
    kind: str,
    epochs: int,
    save_path: str,
    device: torch.device,
    encoder_lr: float = 1e-3,
    classifier_lr: float = 1e-3,
    weight_decay: float = 1e-2,
    mse_weight: float = 1.0,
    ce_weight: float = 0.0,
    kd_weight: float = 0.0,
    T: float = 4.0,
    patience: int = 5,
) -> dict:
    """End-to-end single-stage training. Saves the best (by val_acc) checkpoint.

    Returns a dict with ``best_val_acc`` and the ``val_mse`` at that epoch.
    """
    teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False

    student.to(device)

    # Encoder (everything but the glued head) vs. head, with separate LRs.
    encoder_params = [p for name, p in student.named_parameters()
                      if "classifier" not in name and p.requires_grad]
    classifier_params = [p for name, p in student.named_parameters()
                         if "classifier" in name and p.requires_grad]
    groups = [{"params": encoder_params, "lr": encoder_lr}]
    if classifier_params:
        groups.append({"params": classifier_params, "lr": classifier_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    use_head_loss = ce_weight > 0 or kd_weight > 0
    best_val_acc = -1.0
    best_val_mse = float("inf")
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        student.train()
        run_loss = 0.0
        train_correct = 0
        n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                # Run the student encoder once; derive both the projected
                # (post-GAP) features and the head logits from the same map.
                fmap = student.encoder(x)
                fs = torch.flatten(student.classifier[0](fmap), 1)
                ft = teacher_encode(teacher, x, kind)
                loss = mse_weight * F.mse_loss(fs.float(), ft.float())

                if use_head_loss:
                    logits = student.classifier(fmap)
                    if ce_weight > 0:
                        loss = loss + ce_weight * F.cross_entropy(logits, y)
                    if kd_weight > 0:
                        teacher_logits = teacher(x)
                        loss = loss + kd_weight * F.kl_div(
                            F.log_softmax(logits.float() / T, dim=1),
                            F.softmax(teacher_logits.float() / T, dim=1),
                            reduction="batchmean",
                        ) * (T * T)
                else:
                    # MSE-only: logits aren't in the loss, so skip the head's
                    # autograd graph -- still need them for train_acc.
                    with torch.no_grad():
                        logits = student.classifier(fmap)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        scheduler.step()

        train_loss = run_loss / n
        train_acc = train_correct / n
        val_loss, val_mse, val_acc = evaluate(
            teacher, student, val_loader, kind, device,
            mse_weight=mse_weight, ce_weight=ce_weight, kd_weight=kd_weight, T=T,
        )
        enc_lr = scheduler.get_last_lr()[0]
        cls_lr = scheduler.get_last_lr()[-1]
        print(f"{kind}] epoch {epoch:02d} | train_loss {train_loss:.5f} | train_acc {train_acc:.4f} | "
              f"val_loss {val_loss:.5f} | val_acc {val_acc:.4f} | "
              f"enc_lr {enc_lr:.2e} | cls_lr {cls_lr:.2e}")

        # Save on best accuracy; reset patience on improved accuracy OR improved
        # full validation loss.
        improved_acc = val_acc > best_val_acc
        improved_loss = val_loss < best_val_loss
        if improved_acc:
            best_val_acc = val_acc
            best_val_mse = val_mse
            torch.save(student.state_dict(), save_path)
        if improved_loss:
            best_val_loss = val_loss
        if improved_acc or improved_loss:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best_val_acc={best_val_acc:.4f})")
                break

    print(f"best val_acc = {best_val_acc:.4f} (val_mse {best_val_mse:.5f}, "
          f"best val_loss {best_val_loss:.5f}) -> {save_path}\n")
    return {"best_val_acc": best_val_acc, "val_mse": best_val_mse, "best_val_loss": best_val_loss}
