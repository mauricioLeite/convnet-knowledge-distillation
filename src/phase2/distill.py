"""Single-stage distillation for the teacher-head student (grouped trainer).

The student (``student_teacher_head.StudentTeacherHead``) ends in a
``[teacher_dim, 7, 7]`` conv map; ``project(x)`` global-average-pools it to a
post-GAP vector, and ``forward(x)`` runs the glued teacher head.

Training is end-to-end with a combinable loss::

    loss = mse_weight * MSE(student.project(x), teacher_post_gap(x))
         + ce_weight  * CE(student(x), y)
         + kd_weight  * KD_KL(student(x), teacher(x); T)

By default only the MSE (post-GAP feature distillation) term is active; setting
``ce_weight`` / ``kd_weight`` > 0 brings the glued head into the loss.

Grouped training
----------------
The teacher is frozen, so its target ``ft`` for a given batch is identical for
every student that distills from it. :func:`train_student_group` exploits this:
each batch runs the teacher **once** and reuses the result across all students of
that teacher (and across the optional KD logits), instead of one teacher forward
per student. Per-epoch augmentation is preserved -- every student sees the same
freshly-augmented batch and is matched against the teacher target for that exact
view. Students are stepped sequentially within a batch (independent optimizers /
schedulers / early-stopping), so peak activation memory is ~one student's graph.
"""

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def teacher_encode(model: nn.Module, x: torch.Tensor, kind: str) -> torch.Tensor:
    """Distillation target: the representation the teacher's classifier consumes.

    Post-GAP ``[teacher_dim]`` for ResNet-50; post-GAP **+ LayerNorm** for
    ConvNeXt-Base (its head normalizes the pooled map before the Linear); and for
    VGG-16 the flattened ``avgpool(7)`` map (``512*7*7 = 25088``), which is what
    its classifier actually uses -- not a post-GAP vector.

    ``StudentTeacherHead.pooled_feature`` reproduces this same tail on the
    student's conv map -- keep the two in sync.
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
def teacher_forward(teacher: nn.Module, x: torch.Tensor, kind: str, need_logits: bool):
    """One teacher pass per batch, shared across every student of that teacher.

    Returns ``(features, logits_or_None)``. ``features`` is the distillation
    target from :func:`teacher_encode`; ``logits`` is only computed when a KD
    term needs it (a second teacher pass -- KD is off by default).
    """
    features = teacher_encode(teacher, x, kind)
    logits = teacher(x) if need_logits else None
    return features, logits


class _StudentState:
    """Per-student training state for grouped (shared-teacher) distillation."""

    def __init__(self, student, save_path, sid, optimizer, scheduler):
        self.student = student
        self.save_path = save_path
        self.id = sid
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.amp.GradScaler("cuda")
        self.best_val_acc = -1.0
        self.best_val_mse = float("inf")
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.active = True
        self.run_loss = 0.0
        self.train_correct = 0
        self.n = 0

    def reset_epoch(self):
        self.run_loss = 0.0
        self.train_correct = 0
        self.n = 0


@torch.no_grad()
def evaluate_group(teacher, students, loader, kind, device,
                   mse_weight=1.0, ce_weight=0.0, kd_weight=0.0, T=4.0):
    """Evaluates several students against one shared teacher in a single pass.

    Returns a list of ``(val_loss, val_mse, val_acc)`` aligned with ``students``.
    The teacher target (and optional KD logits) is computed once per batch and
    reused across all students -- the per-student math matches the original
    single-student ``evaluate``.
    """
    for student in students:
        student.eval()
    k = len(students)
    loss_sum = [0.0] * k
    mse_sum = [0.0] * k
    correct = [0] * k
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bs = x.size(0)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            ft, teacher_logits = teacher_forward(teacher, x, kind, kd_weight > 0)
            for i, student in enumerate(students):
                fmap = student.encoder(x)
                fs = student.pooled_feature(fmap)  # mirrors teacher_encode's tail
                mse = F.mse_loss(fs.float(), ft.float())
                logits = student.classifier(fmap)
                loss = mse_weight * mse
                if ce_weight > 0:
                    loss = loss + ce_weight * F.cross_entropy(logits, y)
                if kd_weight > 0:
                    loss = loss + kd_weight * F.kl_div(
                        F.log_softmax(logits.float() / T, dim=1),
                        F.softmax(teacher_logits.float() / T, dim=1),
                        reduction="batchmean",
                    ) * (T * T)
                loss_sum[i] += loss.item() * bs
                mse_sum[i] += mse.item() * bs
                correct[i] += (logits.argmax(1) == y).sum().item()
        n += bs
    return [(loss_sum[i] / n, mse_sum[i] / n, correct[i] / n) for i in range(k)]


def train_student_group(
    teacher: nn.Module,
    students: list[nn.Module],
    save_paths: list[str],
    ids: list[str],
    train_loader,
    val_loader,
    kind: str,
    epochs: int,
    device: torch.device,
    encoder_lr: float = 1e-3,
    classifier_lr: float = 1e-3,
    weight_decay: float = 1e-2,
    mse_weight: float = 1.0,
    ce_weight: float = 0.0,
    kd_weight: float = 0.0,
    T: float = 4.0,
    patience: int = 5,
) -> list[dict]:
    """End-to-end single-stage training of all students sharing one teacher.

    The teacher runs once per batch; every (still-active) student trains against
    that shared target with its own optimizer/scheduler/early-stopping and saves
    its own best-by-val_acc checkpoint. Returns a list of result dicts
    (``best_val_acc``, ``val_mse``, ``best_val_loss``) aligned with ``students``.
    """
    teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Build per-student state: encoder vs. head get separate LRs (matches the
    # original single-student split), an AdamW + cosine schedule, and a scaler.
    states: list[_StudentState] = []
    for student, save_path, sid in zip(students, save_paths, ids):
        student.to(device)
        encoder_params = [p for name, p in student.named_parameters()
                          if "classifier" not in name and p.requires_grad]
        classifier_params = [p for name, p in student.named_parameters()
                             if "classifier" in name and p.requires_grad]
        groups = [{"params": encoder_params, "lr": encoder_lr}]
        if classifier_params:
            groups.append({"params": classifier_params, "lr": classifier_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        states.append(_StudentState(student, save_path, sid, optimizer, scheduler))

    use_head_loss = ce_weight > 0 or kd_weight > 0

    for epoch in range(epochs):
        active = [s for s in states if s.active]
        if not active:
            break
        for s in active:
            s.student.train()
            s.reset_epoch()

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # One teacher pass for the whole group; reused by every student.
            with torch.amp.autocast("cuda", dtype=torch.float16):
                ft, teacher_logits = teacher_forward(teacher, x, kind, kd_weight > 0)

            for s in active:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    # Run the student encoder once; derive both the projected
                    # (post-GAP) features and the head logits from the same map.
                    fmap = s.student.encoder(x)
                    fs = s.student.pooled_feature(fmap)  # mirrors teacher_encode's tail
                    loss = mse_weight * F.mse_loss(fs.float(), ft.float())

                    if use_head_loss:
                        logits = s.student.classifier(fmap)
                        if ce_weight > 0:
                            loss = loss + ce_weight * F.cross_entropy(logits, y)
                        if kd_weight > 0:
                            loss = loss + kd_weight * F.kl_div(
                                F.log_softmax(logits.float() / T, dim=1),
                                F.softmax(teacher_logits.float() / T, dim=1),
                                reduction="batchmean",
                            ) * (T * T)
                    else:
                        # MSE-only: logits aren't in the loss, so skip the head's
                        # autograd graph -- still need them for train_acc.
                        with torch.no_grad():
                            logits = s.student.classifier(fmap)

                s.optimizer.zero_grad()
                s.scaler.scale(loss).backward()
                s.scaler.step(s.optimizer)
                s.scaler.update()
                s.run_loss += loss.item() * x.size(0)
                s.train_correct += (logits.argmax(1) == y).sum().item()
                s.n += x.size(0)

        for s in active:
            s.scheduler.step()

        # Single shared-teacher pass over the val set for all active students.
        val_results = evaluate_group(
            teacher, [s.student for s in active], val_loader, kind, device,
            mse_weight=mse_weight, ce_weight=ce_weight, kd_weight=kd_weight, T=T,
        )
        for s, (val_loss, val_mse, val_acc) in zip(active, val_results):
            train_loss = s.run_loss / s.n
            train_acc = s.train_correct / s.n
            enc_lr = s.scheduler.get_last_lr()[0]
            cls_lr = s.scheduler.get_last_lr()[-1]
            print(f"  [{s.id}] epoch {epoch:02d} | train_loss {train_loss:.5f} | train_acc {train_acc:.4f} | "
                  f"val_loss {val_loss:.5f} | val_acc {val_acc:.4f} | "
                  f"enc_lr {enc_lr:.2e} | cls_lr {cls_lr:.2e}")

            # Save on best accuracy; reset patience on improved accuracy OR
            # improved full validation loss.
            improved_acc = val_acc > s.best_val_acc
            improved_loss = val_loss < s.best_val_loss
            if improved_acc:
                s.best_val_acc = val_acc
                s.best_val_mse = val_mse
                torch.save(s.student.state_dict(), s.save_path)
            if improved_loss:
                s.best_val_loss = val_loss
            if improved_acc or improved_loss:
                s.epochs_no_improve = 0
            else:
                s.epochs_no_improve += 1
                if s.epochs_no_improve >= patience:
                    s.active = False
                    print(f"  [{s.id}] early stopping at epoch {epoch} "
                          f"(best_val_acc={s.best_val_acc:.4f})")

    for s in states:
        print(f"  [{s.id}] best val_acc = {s.best_val_acc:.4f} (val_mse {s.best_val_mse:.5f}, "
              f"best val_loss {s.best_val_loss:.5f}) -> {s.save_path}")
    return [{"best_val_acc": s.best_val_acc, "val_mse": s.best_val_mse,
             "best_val_loss": s.best_val_loss} for s in states]
