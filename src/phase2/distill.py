from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def teacher_forward(
    teacher: nn.Module,
    x: torch.Tensor,
    need_logits: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Returns feature maps, pooled features, and optional logits."""
    teacher.eval()
    feature_map, pooled_vector = teacher.extract_features(x)
    logits = teacher.classifier(pooled_vector) if need_logits else None
    return feature_map, pooled_vector, logits


def _pdist(e: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Pairwise euclidean distance matrix [N, N] with a zeroed diagonal."""
    e_sq = e.pow(2).sum(dim=1)
    dist = (e_sq.unsqueeze(1) + e_sq.unsqueeze(0) - 2.0 * (e @ e.t())).clamp(min=eps).sqrt()
    dist = dist.clone()
    dist[range(len(e)), range(len(e))] = 0.0
    return dist


def nrkd_loss(student_vec: torch.Tensor, teacher_vec: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Neighborhood RKD over teacher-defined k-nearest neighbors."""
    with torch.amp.autocast("cuda", enabled=False):
        s = student_vec.float()
        t = teacher_vec.float()
        N = t.size(0)

        with torch.no_grad():
            td = _pdist(t)
            td_masked = td.clone()
            td_masked[range(N), range(N)] = float("inf")

            actual_k = min(k, N - 1)
            if actual_k <= 0:
                return torch.tensor(0.0, device=student_vec.device)

            _, topk_idx = td_masked.topk(actual_k, dim=1, largest=False)

            mask = torch.zeros_like(td, dtype=torch.bool)
            mask.scatter_(1, topk_idx, True)

            td_k = td[mask]
            td_k = td_k / (td_k.mean() + 1e-8)

        sd = _pdist(s)
        sd_k = sd[mask]
        sd_k = sd_k / (sd_k.mean() + 1e-8)

        loss_d = F.smooth_l1_loss(sd_k, td_k)

        with torch.no_grad():
            te = F.normalize(t.unsqueeze(0) - t.unsqueeze(1), p=2, dim=2)
            t_angle = torch.bmm(te, te.transpose(1, 2))

        se = F.normalize(s.unsqueeze(0) - s.unsqueeze(1), p=2, dim=2)
        s_angle = torch.bmm(se, se.transpose(1, 2))

        mask_3d = mask.unsqueeze(2) & mask.unsqueeze(1)
        diag_idx = torch.arange(N)
        mask_3d[:, diag_idx, diag_idx] = False

        t_angle_k = t_angle[mask_3d]
        s_angle_k = s_angle[mask_3d]

        loss_a = F.smooth_l1_loss(s_angle_k, t_angle_k)

    return loss_d + 2.0 * loss_a

def _pooled_student_vec(student, fs: torch.Tensor) -> torch.Tensor:
    """Post-GAP vector of the student's projected representation."""
    return fs.mean(dim=(2, 3)) if student.target_mode == "pre_gap" else fs


class _StudentState:
    def __init__(self, student, save_path, sid):
        self.student = student
        self.save_path = save_path
        self.id = sid
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
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
def evaluate_group(
    teacher,
    students: list,
    loader,
    device: torch.device,
    mse_weight: float = 1.0,
    ce_weight: float = 1.0,
    kd_weight: float = 0.0,
    rkd_weight: float = 0.0,
    T: float = 4.0,
) -> list[tuple[float, float, float]]:
    """Evaluates all students in one shared-teacher pass. """
    for s in students:
        s.eval()
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
            fmap, vec, teacher_logits = teacher_forward(teacher, x, kd_weight > 0)
            for i, student in enumerate(students):
                target = fmap if student.target_mode == "pre_gap" else vec
                fs = student.project(x)
                mse = F.mse_loss(fs.float(), target.float())
                logits = student(x)
                loss = mse_weight * mse
                if ce_weight > 0:
                    loss = loss + ce_weight * F.cross_entropy(logits, y)
                if kd_weight > 0:
                    loss = loss + kd_weight * F.kl_div(
                        F.log_softmax(logits.float() / T, dim=1),
                        F.softmax(teacher_logits.float() / T, dim=1),
                        reduction="batchmean",
                    ) * (T * T)
                if rkd_weight > 0:
                    loss = loss + rkd_weight * nrkd_loss(
                        _pooled_student_vec(student, fs), vec
                    )
                loss_sum[i] += loss.item() * bs
                mse_sum[i] += mse.item() * bs
                correct[i] += (logits.argmax(1) == y).sum().item()
        n += bs
    return [(loss_sum[i] / n, mse_sum[i] / n, correct[i] / n) for i in range(k)]


def _run_phase(
    *,
    phase_label: str,
    states: list[_StudentState],
    teacher: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    eta_min: float,
    encoder_lr: float,
    classifier_lr: float,
    weight_decay: float,
    mse_weight: float,
    ce_weight: float,
    kd_weight: float,
    rkd_weight: float,
    T: float,
    patience: int,
    early_stop: bool,
) -> None:
    """Runs one training phase with fresh optimizer state per student."""
    need_kd = kd_weight > 0
    need_rkd = rkd_weight > 0
    use_head = ce_weight > 0 or need_kd

    for s in states:
        s.active = True
        s.epochs_no_improve = 0
        s.best_val_loss = float("inf")
        enc_params = [p for name, p in s.student.named_parameters()
                      if "classifier" not in name and p.requires_grad]
        head_params = [p for name, p in s.student.named_parameters()
                       if "classifier" in name and p.requires_grad]
        groups = [{"params": enc_params, "lr": encoder_lr}]
        if head_params:
            groups.append({"params": head_params, "lr": classifier_lr})
        s.optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
        s.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            s.optimizer, T_max=epochs, eta_min=eta_min)
        s.scaler = torch.amp.GradScaler("cuda")

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
            with torch.amp.autocast("cuda", dtype=torch.float16):
                fmap, vec, teacher_logits = teacher_forward(teacher, x, need_kd)

            for s in active:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    target = fmap if s.student.target_mode == "pre_gap" else vec
                    fs = s.student.project(x)
                    loss = mse_weight * F.mse_loss(fs.float(), target.float())

                    if use_head:
                        logits = s.student(x)
                        if ce_weight > 0:
                            loss = loss + ce_weight * F.cross_entropy(logits, y)
                        if need_kd:
                            loss = loss + kd_weight * F.kl_div(
                                F.log_softmax(logits.float() / T, dim=1),
                                F.softmax(teacher_logits.float() / T, dim=1),
                                reduction="batchmean",
                            ) * (T * T)
                    else:
                        with torch.no_grad():
                            logits = s.student(x)

                    if need_rkd:
                        loss = loss + rkd_weight * nrkd_loss(
                            _pooled_student_vec(s.student, fs), vec
                        )

                s.optimizer.zero_grad()
                s.scaler.scale(loss).backward()
                s.scaler.step(s.optimizer)
                s.scaler.update()
                s.run_loss += loss.item() * x.size(0)
                s.train_correct += (logits.argmax(1) == y).sum().item()
                s.n += x.size(0)

        for s in active:
            s.scheduler.step()

        val_results = evaluate_group(
            teacher, [s.student for s in active], val_loader, device,
            mse_weight=mse_weight, ce_weight=ce_weight, kd_weight=kd_weight,
            rkd_weight=rkd_weight, T=T,
        )
        for s, (val_loss, val_mse, val_acc) in zip(active, val_results):
            train_loss = s.run_loss / s.n
            train_acc = s.train_correct / s.n
            enc_lr = s.scheduler.get_last_lr()[0]
            cls_lr = s.scheduler.get_last_lr()[-1]
            print(
                f"  [{s.id}] {phase_label} epoch {epoch:02d} | train_loss {train_loss:.5f} | "
                f"train_acc {train_acc:.4f} | val_loss {val_loss:.5f} | "
                f"val_acc {val_acc:.4f} | enc_lr {enc_lr:.2e} | cls_lr {cls_lr:.2e}"
            )

            improved_acc = val_acc > s.best_val_acc
            improved_loss = val_loss < s.best_val_loss
            if improved_acc:
                s.best_val_acc = val_acc
                s.best_val_mse = val_mse
                torch.save(s.student.state_dict(), s.save_path)
            if improved_loss:
                s.best_val_loss = val_loss
            if early_stop:
                if improved_acc or improved_loss:
                    s.epochs_no_improve = 0
                else:
                    s.epochs_no_improve += 1
                    if s.epochs_no_improve >= patience:
                        s.active = False
                        print(f"  [{s.id}] early stopping at {phase_label} epoch {epoch} "
                              f"(best_val_acc={s.best_val_acc:.4f})")


def train_student_group(
    teacher: nn.Module,
    students: list,
    save_paths: list[str],
    ids: list[str],
    train_loader,
    val_loader,
    epochs: int,
    device: torch.device,
    encoder_lr: float = 1e-3,
    classifier_lr: float = 1e-5,
    weight_decay: float = 1e-2,
    mse_weight: float = 1.0,
    ce_weight: float = 1.0,
    kd_weight: float = 0.0,
    rkd_weight: float = 0.0,
    T: float = 4.0,
    patience: int = 5,
    eta_min: float = 1e-6,
    phase1_epochs: int = 0,
    phase1_eta_min: float | None = None,
) -> list[dict]:
    """Trains students that share a frozen teacher."""
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    states: list[_StudentState] = []
    for student, save_path, sid in zip(students, save_paths, ids):
        student.to(device)
        states.append(_StudentState(student, save_path, sid))

    common = dict(
        states=states, teacher=teacher, train_loader=train_loader,
        val_loader=val_loader, device=device,
        encoder_lr=encoder_lr, classifier_lr=classifier_lr,
        weight_decay=weight_decay, T=T, patience=patience,
    )
    p1_eta_min = phase1_eta_min if phase1_eta_min is not None else eta_min

    if phase1_epochs > 0:
        _run_phase(
            phase_label="P1-mse", epochs=phase1_epochs, eta_min=p1_eta_min,
            early_stop=False,
            mse_weight=mse_weight, ce_weight=0.0, kd_weight=0.0, rkd_weight=0.0,
            **common,
        )
        _run_phase(
            phase_label="P2-full", epochs=epochs, eta_min=eta_min, early_stop=True,
            mse_weight=mse_weight, ce_weight=ce_weight, kd_weight=kd_weight,
            rkd_weight=rkd_weight, **common,
        )
    else:
        _run_phase(
            phase_label="train", epochs=epochs, eta_min=eta_min, early_stop=True,
            mse_weight=mse_weight, ce_weight=ce_weight, kd_weight=kd_weight,
            rkd_weight=rkd_weight, **common,
        )

    for s in states:
        print(f"  [{s.id}] best val_acc={s.best_val_acc:.4f} "
              f"(val_mse {s.best_val_mse:.5f}, best_val_loss {s.best_val_loss:.5f}) "
              f"-> {s.save_path}")
    return [{"best_val_acc": s.best_val_acc, "val_mse": s.best_val_mse,
             "best_val_loss": s.best_val_loss} for s in states]
