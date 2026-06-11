"""Grouped distillation trainer for Phase-2 students.

One teacher forward per batch is shared across every student in the group
(both pre_gap and post_gap students of the same teacher), avoiding redundant
teacher passes. ``TeacherModel.extract_features`` returns both targets in a
single forward so no second pass is ever needed:

    feature_map,   [N, C, 7, 7]  -> pre_gap students
    pooled_vector, [N, C]         -> post_gap students

Loss (configurable per run):
    loss = mse_weight  * MSE(student.project(x), teacher_target)
         + ce_weight   * CE(student(x), y)
         + kd_weight   * KL(student(x)/T, teacher(x)/T) * T^2
         + rkd_weight  * RKD(student_vec, teacher_vec)

CE is on by default (ce_weight=1); KD and RKD are off by default.

RKD (Park et al., CVPR 2019) matches *relations* between samples in the batch
instead of individual representations: a Huber loss on normalized pairwise
distances (RKD-D) plus a Huber loss on triplet angles (RKD-A), computed on the
post-GAP vectors of teacher and student (pre_gap students are pooled first).
The internal distance:angle ratio is 1:2 as in the paper, so the paper's
(lambda_d=25, lambda_a=50) corresponds to ``rkd_weight=25``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Teacher target extraction (uses Phase-1 TeacherModel.extract_features)
# ---------------------------------------------------------------------------

@torch.no_grad()
def teacher_forward(
    teacher: nn.Module,
    x: torch.Tensor,
    need_logits: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Single teacher pass that yields both pre- and post-GAP targets.

    Returns ``(feature_map, pooled_vector, logits_or_None)``.

    ``feature_map``    -- ``[N, C, 7, 7]``  pre-GAP target.
    ``pooled_vector``  -- ``[N, C]``         post-GAP target (= GAP(feature_map)).
    ``logits``         -- teacher logits, only when ``need_logits=True`` (KD).
    """
    teacher.eval()
    feature_map, pooled_vector = teacher.extract_features(x)
    logits = teacher.classifier(pooled_vector) if need_logits else None
    return feature_map, pooled_vector, logits


# ---------------------------------------------------------------------------
# Relational Knowledge Distillation (Park et al., CVPR 2019)
# ---------------------------------------------------------------------------

def _pdist(e: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Pairwise euclidean distance matrix ``[N, N]`` with a zeroed diagonal."""
    e_sq = e.pow(2).sum(dim=1)
    dist = (e_sq.unsqueeze(1) + e_sq.unsqueeze(0) - 2.0 * (e @ e.t())).clamp(min=eps).sqrt()
    dist = dist.clone()
    dist[range(len(e)), range(len(e))] = 0.0
    return dist


def rkd_loss(student_vec: torch.Tensor, teacher_vec: torch.Tensor) -> torch.Tensor:
    """RKD-D (pairwise distances) + 2x RKD-A (triplet angles), both Huber.

    Inputs are ``[N, D]`` post-GAP vectors; teacher relations are targets
    (no grad). Forced to fp32 — normalized relation matrices are too
    ill-conditioned for fp16 autocast.
    """
    with torch.amp.autocast("cuda", enabled=False):
        s = student_vec.float()
        t = teacher_vec.float()

        # RKD-D: distances normalized by their batch mean.
        with torch.no_grad():
            td = _pdist(t)
            td = td / td[td > 0].mean()
        sd = _pdist(s)
        sd = sd / sd[sd > 0].mean()
        loss_d = F.smooth_l1_loss(sd, td)

        # RKD-A: cosine of angles formed by every (i, j, k) triplet.
        with torch.no_grad():
            te = F.normalize(t.unsqueeze(0) - t.unsqueeze(1), p=2, dim=2)
            t_angle = torch.bmm(te, te.transpose(1, 2)).view(-1)
        se = F.normalize(s.unsqueeze(0) - s.unsqueeze(1), p=2, dim=2)
        s_angle = torch.bmm(se, se.transpose(1, 2)).view(-1)
        loss_a = F.smooth_l1_loss(s_angle, t_angle)

    return loss_d + 2.0 * loss_a


def _pooled_student_vec(student, fs: torch.Tensor) -> torch.Tensor:
    """Post-GAP vector of the student's projected representation."""
    return fs.mean(dim=(2, 3)) if student.target_mode == "pre_gap" else fs


# ---------------------------------------------------------------------------
# Per-student training state
# ---------------------------------------------------------------------------

class _StudentState:
    def __init__(self, student, save_path, sid):
        self.student    = student
        self.save_path  = save_path
        self.id         = sid
        # optimizer / scheduler / scaler are (re)built per training phase.
        self.optimizer  = None
        self.scheduler  = None
        self.scaler     = None
        # best_val_acc and the saved checkpoint persist across phases (acc is the
        # same metric every phase); best_val_loss resets per phase since the loss
        # definition changes between phases.
        self.best_val_acc   = -1.0
        self.best_val_mse   = float("inf")
        self.best_val_loss  = float("inf")
        self.epochs_no_improve = 0
        self.active     = True
        self.run_loss   = 0.0
        self.train_correct = 0
        self.n          = 0

    def reset_epoch(self):
        self.run_loss      = 0.0
        self.train_correct = 0
        self.n             = 0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_group(
    teacher,
    students: list,
    loader,
    device: torch.device,
    mse_weight: float = 1.0,
    ce_weight:  float = 1.0,
    kd_weight:  float = 0.0,
    rkd_weight: float = 0.0,
    T: float = 4.0,
) -> list[tuple[float, float, float]]:
    """Evaluates all students in one shared-teacher pass.

    Returns ``[(val_loss, val_mse, val_acc)]`` aligned with ``students``.
    """
    for s in students:
        s.eval()
    k = len(students)
    loss_sum = [0.0] * k
    mse_sum  = [0.0] * k
    correct  = [0]   * k
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bs = x.size(0)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            fmap, vec, teacher_logits = teacher_forward(teacher, x, kd_weight > 0)
            for i, student in enumerate(students):
                target = fmap if student.target_mode == "pre_gap" else vec
                fs     = student.project(x)
                mse    = F.mse_loss(fs.float(), target.float())
                logits = student(x)
                loss   = mse_weight * mse
                if ce_weight > 0:
                    loss = loss + ce_weight * F.cross_entropy(logits, y)
                if kd_weight > 0:
                    loss = loss + kd_weight * F.kl_div(
                        F.log_softmax(logits.float() / T, dim=1),
                        F.softmax(teacher_logits.float() / T, dim=1),
                        reduction="batchmean",
                    ) * (T * T)
                if rkd_weight > 0:
                    loss = loss + rkd_weight * rkd_loss(
                        _pooled_student_vec(student, fs), vec
                    )
                loss_sum[i] += loss.item() * bs
                mse_sum[i]  += mse.item()  * bs
                correct[i]  += (logits.argmax(1) == y).sum().item()
        n += bs
    return [(loss_sum[i] / n, mse_sum[i] / n, correct[i] / n) for i in range(k)]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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
    """Runs one training phase over all states with a *fresh* optimizer +
    cosine scheduler (and a fresh GradScaler) per student.

    ``best_val_acc`` and the saved checkpoint persist across calls (overall best
    is kept); ``best_val_loss`` and the patience counter reset each phase because
    the loss definition (and scale) changes between phases.
    """
    need_kd  = kd_weight > 0
    need_rkd = rkd_weight > 0
    use_head = ce_weight > 0 or need_kd

    for s in states:
        s.active = True
        s.epochs_no_improve = 0
        s.best_val_loss = float("inf")
        enc_params  = [p for name, p in s.student.named_parameters()
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
            # One shared teacher forward for the whole group.
            with torch.amp.autocast("cuda", dtype=torch.float16):
                fmap, vec, teacher_logits = teacher_forward(teacher, x, need_kd)

            for s in active:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    target = fmap if s.student.target_mode == "pre_gap" else vec
                    fs     = s.student.project(x)
                    loss   = mse_weight * F.mse_loss(fs.float(), target.float())

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
                        loss = loss + rkd_weight * rkd_loss(
                            _pooled_student_vec(s.student, fs), vec
                        )

                s.optimizer.zero_grad()
                s.scaler.scale(loss).backward()
                s.scaler.step(s.optimizer)
                s.scaler.update()
                s.run_loss      += loss.item() * x.size(0)
                s.train_correct += (logits.argmax(1) == y).sum().item()
                s.n             += x.size(0)

        for s in active:
            s.scheduler.step()

        val_results = evaluate_group(
            teacher, [s.student for s in active], val_loader, device,
            mse_weight=mse_weight, ce_weight=ce_weight, kd_weight=kd_weight,
            rkd_weight=rkd_weight, T=T,
        )
        for s, (val_loss, val_mse, val_acc) in zip(active, val_results):
            train_loss = s.run_loss      / s.n
            train_acc  = s.train_correct / s.n
            enc_lr     = s.scheduler.get_last_lr()[0]
            cls_lr     = s.scheduler.get_last_lr()[-1]
            print(
                f"  [{s.id}] {phase_label} epoch {epoch:02d} | train_loss {train_loss:.5f} | "
                f"train_acc {train_acc:.4f} | val_loss {val_loss:.5f} | "
                f"val_acc {val_acc:.4f} | enc_lr {enc_lr:.2e} | cls_lr {cls_lr:.2e}"
            )

            improved_acc  = val_acc  > s.best_val_acc
            improved_loss = val_loss < s.best_val_loss
            if improved_acc:
                s.best_val_acc  = val_acc
                s.best_val_mse  = val_mse
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
    encoder_lr:    float = 1e-3,
    classifier_lr: float = 1e-5,
    weight_decay:  float = 1e-2,
    mse_weight:    float = 1.0,
    ce_weight:     float = 1.0,
    kd_weight:     float = 0.0,
    rkd_weight:    float = 0.0,
    T:             float = 4.0,
    patience:      int   = 5,
    eta_min:       float = 1e-6,
    phase1_epochs: int   = 0,
    phase1_eta_min: float | None = None,
) -> list[dict]:
    """Train all students sharing one frozen teacher.

    The teacher runs **once per batch**; both ``feature_map`` and
    ``pooled_vector`` are extracted in that single pass and handed to each
    student according to its ``target_mode``. Students are stepped sequentially
    within the batch so peak activation memory stays ~one student's graph.

    Two training schemes:
    - ``phase1_epochs == 0`` (default): single phase of ``epochs`` with the
      configured loss weights and early stopping.
    - ``phase1_epochs > 0``: **Phase 1** = ``phase1_epochs`` epochs of MSE only
      (a fixed warm-up, no early stopping), then **Phase 2** = ``epochs`` epochs
      with the configured weights and early stopping. Each phase gets a fresh
      optimizer + cosine schedule (``encoder_lr`` -> phase ``eta_min``); Phase 1
      anneals to ``phase1_eta_min`` (falls back to ``eta_min`` when None) and
      Phase 2 to ``eta_min``.

    Returns a list of result dicts (``best_val_acc``, ``val_mse``,
    ``best_val_loss``) aligned with ``students``.
    """
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
        # Phase 1: MSE-only warm-up (fixed length, no early stopping).
        _run_phase(
            phase_label="P1-mse", epochs=phase1_epochs, eta_min=p1_eta_min,
            early_stop=False,
            mse_weight=mse_weight, ce_weight=0.0, kd_weight=0.0, rkd_weight=0.0,
            **common,
        )
        # Phase 2: full configured loss, with early stopping.
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
