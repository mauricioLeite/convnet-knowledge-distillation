from phase1.teachers import TeacherModel

from .student import Student
from torch import nn
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def teacher_encode(model: nn.Module, x: DataLoader, kind: str, post_gap: bool = True) -> torch.Tensor:
    """Extracts the teacher's post GAP features for a batch of images."""
    model.eval()

    if kind == "resnet50":
        x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
        if post_gap:
            x = model.avgpool(x)
    elif kind == "convnext_base":
        x = (model.features(x))
        if post_gap:
            x = model.avgpool(x)
    elif kind == "vgg16":
        x = model.features(x)
        if post_gap:
            x = F.adaptive_avg_pool2d(x, 1)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return torch.flatten(x, 1)


def eval_distill_mse(teacher_model: TeacherModel, student: Student,  loader, kind: str, device: torch.device) -> float:
    """Evaluates the MSE between the teacher's post GAP features and the student's features for a batch of images."""
    student.eval()
    total = 0.0
    n = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            ft = teacher_encode(teacher_model, x, kind)
            fs = student.project(x)
        loss = F.mse_loss(F.normalize(fs.float(), dim=1), F.normalize(ft.float(), dim=1))
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


def destill_encoder_features(teacher: TeacherModel, student: Student, train_loader, validation_loader,
                     kind: str, epochs: int, save_path: str, device: torch.device, lr: float = 1e-3, weight_decay: float = 1e-2) -> float:
    teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False

    student.to(device)
    encoder_parameters = [param for name, param in student.named_parameters() if "classifier" not in name]
    optimizer = torch.optim.AdamW(encoder_parameters, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    best_val_mse = float("inf")

    for epoch in range(epochs):
        student.train()
        run = 0.0
        n = 0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                ft = teacher_encode(teacher, x, kind)
                fs = student.project(x)
            loss = F.mse_loss(F.normalize(fs.float(), dim=1), F.normalize(ft.float(), dim=1))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()

        train_mse = run / n
        val_mse = eval_distill_mse(teacher, student, validation_loader, kind, device)
        if val_mse < best_val_mse:                       # menor val_MSE -> salva
            best_val_mse = val_mse
            torch.save(student.state_dict(), save_path)
        print(f"{kind}] epoch {epoch:02d} | train_mse {train_mse:.5f} | val_mse {val_mse:.5f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e}")

    # recarrega o MELHOR encoder p/ a memoria (a Fase 2 deve partir do melhor, nao do ultimo)
    student.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    print(f"melhor val_MSE = {best_val_mse:.5f} (encoder recarregado de {save_path})\n")
    return best_val_mse