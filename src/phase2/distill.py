from src.phase1.teachers import TeacherModel

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


@torch.no_grad()
def evaluate_cls(model, loader, device):
    """Calcula a Loss (CrossEntropy) e a Acurácia no conjunto de validação."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    
    return loss_sum / n, correct / n


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


def destill_students(teacher: TeacherModel, student: Student, train_loader, validation_loader, 
                     kind: str, epochs: int, weights_path: str, save_path: str, device: torch.device, 
                     lr: float = 1e-3, weight_decay: float = 1e-2, T: float = 4.0, alpha: float = 0.8):
    
    teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # O aluno precisa estar com tudo descongelado para o fine-tuning
    student.to(device)
    for param in student.parameters():
        param.requires_grad = True

    student.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

    # LRs separados: o encoder já está inteligente (1e-5), o classificador é burro (1e-3)
    encoder_parameters = [param for name, param in student.named_parameters() if "classifier" not in name]
    classifier_parameters = [param for name, param in student.named_parameters() if "classifier" in name]
    optimizer = torch.optim.AdamW([
        {'params': encoder_parameters, 'lr': 1e-5},
        {'params': classifier_parameters, 'lr': 1e-3}
    ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", dtype=torch.float16)

    best_val_acc = -1.0

    for epoch in range(epochs):
        student.train()
        run_loss = 0.0
        correct = 0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                student_pred = student(x)

                with torch.no_grad():
                    teacher_pred = teacher(x)
                loss_ce = F.cross_entropy(student_pred, y)
                loss_kd = F.kl_div(
                    F.log_softmax(student_pred / T, dim=1),
                    F.softmax(teacher_pred / T, dim=1),
                    reduction="batchmean"
                ) * (T * T)
                loss = (1 - alpha) * loss_ce + alpha * loss_kd

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item() * x.size(0)
            correct += (student_pred.argmax(1) == y).sum().item()
            n += x.size(0)

        scheduler.step()

        train_loss = run_loss / n
        train_acc = correct / n
        loss_val, val_acc = evaluate_cls(student, validation_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), save_path)

        print(f"{kind}] epoch {epoch:02d} | train_loss {train_loss:.5f} | train_acc {train_acc:.4f} | "
              f"val_loss {loss_val:.5f} | val_acc {val_acc:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

    return best_val_acc