"""Phase 2 orchestrator: distill every student architecture on Oxford-IIIT Pet.

Iterates over the architectures described in ``random_students.json``. For each
one it builds the :class:`Student`, loads the matching teacher (the
``*_pets.pth`` checkpoints in the project root) and runs feature distillation
with :func:`destill_encoder_features`. The best encoder of each student is saved
under ``students_weights/encoder_train/<id>.pth``.

    python src/phase2/train_students.py
    python src/phase2/train_students.py --epochs 40 --batch-size 256
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision.models import convnext_base, resnet50, vgg16

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1.utils import count_parameters, get_dataloaders, set_seed
from src.phase2.distill import destill_encoder_features, destill_students
from src.phase2.student import Student

SEED = 291652
DATA_ROOT = PROJECT_ROOT / "data"
STUDENTS_JSON = Path(__file__).resolve().parent / "random_students.json"
WEIGHTS_DIR_PHASE1 = PROJECT_ROOT / "students_weights" / "encoder_train"
WEIGHTS_DIR_PHASE2 = PROJECT_ROOT / "students_weights" / "classifier_train"

TEACHERS = {
    "resnet": (lambda n: resnet50(weights=None, num_classes=n), "resnet50", "resnet50_pets.pth"),
    "convnext": (lambda n: convnext_base(weights=None, num_classes=n), "convnext_base", "convnext_base_pets.pth"),
    "vgg": (lambda n: vgg16(weights=None, num_classes=n), "vgg16", "vgg16_pets.pth"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--T", type=float, default=4.0, help="KD temperature.")
    parser.add_argument("--alpha", type=float, default=0.8, help="KD loss weight (1-alpha goes to CE).")
    parser.add_argument(
        "--phase",
        choices=("encoder", "classifier"),
        default="classifier",
        help="Which Phase 2 stage to run.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (defaults to cuda when available).",
    )
    return parser.parse_args()


def load_teacher(teacher_key: str, num_classes: int, device: torch.device):
    """Builds a torchvision teacher and loads its Oxford-Pets checkpoint."""
    builder, kind, checkpoint_name = TEACHERS[teacher_key]
    checkpoint_path = Path(PROJECT_ROOT / checkpoint_name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing teacher checkpoint: {checkpoint_path}")
    model = builder(num_classes)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, kind


def train_students_encoder() -> None:
    args = parse_args()
    device = torch.device(args.device)
    WEIGHTS_DIR_PHASE1.mkdir(parents=True, exist_ok=True)

    students = json.loads(STUDENTS_JSON.read_text(encoding="utf-8"))
    print(f"Device: {device} | students: {len(students)} | weights -> {WEIGHTS_DIR_PHASE1}")

    # Oxford-Pets is identical for every student, so build the loaders once.
    set_seed(SEED)
    train_loader, val_loader, _, num_classes = get_dataloaders(
        dataset_name="oxford-pets",
        data_root=DATA_ROOT,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    teacher_cache: dict[str, tuple] = {}
    summary: list[dict] = []

    for index, config in enumerate(students, start=1):
        teacher_key = config["teacher"]
        teacher_dim = config["teacher_dim"]

        print("=" * 100)
        print(
            f"[{index}/{len(students)}] {config['id']} | "
            f"teacher={teacher_key} dim={teacher_dim} convs={config['n_convs']}"
        )

        # Reseed before each build so a student's init is identical regardless
        # of how many ran before it.
        set_seed(SEED)
        student = Student(
            layers=config["layers"],
            teacher_dim=teacher_dim,
            num_classes=config["num_classes"],
        )
        print(f"  trainable params: {count_parameters(student):,}")

        if teacher_key not in teacher_cache:
            teacher_cache[teacher_key] = load_teacher(teacher_key, num_classes, device)
        teacher, kind = teacher_cache[teacher_key]

        save_path = WEIGHTS_DIR_PHASE1 / f"{config['id']}.pth"
        best_val_mse = destill_encoder_features(
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            validation_loader=val_loader,
            kind=kind,
            epochs=args.epochs,
            save_path=str(save_path),
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        print(f"  best val_MSE = {best_val_mse:.5f} -> {save_path.name}")
        summary.append({
            "id": config["id"],
            "teacher": teacher_key,
            "teacher_dim": teacher_dim,
            "n_convs": config["n_convs"],
            "trainable_params": count_parameters(student),
            "best_val_mse": best_val_mse,
            "weights": str(save_path),
        })

    summary_path = WEIGHTS_DIR_PHASE1 / "distillation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 100)
    print(f"Done. Summary -> {summary_path}")


def train_students_classifier() -> None:
    args = parse_args()
    device = torch.device(args.device)
    WEIGHTS_DIR_PHASE2.mkdir(parents=True, exist_ok=True)

    students = json.loads(STUDENTS_JSON.read_text(encoding="utf-8"))
    print(
        f"Device: {device} | students: {len(students)} | "
        f"encoders <- {WEIGHTS_DIR_PHASE1} | weights -> {WEIGHTS_DIR_PHASE2}"
    )

    set_seed(SEED)
    train_loader, val_loader, _, num_classes = get_dataloaders(
        dataset_name="oxford-pets",
        data_root=DATA_ROOT,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    teacher_cache: dict[str, tuple] = {}
    summary: list[dict] = []

    for index, config in enumerate(students, start=1):
        teacher_key = config["teacher"]
        teacher_dim = config["teacher_dim"]

        print("=" * 100)
        print(
            f"[{index}/{len(students)}] {config['id']} | "
            f"teacher={teacher_key} dim={teacher_dim} convs={config['n_convs']}"
        )

        set_seed(SEED)
        student = Student(
            layers=config["layers"],
            teacher_dim=teacher_dim,
            num_classes=config["num_classes"],
        )
        print(f"  trainable params: {count_parameters(student):,}")

        encoder_weights = WEIGHTS_DIR_PHASE1 / f"{config['id']}.pth"
        if not encoder_weights.exists():
            raise FileNotFoundError(
                f"Missing Phase 2.1 encoder weights for {config['id']}: {encoder_weights}"
            )

        if teacher_key not in teacher_cache:
            teacher_cache[teacher_key] = load_teacher(teacher_key, num_classes, device)
        teacher, kind = teacher_cache[teacher_key]

        save_path = WEIGHTS_DIR_PHASE2 / f"{config['id']}.pth"
        best_val_acc = destill_students(
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            validation_loader=val_loader,
            kind=kind,
            epochs=args.epochs,
            weights_path=str(encoder_weights),
            save_path=str(save_path),
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            T=args.T,
            alpha=args.alpha,
        )
        print(f"  best val_acc = {best_val_acc:.4f} -> {save_path.name}")
        summary.append({
            "id": config["id"],
            "teacher": teacher_key,
            "teacher_dim": teacher_dim,
            "n_convs": config["n_convs"],
            "trainable_params": count_parameters(student),
            "best_val_acc": best_val_acc,
            "weights": str(save_path),
        })

    summary_path = WEIGHTS_DIR_PHASE2 / "classifier_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 100)
    print(f"Done. Summary -> {summary_path}")


if __name__ == "__main__":
    phase = parse_args().phase
    if phase == "encoder":
        train_students_encoder()
    else:
        train_students_classifier()
