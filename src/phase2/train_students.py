"""Teacher-head pipeline orchestrator.

Single-stage distillation of every student backbone in ``students.json`` on
Oxford-IIIT Pet. Each student is a :class:`StudentTeacherHead` (conv backbone ->
``[teacher_dim, 7, 7]`` map -> glued teacher head); training matches the post-GAP
features to the teacher (MSE) and, optionally, adds CE/KD on the head.

Students are trained in per-teacher groups: the frozen teacher runs once per
batch and its target is reused across all students that distill from it (instead
of one teacher forward per student), while each student keeps its own optimizer,
schedule, and early-stopping. Per-epoch augmentation is preserved.

    uv run train_students.py
    uv run train_students.py --ce-weight 0.5 --kd-weight 0.5
    uv run train_students.py --start-at 7 --epochs 30
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
from src.phase2.distill import train_student_group
from src.phase2.student_teacher_head import StudentTeacherHead, teacher_head

SEED = 291652
DATA_ROOT = PROJECT_ROOT / "data"
STUDENTS_JSON = Path(__file__).resolve().parent / "students.json"
WEIGHTS_DIR = PROJECT_ROOT / "students_weights" / "teacher_head_train"

TEACHERS = {
    "resnet": (lambda n: resnet50(weights=None, num_classes=n), "resnet50", "resnet50_pets.pth"),
    "convnext": (lambda n: convnext_base(weights=None, num_classes=n), "convnext_base", "convnext_base_pets.pth"),
    "vgg": (lambda n: vgg16(weights=None, num_classes=n), "vgg16", "vgg16_pets.pth"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--students-json", default=str(STUDENTS_JSON),
                        help="Path to the student backbones JSON.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--encoder-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--mse-weight", type=float, default=1.0,
                        help="Weight of the post-GAP feature MSE term.")
    parser.add_argument("--ce-weight", type=float, default=0.0,
                        help="Weight of the cross-entropy term on the glued head (0 = off).")
    parser.add_argument("--kd-weight", type=float, default=0.0,
                        help="Weight of the KD (KL) term against the teacher logits (0 = off).")
    parser.add_argument("--T", type=float, default=4.0, help="KD temperature.")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience (epochs).")
    parser.add_argument("--freeze-head", action="store_true", default=True,
                        help="Keep the glued teacher head frozen (default: fine-tune).")
    parser.add_argument("--start-at", type=int, default=1,
                        help="1-based index of the student to (re)start from; "
                             "earlier students are skipped and their summary preserved.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_teacher(teacher_key: str, num_classes: int, device: torch.device):
    builder, kind, checkpoint_name = TEACHERS[teacher_key]
    checkpoint_path = PROJECT_ROOT / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing teacher checkpoint: {checkpoint_path}")
    model = builder(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model, kind


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    students = json.loads(Path(args.students_json).read_text(encoding="utf-8"))
    start_at = args.start_at
    if not 1 <= start_at <= len(students):
        raise ValueError(f"--start-at must be in [1, {len(students)}], got {start_at}.")
    print(
        f"Device: {device} | students: {len(students)} | start-at: {start_at} | "
        f"loss: mse={args.mse_weight} ce={args.ce_weight} kd={args.kd_weight} | weights -> {WEIGHTS_DIR}"
    )

    summary_path = WEIGHTS_DIR / "teacher_head_summary.json"
    summary_by_id: dict[str, dict] = {}
    if start_at > 1 and summary_path.exists():
        for entry in json.loads(summary_path.read_text(encoding="utf-8")):
            summary_by_id[entry["id"]] = entry

    set_seed(SEED)
    train_loader, val_loader, _, num_classes = get_dataloaders(
        dataset_name="oxford-pets",
        data_root=DATA_ROOT,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Group students by teacher: the frozen teacher's target is identical for
    # every student that distills from it, so each teacher runs once per batch
    # for its whole group (see distill.train_student_group). Students keep their
    # original JSON order within a group for a stable, ordered summary.
    groups: dict[str, list[tuple[int, dict]]] = {}
    for index, config in enumerate(students, start=1):
        if index < start_at:
            continue
        groups.setdefault(config["teacher"], []).append((index, config))

    teacher_cache: dict[str, tuple] = {}

    for teacher_key, members in groups.items():
        if teacher_key not in teacher_cache:
            teacher_cache[teacher_key] = load_teacher(teacher_key, num_classes, device)
        teacher, kind = teacher_cache[teacher_key]
        teacher_dim = members[0][1]["teacher_dim"]

        print("=" * 100)
        print(f"Teacher group {teacher_key} (kind={kind}, dim={teacher_dim}) | "
              f"{len(members)} student(s)")

        # When the head is frozen it is identical across the group (a deep copy
        # of the same teacher head), so share a single instance instead of one
        # per student -- this matters for VGG (its head is ~120M params).
        shared_head = teacher_head(teacher, kind) if args.freeze_head else None

        models: list[StudentTeacherHead] = []
        save_paths: list[str] = []
        ids: list[str] = []
        for index, config in members:
            set_seed(SEED)
            head = shared_head if args.freeze_head else teacher_head(teacher, kind)
            student = StudentTeacherHead(
                layers=config["layers"],
                teacher_dim=config["teacher_dim"],
                num_classes=config["num_classes"],
                head=head,
                kind=kind,
                freeze_head=args.freeze_head,
            )
            print(f"  [{index}/{len(students)}] {config['id']} | convs={config['n_convs']} | "
                  f"trainable params: {count_parameters(student):,}")
            models.append(student)
            save_paths.append(str(WEIGHTS_DIR / f"{config['id']}.pth"))
            ids.append(config["id"])

        results = train_student_group(
            teacher=teacher,
            students=models,
            save_paths=save_paths,
            ids=ids,
            train_loader=train_loader,
            val_loader=val_loader,
            kind=kind,
            epochs=args.epochs,
            device=device,
            encoder_lr=args.encoder_lr,
            classifier_lr=args.classifier_lr,
            weight_decay=args.weight_decay,
            mse_weight=args.mse_weight,
            ce_weight=args.ce_weight,
            kd_weight=args.kd_weight,
            T=args.T,
            patience=args.patience,
        )

        for (index, config), model, save_path, result in zip(members, models, save_paths, results):
            summary_by_id[config["id"]] = {
                "id": config["id"],
                "teacher": teacher_key,
                "teacher_dim": config["teacher_dim"],
                "n_convs": config["n_convs"],
                "trainable_params": count_parameters(model),
                "best_val_acc": result["best_val_acc"],
                "val_mse": result["val_mse"],
                "best_val_loss": result["best_val_loss"],
                "weights": save_path,
            }
        # Incremental, ordered write so a crash keeps progress per teacher group.
        summary = [summary_by_id[c["id"]] for c in students if c["id"] in summary_by_id]
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 100)
    print(f"Done. Summary -> {summary_path}")


if __name__ == "__main__":
    main()
