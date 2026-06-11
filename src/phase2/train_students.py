"""Phase-2 distillation orchestrator.

Edit the config block below to choose which teachers, datasets, and
checkpoint modes to use, then just run the script:

    uv run src/phase2/train_students.py
    uv run src/phase2/train_students.py --teachers resnet --datasets oxford-pets
    uv run src/phase2/train_students.py --epochs 5 --start-at 7

Hyperparameter precedence (lowest → highest):
    DEFAULTS  →  RUN_OVERRIDES[(teacher, dataset)]  →  CLI flags
CLI flags only win when explicitly passed (they default to None).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1.teachers import load_teacher_checkpoint
from src.phase1.utils import count_parameters, get_dataloaders, set_seed
from src.phase2.distill import train_student_group
from src.phase2.student import Student

# ── paths ────────────────────────────────────────────────────────────────────
SEED        = 291652
DATA_ROOT   = PROJECT_ROOT / "data"
CKPT_DIR    = PROJECT_ROOT / "outputs" / "checkpoints" / "teachers"
WEIGHTS_DIR = PROJECT_ROOT / "outputs" / "students"

# Map JSON teacher key -> backbone name understood by TeacherModel
_BACKBONE = {
    "resnet":   "resnet50",
    "convnext": "convnext_base",
    "vgg":      "vgg16_bn",
}

# ── what to run ───────────────────────────────────────────────────────────────
TEACHERS = ["resnet", "convnext", "vgg"]
DATASETS = ["oxford-pets", "flowers-102"]

# Per-teacher Phase-1 checkpoint mode.
# VGG has no finetune checkpoint — must stay "frozen".
CKPT_MODE: dict[str, str] = {
    "resnet":   "finetune",
    "convnext": "finetune",
    "vgg":      "frozen",
}

# ── default hyperparameters ───────────────────────────────────────────────────
DEFAULTS: dict[str, Any] = {
    "epochs":        30,
    "batch_size":    128,
    "encoder_lr":    1e-3,
    "classifier_lr": 1e-5,
    "weight_decay":  1e-2,
    "mse_weight":    1.0,
    "ce_weight":     1.0,
    "kd_weight":     0.0,
    "rkd_weight":    0.0,
    "T":             4.0,
    "patience":      5,
}

# ── per-(teacher, dataset) overrides ─────────────────────────────────────────
RUN_OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {
    ("convnext", "flowers-102"): {"epochs": 40, "batch_size": 64},
    ("resnet",   "flowers-102"): {"epochs": 40, "batch_size": 64},
    ("vgg",      "flowers-102"): {"epochs": 40, "batch_size": 64},
}


# ── config merge ─────────────────────────────────────────────────────────────

def run_config(teacher_key: str, dataset: str, cli_overrides: dict[str, Any]) -> dict[str, Any]:
    """Merges DEFAULTS, per-run overrides, and explicit CLI flags."""
    cfg = {**DEFAULTS, **RUN_OVERRIDES.get((teacher_key, dataset), {})}
    cfg.update({k: v for k, v in cli_overrides.items() if v is not None})
    return cfg


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teachers",  nargs="*", default=None,
                   help="Teacher keys to run (default: TEACHERS list above).")
    p.add_argument("--datasets",  nargs="*", default=None,
                   help="Datasets to run (default: DATASETS list above).")
    p.add_argument("--students-json", default=None,
                   help="Explicit path to student JSON (overrides auto-resolution).")
    # Hyperparams — all default to None so they only override when explicitly passed.
    p.add_argument("--epochs",        type=int,   default=None)
    p.add_argument("--batch-size",    type=int,   default=None)
    p.add_argument("--num-workers",   type=int,   default=6)
    p.add_argument("--encoder-lr",    type=float, default=None)
    p.add_argument("--classifier-lr", type=float, default=None)
    p.add_argument("--weight-decay",  type=float, default=None)
    p.add_argument("--mse-weight",    type=float, default=None)
    p.add_argument("--ce-weight",     type=float, default=None)
    p.add_argument("--kd-weight",     type=float, default=None)
    p.add_argument("--rkd-weight",    type=float, default=None)
    p.add_argument("--T",             type=float, default=None)
    p.add_argument("--patience",      type=int,   default=None)
    p.add_argument("--freeze-classifier", action="store_true", default=True)
    p.add_argument("--start-at", type=int, default=1,
                   help="1-based student index to (re)start from within each dataset run.")
    p.add_argument("--tag", default=None,
                   help="Ablation tag: weights/summary go to "
                        "outputs/students/<dataset>/ablation_<tag>/ instead of "
                        "the main run directory.")
    p.add_argument("--ids", nargs="*", default=None,
                   help="fnmatch pattern(s) on student ids, e.g. "
                        "'arch6_6conv_res__*_pre_gap'. Default: all students.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Extracts only the hyperparameter flags (those that may be None)."""
    return {
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "encoder_lr":    args.encoder_lr,
        "classifier_lr": args.classifier_lr,
        "weight_decay":  args.weight_decay,
        "mse_weight":    args.mse_weight,
        "ce_weight":     args.ce_weight,
        "kd_weight":     args.kd_weight,
        "rkd_weight":    args.rkd_weight,
        "T":             args.T,
        "patience":      args.patience,
    }


def _load_teacher(teacher_key: str, dataset: str, num_classes: int, device: torch.device):
    return load_teacher_checkpoint(
        backbone_name=_BACKBONE[teacher_key],
        dataset_name=dataset,
        num_classes=num_classes,
        checkpoint_dir=CKPT_DIR,
        mode=CKPT_MODE[teacher_key],
        device=device,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args        = parse_args()
    device      = torch.device(args.device)
    cli_ovr     = _cli_overrides(args)
    active_teachers = args.teachers or TEACHERS
    active_datasets = args.datasets or DATASETS

    for dataset in active_datasets:
        # Resolve the student JSON for this dataset.
        if args.students_json:
            json_path = Path(args.students_json)
        else:
            json_path = Path(__file__).resolve().parent / f"students_{dataset}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Student JSON not found: {json_path}\n"
                f"Run: python src/phase2/gen_students.py --datasets {dataset}"
            )
        students_cfg = json.loads(json_path.read_text(encoding="utf-8"))
        num_classes  = students_cfg[0]["num_classes"]

        out_dir = WEIGHTS_DIR / dataset
        if args.tag:
            out_dir = out_dir / f"ablation_{args.tag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        start_at = args.start_at
        if not 1 <= start_at <= len(students_cfg):
            raise ValueError(f"--start-at must be in [1, {len(students_cfg)}].")

        summary_path   = out_dir / "summary.json"
        summary_by_id: dict[str, dict] = {}
        if summary_path.exists():
            for entry in json.loads(summary_path.read_text(encoding="utf-8")):
                summary_by_id[entry["id"]] = entry

        print("=" * 100)
        print(f"DATASET: {dataset} | classes: {num_classes} | "
              f"teachers: {active_teachers} | students: {len(students_cfg)} | "
              f"start-at: {start_at} | device: {device} | weights -> {out_dir}")

        # Group students by teacher, filtered to active_teachers.
        groups: dict[str, list[tuple[int, dict]]] = {}
        for idx, cfg in enumerate(students_cfg, start=1):
            if idx < start_at:
                continue
            if cfg["teacher"] not in active_teachers:
                continue
            if args.ids and not any(fnmatch(cfg["id"], pat) for pat in args.ids):
                continue
            groups.setdefault(cfg["teacher"], []).append((idx, cfg))

        set_seed(SEED)
        teacher_cache: dict[str, Any] = {}

        for teacher_key, members in groups.items():
            cfg_run = run_config(teacher_key, dataset, cli_ovr)

            # Load data here so batch_size from cfg_run is used.
            train_loader, val_loader, _, _ = get_dataloaders(
                dataset_name=dataset,
                data_root=DATA_ROOT,
                batch_size=cfg_run["batch_size"],
                num_workers=args.num_workers,
            )

            if teacher_key not in teacher_cache:
                teacher_cache[teacher_key] = _load_teacher(
                    teacher_key, dataset, num_classes, device
                )
            teacher = teacher_cache[teacher_key]

            print("-" * 100)
            print(f"Teacher: {teacher_key} ({_BACKBONE[teacher_key]}, "
                  f"ckpt={CKPT_MODE[teacher_key]}) | {len(members)} student(s) | "
                  f"epochs={cfg_run['epochs']} batch={cfg_run['batch_size']} "
                  f"enc_lr={cfg_run['encoder_lr']} mse={cfg_run['mse_weight']} "
                  f"ce={cfg_run['ce_weight']} kd={cfg_run['kd_weight']} "
                  f"rkd={cfg_run['rkd_weight']}")

            models:     list[Student] = []
            save_paths: list[str]     = []
            ids:        list[str]     = []

            for idx, scfg in members:
                set_seed(SEED)
                student = Student(
                    layers=scfg["layers"],
                    teacher_dim=scfg["teacher_dim"],
                    num_classes=scfg["num_classes"],
                    classifier=copy.deepcopy(teacher.classifier),
                    target_mode=scfg["target_mode"],
                    freeze_classifier=args.freeze_classifier,
                )
                enc_p = count_parameters(student.encoder) + count_parameters(student.predictor)
                print(f"  [{idx}/{len(students_cfg)}] {scfg['id']} | "
                      f"mode={scfg['target_mode']} convs={scfg['n_convs']} | "
                      f"trainable enc+pred: {enc_p:,}")
                models.append(student)
                save_paths.append(str(out_dir / f"{scfg['id']}.pth"))
                ids.append(scfg["id"])

            results = train_student_group(
                teacher=teacher,
                students=models,
                save_paths=save_paths,
                ids=ids,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=cfg_run["epochs"],
                device=device,
                encoder_lr=cfg_run["encoder_lr"],
                classifier_lr=cfg_run["classifier_lr"],
                weight_decay=cfg_run["weight_decay"],
                mse_weight=cfg_run["mse_weight"],
                ce_weight=cfg_run["ce_weight"],
                kd_weight=cfg_run["kd_weight"],
                rkd_weight=cfg_run["rkd_weight"],
                T=cfg_run["T"],
                patience=cfg_run["patience"],
            )

            for (idx, scfg), model, save_path, result in zip(
                members, models, save_paths, results
            ):
                enc_p = count_parameters(model.encoder) + count_parameters(model.predictor)
                summary_by_id[scfg["id"]] = {
                    "id":               scfg["id"],
                    "dataset":          dataset,
                    "teacher":          teacher_key,
                    "teacher_backbone": _BACKBONE[teacher_key],
                    "teacher_ckpt_mode": CKPT_MODE[teacher_key],
                    "teacher_dim":      scfg["teacher_dim"],
                    "target_mode":      scfg["target_mode"],
                    "n_convs":          scfg["n_convs"],
                    "trainable_params": enc_p,
                    "best_val_acc":     result["best_val_acc"],
                    "val_mse":          result["val_mse"],
                    "best_val_loss":    result["best_val_loss"],
                    "weights":          save_path,
                    "tag":              args.tag,
                    "loss": {k: cfg_run[k] for k in
                             ("mse_weight", "ce_weight", "kd_weight",
                              "rkd_weight", "T")},
                }
            # Incremental write per teacher group — crash-safe.
            summary = [summary_by_id[c["id"]] for c in students_cfg
                       if c["id"] in summary_by_id]
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("=" * 100)
        print(f"Done [{dataset}]. Summary -> {summary_path}")


if __name__ == "__main__":
    main()
