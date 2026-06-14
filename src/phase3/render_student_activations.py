"""Phase-3 activation visualizer.

Mirrors the Phase-1 ``render_feature_maps`` (src/phase1/01_train_teachers.py)
but for the distilled students. For each dataset it picks a single test image
and renders one figure with:

    row 0      : the teacher's expected feature map (the distillation target)
    row 1..N   : each trained student's final conv activation

Columns are the three teachers (resnet, convnext, vgg). Student rows are
ordered largest -> smallest (bottom row = smallest student); within each
architecture the ``pre_gap`` student comes before the ``post_gap`` one. The
``pre_gap`` / ``post_gap`` label is only the training method (which student
sits in that row) -- the activation shown is always the encoder's final conv
output, mean-reduced over channels, exactly like Phase-1.

Usage:
    uv run src/phase3/render_student_activations.py
    uv run src/phase3/render_student_activations.py --datasets oxford-pets --tag mse_only
    uv run src/phase3/render_student_activations.py --ids 'arch6_6conv_res__*'
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from fnmatch import fnmatch
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1.teachers import load_teacher_checkpoint
from src.phase1.utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_dataloaders,
    set_seed,
)
from src.phase2.student import Student

# ── paths / constants (mirror test_students.py) ───────────────────────────────
SEED        = 291652
DATA_ROOT   = PROJECT_ROOT / "data"
CKPT_DIR    = PROJECT_ROOT / "outputs" / "checkpoints" / "teachers"
WEIGHTS_DIR = PROJECT_ROOT / "outputs" / "students"
FIGURE_DIR  = PROJECT_ROOT / "outputs" / "figures"

_BACKBONE = {
    "resnet":   "resnet50",
    "convnext": "convnext_base",
    "vgg":      "vgg16_bn",
}
TEACHER_ORDER = ["resnet", "convnext", "vgg"]


# ── helpers ───────────────────────────────────────────────────────────────────

def unnormalize(image: torch.Tensor) -> np.ndarray:
    """Converts an ImageNet-normalized CHW tensor to an HWC NumPy image."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD, dtype=image.dtype).view(3, 1, 1)
    image = (image.cpu() * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def mean_activation(feature_map: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """Channel-mean activation, upsampled to ``size`` and min-max normalized."""
    activation = feature_map.mean(dim=1, keepdim=True)
    activation = F.interpolate(
        activation, size=size, mode="bilinear", align_corners=False,
    ).squeeze().float().cpu()
    activation = (activation - activation.min()) / (
        activation.max() - activation.min() + 1e-8
    )
    return activation.numpy()


def _overlay(axis, image_hwc: np.ndarray, activation: np.ndarray, title: str) -> None:
    axis.imshow(image_hwc)
    axis.imshow(activation, cmap="magma", alpha=0.45)
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def _pick_image(loader, index: int) -> torch.Tensor:
    for i, (image, _) in enumerate(loader):
        if i == index:
            return image
    raise IndexError(f"--image-index {index} out of range for this test split.")


# ── per-dataset render ─────────────────────────────────────────────────────────

def render_dataset(dataset: str, args: argparse.Namespace, device: torch.device) -> None:
    json_path = Path(args.students_json) if args.students_json else (
        Path(__file__).resolve().parents[1] / "phase2" / f"students_{dataset}.json"
    )
    if not json_path.exists():
        raise FileNotFoundError(f"Student JSON not found: {json_path}")
    students_cfg = json.loads(json_path.read_text(encoding="utf-8"))
    num_classes  = students_cfg[0]["num_classes"]

    weights_dir = Path(args.weights_dir) if args.weights_dir else (
        WEIGHTS_DIR / dataset / f"ablation_{args.tag}" if args.tag
        else WEIGHTS_DIR / dataset
    )
    summary_path = weights_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Training summary not found: {summary_path}\nRun train_students.py first."
        )
    trained = {e["id"]: e for e in json.loads(summary_path.read_text(encoding="utf-8"))}

    # Test-set accuracy for the cell titles comes from test_results.json
    # (written by test_students.py into the same weights dir). Optional: if the
    # test step hasn't run yet, titles fall back to "n/a".
    results_path = weights_dir / "test_results.json"
    test_acc_by_id: dict[str, float] = {}
    if results_path.exists():
        test_acc_by_id = {
            e["id"]: e["test_acc"]
            for e in json.loads(results_path.read_text(encoding="utf-8"))
        }
    else:
        print(f"  [warn] {results_path} not found — titles show test_acc=n/a")

    cfg_by_id = {cfg["id"]: cfg for cfg in students_cfg}

    def _rank(sid: str) -> float:
        # Prefer test_acc; fall back to validation accuracy when the test step
        # hasn't run for this run yet.
        return test_acc_by_id.get(sid, trained[sid].get("best_val_acc", 0.0))

    # Best student per teacher: highest test_acc among trained students that
    # have weights on disk (optionally restricted by --ids).
    best_by_teacher: dict[str, str | None] = {}
    for teacher_key in TEACHER_ORDER:
        candidates = [
            sid for sid in trained
            if trained[sid]["teacher"] == teacher_key
            and sid in cfg_by_id
            and (weights_dir / f"{sid}.pth").exists()
            and (not args.ids or any(fnmatch(sid, pat) for pat in args.ids))
        ]
        best_by_teacher[teacher_key] = max(candidates, key=_rank) if candidates else None

    # Teacher checkpoint mode per teacher (taken from the summary so the row-0
    # target matches what the students were actually distilled against).
    teacher_mode: dict[str, str] = {}
    for e in trained.values():
        teacher_mode.setdefault(e["teacher"], e.get("teacher_ckpt_mode", "frozen"))

    set_seed(SEED)
    _, _, test_loader, _ = get_dataloaders(
        dataset_name=dataset, data_root=DATA_ROOT,
        batch_size=1, num_workers=args.num_workers,
    )
    image = _pick_image(test_loader, args.image_index)
    image_hwc = unnormalize(image[0])
    size = image.shape[-2:]

    n_rows = 2  # row 0: teacher target | row 1: best student
    n_cols = len(TEACHER_ORDER)
    figure, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.4 * n_rows), squeeze=False,
    )

    teacher_cache: dict[str, object] = {}

    def get_teacher(teacher_key: str):
        if teacher_key not in teacher_cache:
            teacher_cache[teacher_key] = load_teacher_checkpoint(
                backbone_name=_BACKBONE[teacher_key],
                dataset_name=dataset,
                num_classes=num_classes,
                checkpoint_dir=CKPT_DIR,
                mode=teacher_mode.get(teacher_key, "frozen"),
                device=device,
            )
        return teacher_cache[teacher_key]

    # ── row 0: teacher expected feature maps ──────────────────────────────────
    axes[0][0].set_ylabel("TEACHER\n(target)", fontsize=10, rotation=0,
                          ha="right", va="center", labelpad=40)
    for col, teacher_key in enumerate(TEACHER_ORDER):
        axis = axes[0][col]
        teacher = get_teacher(teacher_key)
        teacher.eval()
        with torch.no_grad():
            feature_map, _ = teacher.extract_features(image.to(device))
        _overlay(axis, image_hwc, mean_activation(feature_map, size),
                 f"{teacher_key} ({_BACKBONE[teacher_key]})")

    # ── row 1: best student per teacher ───────────────────────────────────────
    axes[1][0].set_ylabel("BEST\nSTUDENT", fontsize=10, rotation=0,
                          ha="right", va="center", labelpad=40)
    for col, teacher_key in enumerate(TEACHER_ORDER):
        axis = axes[1][col]
        axis.set_xticks([])
        axis.set_yticks([])

        sid = best_by_teacher[teacher_key]
        if sid is None:
            axis.axis("off")
            axis.set_title("no student", fontsize=8)
            continue

        cfg = cfg_by_id[sid]
        teacher = get_teacher(teacher_key)
        student = Student(
            layers=cfg["layers"],
            teacher_dim=cfg["teacher_dim"],
            num_classes=cfg["num_classes"],
            classifier=copy.deepcopy(teacher.classifier),
            target_mode=cfg["target_mode"],
            freeze_classifier=True,
        ).to(device)
        student.load_state_dict(
            torch.load(weights_dir / f"{sid}.pth", map_location=device, weights_only=True)
        )
        student.eval()
        with torch.no_grad():
            activation_map = student.encoder(image.to(device))

        # Student name, e.g. "pre_gap mse only" -> target_mode + method (tag).
        method = (args.tag or trained[sid].get("tag") or "baseline").replace("_", " ")
        acc = test_acc_by_id.get(sid)
        acc_str = f"{acc:.3f}" if acc is not None else "n/a"
        _overlay(axis, image_hwc, mean_activation(activation_map, size),
                 f"{cfg['target_mode']} {method}\ntest_acc={acc_str}")
        del student

    figure.suptitle(
        f"Student conv activations vs teacher target — {dataset}"
        + (f" (ablation: {args.tag})" if args.tag else ""),
        fontsize=13,
    )
    figure.tight_layout(rect=[0.04, 0, 1, 0.99])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = FIGURE_DIR / f"student_activations_{dataset}{suffix}.png"
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  saved -> {out_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="*", default=["oxford-pets", "flowers-102"])
    p.add_argument("--tag", default=None,
                   help="Ablation tag: read weights from "
                        "outputs/students/<dataset>/ablation_<tag>/.")
    p.add_argument("--weights-dir", default=None,
                   help="Explicit weights dir (overrides --tag resolution).")
    p.add_argument("--students-json", default=None,
                   help="Explicit student JSON path (overrides auto-resolution).")
    p.add_argument("--ids", nargs="*", default=None,
                   help="fnmatch pattern(s) on student ids to include.")
    p.add_argument("--image-index", type=int, default=0,
                   help="Which test image to visualize (default: first).")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    for dataset in args.datasets:
        print(f"DATASET: {dataset}")
        render_dataset(dataset, args, device)


if __name__ == "__main__":
    main()
