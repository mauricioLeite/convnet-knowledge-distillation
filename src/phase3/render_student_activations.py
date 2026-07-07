from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from fnmatch import fnmatch
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.nn import functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1.teachers import load_teacher_checkpoint
from src.common.utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_dataloaders,
    set_seed,
)
from src.phase2.student import Student

SEED = 291652
DATA_ROOT = PROJECT_ROOT / "data"
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "teachers"
WEIGHTS_DIR = PROJECT_ROOT / "outputs" / "students"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

_BACKBONE = {
    "resnet":   "resnet50",
    "convnext": "convnext_base",
    "vgg":      "vgg16_bn",
}
TEACHER_ORDER = ["resnet", "convnext", "vgg"]


def unnormalize(image: torch.Tensor) -> np.ndarray:
    """Converts an ImageNet-normalized CHW tensor to an HWC NumPy image."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD, dtype=image.dtype).view(3, 1, 1)
    image = (image.cpu() * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def mean_activation(feature_map: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """Channel-mean activation, upsampled to size and min-max normalized."""
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
    dataset = loader.dataset
    if not 0 <= index < len(dataset):
        raise IndexError(
            f"image index {index} out of range for this test split (n={len(dataset)})."
        )
    image, _ = dataset[index]
    return image.unsqueeze(0)


def render_dataset(dataset: str, args: argparse.Namespace, device: torch.device) -> None:
    json_path = Path(args.students_json) if args.students_json else (
        Path(__file__).resolve().parents[1] / "phase2" / f"students_{dataset}.json"
    )
    if not json_path.exists():
        raise FileNotFoundError(f"Student JSON not found: {json_path}")
    students_cfg = json.loads(json_path.read_text(encoding="utf-8"))
    num_classes = students_cfg[0]["num_classes"]

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
        return trained[sid].get("best_val_acc", 0.0)

    best_by_teacher: dict[str, str | None] = {}
    for teacher_key in TEACHER_ORDER:
        candidates = [
            sid for sid in trained
            if trained[sid]["teacher"] == teacher_key
            and sid in cfg_by_id
            and cfg_by_id[sid]["target_mode"] == "pre_gap"
            and (weights_dir / f"{sid}.pth").exists()
            and (not args.ids or any(fnmatch(sid, pat) for pat in args.ids))
        ]
        best_by_teacher[teacher_key] = max(candidates, key=_rank) if candidates else None

    teacher_mode: dict[str, str] = {}
    for e in trained.values():
        teacher_mode.setdefault(e["teacher"], e.get("teacher_ckpt_mode", "frozen"))

    set_seed(SEED)
    _, _, test_loader, _ = get_dataloaders(
        dataset_name=dataset, data_root=DATA_ROOT,
        batch_size=1, num_workers=args.num_workers,
    )
    n_images = len(test_loader.dataset)
    if args.image_index is not None:
        image_indices = [args.image_index]
    elif args.image_indices:
        image_indices = args.image_indices
    else:
        image_indices = [random.Random(args.seed).randrange(n_images)]
    for idx in image_indices:
        if not 0 <= idx < n_images:
            raise IndexError(f"image index {idx} out of range for test split (n={n_images})")
    print(f"  image indices: {image_indices} / {n_images}")

    n_rows = 2 * len(image_indices)
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

    student_cache: dict[str, tuple[str, dict, Student]] = {}

    def get_student(teacher_key: str) -> tuple[str, dict, Student] | None:
        sid = best_by_teacher[teacher_key]
        if sid is None:
            return None
        if sid not in student_cache:
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
            student_cache[sid] = (sid, cfg, student)
        return student_cache[sid]

    for image_pos, idx in enumerate(image_indices):
        image = _pick_image(test_loader, idx)
        image_hwc = unnormalize(image[0])
        size = image.shape[-2:]
        teacher_features: dict[str, torch.Tensor] = {}

        teacher_row = image_pos * 2
        student_row = teacher_row + 1
        axes[teacher_row][0].set_ylabel(
            f"idx {idx}\nTEACHER", fontsize=10, rotation=0,
            ha="right", va="center", labelpad=42,
        )
        axes[student_row][0].set_ylabel(
            f"idx {idx}\nSTUDENT", fontsize=10, rotation=0,
            ha="right", va="center", labelpad=42,
        )

        for col, teacher_key in enumerate(TEACHER_ORDER):
            axis = axes[teacher_row][col]
            teacher = get_teacher(teacher_key)
            teacher.eval()
            with torch.no_grad():
                feature_map, _ = teacher.extract_features(image.to(device))
            teacher_features[teacher_key] = feature_map.detach()
            _overlay(axis, image_hwc, mean_activation(feature_map, size),
                     f"{teacher_key} ({_BACKBONE[teacher_key]})")

        for col, teacher_key in enumerate(TEACHER_ORDER):
            axis = axes[student_row][col]
            axis.set_xticks([])
            axis.set_yticks([])

            packed = get_student(teacher_key)
            if packed is None:
                axis.axis("off")
                axis.set_title("no pre-GAP student", fontsize=8)
                continue

            sid, cfg, student = packed
            target = teacher_features[teacher_key]
            with torch.no_grad():
                projected = student.project(image.to(device)).detach()
                mse = F.mse_loss(projected.float(), target.float()).item()
                cosine = F.cosine_similarity(
                    projected.flatten(1).float(),
                    target.flatten(1).float(),
                    dim=1,
                ).mean().item()

            method = (args.tag or trained[sid].get("tag") or "baseline").replace("_", " ")
            acc = test_acc_by_id.get(sid)
            acc_str = f"{acc:.3f}" if acc is not None else "n/a"
            val_str = f"{trained[sid].get('best_val_acc', float('nan')):.3f}"
            _overlay(
                axis,
                image_hwc,
                mean_activation(projected, size),
                f"{cfg['target_mode']} {method}\nval={val_str} test={acc_str} mse={mse:.3f} cos={cosine:.3f}",
            )

    if not args.no_title:
        figure.suptitle(
            f"Validation-selected student projected maps vs teacher targets: {dataset}" + (f" (ablation: {args.tag})" if args.tag else ""),
            fontsize=13,
        )
    figure.tight_layout(rect=[0.04, 0, 1, 0.99])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = FIGURE_DIR / f"student_teacher_activations_{dataset}{suffix}.png"
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  saved -> {out_path}")


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
    p.add_argument("--image-index", type=int, default=None,
                   help="Pin one specific test image. Overrides --image-indices.")
    p.add_argument("--image-indices", nargs="*", type=int, default=[0],
                   help="Fixed test-image indices to render. Default: 0.")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Seed used only when --image-indices is empty.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--no-title",
        action="store_true",
        help="Omit the figure-level title and overwrite the canonical figure file.",
    )
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
