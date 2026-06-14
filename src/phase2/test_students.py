"""Phase-3 evaluation: test all trained students and write a results table.

Loads each trained student (encoder + predictor) from its checkpoint, attaches
the frozen Phase-1 classifier, and evaluates top-1 accuracy on the test split.

Usage
-----
    python src/phase2/test_students.py
    python src/phase2/test_students.py --datasets oxford-pets flowers-102
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1.teachers import load_teacher_checkpoint
from src.phase1.utils import compute_flops, get_dataloaders, set_seed
from src.phase2.student import Student

SEED        = 291652
DATA_ROOT   = PROJECT_ROOT / "data"
CKPT_DIR    = PROJECT_ROOT / "outputs" / "checkpoints" / "teachers"
WEIGHTS_DIR = PROJECT_ROOT / "outputs" / "students"
TABLE_DIR   = PROJECT_ROOT / "outputs" / "tables"

_BACKBONE = {
    "resnet":   "resnet50",
    "convnext": "convnext_base",
    "vgg":      "vgg16_bn",
}

DATASETS = ["oxford-pets", "flowers-102"]


@torch.no_grad()
def evaluate_test(model, loader, device: torch.device) -> tuple[float, float]:
    """Returns (test_loss, test_acc) on the given loader."""
    import torch.nn.functional as F
    model.eval()
    loss_sum = 0.0; correct = 0; n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            loss   = F.cross_entropy(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        n        += x.size(0)
    return loss_sum / n, correct / n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets",    nargs="*", default=DATASETS,
                   help="Datasets to evaluate (default: all).")
    p.add_argument("--ckpt-mode",  default="frozen", choices=["frozen", "finetune"])
    p.add_argument("--students-json", default=None)
    p.add_argument("--weights-dir",   default=None,
                   help="Directory containing trained student .pth files.")
    p.add_argument("--tag", default=None,
                   help="Ablation tag: read weights/summary from "
                        "outputs/students/<dataset>/ablation_<tag>/ and suffix "
                        "the result CSVs with __<tag>.")
    p.add_argument("--batch-size",  type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _csv_row(r: dict) -> dict:
    """Flattens one test result into a CSV row."""
    lw = r.get("loss_weights") or {}
    return {
        "id":               r["id"],
        "dataset":          r["dataset"],
        "teacher":          r["teacher"],
        "target_mode":      r["target_mode"],
        "arch":             r["id"].split("__")[0],
        "n_convs":          r["n_convs"],
        "trainable_params": r["trainable_params"],
        "gflops":           r["gflops"],
        "best_val_acc":     r["best_val_acc"],
        "test_acc":         r["test_acc"],
        "test_loss":        r["test_loss"],
        "tag":              r.get("tag"),
        "mse_weight":       lw.get("mse_weight"),
        "ce_weight":        lw.get("ce_weight"),
        "kd_weight":        lw.get("kd_weight"),
        "rkd_weight":       lw.get("rkd_weight"),
    }


def _run_dataset(
    dataset: str,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict]:
    import pandas as pd

    json_path = Path(args.students_json) if args.students_json else (
        Path(__file__).resolve().parent / f"students_{dataset}.json"
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
            f"Training summary not found: {summary_path}\n"
            "Run train_students.py first."
        )
    trained = {e["id"]: e for e in json.loads(summary_path.read_text(encoding="utf-8"))}

    set_seed(SEED)
    _, _, test_loader, _ = get_dataloaders(
        dataset_name=dataset,
        data_root=DATA_ROOT,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    teacher_cache: dict[str, object] = {}
    results = []

    for cfg in students_cfg:
        sid = cfg["id"]
        if sid not in trained:
            print(f"  [SKIP] {sid} — not in training summary")
            continue
        # Resolve weights relative to weights_dir by id, rather than trusting
        # the absolute path stored in summary.json (which is OS-specific and
        # breaks when the summary was generated on another machine).
        weight_path = weights_dir / f"{sid}.pth"
        if not weight_path.exists():
            print(f"  [SKIP] {sid} — weights not found at {weight_path}")
            continue

        teacher_key = cfg["teacher"]
        mode = trained[sid].get("teacher_ckpt_mode", args.ckpt_mode)
        cache_key = (teacher_key, mode)
        if cache_key not in teacher_cache:
            teacher_cache[cache_key] = load_teacher_checkpoint(
                backbone_name=_BACKBONE[teacher_key],
                dataset_name=dataset,
                num_classes=num_classes,
                checkpoint_dir=CKPT_DIR,
                mode=mode,
                device=device,
            )
        teacher = teacher_cache[cache_key]

        student = Student(
            layers=cfg["layers"],
            teacher_dim=cfg["teacher_dim"],
            num_classes=cfg["num_classes"],
            classifier=copy.deepcopy(teacher.classifier),
            target_mode=cfg["target_mode"],
            freeze_classifier=True,
        ).to(device)
        student.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True)
        )

        test_loss, test_acc = evaluate_test(student, test_loader, device)
        gflops = compute_flops(student)
        print(f"  {sid:60s} | test_acc={test_acc:.4f} | "
              f"test_loss={test_loss:.5f} | gflops={gflops:.3f}")

        results.append({
            **{k: trained[sid][k] for k in
               ("id", "teacher", "teacher_backbone", "teacher_dim",
                "target_mode", "n_convs", "trainable_params",
                "best_val_acc", "val_mse")},
            "dataset":   dataset,
            "test_acc":  test_acc,
            "test_loss": test_loss,
            "gflops":    gflops,
            "loss_weights": trained[sid].get("loss"),
            "tag":          trained[sid].get("tag"),
        })

    # Write per-dataset JSON (unchanged format for downstream compat).
    out_path = weights_dir / "test_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} test results -> {out_path}")

    # Write per-dataset CSV (Phase-1 style).
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([_csv_row(r) for r in results])
    suffix = f"__{args.tag}" if args.tag else ""
    csv_path = TABLE_DIR / f"student_results_{dataset}{suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV -> {csv_path}")

    # Print quick summary table.
    print(f"\n{'id':60s} {'teacher':10s} {'mode':8s} {'val_acc':>8} {'test_acc':>9}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: -x["test_acc"]):
        print(f"  {r['id']:58s} {r['teacher']:10s} {r['target_mode']:8s} "
              f"{r['best_val_acc']:8.4f} {r['test_acc']:9.4f}")

    return results


def main() -> None:
    import pandas as pd

    args   = parse_args()
    device = torch.device(args.device)

    all_results = []
    for dataset in args.datasets:
        print("=" * 100)
        print(f"DATASET: {dataset}")
        results = _run_dataset(dataset, args, device)
        all_results.extend(results)

    # Combined CSV across all datasets.
    if len(args.datasets) > 1:
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        combined_df = pd.DataFrame([_csv_row(r) for r in all_results])
        suffix = f"__{args.tag}" if args.tag else ""
        combined_path = TABLE_DIR / f"student_results{suffix}.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"\nSaved combined CSV -> {combined_path}")

    print("=" * 100)
    print("Done.")


if __name__ == "__main__":
    main()
