"""Phase-2/3 report figures and analysis tables.

Reads per-dataset test_results.json and teacher_results_comparison.csv,
then writes 3 figures to outputs/figures/ and 2 CSV tables to outputs/tables/.

Usage
-----
    python src/phase2/plot_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STUDENTS_DIR  = PROJECT_ROOT / "outputs" / "students"
TABLE_DIR     = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR    = PROJECT_ROOT / "outputs" / "figures"
TEACHER_CSV   = TABLE_DIR / "teacher_results_comparison.csv"

DATASETS = ["oxford-pets", "flowers-102"]

# Checkpoint mode used per teacher key during distillation (mirrors train_students.py).
CKPT_MODE = {"resnet": "finetune", "convnext": "finetune", "vgg": "frozen"}

# Full backbone names as they appear in teacher_results_comparison.csv.
_BACKBONE_NAME = {"resnet": "resnet50", "convnext": "convnext_base", "vgg": "vgg16_bn"}

# Arch display order (ascending capacity).
ARCH_ORDER = [
    "arch1_3conv_narrow",
    "arch2_3conv",
    "arch3_4conv",
    "arch4_4conv_res",
    "arch5_5conv_res",
    "arch6_6conv_res",
]
ARCH_LABELS = ["arch1\n3conv", "arch2\n3conv", "arch3\n4conv",
               "arch4\n4res", "arch5\n5res", "arch6\n6res"]

TEACHER_COLORS = {"resnet": "#4878CF", "convnext": "#6ACC65", "vgg": "#D65F5F"}
MODE_COLORS    = {"pre_gap": "#E07B39", "post_gap": "#4878CF"}
TEACHER_MARKERS = {"resnet": "o", "convnext": "s", "vgg": "^"}

DPI = 150


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_student_data() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        path = STUDENTS_DIR / ds / "test_results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}\nRun test_students.py first.")
        for r in json.loads(path.read_text()):
            r["dataset"] = ds
            r["arch"] = r["id"].split("__")[0]
            rows.append(r)
    df = pd.DataFrame(rows)
    return df


def load_teacher_accs() -> dict[tuple[str, str], float]:
    """Returns {(teacher_key, dataset): test_acc} using the correct ckpt mode."""
    tdf = pd.read_csv(TEACHER_CSV)
    out: dict[tuple[str, str], float] = {}
    for tk, mode in CKPT_MODE.items():
        backbone = _BACKBONE_NAME[tk]
        for ds in DATASETS:
            row = tdf[(tdf["Mode"] == mode) &
                      (tdf["Teacher"] == backbone) &
                      (tdf["Dataset"] == ds)]
            if len(row) == 1:
                out[(tk, ds)] = float(row["Test Acc (%)"].iloc[0]) / 100.0
    return out


# ---------------------------------------------------------------------------
# Figure 1: Q2 — Δ(pre_gap - post_gap) vs arch capacity
# ---------------------------------------------------------------------------

def plot_q2_delta(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, ds in zip(axes, DATASETS):
        sub = df[df["dataset"] == ds]
        for tk in ["resnet", "convnext", "vgg"]:
            deltas = []
            for arch in ARCH_ORDER:
                pre  = sub[(sub["arch"] == arch) & (sub["teacher"] == tk) &
                           (sub["target_mode"] == "pre_gap")]["test_acc"].values
                post = sub[(sub["arch"] == arch) & (sub["teacher"] == tk) &
                           (sub["target_mode"] == "post_gap")]["test_acc"].values
                if len(pre) == 1 and len(post) == 1:
                    deltas.append((pre[0] - post[0]) * 100)
                else:
                    deltas.append(np.nan)
            ax.plot(range(len(ARCH_ORDER)), deltas,
                    marker="o", label=tk, color=TEACHER_COLORS[tk], linewidth=1.8)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(ARCH_ORDER)))
        ax.set_xticklabels(ARCH_LABELS, fontsize=8)
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("Architecture (increasing capacity)")
        ax.set_ylabel("Δ test_acc  pre_gap − post_gap  (pp)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.4)

    fig.suptitle("Q2: pre_gap vs post_gap — advantage by architecture and teacher",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = FIGURE_DIR / "q2_pre_vs_post_delta.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# Figure 2: Q1 — best absolute + retention per teacher × dataset
# ---------------------------------------------------------------------------

def plot_q1_teacher(df: pd.DataFrame, teacher_accs: dict) -> None:
    teachers = ["resnet", "convnext", "vgg"]
    n_teachers = len(teachers)
    n_datasets = len(DATASETS)

    fig, axes = plt.subplots(1, n_datasets, figsize=(10, 4.5), sharey=False)

    bar_width = 0.35
    x = np.arange(n_teachers)

    for ax, ds in zip(axes, DATASETS):
        best_abs  = []
        retentions = []
        for tk in teachers:
            sub = df[(df["dataset"] == ds) & (df["teacher"] == tk)]
            best = sub["test_acc"].max() if len(sub) else np.nan
            teacher_acc = teacher_accs.get((tk, ds), np.nan)
            best_abs.append(best * 100)
            retentions.append((best / teacher_acc * 100) if not np.isnan(teacher_acc) else np.nan)

        bars1 = ax.bar(x - bar_width / 2, best_abs, bar_width,
                       label="Best student acc (%)", color=[TEACHER_COLORS[t] for t in teachers],
                       alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + bar_width / 2, retentions, bar_width,
                       label="Retention (student/teacher, %)",
                       color=[TEACHER_COLORS[t] for t in teachers],
                       alpha=0.45, edgecolor="black", linewidth=0.8, hatch="///")

        # Value labels
        for bar in bars1:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)
        for bar in bars2:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(teachers)
        ax.set_ylim(0, 105)
        ax.set_title(ds, fontsize=10)
        ax.set_ylabel("Accuracy / Retention (%)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.4)

    fig.suptitle("Q1: Teacher transfer — best student accuracy and retention by teacher",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = FIGURE_DIR / "q1_teacher_transfer.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# Figure 3: accuracy vs trainable params scatter
# ---------------------------------------------------------------------------

def plot_acc_vs_params(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, ds in zip(axes, DATASETS):
        sub = df[df["dataset"] == ds]
        legend_handles = []
        for mode, mcolor in MODE_COLORS.items():
            for tk, marker in TEACHER_MARKERS.items():
                pts = sub[(sub["target_mode"] == mode) & (sub["teacher"] == tk)]
                sc = ax.scatter(
                    pts["trainable_params"], pts["test_acc"] * 100,
                    color=mcolor, marker=marker, s=60, alpha=0.85,
                    edgecolors="white", linewidths=0.5,
                    label=f"{mode} / {tk}",
                )
                legend_handles.append(sc)

        ax.set_xscale("log")
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("Trainable params (enc + predictor)")
        ax.set_ylabel("Test accuracy (%)")
        ax.legend(fontsize=7, ncol=2, loc="lower right")
        ax.grid(alpha=0.35)

    fig.suptitle("Accuracy vs model size — color: target mode, marker: teacher",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = FIGURE_DIR / "acc_vs_params.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# Figure 4: Q3 — accuracy vs GFLOPs (student vs teacher savings)
# ---------------------------------------------------------------------------

def plot_acc_vs_gflops(df: pd.DataFrame) -> None:
    if "gflops" not in df.columns or df["gflops"].isna().all():
        print("  [skip] no gflops column — re-run test_students.py to populate it.")
        return

    # Teacher GFLOPs for reference lines (from teacher_results_comparison.csv).
    tdf = pd.read_csv(TEACHER_CSV)
    teacher_gflops = {}
    for tk, mode in CKPT_MODE.items():
        row = tdf[(tdf["Mode"] == mode) & (tdf["Teacher"] == _BACKBONE_NAME[tk])]
        if len(row):
            col = "GFLOPs" if "GFLOPs" in tdf.columns else "GFLOPS"
            teacher_gflops[tk] = float(row[col].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        sub = df[df["dataset"] == ds]
        for mode, mcolor in MODE_COLORS.items():
            for tk, marker in TEACHER_MARKERS.items():
                pts = sub[(sub["target_mode"] == mode) & (sub["teacher"] == tk)]
                ax.scatter(pts["gflops"], pts["test_acc"] * 100,
                           color=mcolor, marker=marker, s=60, alpha=0.85,
                           edgecolors="white", linewidths=0.5,
                           label=f"{mode} / {tk}")
        for tk, g in teacher_gflops.items():
            ax.axvline(g, color=TEACHER_COLORS[tk], linestyle="--",
                       linewidth=1.0, alpha=0.7)
            ax.text(g, ax.get_ylim()[0], f" {tk} teacher\n {g:.1f}G",
                    color=TEACHER_COLORS[tk], fontsize=6.5, va="bottom", ha="left")
        ax.set_xscale("log")
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("GFLOPs (student fwd @224, log scale)")
        ax.set_ylabel("Test accuracy (%)")
        ax.legend(fontsize=7, ncol=2, loc="lower right")
        ax.grid(alpha=0.35)

    fig.suptitle("Q3: accuracy vs compute — dashed lines mark teacher GFLOPs",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = FIGURE_DIR / "acc_vs_gflops.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# Analysis tables
# ---------------------------------------------------------------------------

def write_q2_table(df: pd.DataFrame) -> None:
    rows = []
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        for arch in ARCH_ORDER:
            for tk in ["resnet", "convnext", "vgg"]:
                pre  = sub[(sub["arch"] == arch) & (sub["teacher"] == tk) &
                           (sub["target_mode"] == "pre_gap")]
                post = sub[(sub["arch"] == arch) & (sub["teacher"] == tk) &
                           (sub["target_mode"] == "post_gap")]
                if len(pre) != 1 or len(post) != 1:
                    continue
                rows.append({
                    "dataset":       ds,
                    "arch":          arch,
                    "teacher":       tk,
                    "pre_gap_acc":   round(pre["test_acc"].iloc[0], 4),
                    "post_gap_acc":  round(post["test_acc"].iloc[0], 4),
                    "delta_pp":      round((pre["test_acc"].iloc[0] - post["test_acc"].iloc[0]) * 100, 2),
                    "pre_gap_params": int(pre["trainable_params"].iloc[0]),
                    "post_gap_params": int(post["trainable_params"].iloc[0]),
                })
    out = TABLE_DIR / "q2_pre_vs_post.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved -> {out}")


def write_q1_table(df: pd.DataFrame, teacher_accs: dict) -> None:
    rows = []
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        for tk in ["resnet", "convnext", "vgg"]:
            pts = sub[sub["teacher"] == tk]["test_acc"]
            teacher_acc = teacher_accs.get((tk, ds), np.nan)
            best = pts.max() if len(pts) else np.nan
            rows.append({
                "dataset":          ds,
                "teacher":          tk,
                "ckpt_mode":        CKPT_MODE[tk],
                "teacher_test_acc": round(teacher_acc, 4) if not np.isnan(teacher_acc) else None,
                "best_student_acc": round(best, 4) if not np.isnan(best) else None,
                "mean_student_acc": round(pts.mean(), 4) if len(pts) else None,
                "retention_pct":    round(best / teacher_acc * 100, 1) if not np.isnan(teacher_acc) else None,
            })
    out = TABLE_DIR / "q1_teacher_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading student test results...")
    df = load_student_data()
    print(f"  {len(df)} student results loaded ({df['dataset'].nunique()} datasets, "
          f"{df['arch'].nunique()} archs, {df['teacher'].nunique()} teachers)")

    print("Loading teacher accuracies...")
    teacher_accs = load_teacher_accs()
    for (tk, ds), acc in sorted(teacher_accs.items()):
        print(f"  {tk:10s} {ds:15s} -> {acc:.4f}")

    print("\nFigure 1: Q2 pre_gap vs post_gap delta...")
    plot_q2_delta(df)

    print("Figure 2: Q1 teacher transfer...")
    plot_q1_teacher(df, teacher_accs)

    print("Figure 3: acc vs params scatter...")
    plot_acc_vs_params(df)

    print("Figure 4: Q3 acc vs gflops...")
    plot_acc_vs_gflops(df)

    print("\nTable: q2_pre_vs_post.csv...")
    write_q2_table(df)

    print("Table: q1_teacher_summary.csv...")
    write_q1_table(df, teacher_accs)

    print("\nDone. Check outputs/figures/ and outputs/tables/.")


if __name__ == "__main__":
    main()
