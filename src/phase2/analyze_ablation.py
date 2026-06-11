"""Q4/Q5 loss-ablation analysis.

Combines the four ablation runs (tags: mse_only, mse_ce_kd, ce_kd, mse_ce_rkd)
with the existing main-run baseline (mse=1, ce=1) on the reduced grid
``arch6_6conv_res`` x {resnet, convnext, vgg} x pre_gap x {oxford-pets,
flowers-102}, then writes a tidy comparison table and a grouped bar chart.

    outputs/tables/q4_loss_ablation.csv
    outputs/figures/q4_loss_ablation.png

Usage
-----
    python src/phase2/analyze_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STUDENTS_DIR = PROJECT_ROOT / "outputs" / "students"
TABLE_DIR    = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR   = PROJECT_ROOT / "outputs" / "figures"

DATASETS = ["oxford-pets", "flowers-102"]
TEACHERS = ["resnet", "convnext", "vgg"]
ARCH     = "arch6_6conv_res"

# Display order + human labels for the loss configurations.
CONFIG_ORDER = ["mse_only", "baseline", "mse_ce_kd", "ce_kd", "mse_ce_rkd"]
CONFIG_LABEL = {
    "mse_only":   "MSE only\n(feat)",
    "baseline":   "MSE+CE\n(main)",
    "mse_ce_kd":  "MSE+CE+KD\n(T=4)",
    "ce_kd":      "CE+KD\n(logit only)",
    "mse_ce_rkd": "MSE+CE+RKD",
}
CONFIG_COLORS = {
    "mse_only":   "#9E9E9E",
    "baseline":   "#4878CF",
    "mse_ce_kd":  "#6ACC65",
    "ce_kd":      "#D65F5F",
    "mse_ce_rkd": "#E07B39",
}
DPI = 150


def _baseline_rows() -> list[dict]:
    """arch6 pre_gap rows from the main run (mse=1, ce=1, kd=0)."""
    rows = []
    for ds in DATASETS:
        path = STUDENTS_DIR / ds / "test_results.json"
        if not path.exists():
            print(f"  [WARN] missing baseline {path} — skipping {ds}")
            continue
        for r in json.loads(path.read_text()):
            if r["id"].startswith(ARCH) and r["target_mode"] == "pre_gap":
                rows.append({
                    "dataset":   ds,
                    "teacher":   r["teacher"],
                    "config":    "baseline",
                    "test_acc":  r["test_acc"],
                    "best_val_acc": r["best_val_acc"],
                    "gflops":    r.get("gflops"),
                })
    return rows


def _ablation_rows() -> list[dict]:
    rows = []
    for tag in ["mse_only", "mse_ce_kd", "ce_kd", "mse_ce_rkd"]:
        for ds in DATASETS:
            csv = TABLE_DIR / f"student_results_{ds}__{tag}.csv"
            if not csv.exists():
                print(f"  [WARN] missing ablation CSV {csv}")
                continue
            df = pd.read_csv(csv)
            df = df[(df["arch"] == ARCH) & (df["target_mode"] == "pre_gap")]
            for _, r in df.iterrows():
                rows.append({
                    "dataset":   ds,
                    "teacher":   r["teacher"],
                    "config":    tag,
                    "test_acc":  r["test_acc"],
                    "best_val_acc": r["best_val_acc"],
                    "gflops":    r.get("gflops"),
                })
    return rows


def build_table() -> pd.DataFrame:
    df = pd.DataFrame(_baseline_rows() + _ablation_rows())
    df["config"] = pd.Categorical(df["config"], categories=CONFIG_ORDER, ordered=True)
    df = df.sort_values(["dataset", "teacher", "config"]).reset_index(drop=True)
    out = TABLE_DIR / "q4_loss_ablation.csv"
    df.to_csv(out, index=False)
    print(f"Saved -> {out}  ({len(df)} rows)")
    return df


def plot_table(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(13, 4.8), sharey=False)
    configs = [c for c in CONFIG_ORDER if c in df["config"].unique().tolist()]
    width = 0.8 / len(configs)
    x = np.arange(len(TEACHERS))

    for ax, ds in zip(axes, DATASETS):
        sub = df[df["dataset"] == ds]
        for j, cfg in enumerate(configs):
            accs = []
            for tk in TEACHERS:
                v = sub[(sub["teacher"] == tk) & (sub["config"] == cfg)]["test_acc"]
                accs.append(float(v.iloc[0]) * 100 if len(v) else np.nan)
            bars = ax.bar(x + (j - (len(configs) - 1) / 2) * width, accs, width,
                          label=CONFIG_LABEL[cfg], color=CONFIG_COLORS[cfg],
                          edgecolor="white", linewidth=0.5)
            for b in bars:
                h = b.get_height()
                if not np.isnan(h):
                    ax.text(b.get_x() + b.get_width() / 2, h + 0.4, f"{h:.0f}",
                            ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(TEACHERS)
        ax.set_title(ds, fontsize=10)
        ax.set_ylabel("Test accuracy (%)")
        ax.grid(axis="y", alpha=0.4)
        ax.legend(fontsize=7, ncol=2, loc="upper right")

    fig.suptitle(f"Q4/Q5: loss-function ablation on {ARCH} (pre_gap)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = FIGURE_DIR / "q4_loss_ablation.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = build_table()
    if df.empty:
        print("No data — run the ablation first.")
        return
    plot_table(df)
    # Compact console summary: best config per (dataset, teacher).
    print("\nBest loss config per (dataset, teacher):")
    for ds in DATASETS:
        for tk in TEACHERS:
            sub = df[(df["dataset"] == ds) & (df["teacher"] == tk)]
            if sub.empty:
                continue
            best = sub.loc[sub["test_acc"].idxmax()]
            print(f"  {ds:13s} {tk:9s} -> {best['config']:11s} "
                  f"({best['test_acc']*100:.1f}%)")


if __name__ == "__main__":
    main()
