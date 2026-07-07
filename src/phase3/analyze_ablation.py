from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STUDENTS_DIR = PROJECT_ROOT / "outputs" / "students"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

DATASETS = ["oxford-pets", "flowers-102", "tiny-imagenet-200"]
TEACHERS = ["resnet", "convnext", "vgg"]
ARCH = "arch6_6conv_res"

CONFIG_ORDER = ["mse_only", "baseline", "mse_ce_kd", "ce_kd", "mse_ce_nrkd",
                "mse_kd_2stage", "ce_kd_nrkd_1stage"]
CONFIG_LABEL = {
    "mse_only":         "MSE only\n(feat)",
    "baseline":         "MSE+CE\n(main)",
    "mse_ce_kd":        "MSE+CE+KD\n(T=4)",
    "ce_kd":            "CE+KD\n(logit only)",
    "mse_ce_nrkd":       "MSE+CE+NRKD",
    "mse_kd_2stage":    "MSE→MSE+CE+KD\n(2-stage)",
    "ce_kd_nrkd_1stage": "CE+KD+NRKD\n(1-stage)",
}
CONFIG_COLORS = {
    "mse_only":         "#9E9E9E",
    "baseline":         "#4878CF",
    "mse_ce_kd":        "#6ACC65",
    "ce_kd":            "#D65F5F",
    "mse_ce_nrkd":       "#E07B39",
    "mse_kd_2stage":    "#8E6FCF",
    "ce_kd_nrkd_1stage": "#3FB8AF",
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
    for tag in ["mse_only", "mse_ce_kd", "ce_kd", "mse_ce_nrkd",
                "mse_kd_2stage", "ce_kd_nrkd_1stage"]:
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


def plot_table(df: pd.DataFrame, no_title: bool = False) -> None:
    datasets = [ds for ds in DATASETS if (df["dataset"] == ds).any()]
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(6.5 * len(datasets), 4.8),
        sharey=False,
        squeeze=False,
    )
    axes = axes[0]
    configs = [c for c in CONFIG_ORDER if c in df["config"].unique().tolist()]
    width = 0.8 / len(configs)
    x = np.arange(len(TEACHERS))

    for ax, ds in zip(axes, datasets):
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

    if not no_title:
        fig.suptitle(f"Q4/Q5: loss-function ablation on {ARCH} (pre_gap)",
                     fontsize=11, y=1.02)
    fig.tight_layout()
    out = FIGURE_DIR / "q4_loss_ablation.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_delta_heatmap(df: pd.DataFrame, no_title: bool = False) -> None:
    """Heatmap of test-accuracy change from the matched MSE+CE baseline."""
    if df.empty:
        return

    configs = [c for c in CONFIG_ORDER if c in df["config"].astype(str).unique().tolist()]
    columns = [
        (ds, tk)
        for ds in DATASETS
        for tk in TEACHERS
        if not df[(df["dataset"] == ds) & (df["teacher"] == tk)].empty
    ]

    rows = []
    for cfg in configs:
        row = {"config": cfg}
        for ds, tk in columns:
            base = df[
                (df["dataset"] == ds)
                & (df["teacher"] == tk)
                & (df["config"].astype(str) == "baseline")
            ]["test_acc"]
            cur = df[
                (df["dataset"] == ds)
                & (df["teacher"] == tk)
                & (df["config"].astype(str) == cfg)
            ]["test_acc"]
            col = f"{ds}__{tk}"
            if len(base) and len(cur):
                row[col] = (float(cur.iloc[0]) - float(base.iloc[0])) * 100
            else:
                row[col] = np.nan
        rows.append(row)

    delta_df = pd.DataFrame(rows)
    out_csv = TABLE_DIR / "q4_loss_delta_heatmap.csv"
    delta_df.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}")

    value_cols = [f"{ds}__{tk}" for ds, tk in columns]
    values = delta_df[value_cols].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        print("  [skip] no finite deltas for heatmap.")
        return
    vmax = max(abs(float(finite.min())), abs(float(finite.max())))
    if vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(1.25 * len(value_cols) + 3.2, 0.58 * len(configs) + 2.2))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(value_cols)))
    ax.set_xticklabels([f"{ds}\n{tk}" for ds, tk in columns], fontsize=8)
    ax.set_yticks(np.arange(len(configs)))
    ax.set_yticklabels([CONFIG_LABEL[c].replace("\n", " ") for c in configs], fontsize=8)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                color = "white" if abs(v) > 0.55 * vmax else "black"
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_xlabel("Dataset / teacher")
    ax.set_ylabel("Configuration")
    if not no_title:
        ax.set_title("Loss/training-strategy delta over MSE+CE baseline (pp)", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Test accuracy delta (pp)")
    fig.tight_layout()
    out = FIGURE_DIR / "q4_loss_delta_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit the figure-level title and overwrite q4_loss_ablation.png.",
    )
    parser.add_argument(
        "--improvements-only",
        action="store_true",
        help="Generate only additive improvement artifacts with new filenames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    existing = TABLE_DIR / "q4_loss_ablation.csv"
    if args.improvements_only and existing.exists():
        df = pd.read_csv(existing)
        df["config"] = pd.Categorical(df["config"], categories=CONFIG_ORDER, ordered=True)
        print(f"Loaded -> {existing}  ({len(df)} rows)")
    else:
        df = build_table()
    if df.empty:
        print("No data — run the ablation first.")
        return
    if not args.improvements_only:
        plot_table(df, no_title=args.no_title)
    plot_delta_heatmap(df, no_title=args.no_title)
    if args.improvements_only:
        return
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
