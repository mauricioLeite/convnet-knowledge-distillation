from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STUDENTS_DIR = PROJECT_ROOT / "outputs" / "students"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TEACHER_CSV = TABLE_DIR / "teacher_results_comparison.csv"

DATASETS = ["oxford-pets", "flowers-102", "tiny-imagenet-200"]

CKPT_MODE = {"resnet": "finetune", "convnext": "finetune", "vgg": "frozen"}

_BACKBONE_NAME = {"resnet": "resnet50", "convnext": "convnext_base", "vgg": "vgg16_bn"}

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
MODE_COLORS = {"pre_gap": "#E07B39", "post_gap": "#4878CF"}
TEACHER_MARKERS = {"resnet": "o", "convnext": "s", "vgg": "^"}
DATASET_COLORS = {
    "oxford-pets": "#4878CF",
    "flowers-102": "#D65F5F",
    "tiny-imagenet-200": "#6ACC65",
}
NUM_CLASSES = {"oxford-pets": 37, "flowers-102": 102, "tiny-imagenet-200": 200}
CONFIG_SHORT = {
    "baseline": "baseline",
    "mse_only": "MSE only",
    "mse_ce_kd": "MSE+CE+KD",
    "ce_kd": "CE+KD",
    "mse_ce_nrkd": "MSE+CE+NRKD",
    "mse_kd_2stage": "2-stage",
    "ce_kd_nrkd_1stage": "CE+KD+NRKD",
}

DPI = 150


def _figure_path(stem: str) -> Path:
    return FIGURE_DIR / f"{stem}.png"


def load_student_data() -> pd.DataFrame:
    global DATASETS
    rows = []
    present = []
    for ds in DATASETS:
        path = STUDENTS_DIR / ds / "test_results.json"
        if not path.exists():
            print(f"  [WARN] missing {path} — skipping {ds}")
            continue
        present.append(ds)
        for r in json.loads(path.read_text()):
            r["dataset"] = ds
            r["arch"] = r["id"].split("__")[0]
            rows.append(r)
    if not present:
        raise FileNotFoundError(
            f"No test_results.json found under {STUDENTS_DIR}\nRun test_students.py first."
        )
    DATASETS = present
    df = pd.DataFrame(rows)

    ablation_tags = [
        "mse_only", "mse_ce_kd", "ce_kd", "mse_ce_nrkd",
        "mse_kd_2stage", "ce_kd_nrkd_1stage",
    ]
    abl_rows = []
    for ds in DATASETS:
        for tag in ablation_tags:
            csv = TABLE_DIR / f"student_results_{ds}__{tag}.csv"
            if not csv.exists():
                continue
            adf = pd.read_csv(csv)
            adf["dataset"] = ds
            adf["arch"]    = adf["id"].apply(lambda x: x.split("__")[0])
            adf["tag"]     = tag
            abl_rows.append(adf)

    if abl_rows:
        abl_df = pd.concat(abl_rows, ignore_index=True)
        best_abl = (
            abl_df.groupby(["dataset", "arch", "teacher", "target_mode"], as_index=False)
            .agg(best_ablation_acc=("test_acc", "max"))
        )
        df = df.merge(best_abl, on=["dataset", "arch", "teacher", "target_mode"], how="left")
        df["baseline_acc"] = df["test_acc"]
        df["best_student_acc"] = df[["test_acc", "best_ablation_acc"]].max(axis=1)
        if "gflops" not in df.columns or df["gflops"].isna().all():
            gfl = abl_df.groupby(["dataset", "arch", "teacher", "target_mode"],
                                  as_index=False)["gflops"].first()
            df = df.merge(gfl, on=["dataset", "arch", "teacher", "target_mode"],
                          how="left", suffixes=("", "_abl"))
            df["gflops"] = df.get("gflops_abl", df.get("gflops"))
            df.drop(columns=[c for c in df.columns if c.endswith("_abl")], inplace=True)
        else:
            mask = df["gflops"].isna()
            if mask.any():
                gfl = abl_df.groupby(["dataset", "arch", "teacher", "target_mode"],
                                      as_index=False)["gflops"].first()
                tmp = df[mask].merge(gfl, on=["dataset", "arch", "teacher", "target_mode"],
                                     how="left", suffixes=("", "_abl"))
                df.loc[mask, "gflops"] = tmp["gflops_abl"].values
        n = (df["best_ablation_acc"].notna() & (df["best_ablation_acc"] > df["baseline_acc"])).sum()
        print(f"  Ablation merge: {n} rows upgraded above baseline accuracy")

    return df


def load_teacher_accs() -> dict[tuple[str, str], float]:
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


def load_ablation_data(include_baseline: bool = True) -> pd.DataFrame:
    tags = [
        "mse_only", "mse_ce_kd", "ce_kd", "mse_ce_nrkd",
        "mse_kd_2stage", "ce_kd_nrkd_1stage",
    ]
    rows: list[dict] = []

    if include_baseline:
        for ds in DATASETS:
            path = STUDENTS_DIR / ds / "test_results.json"
            if not path.exists():
                continue
            for r in json.loads(path.read_text(encoding="utf-8")):
                if r["id"].startswith("arch6_6conv_res") and r["target_mode"] == "pre_gap":
                    rows.append({
                        "dataset": ds,
                        "teacher": r["teacher"],
                        "config": "baseline",
                        "id": r["id"],
                        "arch": "arch6_6conv_res",
                        "target_mode": r["target_mode"],
                        "best_val_acc": r["best_val_acc"],
                        "test_acc": r["test_acc"],
                        "gflops": r.get("gflops"),
                        "trainable_params": r.get("trainable_params"),
                        "teacher_dim": r.get("teacher_dim"),
                    })

    for ds in DATASETS:
        for tag in tags:
            csv = TABLE_DIR / f"student_results_{ds}__{tag}.csv"
            if not csv.exists():
                continue
            adf = pd.read_csv(csv)
            adf = adf[(adf["arch"] == "arch6_6conv_res") & (adf["target_mode"] == "pre_gap")]
            for _, r in adf.iterrows():
                teacher_dim = {"resnet": 2048, "convnext": 1024, "vgg": 512}.get(r["teacher"])
                rows.append({
                    "dataset": ds,
                    "teacher": r["teacher"],
                    "config": tag,
                    "id": r["id"],
                    "arch": r["arch"],
                    "target_mode": r["target_mode"],
                    "best_val_acc": r["best_val_acc"],
                    "test_acc": r["test_acc"],
                    "gflops": r.get("gflops"),
                    "trainable_params": r.get("trainable_params"),
                    "teacher_dim": teacher_dim,
                })

    return pd.DataFrame(rows)


def load_teacher_points() -> pd.DataFrame:
    """Teacher accuracy/GFLOPs points using the checkpoint modes used by students."""
    tdf = pd.read_csv(TEACHER_CSV)
    rows = []
    for tk, mode in CKPT_MODE.items():
        backbone = _BACKBONE_NAME[tk]
        for ds in DATASETS:
            row = tdf[
                (tdf["Mode"] == mode)
                & (tdf["Teacher"] == backbone)
                & (tdf["Dataset"] == ds)
            ]
            if row.empty:
                continue
            gflops_col = "GFLOPs" if "GFLOPs" in row.columns else "GFLOPS"
            rows.append({
                "dataset": ds,
                "teacher": tk,
                "teacher_label": backbone,
                "ckpt_mode": mode,
                "test_acc": float(row["Test Acc (%)"].iloc[0]) / 100.0,
                "gflops": float(row[gflops_col].iloc[0]),
            })
    return pd.DataFrame(rows)


def _classifier_params(row: pd.Series) -> float:
    teacher_dim = row.get("teacher_dim")
    if pd.isna(teacher_dim):
        teacher_dim = {"resnet": 2048, "convnext": 1024, "vgg": 512}.get(row["teacher"])
    num_classes = NUM_CLASSES.get(row["dataset"])
    if teacher_dim is None or num_classes is None:
        return np.nan
    return int(teacher_dim) * int(num_classes) + int(num_classes)


def _add_deployment_params(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["classifier_params"] = out.apply(_classifier_params, axis=1)
    out["deployed_params"] = out["trainable_params"] + out["classifier_params"]
    out["deployed_params_m"] = out["deployed_params"] / 1_000_000
    return out


def _validation_selected(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Select one row per group by validation accuracy, reporting test accuracy."""
    if df.empty:
        return df.copy()
    valid = df.dropna(subset=["best_val_acc"]).copy()
    idx = valid.groupby(group_cols)["best_val_acc"].idxmax()
    return valid.loc[idx].reset_index(drop=True)


def plot_pre_vs_post_delta(df: pd.DataFrame, no_title: bool = False) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 4.5),
                             sharey=False, squeeze=False)
    axes = axes[0]

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

    if not no_title:
        fig.suptitle("pre_gap vs post_gap: advantage by architecture and teacher",
                     fontsize=11, y=1.01)
    fig.tight_layout()
    out = _figure_path("pre_vs_post_delta")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_teacher_transfer(
    df: pd.DataFrame,
    teacher_accs: dict,
    no_title: bool = False,
) -> None:
    teachers   = ["resnet", "convnext", "vgg"]
    n_datasets = len(DATASETS)

    has_ablation = "baseline_acc" in df.columns

    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 4.5),
                             sharey=False, squeeze=False)
    axes = axes[0]

    bar_width = 0.25 if has_ablation else 0.35
    x = np.arange(len(teachers))

    for ax, ds in zip(axes, DATASETS):
        base_vals, best_vals, retentions = [], [], []
        dataset_has_ablation = False
        for tk in teachers:
            sub = df[(df["dataset"] == ds) & (df["teacher"] == tk)]
            teacher_acc = teacher_accs.get((tk, ds), np.nan)
            base = (sub["baseline_acc"].max()
                    if has_ablation and "baseline_acc" in sub.columns and len(sub)
                    else sub["test_acc"].max() if len(sub) else np.nan)
            ablation_values = (sub["best_ablation_acc"].dropna()
                               if "best_ablation_acc" in sub.columns else pd.Series(dtype=float))
            if len(ablation_values):
                dataset_has_ablation = True
                best = max(base, ablation_values.max())
            else:
                best = np.nan
            base_vals.append(base * 100 if not np.isnan(base) else np.nan)
            best_vals.append(best * 100 if not np.isnan(best) else np.nan)
            retentions.append(
                (best / teacher_acc * 100)
                if not np.isnan(best) and not np.isnan(teacher_acc) else np.nan
            )

        colors = [TEACHER_COLORS[t] for t in teachers]

        if has_ablation and dataset_has_ablation:
            offsets = [-bar_width, 0, bar_width]
            bar_groups = [
                (offsets[0], base_vals,  "Baseline grid best (%)",         0.55, None,  ""),
                (offsets[1], best_vals,  "Ablation best (%)",               0.90, None,  ""),
                (offsets[2], retentions, "Ablation retention (teacher=100%)", 0.45, "black", "///"),
            ]
            plot_bar_width = bar_width
        elif has_ablation:
            bar_groups = [
                (0, base_vals, "Baseline grid best (%)", 0.55, None, ""),
            ]
            plot_bar_width = 0.35
        else:
            offsets = [-bar_width / 2, bar_width / 2]
            bar_groups = [
                (offsets[0], best_vals,  "Best student acc (%)",            0.85, "white", ""),
                (offsets[1], retentions, "Retention (student/teacher, %)",  0.45, "black", "///"),
            ]
            plot_bar_width = bar_width

        for offset, vals, label, alpha, ec, hatch in bar_groups:
            bars = ax.bar(x + offset, vals, plot_bar_width, label=label,
                          color=colors, alpha=alpha,
                          edgecolor=ec if ec else "none",
                          linewidth=0.8 if ec else 0,
                          hatch=hatch if hatch else "")
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                            f"{h:.1f}", ha="center", va="bottom", fontsize=6.5)

        ax.set_xticks(x)
        ax.set_xticklabels(teachers)
        ax.set_ylim(0, 105)
        ax.set_title(ds, fontsize=10)
        ax.set_ylabel("Accuracy / Retention (%)")
        ax.legend(fontsize=7.5)
        ax.grid(axis="y", alpha=0.4)
        if has_ablation and not dataset_has_ablation:
            ax.text(
                0.5, 0.75, "Ablation not run",
                transform=ax.transAxes, ha="center", va="center", fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
            )

    if not no_title:
        fig.suptitle(
            "Teacher transfer: baseline vs ablation-best student accuracy and retention",
            fontsize=11,
            y=1.01,
        )
    fig.tight_layout()
    out = _figure_path("teacher_transfer")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_acc_vs_params(df: pd.DataFrame, no_title: bool = False) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 4.5),
                             sharey=False, squeeze=False)
    axes = axes[0]

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

    if not no_title:
        fig.suptitle("Accuracy vs model size: color by target mode, marker by teacher",
                     fontsize=11, y=1.01)
    fig.tight_layout()
    out = _figure_path("acc_vs_params")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_acc_vs_gflops(df: pd.DataFrame, no_title: bool = False) -> None:
    if "gflops" not in df.columns or df["gflops"].isna().all():
        print("  [skip] no gflops column — re-run test_students.py to populate it.")
        return
    if df["gflops"].isna().any():
        available = int(df["gflops"].notna().sum())
        print(f"  [skip] incomplete gflops data ({available}/{len(df)} rows) — "
              "re-run test_students.py before plotting the full grid.")
        return

    tdf = pd.read_csv(TEACHER_CSV)
    teacher_gflops = {}
    for tk, mode in CKPT_MODE.items():
        row = tdf[(tdf["Mode"] == mode) & (tdf["Teacher"] == _BACKBONE_NAME[tk])]
        if len(row):
            col = "GFLOPs" if "GFLOPs" in tdf.columns else "GFLOPS"
            teacher_gflops[tk] = float(row[col].iloc[0])

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 4.5),
                             sharey=False, squeeze=False)
    axes = axes[0]
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

    if not no_title:
        fig.suptitle("accuracy vs compute: dashed lines mark teacher GFLOPs",
                     fontsize=11, y=1.01)
    fig.tight_layout()
    out = _figure_path("acc_vs_gflops")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def _pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Efficient frontier for low x / high y."""
    keep = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        dominated = (
            (x <= x[i])
            & (y >= y[i])
            & ((x < x[i]) | (y > y[i]))
        )
        keep[i] = not dominated.any()
    return keep


def plot_acc_vs_gflops_pareto(
    df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    teacher_df: pd.DataFrame,
    no_title: bool = False,
) -> None:
    """Global accuracy-vs-compute plot with teachers and validation-selected tuned rows."""
    if "gflops" not in df.columns or df["gflops"].isna().all():
        print("  [skip] no gflops column for Pareto plot.")
        return

    selected_abl = pd.DataFrame()
    if not ablation_df.empty:
        has_tuned = (
            ablation_df.groupby(["dataset", "teacher"])["config"]
            .transform(lambda s: (s != "baseline").any())
        )
        selected_abl = _validation_selected(
            ablation_df[has_tuned],
            ["dataset", "teacher"],
        )

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.5 * len(DATASETS), 5),
                             sharey=False, squeeze=False)
    axes = axes[0]

    for ax, ds in zip(axes, DATASETS):
        sub = df[df["dataset"] == ds].dropna(subset=["gflops", "test_acc"])
        points = []

        for mode, color in MODE_COLORS.items():
            for tk, marker in TEACHER_MARKERS.items():
                pts = sub[(sub["target_mode"] == mode) & (sub["teacher"] == tk)]
                if pts.empty:
                    continue
                ax.scatter(
                    pts["gflops"], pts["test_acc"] * 100,
                    color=color, marker=marker, s=42, alpha=0.45,
                    edgecolors="white", linewidths=0.4,
                    label=f"baseline {mode}" if tk == "resnet" else None,
                )
                for _, r in pts.iterrows():
                    points.append({
                        "x": float(r["gflops"]),
                        "y": float(r["test_acc"]) * 100,
                        "kind": "baseline",
                        "label": r["arch"].replace("_", " "),
                    })

        arch4 = sub[sub["arch"] == "arch4_4conv_res"]
        if not arch4.empty:
            ax.scatter(
                arch4["gflops"], arch4["test_acc"] * 100,
                marker="x", s=52, color="black", alpha=0.55, linewidths=0.9,
                label="arch4 baseline",
            )

        tuned = selected_abl[selected_abl["dataset"] == ds].sort_values("test_acc")
        offset_values = [-12, 0, 12]
        tuned_offsets = {
            idx: (6, offset_values[min(pos, len(offset_values) - 1)])
            for pos, idx in enumerate(tuned.index)
        }
        for _, r in tuned.iterrows():
            ax.scatter(
                r["gflops"], r["test_acc"] * 100,
                marker="*", s=185, color=TEACHER_COLORS[r["teacher"]],
                edgecolors="black", linewidths=0.8, zorder=5,
                label="val-selected tuned" if r["teacher"] == "resnet" else None,
            )
            ax.annotate(
                f"{r['teacher']} {CONFIG_SHORT.get(r['config'], r['config'])}",
                xy=(r["gflops"], r["test_acc"] * 100),
                xytext=tuned_offsets.get(r.name, (6, 0)),
                textcoords="offset points",
                fontsize=6.5,
                va="center",
            )
            points.append({
                "x": float(r["gflops"]),
                "y": float(r["test_acc"]) * 100,
                "kind": "selected_ablation",
                "label": f"{r['teacher']} {r['config']}",
            })

        teachers = teacher_df[teacher_df["dataset"] == ds]
        teacher_offsets = {"resnet": (6, -9), "convnext": (6, 7), "vgg": (6, -17)}
        for _, r in teachers.iterrows():
            ax.scatter(
                r["gflops"], r["test_acc"] * 100,
                marker="P", s=160, color=TEACHER_COLORS[r["teacher"]],
                edgecolors="black", linewidths=0.8, zorder=6,
                label="teacher" if r["teacher"] == "resnet" else None,
            )
            ax.annotate(
                f"{r['teacher']} teacher",
                xy=(r["gflops"], r["test_acc"] * 100),
                xytext=teacher_offsets.get(r["teacher"], (6, 0)),
                textcoords="offset points",
                fontsize=6.5,
                va="center",
            )
            points.append({
                "x": float(r["gflops"]),
                "y": float(r["test_acc"]) * 100,
                "kind": "teacher",
                "label": f"{r['teacher']} teacher",
            })

        if points:
            p = pd.DataFrame(points)
            mask = _pareto_mask(p["x"].to_numpy(), p["y"].to_numpy())
            frontier = p[mask].sort_values("x")
            ax.plot(frontier["x"], frontier["y"], color="black", linewidth=1.2,
                    alpha=0.75, label="Pareto frontier")
            ax.scatter(frontier["x"], frontier["y"], facecolors="none",
                       edgecolors="black", s=95, linewidths=1.0, zorder=7)

        ax.set_xscale("log")
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("GFLOPs (forward @224, log scale)")
        ax.set_ylabel("Test accuracy (%)")
        ax.grid(alpha=0.35)
        ax.set_ylim(15, 100)
        ax.legend(fontsize=7, loc="lower right")

    if not no_title:
        fig.suptitle("Accuracy-GFLOPs Pareto frontier: baseline, tuned students, and teachers",
                     fontsize=11, y=1.01)
    fig.tight_layout()
    out = _figure_path("acc_vs_gflops_pareto")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def write_validation_test_gap_tables(df: pd.DataFrame) -> None:
    gap = df.copy()
    gap["val_acc_pct"] = gap["best_val_acc"] * 100
    gap["test_acc_pct"] = gap["test_acc"] * 100
    gap["val_minus_test_pp"] = gap["val_acc_pct"] - gap["test_acc_pct"]
    cols = [
        "dataset", "teacher", "target_mode", "arch", "id",
        "val_acc_pct", "test_acc_pct", "val_minus_test_pp",
    ]
    out = TABLE_DIR / "validation_test_gaps.csv"
    gap[cols].to_csv(out, index=False)
    print(f"Saved -> {out}")

    summary = (
        gap.groupby("dataset", as_index=False)
        .agg(
            mean_gap_pp=("val_minus_test_pp", "mean"),
            median_gap_pp=("val_minus_test_pp", "median"),
            max_gap_pp=("val_minus_test_pp", "max"),
            rows=("id", "count"),
        )
    )
    out_summary = TABLE_DIR / "validation_test_gap_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"Saved -> {out_summary}")


def plot_validation_vs_test(df: pd.DataFrame, no_title: bool = False) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.6 * len(DATASETS), 4.8),
                             sharey=False, squeeze=False)
    axes = axes[0]

    for ax, ds in zip(axes, DATASETS):
        sub = df[df["dataset"] == ds].copy()
        x = sub["best_val_acc"] * 100
        y = sub["test_acc"] * 100
        lo = max(0, min(x.min(), y.min()) - 5)
        hi = min(100, max(x.max(), y.max()) + 5)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.0)

        for mode, color in MODE_COLORS.items():
            for tk, marker in TEACHER_MARKERS.items():
                pts = sub[(sub["target_mode"] == mode) & (sub["teacher"] == tk)]
                ax.scatter(
                    pts["best_val_acc"] * 100, pts["test_acc"] * 100,
                    color=color, marker=marker, s=52, alpha=0.8,
                    edgecolors="white", linewidths=0.5,
                    label=f"{mode} / {tk}",
                )

        mean_gap = ((sub["best_val_acc"] - sub["test_acc"]) * 100).mean()
        ax.text(0.03, 0.94, f"mean val-test gap: {mean_gap:.2f} pp",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85})
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("Best validation accuracy (%)")
        ax.set_ylabel("Held-out test accuracy (%)")
        ax.grid(alpha=0.35)
        ax.legend(fontsize=6.5, ncol=2, loc="lower right")

    if not no_title:
        fig.suptitle("Validation-selected checkpoints vs held-out test accuracy",
                     fontsize=11, y=1.01)
    fig.tight_layout()
    out = _figure_path("val_vs_test_scatter")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def write_deployed_params_table(df: pd.DataFrame, ablation_df: pd.DataFrame) -> None:
    base = _add_deployment_params(df)
    base["source"] = "baseline_grid"
    tables = [base]
    if not ablation_df.empty:
        abl = _add_deployment_params(ablation_df)
        abl["source"] = "ablation"
        tables.append(abl)
    out_df = pd.concat(tables, ignore_index=True, sort=False)
    cols = [
        "source", "dataset", "teacher", "config", "target_mode", "arch", "id",
        "trainable_params", "classifier_params", "deployed_params",
        "deployed_params_m", "gflops", "best_val_acc", "test_acc",
    ]
    cols = [c for c in cols if c in out_df.columns]
    out = TABLE_DIR / "student_deployed_params.csv"
    out_df[cols].to_csv(out, index=False)
    print(f"Saved -> {out}")


def write_teacher_transfer_val_selected(
    df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    teacher_accs: dict,
) -> pd.DataFrame:
    baseline = _validation_selected(df, ["dataset", "teacher"])
    baseline = _add_deployment_params(baseline)

    tuned = pd.DataFrame()
    if not ablation_df.empty:
        has_tuned = (
            ablation_df.groupby(["dataset", "teacher"])["config"]
            .transform(lambda s: (s != "baseline").any())
        )
        tuned = _validation_selected(ablation_df[has_tuned], ["dataset", "teacher"])
        tuned = _add_deployment_params(tuned)

    rows = []
    for ds in DATASETS:
        for tk in ["resnet", "convnext", "vgg"]:
            base = baseline[(baseline["dataset"] == ds) & (baseline["teacher"] == tk)]
            tune = tuned[(tuned["dataset"] == ds) & (tuned["teacher"] == tk)]
            teacher_acc = teacher_accs.get((tk, ds), np.nan)
            if base.empty:
                continue
            b = base.iloc[0]
            row = {
                "dataset": ds,
                "teacher": tk,
                "ckpt_mode": CKPT_MODE[tk],
                "teacher_test_acc": teacher_acc,
                "baseline_selected_id": b["id"],
                "baseline_selected_arch": b["arch"],
                "baseline_val_acc": b["best_val_acc"],
                "baseline_test_acc": b["test_acc"],
                "baseline_retention_pct": b["test_acc"] / teacher_acc * 100
                if not np.isnan(teacher_acc) else np.nan,
                "baseline_teacher_student_gap_pp": (teacher_acc - b["test_acc"]) * 100
                if not np.isnan(teacher_acc) else np.nan,
                "baseline_deployed_params_m": b["deployed_params_m"],
                "baseline_gflops": b["gflops"],
            }
            if not tune.empty:
                t = tune.iloc[0]
                row.update({
                    "tuned_selected_config": t["config"],
                    "tuned_selected_id": t["id"],
                    "tuned_val_acc": t["best_val_acc"],
                    "tuned_test_acc": t["test_acc"],
                    "tuned_gain_over_baseline_pp": (t["test_acc"] - b["test_acc"]) * 100,
                    "tuned_retention_pct": t["test_acc"] / teacher_acc * 100
                    if not np.isnan(teacher_acc) else np.nan,
                    "tuned_teacher_student_gap_pp": (teacher_acc - t["test_acc"]) * 100
                    if not np.isnan(teacher_acc) else np.nan,
                    "tuned_deployed_params_m": t["deployed_params_m"],
                    "tuned_gflops": t["gflops"],
                })
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out = TABLE_DIR / "teacher_summary_val_selected.csv"
    out_df.to_csv(out, index=False)
    print(f"Saved -> {out}")
    return out_df


def plot_retention_vs_teacher_acc(summary_df: pd.DataFrame, no_title: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for _, r in summary_df.iterrows():
        ds = r["dataset"]
        tk = r["teacher"]
        x = r["teacher_test_acc"] * 100
        y = r["baseline_retention_pct"]
        ax.scatter(
            x, y, marker=TEACHER_MARKERS[tk], s=85,
            facecolors="none", edgecolors=DATASET_COLORS.get(ds, "gray"),
            linewidths=1.4, label=f"{ds} baseline" if tk == "resnet" else None,
        )
        ax.text(x + 0.25, y, tk, fontsize=7, va="center")

        if "tuned_retention_pct" in r and not pd.isna(r.get("tuned_retention_pct")):
            ax.scatter(
                x, r["tuned_retention_pct"], marker="*", s=170,
                color=DATASET_COLORS.get(ds, "gray"), edgecolors="black",
                linewidths=0.7, label=f"{ds} tuned" if tk == "resnet" else None,
            )

    ax.set_xlabel("Teacher test accuracy (%)")
    ax.set_ylabel("Student retention (%)")
    ax.grid(alpha=0.35)
    ax.set_ylim(40, 95)
    ax.legend(fontsize=8, loc="lower left")
    if not no_title:
        ax.set_title("Retention is partly inflated for weaker teachers", fontsize=11)

    out = _figure_path("retention_vs_teacher_acc")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_feature_fidelity_vs_accuracy(df: pd.DataFrame, no_title: bool = False) -> None:
    """Plot normalized validation MSE against held-out accuracy."""
    if "val_mse" not in df.columns or df["val_mse"].isna().all():
        print("  [skip] no val_mse column for fidelity plot.")
        return

    work = df.copy()
    work["val_mse_norm"] = np.nan
    for _, idx in work.groupby(["dataset", "teacher", "target_mode"]).groups.items():
        vals = work.loc[idx, "val_mse"].astype(float)
        span = vals.max() - vals.min()
        if span == 0:
            work.loc[idx, "val_mse_norm"] = 0.0
        else:
            work.loc[idx, "val_mse_norm"] = (vals - vals.min()) / span

    rows = []
    for (ds, mode), sub in work.groupby(["dataset", "target_mode"]):
        if len(sub) >= 2:
            rows.append({
                "dataset": ds,
                "target_mode": mode,
                "pearson_norm_val_mse_vs_test_acc": sub["val_mse_norm"].corr(sub["test_acc"]),
                "rows": len(sub),
            })
    out_summary = TABLE_DIR / "feature_fidelity_accuracy_summary.csv"
    pd.DataFrame(rows).to_csv(out_summary, index=False)
    print(f"Saved -> {out_summary}")

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.6 * len(DATASETS), 4.6),
                             sharey=False, squeeze=False)
    axes = axes[0]
    for ax, ds in zip(axes, DATASETS):
        sub = work[work["dataset"] == ds]
        for mode, color in MODE_COLORS.items():
            for tk, marker in TEACHER_MARKERS.items():
                pts = sub[(sub["target_mode"] == mode) & (sub["teacher"] == tk)]
                ax.scatter(
                    pts["val_mse_norm"], pts["test_acc"] * 100,
                    color=color, marker=marker, s=58, alpha=0.82,
                    edgecolors="white", linewidths=0.5,
                    label=f"{mode} / {tk}",
                )
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("Validation MSE, min-max normalized within teacher/mode")
        ax.set_ylabel("Held-out test accuracy (%)")
        ax.grid(alpha=0.35)
        ax.legend(fontsize=6.5, ncol=2, loc="lower right")

    if not no_title:
        fig.suptitle("Feature matching fidelity vs downstream accuracy",
                     fontsize=11, y=1.01)
    fig.tight_layout()
    out = _figure_path("feature_fidelity_vs_accuracy")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def write_augmentation_summary() -> None:
    src = PROJECT_ROOT / "outputs" / "transform_test" / "tables" / "results.csv"
    if not src.exists():
        print(f"  [skip] augmentation sweep table not found: {src}")
        return
    df = pd.read_csv(src)
    summary = (
        df.groupby("Setup", as_index=False)
        .agg(
            mean_test_acc_pct=("Test Acc (%)", "mean"),
            median_test_acc_pct=("Test Acc (%)", "median"),
            min_test_acc_pct=("Test Acc (%)", "min"),
            max_test_acc_pct=("Test Acc (%)", "max"),
            runs=("Test Acc (%)", "count"),
        )
        .sort_values("mean_test_acc_pct", ascending=False)
    )
    out = TABLE_DIR / "augmentation_sweep_summary.csv"
    summary.to_csv(out, index=False)
    print(f"Saved -> {out}")


def run_improvement_outputs(
    df: pd.DataFrame,
    teacher_accs: dict,
    no_title: bool = False,
) -> None:
    ablation_df = load_ablation_data(include_baseline=True)
    teacher_df = load_teacher_points()

    print("\nImprovement figure: accuracy-GFLOPs Pareto...")
    plot_acc_vs_gflops_pareto(df, ablation_df, teacher_df, no_title=no_title)

    print("Improvement figure/table: validation vs test...")
    write_validation_test_gap_tables(df)
    plot_validation_vs_test(df, no_title=no_title)

    print("Improvement table: deployed parameter counts...")
    write_deployed_params_table(df, ablation_df)

    print("Improvement figure/table: retention vs teacher accuracy...")
    summary = write_teacher_transfer_val_selected(df, ablation_df, teacher_accs)
    plot_retention_vs_teacher_acc(summary, no_title=no_title)

    print("Improvement figure/table: feature fidelity vs accuracy...")
    plot_feature_fidelity_vs_accuracy(df, no_title=no_title)

    print("Improvement table: augmentation sweep summary...")
    write_augmentation_summary()


def write_pre_vs_post_table(df: pd.DataFrame) -> None:
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
    out = TABLE_DIR / "pre_vs_post.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved -> {out}")


def write_teacher_summary_table(df: pd.DataFrame, teacher_accs: dict) -> None:
    has_ablation = "baseline_acc" in df.columns
    rows = []
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        for tk in ["resnet", "convnext", "vgg"]:
            pts = sub[sub["teacher"] == tk]
            teacher_acc = teacher_accs.get((tk, ds), np.nan)
            base_best = (pts["baseline_acc"].max()
                         if has_ablation and "baseline_acc" in pts.columns and len(pts)
                         else pts["test_acc"].max() if len(pts) else np.nan)
            ablation_values = (pts["best_ablation_acc"].dropna()
                               if "best_ablation_acc" in pts.columns else pd.Series(dtype=float))
            abl_best = (max(base_best, ablation_values.max())
                        if len(ablation_values) else np.nan)
            mean_acc  = pts["baseline_acc"].mean() if has_ablation and len(pts) else pts["test_acc"].mean() if len(pts) else np.nan
            row = {
                "dataset":              ds,
                "teacher":              tk,
                "ckpt_mode":            CKPT_MODE[tk],
                "teacher_test_acc":     round(teacher_acc, 4) if not np.isnan(teacher_acc) else None,
                "best_student_acc_baseline": round(base_best, 4) if not np.isnan(base_best) else None,
                "best_student_acc_ablation": round(abl_best, 4) if not np.isnan(abl_best) else None,
                "mean_student_acc":     round(mean_acc, 4) if not np.isnan(mean_acc) else None,
                "retention_pct_baseline": round(base_best / teacher_acc * 100, 1) if not np.isnan(base_best) and not np.isnan(teacher_acc) else None,
                "retention_pct_ablation": round(abl_best / teacher_acc * 100, 1) if not np.isnan(abl_best) and not np.isnan(teacher_acc) else None,
            }
            rows.append(row)
    out = TABLE_DIR / "teacher_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved -> {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit the figure-level title and overwrite the canonical figure files.",
    )
    parser.add_argument(
        "--improvements-only",
        action="store_true",
        help="Generate only additive sec5 improvement artifacts with new filenames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    print("\nFigure 1: pre_gap vs post_gap delta...")
    plot_pre_vs_post_delta(df, no_title=args.no_title)

    print("Figure 2: teacher transfer...")
    plot_teacher_transfer(df, teacher_accs, no_title=args.no_title)

    print("Figure 3: acc vs params scatter...")
    plot_acc_vs_params(df, no_title=args.no_title)

    print("Figure 4: acc vs gflops...")
    plot_acc_vs_gflops(df, no_title=args.no_title)

    print("\nTable: pre_vs_post.csv...")
    write_pre_vs_post_table(df)

    print("Table: teacher_summary.csv...")
    write_teacher_summary_table(df, teacher_accs)

    print("\nAdditive improvement outputs...")
    run_improvement_outputs(df, teacher_accs, no_title=args.no_title)

    print("\nDone. Check outputs/figures/ and outputs/tables/.")


if __name__ == "__main__":
    main()
