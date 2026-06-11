"""Generates the student backbone configs for Phase-2 distillation.

6 architectures x 3 teachers x 2 target modes x N datasets.

Output: one JSON file per dataset under ``src/phase2/``, e.g.
    students_oxford-pets.json
    students_flowers-102.json

Structure:
- Small archs (1-3): plain Conv3x3→BN→GELU stages (+ MaxPool on the stem).
- Large archs (4-6): conv stem + ResidualBlock stages for better gradient flow.
  The widest block uses grouped convs (groups=2) to stay within the 3M budget.
- Every backbone ends with a plain conv encoder; the predictor (GAP+dense Linear
  for post_gap, or pool7+1x1conv for pre_gap) is built at runtime by Student
  and is NOT in the JSON.

Hard constraint: encoder params < 3 M (asserted by the generator).
Conv-count rule: 3x3 / 7x7 convs only; 1x1 and shortcut 1x1 do not count.
  A ResidualBlock contributes 2 (its two 3x3 convs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PARAM_BUDGET = 3_000_000
TEACHERS = [("resnet", 2048), ("convnext", 1024), ("vgg", 512)]
TARGET_MODES = ["pre_gap", "post_gap"]

DATASETS: dict[str, int] = {
    "oxford-pets":  37,
    "flowers-102": 102,
}


# ---------------------------------------------------------------------------
# Architecture definitions (encoder stages only — no predictor or head)
# ---------------------------------------------------------------------------

def conv(cin, cout, k, s, p, maxpool=None):
    block = [
        {"type": "conv2d", "in_channels": cin, "out_channels": cout,
         "kernel_size": k, "stride": s, "padding": p},
        {"type": "activation", "activation": "gelu"},
    ]
    if maxpool is not None:
        block.append({"type": "MaxPool2d", "kernel_size": maxpool[0],
                      "stride": maxpool[1], "padding": maxpool[2]})
    return block


def res(cin, cout, stride=1, groups=1):
    return [{"type": "residual_block", "in_channels": cin, "out_channels": cout,
             "stride": stride, "groups": groups}]


ARCHS: dict[str, list] = {
    "arch1_3conv_narrow": [          # 3 convs
        conv(3, 16, 7, 2, 3, (3, 2, 1)),
        conv(16, 32, 3, 2, 1),
        conv(32, 64, 3, 2, 1),
    ],
    "arch2_3conv": [                 # 3 convs
        conv(3, 32, 7, 2, 3, (3, 2, 1)),
        conv(32, 64, 3, 2, 1),
        conv(64, 128, 3, 2, 1),
    ],
    "arch3_4conv": [                 # 4 convs
        conv(3, 32, 7, 2, 3, (3, 2, 1)),
        conv(32, 64, 3, 2, 1),
        conv(64, 128, 3, 2, 1),
        conv(128, 256, 3, 2, 1),
    ],
    "arch4_4conv_res": [             # 2 ResBlocks = 4 convs
        res(3, 128, stride=2),
        res(128, 256, stride=2),
    ],
    "arch5_5conv_res": [             # stem + 2 ResBlocks = 5 convs
        conv(3, 64, 7, 2, 3, (3, 2, 1)),
        res(64, 128, stride=2),
        res(128, 256, stride=2),
    ],
    "arch6_6conv_res": [             # stem(1) + 2 ResBlocks(4) + conv(1) = 6 convs
        conv(3, 64, 7, 2, 3, (3, 2, 1)),
        res(64, 128, stride=2),
        res(128, 256, stride=2),
        conv(256, 256, 3, 1, 1),
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stage_out_channels(stage: list[dict]) -> int | None:
    out = None
    for a in stage:
        if a["type"] in ("conv2d", "residual_block"):
            out = a["out_channels"]
    return out


def build_layers(stages: list[list[dict]]) -> list[dict]:
    """Wraps each stage in a named block dict (layer1, layer2, …)."""
    return [{f"layer{i}": stage} for i, stage in enumerate(stages, start=1)]


def _attr_params(a: dict) -> int:
    if a["type"] == "conv2d":
        g = a.get("groups", 1)
        return (a["in_channels"] * a["out_channels"] * a["kernel_size"] ** 2 // g
                + 2 * a["out_channels"])
    if a["type"] == "residual_block":
        cin, cout, g = a["in_channels"], a["out_channels"], a.get("groups", 1)
        p = cin * cout * 9 // g + cout * cout * 9 // g + 4 * cout
        if a.get("stride", 1) != 1 or cin != cout:
            p += cin * cout + 2 * cout
        return p
    return 0


def encoder_params(layers: list[dict]) -> int:
    return sum(_attr_params(a) for block in layers for stage in block.values() for a in stage)


def n_conv_layers(stages: list[list[dict]]) -> int:
    total = 0
    for stage in stages:
        for a in stage:
            if a["type"] == "conv2d" and a["kernel_size"] > 1:
                total += 1
            elif a["type"] == "residual_block":
                total += 2
    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate(dataset_name: str, num_classes: int, out_dir: Path) -> list[dict]:
    students = []
    for arch_name, stages in ARCHS.items():
        for teacher, teacher_dim in TEACHERS:
            layers = build_layers(stages)
            params = encoder_params(layers)
            if params > PARAM_BUDGET:
                raise ValueError(
                    f"{arch_name}/{teacher}: encoder has {params:,} params > {PARAM_BUDGET:,}."
                )
            last_ch = _stage_out_channels(stages[-1])
            for mode in TARGET_MODES:
                students.append({
                    "id":           f"{arch_name}__{teacher}_td{teacher_dim}_{mode}",
                    "arch":         arch_name,
                    "teacher":      teacher,
                    "teacher_dim":  teacher_dim,
                    "num_classes":  num_classes,
                    "target_mode":  mode,
                    "n_convs":      n_conv_layers(stages),
                    "last_enc_channels": last_ch,
                    "encoder_params":    params,
                    "layers":       layers,
                })
    out_path = out_dir / f"students_{dataset_name}.json"
    out_path.write_text(json.dumps(students, indent=2), encoding="utf-8")
    print(f"[{dataset_name}] Wrote {len(students)} configs "
          f"({len(ARCHS)} archs × {len(TEACHERS)} teachers × {len(TARGET_MODES)} modes) "
          f"→ {out_path}")
    for s in students[:6]:  # print a sample
        print(f"  {s['id']:55s} convs={s['n_convs']} enc={s['encoder_params']/1e6:.2f}M "
              f"mode={s['target_mode']}")
    print("  ...")
    return students


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Which dataset JSONs to generate (default: all).",
    )
    args = parser.parse_args()
    out_dir = Path(__file__).resolve().parent
    for ds in args.datasets:
        generate(ds, DATASETS[ds], out_dir)


if __name__ == "__main__":
    main()
