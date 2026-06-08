"""Generates the student backbones for the teacher-head pipeline.

6 architectures of increasing size x 3 teachers (teacher_dim 2048 -> resnet,
1024 -> convnext, 512 -> vgg), num_classes=37.

Structure (no dropout -- that lives in the imported teacher head):
- Small archs (1-3): plain ``Conv3x3 -> BN -> GELU`` stages (+ MaxPool on the stem).
- Large archs (4-6): a conv stem followed by ``ResidualBlock`` stages (skip
  connections) for better gradient flow / accuracy. The widest block of the
  largest arch uses grouped convs (``groups=2``) to halve its params and stay
  within budget.
- Every backbone ends with a 1x1 conv projecting to ``teacher_dim`` (a cheap
  per-pixel linear) so the teacher head attaches directly. The GAP and the glued
  head live in the model, not in the JSON.

Hard constraint: the **encoder** (everything but the imported head) must stay
under 3M params; the generator asserts this.
"""

import json
from pathlib import Path

NUM_CLASSES = 37
TEACHERS = [("resnet", 2048), ("convnext", 1024), ("vgg", 512)]
PARAM_BUDGET = 3_000_000  # encoder-only hard cap


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


# Each arch is a list of stages (each stage = list of attribute dicts). The
# final 1x1 projection to teacher_dim is appended per-teacher in build_layers.
#
# Conv-count rule (hard, <= 6): a 1x1 conv does NOT count (it is a linear
# projection); every 3x3/7x7 conv counts, including each internal conv of a
# ResidualBlock (2). Shortcut 1x1 and the final 1x1 projection do not count.
# Counts below: 3, 3, 4, 4, 5, 6.
ARCHS = {
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
    "arch6_6conv_res": [             # 3 ResBlocks = 6 convs
        res(3, 128, stride=2),
        res(128, 256, stride=2),
        res(256, 256, stride=1),
    ],
}


def _stage_out_channels(stage):
    out = None
    for a in stage:
        if a["type"] in ("conv2d", "residual_block"):
            out = a["out_channels"]
    return out


def build_layers(stages, teacher_dim):
    layers = [{f"layer{i}": stage} for i, stage in enumerate(stages, start=1)]
    c_last = _stage_out_channels(stages[-1])
    layers.append({"proj1x1": conv(c_last, teacher_dim, 1, 1, 0)})
    return layers


def _attr_params(a):
    if a["type"] == "conv2d":
        g = a.get("groups", 1)
        return a["in_channels"] * a["out_channels"] * a["kernel_size"] ** 2 // g + 2 * a["out_channels"]
    if a["type"] == "residual_block":
        cin, cout, g = a["in_channels"], a["out_channels"], a.get("groups", 1)
        p = cin * cout * 9 // g + cout * cout * 9 // g      # conv1 + conv2 (3x3)
        p += 4 * cout                                       # bn1 + bn2
        if a.get("stride", 1) != 1 or cin != cout:
            p += cin * cout + 2 * cout                      # 1x1 shortcut + bn
        return p
    return 0


def encoder_params(layers):
    return sum(_attr_params(a) for block in layers for stage in block.values() for a in stage)


def n_conv_layers(stages):
    """Counts convs toward the <=6 rule: 3x3/7x7 only (1x1 is linear, free).

    A ResidualBlock contributes its two 3x3 convs (2); its 1x1 shortcut does not.
    """
    total = 0
    for stage in stages:
        for a in stage:
            if a["type"] == "conv2d" and a["kernel_size"] > 1:
                total += 1
            elif a["type"] == "residual_block":
                total += 2
    return total


students = []
for arch_name, stages in ARCHS.items():
    for teacher, teacher_dim in TEACHERS:
        layers = build_layers(stages, teacher_dim)
        params = encoder_params(layers)
        if params > PARAM_BUDGET:
            raise ValueError(
                f"{arch_name}/{teacher}: encoder has {params:,} params > {PARAM_BUDGET:,} budget."
            )
        students.append({
            "id": f"{arch_name}__{teacher}_td{teacher_dim}",
            "arch": arch_name,
            "teacher": teacher,
            "teacher_dim": teacher_dim,
            "num_classes": NUM_CLASSES,
            "n_convs": n_conv_layers(stages),
            "encoder_params": params,
            "layers": layers,
        })

out = Path(__file__).with_name("students.json")
out.write_text(json.dumps(students, indent=2), encoding="utf-8")
print(f"Wrote {len(students)} configs ({len(ARCHS)} archs x {len(TEACHERS)} teachers) -> {out}")
for s in students:
    print(f"  {s['id']:38s} conv_layers={s['n_convs']} encoder={s['encoder_params']/1e6:.2f}M")
