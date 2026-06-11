"""Unified config-driven student for Phase 2 distillation.

Two target modes share one class:

post_gap
    conv encoder -> GAP(1) -> flatten -> Linear(enc_dim, teacher_dim)
    ``project(x)`` returns a ``[N, teacher_dim]`` vector matched (MSE) against
    the teacher's post-GAP pooled vector.

pre_gap
    conv encoder -> AdaptiveAvgPool2d(7)
                 -> Conv2d(3x3, last_ch->last_ch) -> BN -> GELU   (spatial block)
                 -> Conv2d(1x1, last_ch->teacher_dim) -> BN        (channel projection)
    ``project(x)`` returns a ``[N, teacher_dim, 7, 7]`` map matched (MSE)
    against the teacher's pre-GAP feature map.

In both modes the frozen Phase-1 classifier (``Linear(teacher_dim, num_classes)``,
deep-copied from the teacher) is attached as ``self.classifier``.  The
classification path is:

    pre_gap:  map  -> GAP(1) -> flatten -> classifier
    post_gap: vector -> classifier

The optimizer name-based split (any param whose name contains ``"classifier"``
goes to the head group; everything else is encoder) works for both modes
because all backbone + predictor params carry non-``classifier`` names.
"""

from __future__ import annotations

import copy
import torch
from torch import nn, Tensor


# ---------------------------------------------------------------------------
# Layer builder (shared with gen_students.py)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.gelu  = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=groups, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.gelu(out)


def _build_modules(attribute: dict) -> list[nn.Module]:
    kind = attribute["type"]
    if kind == "conv2d":
        return [
            nn.Conv2d(
                attribute["in_channels"], attribute["out_channels"],
                attribute["kernel_size"], stride=attribute["stride"],
                padding=attribute["padding"],
                groups=attribute.get("groups", 1), bias=attribute.get("bias", False),
            ),
            nn.BatchNorm2d(attribute["out_channels"]),
        ]
    if kind == "activation":
        return [{"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[attribute["activation"]]()]
    if kind == "MaxPool2d":
        return [nn.MaxPool2d(attribute["kernel_size"], stride=attribute["stride"], padding=attribute["padding"])]
    if kind == "AdaptiveAvgPool2d":
        return [nn.AdaptiveAvgPool2d(tuple(attribute["output_size"]))]
    if kind == "flatten":
        return [nn.Flatten(attribute.get("start_dim", 1), attribute.get("end_dim", -1))]
    if kind == "residual_block":
        return [ResidualBlock(attribute["in_channels"], attribute["out_channels"],
                              stride=attribute.get("stride", 1), groups=attribute.get("groups", 1))]
    if kind == "linear":
        return [nn.Linear(attribute["in_features"], attribute["out_features"])]
    if kind == "Dropout":
        return [nn.Dropout(p=attribute["p"])]
    raise ValueError(f"Unknown layer type: {kind}")


# ---------------------------------------------------------------------------
# Shared block names that belong to the predictor / head, not the encoder
# ---------------------------------------------------------------------------
_PREDICTOR_BLOCKS = {"avgpool", "flatten", "project", "classifier",
                     "pre_gap_pool", "proj1x1"}


class Student(nn.Module):
    """Conv backbone + predictor + frozen Phase-1 classifier.

    Args:
        layers: JSON-style list of named block dicts (from ``students.json``).
        teacher_dim: Feature dimension of the teacher (2048 / 1024 / 512).
        num_classes: Number of downstream classes.
        classifier: Deep copy of the Phase-1 ``Linear(teacher_dim, num_classes)``.
        target_mode: ``"pre_gap"`` or ``"post_gap"``.
        freeze_classifier: If True (default), freeze the glued head.
    """

    def __init__(
        self,
        layers: list[dict],
        teacher_dim: int,
        num_classes: int,
        classifier: nn.Linear,
        target_mode: str,
        freeze_classifier: bool = True,
    ) -> None:
        super().__init__()
        if target_mode not in ("pre_gap", "post_gap"):
            raise ValueError(f"target_mode must be 'pre_gap' or 'post_gap', got {target_mode!r}")
        self.teacher_dim  = teacher_dim
        self.num_classes  = num_classes
        self.target_mode  = target_mode

        # ---- conv encoder (shared JSON blocks) ---------------------------
        enc_blocks: list[nn.Module] = []
        last_channels: int | None = None
        for block in layers:
            for name, attributes in block.items():
                if name in _PREDICTOR_BLOCKS:
                    continue
                seq = nn.Sequential()
                for attr in attributes:
                    for module in _build_modules(attr):
                        seq.append(module)
                    if attr["type"] in ("conv2d", "residual_block"):
                        last_channels = attr["out_channels"]
                enc_blocks.append(seq)
        if last_channels is None:
            raise ValueError("No conv2d / residual_block found in backbone layers.")
        self.encoder = nn.Sequential(*enc_blocks)

        # ---- predictor ---------------------------------------------------
        if target_mode == "post_gap":
            # encoder -> GAP(1) -> flatten -> Linear(enc_dim -> teacher_dim)
            self.predictor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(last_channels, teacher_dim),
            )
        else:  # pre_gap
            # encoder -> pool(7) -> 3x3 conv (spatial) -> 1x1 proj -> [N,C,7,7]
            self.predictor = nn.Sequential(
                nn.AdaptiveAvgPool2d(7),
                nn.Conv2d(last_channels, last_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(last_channels),
                nn.GELU(),
                nn.Conv2d(last_channels, teacher_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(teacher_dim),
            )

        # ---- frozen Phase-1 classifier -----------------------------------
        self.classifier = copy.deepcopy(classifier)
        if freeze_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = False

        # GAP used by the pre_gap classification path
        if target_mode == "pre_gap":
            self._gap = nn.AdaptiveAvgPool2d(1)

    def project(self, x: Tensor) -> Tensor:
        """Distillation target: the representation matched against the teacher.

        Returns ``[N, teacher_dim]`` for post_gap, ``[N, teacher_dim, 7, 7]``
        for pre_gap.
        """
        return self.predictor(self.encoder(x))

    def forward(self, x: Tensor) -> Tensor:
        """Classification logits via the frozen Phase-1 head."""
        feat = self.project(x)
        if self.target_mode == "pre_gap":
            feat = self._gap(feat).flatten(1)
        return self.classifier(feat)
