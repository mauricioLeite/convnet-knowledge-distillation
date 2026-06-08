"""Student whose final conv matches the teacher's channel count + glued head.

Design
------
The config-driven conv backbone (the JSON ``layer*`` blocks) is built so its
**final conv outputs ``teacher_dim`` channels** (2048 for ResNet-50, 1024 for
ConvNeXt-Base, 512 for VGG-16) -- i.e. the student's last conv map has the same
channel count as the teacher's::

    encoder convs -> [teacher_dim, H, W]

On this map the teacher's *full* pretrained head is glued (its own
avgpool / flatten / linear, see :func:`teacher_head`), so the head's pooling
handles the spatial size (1x1 for ResNet/ConvNeXt, 7x7 for VGG) -- every teacher
works, including VGG (whose head flattens its 7x7 map -> 512*7*7 = 25088).

Distillation target -- :meth:`project` reuses the head's *own* pooling
(``classifier[0]``) so it matches exactly what the teacher's final classifier
consumes:

- ResNet-50 / ConvNeXt: ``AdaptiveAvgPool2d(1)`` -> post-GAP ``[teacher_dim]``.
- VGG-16: ``AdaptiveAvgPool2d(7)`` -> ``[512, 7, 7]`` flattened to 25088 -- the
  spatial map the VGG classifier actually uses (its features feed the flattened
  7x7 map, not a post-GAP vector).

Because there is no separate projection, the backbone's final conv *must* output
``teacher_dim`` channels (asserted in ``__init__``).
"""

import copy

import torch
from torch import nn


def _build_modules(attribute: dict) -> list[nn.Module]:
    """Builds the torch modules for a single layer spec (self-contained)."""
    kind = attribute["type"]

    if kind == "conv2d":
        return [
            nn.Conv2d(
                in_channels=attribute["in_channels"],
                out_channels=attribute["out_channels"],
                kernel_size=attribute["kernel_size"],
                stride=attribute["stride"],
                padding=attribute["padding"],
                groups=attribute.get("groups", 1),
                bias=attribute.get("bias", False),
            ),
            nn.BatchNorm2d(num_features=attribute["out_channels"]),
        ]

    if kind == "activation":
        activations = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        name = attribute["activation"]
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return [activations[name]()]

    if kind == "MaxPool2d":
        return [nn.MaxPool2d(
            kernel_size=attribute["kernel_size"],
            stride=attribute["stride"],
            padding=attribute["padding"],
        )]

    if kind == "Dropout":
        return [nn.Dropout(p=attribute["p"])]

    if kind == "AdaptiveAvgPool2d":
        return [nn.AdaptiveAvgPool2d(output_size=tuple(attribute["output_size"]))]

    if kind == "flatten":
        return [nn.Flatten(
            start_dim=attribute.get("start_dim", 1),
            end_dim=attribute.get("end_dim", -1),
        )]
    
    if kind == "residual_block":
        return [ResidualBlock(
            in_channels=attribute["in_channels"],
            out_channels=attribute["out_channels"],
            stride=attribute.get("stride", 1),
            groups=attribute.get("groups", 1),
        )]

    if kind == "linear":
        return [nn.Linear(
            in_features=attribute["in_features"],
            out_features=attribute["out_features"],
        )]

    raise ValueError(f"Unknown layer type: {kind}")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        # Caminho principal (2 convoluções 3x3). ``groups`` divide os params das
        # convs por g (grouped conv) -- usado nos archs grandes p/ caber em 3M.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Caminho de atalho (skip): 1x1 (groups=1) só quando muda dims/stride.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.gelu(out)
        return out


class StudentTeacherHead(nn.Module):
    """Conv backbone ending in teacher_dim channels + glued teacher head."""

    # Tail blocks are owned by the model/head, not the JSON.
    SKIP_BLOCKS = ("avgpool", "flatten", "project", "classifier")

    def __init__(
        self,
        layers: list[dict],
        teacher_dim: int,
        num_classes: int,
        head: nn.Module,
        freeze_head: bool = False,
    ):
        super().__init__()
        self.teacher_dim = teacher_dim
        self.num_classes = num_classes

        # Conv backbone from the JSON, tracking the last conv's channel count.
        blocks = []
        last_channels = None
        for block in layers:
            for name, attributes in block.items():
                if name in self.SKIP_BLOCKS:
                    continue
                sequential = nn.Sequential()
                for attribute in attributes:
                    for module in _build_modules(attribute):
                        sequential.append(module)
                    if attribute["type"] in ("conv2d", "residual_block"):
                        last_channels = attribute["out_channels"]
                blocks.append(sequential)
        if last_channels is None:
            raise ValueError("No conv2d layer found in the backbone.")
        if last_channels != teacher_dim:
            raise ValueError(
                f"Final conv must output teacher_dim={teacher_dim} channels to match "
                f"the glued head (no projection layer); got {last_channels}."
            )

        self.encoder = nn.Sequential(*blocks)
        # Teacher's full pretrained head ([teacher_dim, H, W] -> logits). Named
        # "classifier" so the optimizer's name-based encoder/head split works.
        # ``head[0]`` is the teacher's avgpool, reused by :meth:`project`.
        self.classifier = head
        if freeze_head:
            for param in self.classifier.parameters():
                param.requires_grad = False

    def project(self, x):
        """Distillation target: the representation fed to the head's classifier.
        """
        return torch.flatten(self.classifier[0](self.encoder(x)), 1)

    def forward(self, x):
        """Logits from the teacher head applied to the conv map."""
        return self.classifier(self.encoder(x))


def teacher_head(teacher: nn.Module, kind: str) -> nn.Module:
    """Returns a deep copy of the teacher's *full* head: [C_t, H, W] -> logits.

    Deep-copied so fine-tuning the student's head does not mutate the (frozen)
    teacher used for KD targets. Each head replicates the teacher's forward from
    the final conv map onward -- its own avgpool, the flatten where the teacher
    keeps one, and the linear classifier.

    Args:
        teacher: A loaded torchvision teacher with its Oxford-Pets head.
        kind: One of ``resnet50``, ``convnext_base``, ``vgg16``.
    """
    if kind == "resnet50":
        # [2048,H,W] -> avgpool(1) -> flatten -> fc
        return copy.deepcopy(nn.Sequential(teacher.avgpool, nn.Flatten(1), teacher.fc))
    if kind == "convnext_base":
        # [1024,H,W] -> avgpool(1) -> classifier(LayerNorm2d, Flatten, Linear)
        return copy.deepcopy(nn.Sequential(teacher.avgpool, teacher.classifier))
    if kind == "vgg16":
        # [512,H,W] -> avgpool(7) -> flatten -> classifier(25088 -> ... -> num_classes)
        return copy.deepcopy(nn.Sequential(teacher.avgpool, nn.Flatten(1), teacher.classifier))
    raise ValueError(f"Unknown teacher kind: {kind}")
