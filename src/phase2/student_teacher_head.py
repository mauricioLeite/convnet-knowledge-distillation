import copy

import torch
from torch import nn


def _build_modules(attribute: dict) -> list[nn.Module]:
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
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

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
    SKIP_BLOCKS = ("avgpool", "flatten", "project", "classifier")

    def __init__(
        self,
        layers: list[dict],
        teacher_dim: int,
        num_classes: int,
        head: nn.Module,
        kind: str,
        freeze_head: bool = False,
    ):
        super().__init__()
        self.teacher_dim = teacher_dim
        self.num_classes = num_classes
        self.kind = kind

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
                f"Final conv must output teacher_dim={teacher_dim} channels to match the glued head (no projection layer); got {last_channels}."
            )

        self.encoder = nn.Sequential(*blocks)
        self.classifier = head
        if freeze_head:
            for param in self.classifier.parameters():
                param.requires_grad = False

    def pooled_feature(self, fmap):
        x = self.classifier[0](fmap)
        if self.kind == "convnext_base":
            x = self.classifier[1][0](x)
        return torch.flatten(x, 1)

    def project(self, x):
        """Distillation target: the representation fed to the head's classifier."""
        return self.pooled_feature(self.encoder(x))

    def forward(self, x):
        """Logits from the teacher head applied to the conv map."""
        return self.classifier(self.encoder(x))


def teacher_head(teacher: nn.Module, kind: str) -> nn.Module:
    """Returns a deep copy of the teacher's full head."""
    if kind == "resnet50":
        return copy.deepcopy(nn.Sequential(teacher.avgpool, nn.Flatten(1), teacher.fc))
    if kind == "convnext_base":
        return copy.deepcopy(nn.Sequential(teacher.avgpool, teacher.classifier))
    if kind == "vgg16":
        return copy.deepcopy(nn.Sequential(teacher.avgpool, nn.Flatten(1), teacher.classifier))
    raise ValueError(f"Unknown teacher kind: {kind}")
