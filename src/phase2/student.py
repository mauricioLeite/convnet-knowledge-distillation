"""Config-driven student network for Phase 2 feature distillation.

A student is described by a list of named blocks (see ``random_students.json``).
Each block maps a name to a list of layer specs, e.g.::

    layers = [
        {"layer1": [
            {"type": "conv2d", "in_channels": 3, "out_channels": 16,
             "kernel_size": 7, "stride": 2, "padding": 3},
            {"type": "activation", "activation": "gelu"},
            {"type": "MaxPool2d", "kernel_size": 3, "stride": 2, "padding": 1},
        ]},
        {"avgpool": [{"type": "AdaptiveAvgPool2d", "output_size": [1, 1]}]},
        {"flatten": [{"type": "flatten"}]},
        {"project": [{"type": "linear", "in_features": 64, "out_features": 2048}]},
        {"classifier": [{"type": "linear", "in_features": 2048, "out_features": 37}]},
    ]

Every block except ``project`` and ``classifier`` is part of the encoder and is
applied, in order, by :meth:`encode`. ``project`` maps the pooled encoder
features to the teacher's feature dimension (used for the distillation MSE) and
``classifier`` produces the final logits.
"""

from torch import nn


def _build_modules(attribute: dict) -> list[nn.Module]:
    """Builds the torch modules for a single layer spec.

    ``conv2d`` expands into a Conv2d followed by a BatchNorm2d so the JSON only
    needs to declare the convolution once.
    """
    kind = attribute["type"]

    if kind == "conv2d":
        return [
            nn.Conv2d(
                in_channels=attribute["in_channels"],
                out_channels=attribute["out_channels"],
                kernel_size=attribute["kernel_size"],
                stride=attribute["stride"],
                padding=attribute["padding"],
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

    if kind == "linear":
        return [nn.Linear(
            in_features=attribute["in_features"],
            out_features=attribute["out_features"],
        )]

    raise ValueError(f"Unknown layer type: {kind}")


class Student(nn.Module):
    """A small convnet built from a JSON-style architecture description."""

    PROJECTION_NAME = "project"
    CLASSIFIER_NAME = "classifier"

    def __init__(self, layers: list[dict], teacher_dim: int, num_classes: int):
        super().__init__()
        self.teacher_dim = teacher_dim
        self.num_classes = num_classes

        # All named blocks live in a single ModuleDict so their parameter names
        # carry the block name (e.g. "blocks.classifier.0.weight"), which the
        # distillation optimizer relies on to exclude the classifier.
        self.blocks = nn.ModuleDict()
        for block in layers:
            for name, attributes in block.items():
                sequential = nn.Sequential()
                for attribute in attributes:
                    for module in _build_modules(attribute):
                        sequential.append(module)
                self.blocks[name] = sequential

        self._encoder_names = [
            name for name in self.blocks
            if name not in (self.PROJECTION_NAME, self.CLASSIFIER_NAME)
        ]

    def encode(self, x):
        """Runs the convolutional encoder, returning the pooled feature vector."""
        for name in self._encoder_names:
            x = self.blocks[name](x)
        return x

    def project(self, x):
        """Projects encoder features into the teacher's feature space."""
        return self.blocks[self.PROJECTION_NAME](self.encode(x))

    def forward(self, x):
        """Returns classification logits."""
        return self.blocks[self.CLASSIFIER_NAME](self.project(x))
