"""Pretrained teacher wrappers for Phase 1 linear probing."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet50_Weights,
    VGG16_BN_Weights,
    convnext_tiny,
    resnet50,
    vgg16_bn,
)


def _normalize_backbone_name(backbone_name: str) -> str:
    name = backbone_name.lower().replace("-", "_")
    aliases = {
        "vgg": "vgg16_bn",
        "vgg16": "vgg16_bn",
        "vgg16_bn": "vgg16_bn",
        "resnet": "resnet50",
        "resnet_50": "resnet50",
        "resnet50": "resnet50",
        "convnext": "convnext_tiny",
        "convnext_tiny": "convnext_tiny",
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. Expected one of: "
            "vgg16_bn, resnet50, convnext_tiny."
        ) from exc


class TeacherModel(nn.Module):
    """Wraps a pretrained ImageNet backbone for Phase 1 linear probing.

    Attributes:
        encoder: Frozen feature extractor, excluding global pooling and head.
        gap: Global average pooling layer.
        classifier: Trainable linear classifier for the target dataset.
        backbone_name: Canonical backbone identifier.
        feature_dim: Dimensionality of the pooled feature vector.
        num_classes: Number of target classes.
    """

    def __init__(self, backbone_name: str, num_classes: int) -> None:
        """Initializes a frozen pretrained backbone and fresh linear head.

        Args:
            backbone_name: One of ``vgg16_bn``, ``resnet50``, or
                ``convnext_tiny``.
            num_classes: Number of classes for the downstream dataset.
        """
        super().__init__()
        self.backbone_name = _normalize_backbone_name(backbone_name)
        self.num_classes = num_classes
        self.encoder, self.feature_dim = self._build_encoder(self.backbone_name)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @staticmethod
    def _build_encoder(backbone_name: str) -> tuple[nn.Module, int]:
        if backbone_name == "vgg16_bn":
            model = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
            return model.features, 512
        if backbone_name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            return nn.Sequential(*list(model.children())[:-2]), 2048
        if backbone_name == "convnext_tiny":
            model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            return model.features, 768
        raise ValueError(f"Unsupported backbone '{backbone_name}'.")

    def train(self, mode: bool = True) -> "TeacherModel":
        """Sets the classifier mode while keeping the encoder in eval mode.

        Args:
            mode: Whether to use training mode for trainable modules.

        Returns:
            This model.
        """
        super().train(mode)
        self.encoder.eval()
        return self

    def unfreeze_top(self, n_blocks: int = 1) -> None:
        """Unfreezes the last ``n_blocks`` blocks of the encoder for fine-tuning.

        Unfreezes only the last ``n_blocks`` sub-blocks *within* the final
        encoder stage (``layer4`` for ResNet-50, ``features[7]`` for
        ConvNeXt-Tiny, the conv5 block for VGG-16-BN), leaving the rest of the
        encoder frozen. This is a light-touch alternative to unfreezing a whole
        stage, which would be ~50–64% of the encoder.

        For VGG-16-BN the encoder is a flat ``nn.Sequential`` with no block
        containers, so the final stage (conv5) is located by index between the
        last two ``MaxPool2d`` boundaries, and each ``Conv2d`` (with its
        following BN/ReLU) counts as one block.

        For ResNet-50 and VGG-16-BN, ``BatchNorm2d`` affine parameters inside
        unfrozen blocks are re-frozen after unfreezing so that running
        statistics are never disturbed. The encoder is kept in eval mode during
        training by ``TeacherModel.train()``, which provides an additional
        guarantee that BN running stats are not updated regardless of
        ``requires_grad``.

        Args:
            n_blocks: Number of blocks to unfreeze, counted from the top of the
                final encoder stage.

        Raises:
            ValueError: If ``n_blocks`` is out of range for the final stage.
        """
        if self.backbone_name == "vgg16_bn":
            self._unfreeze_top_vgg(n_blocks)
            return

        top_stage = list(self.encoder.children())[-1]
        blocks = list(top_stage.children())
        if n_blocks < 1 or n_blocks > len(blocks):
            raise ValueError(
                f"n_blocks must be between 1 and {len(blocks)} "
                f"(blocks in the final stage), got {n_blocks}."
            )
        for block in blocks[-n_blocks:]:
            block.requires_grad_(True)
            if self.backbone_name == "resnet50":
                for module in block.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.requires_grad_(False)

    def _unfreeze_top_vgg(self, n_blocks: int) -> None:
        """Unfreezes the last ``n_blocks`` conv layers of VGG-16-BN's conv5.

        VGG-16-BN's encoder is a flat ``nn.Sequential`` of Conv2d/BatchNorm2d/
        ReLU/MaxPool2d layers. The final conv block (conv5) is the span between
        the last two ``MaxPool2d`` boundaries; each ``Conv2d`` (together with its
        trailing BN and ReLU) is treated as one block. ``BatchNorm2d`` affine
        parameters in the unfrozen region are re-frozen so running statistics
        stay fixed, matching the ResNet-50 handling in ``unfreeze_top``.
        """
        layers = list(self.encoder.children())
        pool_positions = [
            index
            for index, layer in enumerate(layers)
            if isinstance(layer, nn.MaxPool2d)
        ]
        if len(pool_positions) < 2:
            raise ValueError(
                "Expected at least two MaxPool2d layers in the VGG encoder to "
                f"locate the final conv block, found {len(pool_positions)}."
            )
        block_start = pool_positions[-2] + 1
        block_end = pool_positions[-1]  # exclusive; the closing MaxPool
        conv_positions = [
            index
            for index in range(block_start, block_end)
            if isinstance(layers[index], nn.Conv2d)
        ]
        if n_blocks < 1 or n_blocks > len(conv_positions):
            raise ValueError(
                f"n_blocks must be between 1 and {len(conv_positions)} "
                f"(conv layers in the final VGG block), got {n_blocks}."
            )
        first_conv = conv_positions[-n_blocks]
        for layer in layers[first_conv:block_end]:
            layer.requires_grad_(True)
            for module in layer.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.requires_grad_(False)

    def extract_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Extracts pre-GAP feature maps and post-GAP pooled features.

        When the encoder is fully frozen (all ``requires_grad=False``) and the
        input does not require gradients, no autograd graph is built. When
        ``unfreeze_top()`` has been called, gradients flow through the unfrozen
        stages so the fine-tuning optimizer can update them.

        Args:
            x: Input batch with shape ``(N, 3, 224, 224)``.

        Returns:
            Tuple ``(feature_map, pooled_vector)``.
        """
        feature_map = self.encoder(x)
        pooled_vector = self.gap(feature_map).flatten(1)
        return feature_map, pooled_vector

    def forward(self, x: Tensor) -> Tensor:
        """Computes classification logits.

        Args:
            x: Input image batch.

        Returns:
            Logits with shape ``(N, num_classes)``.
        """
        _, pooled_vector = self.extract_features(x)
        return self.classifier(pooled_vector)


def get_teacher(backbone_name: str, num_classes: int) -> TeacherModel:
    """Creates a frozen teacher encoder with a fresh linear classifier.

    Args:
        backbone_name: Teacher backbone identifier.
        num_classes: Number of downstream classes.

    Returns:
        Configured teacher model.
    """
    return TeacherModel(backbone_name=backbone_name, num_classes=num_classes)

