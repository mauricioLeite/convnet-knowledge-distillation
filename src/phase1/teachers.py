from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Tiny_Weights,
    ResNet50_Weights,
    VGG16_BN_Weights,
    convnext_base,
    convnext_tiny,
    resnet50,
    vgg16_bn,
)


def _normalize_backbone_name(backbone_name: str) -> str:
    name = backbone_name.lower().replace("-", "_")
    aliases = {
        "vgg16_bn": "vgg16_bn",
        "resnet50": "resnet50",
        "convnext_base": "convnext_base",
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. Expected one of: vgg16_bn, resnet50, convnext_tiny, convnext_base."
        ) from exc


class TeacherModel(nn.Module):

    def __init__(self, backbone_name: str, num_classes: int) -> None:
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
        if backbone_name == "convnext_base":
            model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
            return model.features, 1024
        raise ValueError(f"Unsupported backbone '{backbone_name}'.")

    def train(self, mode: bool = True) -> "TeacherModel":
        super().train(mode)
        self.encoder.eval()
        return self

    def unfreeze_top(self, n_blocks: int = 1) -> None:
        """Unfreezes the last ``n_blocks`` blocks of the encoder for fine-tuning. """        
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

    def extract_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Extracts pre-GAP feature maps and post-GAP pooled features. """
        feature_map = self.encoder(x)
        pooled_vector = self.gap(feature_map).flatten(1)
        return feature_map, pooled_vector

    def forward(self, x: Tensor) -> Tensor:
        """Computes classification logits. """
        _, pooled_vector = self.extract_features(x)
        return self.classifier(pooled_vector)


def get_teacher(backbone_name: str, num_classes: int) -> TeacherModel:
    return TeacherModel(backbone_name=backbone_name, num_classes=num_classes)


def load_teacher_checkpoint(
    backbone_name: str,
    dataset_name: str,
    num_classes: int,
    checkpoint_dir: "Path | str",
    mode: str = "frozen",
    device: "str | torch.device" = "cpu",
) -> TeacherModel:
    from pathlib import Path

    ckpt_path = Path(checkpoint_dir) / mode / f"{backbone_name}_{dataset_name}.pth"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = TeacherModel(backbone_name=backbone_name, num_classes=num_classes).to(device)
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    if mode == "finetune":
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.encoder.requires_grad_(False)
    model.encoder.eval()
    return model

