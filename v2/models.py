from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleUNet(nn.Module):
    def __init__(self, num_classes: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 8, c * 16)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class TorchvisionSegmentationWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


class HFSegmentationWrapper(nn.Module):
    def __init__(self, model_id: str, num_classes: int = 2) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForSemanticSegmentation
        except ImportError as exc:
            raise ImportError("Install transformers to use model names like hf:nvidia/segformer-b0-finetuned-...") from exc

        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            model_id,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=x)
        logits = output.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


@dataclass(frozen=True)
class ModelInfo:
    name: str
    description: str


AVAILABLE_MODELS = [
    ModelInfo("unet", "Small UNet implemented in this repository."),
    ModelInfo("fcn_resnet50", "torchvision FCN with ResNet-50 backbone."),
    ModelInfo("deeplabv3_resnet50", "torchvision DeepLabV3 with ResNet-50 backbone."),
    ModelInfo("deeplabv3_resnet101", "torchvision DeepLabV3 with ResNet-101 backbone."),
    ModelInfo("lraspp_mobilenet_v3_large", "torchvision LR-ASPP MobileNetV3 segmentation model."),
    ModelInfo("smp:<arch>:<encoder>", "Optional segmentation_models_pytorch model, e.g. smp:Unet:resnet50."),
    ModelInfo("hf:<model_id>", "Optional Hugging Face transformers semantic segmentation model."),
]


def create_model(name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    normalized = name.strip()
    lower = normalized.lower()

    if lower == "unet":
        return SimpleUNet(num_classes=num_classes)

    if lower.startswith("smp:"):
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError("Install segmentation-models-pytorch to use smp:<arch>:<encoder> models.") from exc
        parts = normalized.split(":")
        if len(parts) != 3:
            raise ValueError("SMP model format must be smp:<arch>:<encoder>, e.g. smp:Unet:resnet50")
        _, arch, encoder = parts
        weights = "imagenet" if pretrained else None
        return smp.create_model(arch, encoder_name=encoder, encoder_weights=weights, classes=num_classes, in_channels=3)

    if lower.startswith("hf:"):
        return HFSegmentationWrapper(normalized[3:], num_classes=num_classes)

    from torchvision.models.segmentation import (
        DeepLabV3_ResNet101_Weights,
        DeepLabV3_ResNet50_Weights,
        FCN_ResNet50_Weights,
        LRASPP_MobileNet_V3_Large_Weights,
        deeplabv3_resnet101,
        deeplabv3_resnet50,
        fcn_resnet50,
        lraspp_mobilenet_v3_large,
    )

    weights = None
    weights_backbone = None
    if lower == "fcn_resnet50":
        weights = FCN_ResNet50_Weights.DEFAULT if pretrained else None
        model = fcn_resnet50(weights=weights, weights_backbone=weights_backbone, num_classes=21 if weights else num_classes)
        model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, num_classes, kernel_size=1)
        model.aux_classifier = None
        return TorchvisionSegmentationWrapper(model)
    if lower == "deeplabv3_resnet50":
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        model = deeplabv3_resnet50(weights=weights, weights_backbone=weights_backbone, num_classes=21 if weights else num_classes)
        model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, num_classes, kernel_size=1)
        model.aux_classifier = None
        return TorchvisionSegmentationWrapper(model)
    if lower == "deeplabv3_resnet101":
        weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
        model = deeplabv3_resnet101(weights=weights, weights_backbone=weights_backbone, num_classes=21 if weights else num_classes)
        model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, num_classes, kernel_size=1)
        model.aux_classifier = None
        return TorchvisionSegmentationWrapper(model)
    if lower == "lraspp_mobilenet_v3_large":
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = lraspp_mobilenet_v3_large(weights=weights, weights_backbone=weights_backbone, num_classes=21 if weights else num_classes)
        model.classifier.high_classifier = nn.Conv2d(model.classifier.high_classifier.in_channels, num_classes, kernel_size=1)
        model.classifier.low_classifier = nn.Conv2d(model.classifier.low_classifier.in_channels, num_classes, kernel_size=1)
        return TorchvisionSegmentationWrapper(model)

    choices = ", ".join(item.name for item in AVAILABLE_MODELS)
    raise ValueError(f"Unknown model '{name}'. Available: {choices}")
