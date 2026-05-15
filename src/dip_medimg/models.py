from __future__ import annotations

import torch
import torch.nn as nn


def conv(in_ch: int, out_ch: int, ks: int, stride: int = 1, bias: bool = True, pad: str = "reflect") -> nn.Module:
    if pad == "zero":
        return nn.Conv2d(in_ch, out_ch, ks, stride, ks // 2, bias=bias)
    if pad == "reflect":
        return nn.Sequential(nn.ReflectionPad2d(ks // 2), nn.Conv2d(in_ch, out_ch, ks, stride, 0, bias=bias))
    raise ValueError(f"Unsupported padding: {pad}")


def conv_bn_act(in_ch: int, out_ch: int, ks: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        conv(in_ch, out_ch, ks, stride),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            conv_bn_act(in_ch, out_ch, 3, stride),
            conv(out_ch, out_ch, 3),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.shortcut(x))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        channel = self.sigmoid(self.mlp(avg).unsqueeze(-1).unsqueeze(-1) + self.mlp(mx).unsqueeze(-1).unsqueeze(-1))
        x = x * channel
        spatial = torch.cat([torch.mean(x, dim=1, keepdim=True), torch.amax(x, dim=1, keepdim=True)], dim=1)
        return x * self.sigmoid(self.spatial(spatial))


class SkipNetCBAMEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32, attention: bool = True):
        super().__init__()
        att1 = CBAM(base) if attention else nn.Identity()
        att2 = CBAM(base * 2) if attention else nn.Identity()
        att3 = CBAM(base * 4) if attention else nn.Identity()
        att4 = CBAM(base * 4) if attention else nn.Identity()
        self.stem = conv_bn_act(in_ch, base, 3)
        self.enc1 = nn.Sequential(ResidualBlock(base, base), att1)
        self.enc2 = nn.Sequential(ResidualBlock(base, base * 2, stride=2), att2)
        self.enc3 = nn.Sequential(ResidualBlock(base * 2, base * 4, stride=2), att3)
        self.bottleneck = nn.Sequential(ResidualBlock(base * 4, base * 4), att4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        z = self.bottleneck(x3)
        return z, [x1, x2, x3]


class SkipNetCBAMDecoder(nn.Module):
    def __init__(self, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), conv_bn_act(base * 8, base * 2, 3))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), conv_bn_act(base * 4, base, 3))
        self.refine = nn.Sequential(ResidualBlock(base * 2, base), conv(base, out_ch, 3), nn.Tanh())

    def forward(self, z: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        x1, x2, x3 = skips
        y = self.up2(torch.cat([z, x3], dim=1))
        y = self.up1(torch.cat([y, x2], dim=1))
        return self.refine(torch.cat([y, x1], dim=1))


class ClassificationHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, max(in_ch // 2, 32)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(in_ch // 2, 32), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ImageEnhanceClassifier(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, base: int = 32, dropout: float = 0.4, attention: bool = True):
        super().__init__()
        self.encoder = SkipNetCBAMEncoder(in_ch, base, attention=attention)
        self.decoder = SkipNetCBAMDecoder(in_ch, base)
        self.classifier = ClassificationHead(base * 4, num_classes, dropout)

    def forward(self, x: torch.Tensor, reconstruct: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        z, skips = self.encoder(x)
        logits = self.classifier(z)
        recon = self.decoder(z, skips) if reconstruct else None
        return logits, recon


class SimpleCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, base: int = 32, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_act(in_ch, base, 3),
            nn.MaxPool2d(2),
            conv_bn_act(base, base * 2, 3),
            nn.MaxPool2d(2),
            conv_bn_act(base * 2, base * 4, 3),
            nn.MaxPool2d(2),
            conv_bn_act(base * 4, base * 4, 3),
        )
        self.classifier = ClassificationHead(base * 4, num_classes, dropout)

    def forward(self, x: torch.Tensor, reconstruct: bool = False) -> tuple[torch.Tensor, None]:
        return self.classifier(self.features(x)), None


class ResidualCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, base: int = 32, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_act(in_ch, base, 3),
            ResidualBlock(base, base),
            ResidualBlock(base, base * 2, stride=2),
            ResidualBlock(base * 2, base * 4, stride=2),
            ResidualBlock(base * 4, base * 4),
        )
        self.classifier = ClassificationHead(base * 4, num_classes, dropout)

    def forward(self, x: torch.Tensor, reconstruct: bool = False) -> tuple[torch.Tensor, None]:
        return self.classifier(self.features(x)), None


def build_model(name: str, in_ch: int, num_classes: int, base: int = 32, dropout: float = 0.4) -> nn.Module:
    name = name.lower()
    if name == "simple_cnn":
        return SimpleCNN(in_ch, num_classes, base, dropout)
    if name == "residual_cnn":
        return ResidualCNN(in_ch, num_classes, base, dropout)
    if name == "skipnet_no_cbam":
        return ImageEnhanceClassifier(in_ch, num_classes, base, dropout, attention=False)
    if name in {"proposed", "skipnet_cbam"}:
        return ImageEnhanceClassifier(in_ch, num_classes, base, dropout, attention=True)
    raise ValueError(f"Unknown model name: {name}")
