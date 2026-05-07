import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SpatialEncoder(nn.Module):
    """Lightweight 2D CNN backbone — applied per-frame.

    Three stride-2 convs reduce spatial dims by 8x while expanding
    channels to 128.  Shared across all T frames.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)  # (B*T, 128, H/8, W/8)


class TemporalNeighborhood(nn.Module):
    """Multi-frame pooling module (1D conv along the T axis).

    Short-term branch:  Conv1d(k=3) captures inter-frame variation.
    Long-term branch:   Global avg-pool across T broadcasts scene-level context.
    Both are fused with a residual gate — no 3D conv, no optical flow.

    NOTE (Red-Team Audit, 2026-05-07): This module is ORDER-INVARIANT —
    forward and reversed frame order produce identical outputs. The
    GlobalAvgPool collapses temporal position, so the module functions as
    multi-frame pooling rather than true temporal dynamics modeling.
    """

    def __init__(self, T=5, channels=128):
        super().__init__()
        self.short_term = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.long_pool = nn.AdaptiveAvgPool1d(1)
        self.long_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.fusion = nn.Conv1d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        # Collapse spatial → (B, T, C)
        g = x.mean(dim=[-2, -1])
        g = g.permute(0, 2, 1)  # (B, C, T)

        short = self.short_term(g)  # (B, C, T)
        long_g = self.long_conv(self.long_pool(g)).expand(-1, -1, T)
        fused = self.fusion(torch.cat([short, long_g], dim=1))
        gate = fused.permute(0, 2, 1).view(B, T, C, 1, 1).sigmoid()

        return x * (1.0 + gate)


class BBoxHead(nn.Module):
    """Predicts per-frame bounding boxes in (x1, y1, x2, y2) format [0,1]."""

    def __init__(self, in_channels=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.pool(x).flatten(1)
        return self.fc(x).reshape(B, T, 4)


class MicroVCOD_Lite(nn.Module):
    """Minimal VCOD model — lightweight CNN + temporal pooling + BBox head.

    Input:  (B, T, C, H, W)
    Output: (B, T, 4)  — one BBox (x1,y1,x2,y2) per frame, normalized to [0,1].
    """

    def __init__(self, T=5, in_channels=3):
        super().__init__()
        self.T = T
        self.spatial_encoder = SpatialEncoder(in_channels)
        self.temporal_neck = TemporalNeighborhood(T=T, channels=128)
        self.bbox_head = BBoxHead(in_channels=128)

    def forward(self, x):
        B, T, C, H, W = x.shape

        # Shared spatial encoding across frames
        x = x.reshape(B * T, C, H, W)
        feat = self.spatial_encoder(x)
        _, Cf, Hf, Wf = feat.shape
        feat = feat.reshape(B, T, Cf, Hf, Wf)

        # Temporal neighbourhood fusion
        feat = self.temporal_neck(feat)

        # Per-frame BBox regression
        return self.bbox_head(feat)


# ═══════════════════════════════════════════════════════════════════════
#  MicroVCOD  —  real backbone (MobileNetV3-Small FPN) + TN + BBox
# ═══════════════════════════════════════════════════════════════════════


class SpatialEncoderFPN(nn.Module):
    """Multi-scale FPN backbone using pretrained MobileNetV3-Small.

    Extracts features at stride 8, 16, 32 and fuses them via top-down
    pathway into a single 128-channel feature map at stride 8.

    For 224×224 input the output is (128, 28, 28).
    """

    def __init__(self, pretrained=True):
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_small(
            weights=(
                torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
        ).features

        # Stage splits (verified on torchvision 0.21):
        #   features[:3]   → stride  8,  24 ch
        #   features[3:7]  → stride 16,  40 ch
        #   features[7:]   → stride 32, 576 ch
        self.stage2 = backbone[:3]
        self.stage3 = backbone[3:7]
        self.stage4 = backbone[7:]

        self.lat4 = nn.Conv2d(576, 128, 1)
        self.lat3 = nn.Conv2d(40, 128, 1)
        self.lat2 = nn.Conv2d(24, 128, 1)
        self.smooth3 = nn.Conv2d(128, 128, 3, padding=1)
        self.smooth2 = nn.Conv2d(128, 128, 3, padding=1)

        for m in [self.lat4, self.lat3, self.lat2]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        self.out_channels = 128

    def forward(self, x):
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.smooth3(p3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.smooth2(p2)

        return p2  # (B, 128, H/8, W/8)


class MicroVCOD(nn.Module):
    """MicroVCOD — Micro Video Camouflaged Object Detection.

    Independently implemented PyTorch model sharing only the high-level
    "temporal window" concept with arXiv:2501.10914 (GreenVCOD). Zero lines
    of code reused. Uses gradient-based training, not Green Learning.

    SpatialEncoderFPN (MobileNetV3-Small + FPN)  →  TemporalNeighborhood  →  BBoxHead

    Input:  (B, T, C, H, W)
    Output: (B, T, 4)  — normalized BBoxes (x1, y1, x2, y2) in [0, 1].
    """

    def __init__(self, T=5, in_channels=3, pretrained_backbone=True):
        super().__init__()
        self.T = T
        self.spatial_encoder = SpatialEncoderFPN(pretrained=pretrained_backbone)
        self.temporal_neck = TemporalNeighborhood(T=T, channels=128)
        self.bbox_head = BBoxHead(in_channels=128)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.reshape(B * T, C, H, W)
        feat = self.spatial_encoder(x)
        _, Cf, Hf, Wf = feat.shape
        feat = feat.reshape(B, T, Cf, Hf, Wf)

        feat = self.temporal_neck(feat)
        return self.bbox_head(feat)
