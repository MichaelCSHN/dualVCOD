"""Backbone registry for Phase 1.6 AutoResearch — maps names to torchvision models + FPN config.

Supports: mobilenet_v3_small, mobilenet_v3_large, efficientnet_b0, efficientnet_b1, efficientnet_b2.

Each entry provides:
  - factory: callable(pretrained: bool) -> nn.Module (the backbone features)
  - stage_slices: list of (start, end) indices into backbone.features for FPN stages [s8, s16, s32]
  - stage_channels: expected output channels at each stage boundary
  - fpn_out_channels: FPN output channels (fixed at 128 for downstream compatibility)
"""

import torch.nn as nn
import torchvision


def _mobilenet_v3_small_features(pretrained: bool = True) -> nn.Module:
    return torchvision.models.mobilenet_v3_small(
        weights=(
            torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
    ).features


def _mobilenet_v3_large_features(pretrained: bool = True) -> nn.Module:
    return torchvision.models.mobilenet_v3_large(
        weights=(
            torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
    ).features


def _efficientnet_b0_features(pretrained: bool = True) -> nn.Module:
    return torchvision.models.efficientnet_b0(
        weights=(
            torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
    ).features


def _efficientnet_b1_features(pretrained: bool = True) -> nn.Module:
    return torchvision.models.efficientnet_b1(
        weights=(
            torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
    ).features


def _efficientnet_b2_features(pretrained: bool = True) -> nn.Module:
    return torchvision.models.efficientnet_b2(
        weights=(
            torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
    ).features


def _convnext_tiny_features(pretrained: bool = True) -> nn.Module:
    return torchvision.models.convnext_tiny(
        weights=(
            torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
    ).features


def _shufflenet_v2_x1_5_features(pretrained: bool = True) -> nn.Sequential:
    """Wrap ShuffleNetV2 named stages into a flat Sequential for registry compatibility."""
    m = torchvision.models.shufflenet_v2_x1_5(
        weights=(
            torchvision.models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
    )
    # ShuffleNetV2 has: conv1 (stride 2), maxpool (stride 2), stage2, stage3, stage4, conv5
    # Flatten into sequential: conv1, maxpool, stage2 blocks, stage3 blocks, stage4 blocks, conv5
    seq = nn.Sequential()
    seq.add_module("conv1", m.conv1)
    seq.add_module("maxpool", m.maxpool)
    for i, block in enumerate(m.stage2):
        seq.add_module(f"stage2_{i}", block)
    for i, block in enumerate(m.stage3):
        seq.add_module(f"stage3_{i}", block)
    for i, block in enumerate(m.stage4):
        seq.add_module(f"stage4_{i}", block)
    seq.add_module("conv5", m.conv5)
    return seq


BACKBONE_REGISTRY = {
    # ── MobileNet V3 family ──────────────────────────────────────────
    "mobilenet_v3_small": {
        "factory": _mobilenet_v3_small_features,
        "input_size_range": [224, 640],
        "stage_slices": [(0, 3), (3, 7), (7, None)],  # s8:24ch, s16:40ch, s32:576ch
        "stage_channels": [24, 40, 576],
        "fpn_out_channels": 128,
        "total_params_estimate": "~1.4M",
    },
    "mobilenet_v3_large": {
        "factory": _mobilenet_v3_large_features,
        "input_size_range": [224, 640],
        # MV3-Large features: [0]ConvBnAct(s=2), [1]IR(s=1), [2]IR(s=2→s4), [3]IR(s=1),
        #   [4]IR(s=2→s8), [5-6]IR(s=1), [7]IR(s=2→s16), [8-12]IR(s=1),
        #   [13]IR(s=2→s32), [14-15]IR(s=1), [16]ConvBnAct(s=1)
        # Corrected 2026-05-13: old (0,4) stopped at stride 4 → 56×56, breaking dense_fg_aux
        "stage_slices": [(0, 5), (5, 13), (13, None)],  # s8, s16, s32
        "stage_channels": [40, 112, 960],     # re-probed at corrected slices
        "fpn_out_channels": 128,
        "total_params_estimate": "~3.5M",
    },
    # ── EfficientNet family ──────────────────────────────────────────
    # All EfficientNet variants share identical 9-layer features structure:
    #   [0]Conv(s=2)→112², [1]MBConv(s=1)→112², [2]MBConv(s=2)→56²,
    #   [3]MBConv(s=2)→28², [4]MBConv(s=2)→14², [5]MBConv(s=1)→14²,
    #   [6]MBConv(s=2)→7², [7]MBConv(s=1)→7², [8]Conv(s=1)→7²
    # Corrected 2026-05-13: old (0,3) stopped at stride 4 → 56×56, breaking dense_fg_aux
    "efficientnet_b0": {
        "factory": _efficientnet_b0_features,
        "input_size_range": [224, 640],
        "stage_slices": [(0, 4), (4, 6), (6, None)],  # s8, s16, s32
        "stage_channels": [40, 112, 1280],  # re-probed at corrected slices
        "fpn_out_channels": 128,
        "total_params_estimate": "~4.6M",
        # Multi-scale: stride-4 features for 56×56 dense supervision
        # features[0:3] = Conv(s=2→112²) + MBConv(s=1→112²) + MBConv(s=2→56²) → 24ch
        "stage_slice_s4": [0, 3],
        "stage_channel_s4": 24,
    },
    "efficientnet_b1": {
        "factory": _efficientnet_b1_features,
        "input_size_range": [224, 640],
        "stage_slices": [(0, 4), (4, 6), (6, None)],
        "stage_channels": [40, 112, 1280],  # re-probed at corrected slices
        "fpn_out_channels": 128,
        "total_params_estimate": "~7.1M",
    },
    "efficientnet_b2": {
        "factory": _efficientnet_b2_features,
        "input_size_range": [224, 640],
        "stage_slices": [(0, 4), (4, 6), (6, None)],
        "stage_channels": [48, 120, 1408],  # re-probed at corrected slices
        "fpn_out_channels": 128,
        "total_params_estimate": "~7.7M (backbone) + head -> ~9.1M total",
    },
    # ── ConvNeXt family ──────────────────────────────────────────────────
    "convnext_tiny": {
        "factory": _convnext_tiny_features,
        "input_size_range": [224, 640],
        "stage_slices": [(0, 4), (4, 6), (6, None)],  # s8:192ch, s16:384ch, s32:768ch
        "stage_channels": [192, 384, 768],         # probed at 224×224
        "fpn_out_channels": 128,
        "total_params_estimate": "~28M",
    },
    # ── ShuffleNet V2 family ────────────────────────────────────────────
    "shufflenet_v2_x1_5": {
        "factory": _shufflenet_v2_x1_5_features,
        "input_size_range": [224, 640],
        # After conv1+maxpool: stride4, 24ch → stage2: stride8, 88ch → stage3: stride16, 176ch → stage4+conv5: stride32, 1024ch
        # Flattened: conv1(0), maxpool(1), stage2(2-5), stage3(6-13), stage4(14-17), conv5(18)
        "stage_slices": [(0, 6), (6, 14), (14, None)],  # s8 up to stage2 end, s16 up to stage3 end, s32 rest
        "stage_channels": [176, 352, 1024],     # probed 2026-05-07
        "fpn_out_channels": 128,
        "total_params_estimate": "~3.5M",
    },
}


def get_backbone_config(name: str) -> dict:
    """Return registry entry for backbone name. Raises KeyError if unknown."""
    if name not in BACKBONE_REGISTRY:
        raise KeyError(
            f"Unknown backbone '{name}'. Available: {list(BACKBONE_REGISTRY.keys())}"
        )
    return BACKBONE_REGISTRY[name]


def list_available_backbones() -> list:
    return sorted(BACKBONE_REGISTRY.keys())


def probe_backbone_channels(name: str, input_size: int = 224) -> list:
    """Actually run a forward pass to get real channel counts per stage.

    Returns list of channels at each FPN stage boundary.  Use this to
    verify/supplement the static stage_channels config.
    """
    import torch
    cfg = get_backbone_config(name)
    backbone = cfg["factory"](pretrained=False)
    backbone.eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    channels = []
    x = dummy
    for start, end in cfg["stage_slices"]:
        stage = backbone[start:end] if end is not None else backbone[start:]
        x = stage(x)
        channels.append(x.shape[1])
    return channels
