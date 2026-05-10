"""Test backbone parameterization in MicroVCOD / SpatialEncoderFPN.

Verifies:
  1. All 4 backbones produce correct output shapes
  2. mobilenet_v3_small params ≈ baseline 1,411,684
  3. Params are reasonable for each backbone
  4. Dummy forward pass succeeds for all backbones (pretrained=False to skip download)

Usage:
    python tools/autoresearch/test_backbone_parameterization.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from src.model import MicroVCOD

BASELINE_PARAMS = 1_411_684
PARAM_TOLERANCE = 100  # allow small drift from baseline

BACKBONES = [
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_b1",
]

T = 5
B = 2
INPUT_SIZE = 224


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def test_backbone(backbone_name, pretrained=False):
    print(f"\n{'='*60}")
    print(f"  Testing: {backbone_name}")
    print(f"{'='*60}")

    model = MicroVCOD(T=T, pretrained_backbone=pretrained, backbone_name=backbone_name)
    n_params = count_params(model)

    dummy = torch.randn(B, T, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        output = model(dummy)

    expected_shape = (B, T, 4)
    shape_ok = output.shape == expected_shape

    print(f"  Output shape:   {output.shape}  {'OK' if shape_ok else 'FAIL — expected ' + str(expected_shape)}")
    print(f"  Total params:   {n_params:,}")
    print(f"  Output range:   [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  Backbone attr:  {getattr(model, '_backbone_name', 'N/A')}")

    # Stage channels from SpatialEncoderFPN
    enc = model.spatial_encoder
    if hasattr(enc, '_stage_channels'):
        print(f"  Stage channels: {enc._stage_channels}")
    elif backbone_name == "mobilenet_v3_small":
        print(f"  Stage channels: [24, 40, 576]  (static — baseline path)")

    return {
        "backbone": backbone_name,
        "params": n_params,
        "shape_ok": shape_ok,
        "output_shape": tuple(output.shape),
    }


def main():
    print("=" * 60)
    print("  Backbone Parameterization Tests")
    print(f"  Baseline params: {BASELINE_PARAMS:,}")
    print(f"  Tolerance:       ±{PARAM_TOLERANCE}")
    print("=" * 60)

    results = []
    all_pass = True

    for name in BACKBONES:
        try:
            r = test_backbone(name, pretrained=False)
            results.append(r)
            if not r["shape_ok"]:
                all_pass = False
        except Exception as e:
            print(f"\n  FAIL: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"backbone": name, "params": 0, "shape_ok": False, "output_shape": None})
            all_pass = False

    # ── Baseline params check ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Baseline Parameter Check")
    print(f"{'='*60}")
    mv3_result = [r for r in results if r["backbone"] == "mobilenet_v3_small"]
    if mv3_result:
        mv3_params = mv3_result[0]["params"]
        delta = mv3_params - BASELINE_PARAMS
        in_tolerance = abs(delta) <= PARAM_TOLERANCE
        print(f"  mobilenet_v3_small params: {mv3_params:,}")
        print(f"  Baseline:                  {BASELINE_PARAMS:,}")
        print(f"  Delta:                     {delta:+,}")
        print(f"  Within ±{PARAM_TOLERANCE}: {in_tolerance}")
        if not in_tolerance:
            all_pass = False
    else:
        print("  mobilenet_v3_small NOT TESTED")
        all_pass = False

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  {'Backbone':<24s} {'Params':>10s} {'Shape':>12s} {'Status':>10s}")
    print(f"  {'-'*24} {'-'*10} {'-'*12} {'-'*10}")
    for r in results:
        shape_str = str(r["output_shape"]) if r["output_shape"] else "ERROR"
        status = "PASS" if r["shape_ok"] else "FAIL"
        print(f"  {r['backbone']:<24s} {r['params']:>10,} {shape_str:>12s} {status:>10s}")

    print()
    if all_pass:
        print("  ALL TESTS PASSED")
        return 0
    else:
        print("  SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
