# Phase 1.6-A2 Backbone Parameterization Readiness Report — 2026-05-07 17:47

## 1. Files Modified

| File | Change | Lines |
|---|---|---|
| `src/model.py` | SpatialEncoderFPN: add `backbone_name` param, two init paths (baseline + registry) | +50 |
| `src/model.py` | MicroVCOD: add `backbone_name` param, pass to SpatialEncoderFPN | +2 |
| `tools/autoresearch/run_trial.py` | `build_model()`: pass `backbone_name` to MicroVCOD; update docstring | ~10 |
| `tools/autoresearch/backbone_registry.py` | Update `stage_channels` and `total_params_estimate` from probed values | ~12 |
| `tools/autoresearch/test_backbone_parameterization.py` | **New** — test script for all 4 backbones | +104 |

### src/model.py: SpatialEncoderFPN

Two initialization paths:

- **`backbone_name="mobilenet_v3_small"`** → `_init_mobilenet_v3_small()` — **line-for-line identical to the original code**. Preserved exactly: same `torchvision.models.mobilenet_v3_small(weights=...)`, same stage slicing `[:3]` / `[3:7]` / `[7:]`, same channel constants (576, 40, 24), same kaiming init on lateral convs. Baseline compatibility is guaranteed by construction.

- **Other backbones** → `_init_from_registry()` — queries `backbone_registry.py` for factory + stage_slices, builds backbone, probes channel counts via dummy forward pass, dynamically creates lateral/smooth convs with correct `in_channels`.

The `forward()` method is unified — same FPN top-down pathway for all backbones.

### src/model.py: MicroVCOD

Added `backbone_name="mobilenet_v3_small"` parameter (default preserves existing behavior). Passes through to `SpatialEncoderFPN`. All other components unchanged:
- `TemporalNeighborhood` — NOT modified
- `BBoxHead` (`current_direct_bbox`) — NOT modified
- Forward pass — NOT modified

## 2. Probed Channel Counts

Each backbone was probed via a dummy forward pass through its FPN stages at 224×224 input:

| Backbone | Stage 2 (s8) | Stage 3 (s16) | Stage 4 (s32) | Source |
|---|---|---|---|---|
| mobilenet_v3_small | 24 | 40 | 576 | Static (verified) |
| mobilenet_v3_large | 24 | 40 | 960 | Probed |
| efficientnet_b0 | 24 | 80 | 1280 | Probed |
| efficientnet_b1 | 24 | 80 | 1280 | Probed |

**Note:** The EfficientNet probed channels ([24, 80, 1280]) differ significantly from the original registry estimates ([40, 112, 320]). The dynamic probing mechanism correctly detected the actual channel counts and created lateral convs with matching `in_channels`. The registry has been updated with the correct values.

## 3. Parameter Counts

| Backbone | Total Params | Registry Estimate | Baseline Delta |
|---|---|---|---|
| mobilenet_v3_small | **1,411,684** | ~1.4M | **0** (exact match) |
| mobilenet_v3_large | 3,505,780 | ~3.5M | N/A |
| efficientnet_b0 | 4,587,456 | ~4.6M | N/A |
| efficientnet_b1 | 7,093,092 | ~7.1M | N/A |

All backbones under the 35M param budget. mobilenet_v3_small params are **exactly** 1,411,684 — zero drift from the baseline.

## 4. Dummy Forward Shape Verification

All backbones tested with `(B=2, T=5, C=3, H=224, W=224)` input, `pretrained=False`:

| Backbone | Output Shape | Expected | Result |
|---|---|---|---|
| mobilenet_v3_small | (2, 5, 4) | (2, 5, 4) | ✅ PASS |
| mobilenet_v3_large | (2, 5, 4) | (2, 5, 4) | ✅ PASS |
| efficientnet_b0 | (2, 5, 4) | (2, 5, 4) | ✅ PASS |
| efficientnet_b1 | (2, 5, 4) | (2, 5, 4) | ✅ PASS |

Output is in normalized bbox format `(x1, y1, x2, y2)` in [0, 1] — identical to baseline.

### With pretrained=True (ImageNet weights)

| Backbone | Shape | Params | Delta |
|---|---|---|---|
| mobilenet_v3_small | (1, 5, 4) | 1,411,684 | 0 |

Baseline code path confirmed identical with pretrained weights.

## 5. Test Commands & Results

### Command

```
python tools/autoresearch/test_backbone_parameterization.py
```

### Result

```
============================================================
  Testing: mobilenet_v3_small
============================================================
  Output shape:   torch.Size([2, 5, 4])  OK
  Total params:   1,411,684
  Stage channels: [24, 40, 576]  (static — baseline path)

============================================================
  Testing: mobilenet_v3_large
============================================================
  Output shape:   torch.Size([2, 5, 4])  OK
  Total params:   3,505,780
  Stage channels: [24, 40, 960]

============================================================
  Testing: efficientnet_b0
============================================================
  Output shape:   torch.Size([2, 5, 4])  OK
  Total params:   4,587,456
  Stage channels: [24, 80, 1280]

============================================================
  Testing: efficientnet_b1
============================================================
  Output shape:   torch.Size([2, 5, 4])  OK
  Total params:   7,093,092
  Stage channels: [24, 80, 1280]

============================================================
  Baseline Parameter Check
============================================================
  mobilenet_v3_small params: 1,411,684
  Baseline:                  1,411,684
  Delta:                     +0
  Within ±100: True

ALL TESTS PASSED
```

### Compile checks

```
python -m py_compile src/model.py                     → OK
python -m py_compile tools/autoresearch/backbone_registry.py → OK
python -m py_compile tools/autoresearch/run_trial.py         → OK
```

### Baseline pretrained path

```
python -c "
from src.model import MicroVCOD
model = MicroVCOD(T=5, pretrained_backbone=True, backbone_name='mobilenet_v3_small')
# Params: 1,411,684 (delta: +0)
# Output shape: (1, 5, 4)
# PASS
"
```

## 6. What Was NOT Changed

| Component | Status |
|---|---|
| `BBoxHead` / `current_direct_bbox` | NOT modified |
| `TemporalNeighborhood` | NOT modified |
| `BBoxLoss` | NOT modified |
| Sampler (`window_uniform`) | NOT modified |
| `video_balanced` sampler | NOT implemented (deferred to B2) |
| `objectness_aux_head` | NOT implemented (deferred to B2) |
| DataLoader / canonical_video_id filter | NOT modified |
| Checkpoints | NOT moved, deleted, or overwritten |
| `local_runs/` | NOT touched |
| `run_trial.py` training loop | NOT modified (only `build_model` docstring + call) |

## 7. Readiness for B1 Backbone-Only Smoke

### Is B1 ready?

**YES.** All prerequisites met:

- [x] `SpatialEncoderFPN` accepts `backbone_name` parameter
- [x] `MicroVCOD` accepts `backbone_name` parameter and passes through
- [x] `mobilenet_v3_small` baseline code path is **line-for-line preserved** — zero param drift
- [x] `mobilenet_v3_large`, `efficientnet_b0`, `efficientnet_b1` all produce correct output shapes
- [x] Dynamic channel probing works — lateral convs created with correct `in_channels`
- [x] All 4 backbones within 35M param budget
- [x] All compile checks pass
- [x] Test script passes all 4 backbones
- [x] `run_trial.py` `build_model()` updated to pass `backbone_name`
- [x] `backbone_registry.py` channel counts corrected from probed values
- [x] No sampler/head/loss modifications
- [x] No training executed
- [x] No commit, no push

### B1 smoke trials will use:

| Trial | Backbone | Sz | T | Sampler | Head | LR |
|---|---|---|---|---|---|---|
| smoke_b1_effb0 | efficientnet_b0 | 512 | 5 | window_uniform | current_direct_bbox | 1e-3 |
| smoke_b1_mnv3large | mobilenet_v3_large | 512 | 5 | window_uniform | current_direct_bbox | 1e-3 |
| smoke_b1_effb1 | efficientnet_b1 | 512 | 5 | window_uniform | current_direct_bbox | 3e-4 |

All use `window_uniform` sampler and `current_direct_bbox` head — locked per B1 protocol.

## 8. Git Status

```
 M src/model.py                    ← backbone_name parameterization
?? configs/                        ← Phase 1.6 config (unstaged)
?? tools/autoresearch/             ← AutoResearch tools (unstaged)
?? reports/                        ← Reports (unstaged)
?? local_runs/                     ← Trial outputs (unstaged)
```

No commits. No pushes. `src/model.py` is modified but unstaged. No checkpoints or `local_runs/` in staging area.

## 9. Safety Compliance

| Constraint | Status |
|---|---|
| No training started | ✅ |
| No B1 smoke run | ✅ |
| No sampler modified | ✅ |
| No head modified | ✅ |
| No loss modified | ✅ |
| No objectness_aux_head | ✅ |
| No video_balanced | ✅ |
| No commit | ✅ |
| No push | ✅ |
| No checkpoint touched | ✅ |
| No local_runs/ staged | ✅ |

---

*Report generated at 2026-05-07T17:47:00*
*Phase 1.6-A2 Backbone Parameterization Readiness*
*Verdict: READY for B1 backbone-only smoke*
