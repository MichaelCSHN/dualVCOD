"""Level 0 static screen for Phase 1.6-R Explorer trials.

Cost: seconds per trial. No training. Only checks:
  - Model can be instantiated
  - Dummy forward produces valid bbox shape
  - One-batch loss is finite
  - Peak memory estimate
  - model.to(cuda) succeeds
  - Params counted

Usage:
    python tools/autoresearch/run_l0_screen.py
"""

import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from src.model import MicroVCOD
from src.loss import BBoxLoss
from eval.eval_video_bbox import count_parameters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def l0_screen_one(trial_id: str, trial_config: dict) -> dict:
    """Run a single Level 0 screen. Returns dict with results."""
    result = {
        "trial_id": trial_id,
        "timestamp": datetime.now().isoformat(),
        "passed": True,
        "checks": {},
    }

    backbone = trial_config["backbone"]
    temporal_T = trial_config["temporal_T"]
    head_type = trial_config["head"]
    input_size = trial_config["input_size"]

    try:
        # 1. Model instantiation + .to(cuda)
        t0 = time.time()
        model = MicroVCOD(T=temporal_T, pretrained_backbone=(backbone != "shufflenet_v2_x1_5"),
                         backbone_name=backbone, head_type=head_type).to(DEVICE)
        model.eval()  # ensure objectness head doesn't return tuple during screen
        init_time = time.time() - t0
        result["checks"]["model_init"] = {"passed": True, "time_s": round(init_time, 2)}
    except Exception as e:
        result["checks"]["model_init"] = {"passed": False, "error": str(e)[:200]}
        result["passed"] = False
        return result

    # 2. Params
    try:
        n_params = count_parameters(model)
        result["checks"]["params"] = {"passed": True, "count": n_params}
    except Exception as e:
        result["checks"]["params"] = {"passed": False, "error": str(e)[:200]}
        result["passed"] = False

    # 3. Dummy forward + shape check
    try:
        B, T = 2, temporal_T
        dummy = torch.randn(B, T, 3, input_size, input_size, device=DEVICE)
        with torch.no_grad():
            output = model(dummy)
        expected_shape = (B, T, 4)
        shape_ok = output.shape == expected_shape
        bbox_valid = (output >= 0).all() and (output <= 1).all()
        result["checks"]["dummy_forward"] = {
            "passed": shape_ok and bbox_valid,
            "output_shape": list(output.shape),
            "expected_shape": list(expected_shape),
            "bbox_range_ok": bool(bbox_valid.item() if hasattr(bbox_valid, 'item') else bbox_valid),
        }
    except Exception as e:
        result["checks"]["dummy_forward"] = {"passed": False, "error": str(e)[:200]}
        result["passed"] = False

    # 4. One-batch loss check
    try:
        lw = trial_config.get("loss_weights", {})
        criterion = BBoxLoss(
            smooth_l1_weight=lw.get("smooth_l1", 1.0),
            giou_weight=lw.get("giou", 1.0),
            use_diou=lw.get("use_diou", False),
            center_weight=lw.get("center", 0.0),
            objectness_weight=lw.get("objectness", 0.1) if head_type == "objectness_aux_head" else 0.0,
        )
        gt = torch.rand(B, T, 4, device=DEVICE)
        gt = gt.clamp(min=0.05, max=0.95)
        # Ensure gt has valid format: x1 < x2, y1 < y2
        gt_ordered = torch.zeros_like(gt)
        gt_ordered[..., 0] = torch.min(gt[..., 0], gt[..., 2])
        gt_ordered[..., 1] = torch.min(gt[..., 1], gt[..., 3])
        gt_ordered[..., 2] = torch.max(gt[..., 0], gt[..., 2])
        gt_ordered[..., 3] = torch.max(gt[..., 1], gt[..., 3])
        # For objectness head, model.eval() means forward returns bbox only
        losses = criterion(output, gt_ordered)
        loss_finite = torch.isfinite(losses["loss"])
        result["checks"]["one_batch_loss"] = {
            "passed": bool(loss_finite.item() if hasattr(loss_finite, 'item') else loss_finite),
            "loss_value": round(float(losses["loss"].item()), 6),
            "mean_iou": round(float(losses["mean_iou"].item()), 6),
        }
    except Exception as e:
        result["checks"]["one_batch_loss"] = {"passed": False, "error": str(e)[:200]}
        result["passed"] = False

    # 5. Peak memory estimate
    try:
        torch.cuda.reset_peak_memory_stats()
        B_mem, T_mem = 4, temporal_T  # slightly larger batch for mem estimate
        dummy_mem = torch.randn(B_mem, T_mem, 3, input_size, input_size, device=DEVICE)
        with torch.no_grad():
            _ = model(dummy_mem)
        peak_mem_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        result["checks"]["peak_memory"] = {
            "passed": peak_mem_gib < 20,  # must fit in 24GB with room
            "peak_gib": round(peak_mem_gib, 2),
            "batch_size": B_mem,
        }
    except Exception as e:
        result["checks"]["peak_memory"] = {"passed": False, "error": str(e)[:200]}

    # 6. Cleanup
    del model, dummy
    torch.cuda.empty_cache()

    # Aggregate pass/fail
    result["passed"] = all(v.get("passed", False) for v in result["checks"].values())
    return result


def main():
    print("=" * 72)
    print("  PHASE 1.6-R: LEVEL 0 STATIC SCREENS")
    print(f"  Device: {DEVICE}")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 72)
    print()

    # ── Explorer Trial Configs ───────────────────────────────────────────
    trials = [
        {
            "trial_id": "expl_01_lowlr_warmup_mv3small",
            "config": {
                "backbone": "mobilenet_v3_small",
                "input_size": 224,
                "temporal_T": 5,
                "sampler": "window_uniform",
                "head": "current_direct_bbox",
                "lr": 3e-4,
                "batch_size": 24,
                "epochs": 5,
                "total_epochs": 30,
                "train_seed": 42,
                "warmup_epochs": 5,
                "hypothesis": "Lower LR + warmup allows smoother convergence and earlier recovery"
            }
        },
        {
            "trial_id": "expl_02_mnv3large_recovery",
            "config": {
                "backbone": "mobilenet_v3_large",
                "input_size": 320,
                "temporal_T": 5,
                "sampler": "window_uniform",
                "head": "current_direct_bbox",
                "lr": 3e-4,
                "batch_size": 24,
                "epochs": 5,
                "total_epochs": 30,
                "train_seed": 42,
                "warmup_epochs": 5,
                "hypothesis": "MV3-Large at moderate res with lower LR + warmup avoids B1 overfitting"
            }
        },
        {
            "trial_id": "expl_03_effb0_stable",
            "config": {
                "backbone": "efficientnet_b0",
                "input_size": 320,
                "temporal_T": 5,
                "sampler": "window_uniform",
                "head": "current_direct_bbox",
                "lr": 1e-4,
                "batch_size": 24,
                "epochs": 1,
                "total_epochs": 30,
                "train_seed": 42,
                "warmup_epochs": 5,
                "hypothesis": "EfficientNet-B0 at low LR + FP32 isolates NaN instability source"
            }
        },
        {
            "trial_id": "expl_04_t1_imagemode",
            "config": {
                "backbone": "mobilenet_v3_small",
                "input_size": 224,
                "temporal_T": 1,
                "sampler": "window_uniform",
                "head": "current_direct_bbox",
                "lr": 1e-3,
                "batch_size": 24,
                "epochs": 5,
                "total_epochs": 30,
                "train_seed": 42,
                "hypothesis": "T=1 establishes spatial-only performance lower bound"
            }
        },
        {
            "trial_id": "expl_05_t3_efficiency",
            "config": {
                "backbone": "mobilenet_v3_small",
                "input_size": 224,
                "temporal_T": 3,
                "sampler": "window_uniform",
                "head": "current_direct_bbox",
                "lr": 1e-3,
                "batch_size": 24,
                "epochs": 5,
                "total_epochs": 30,
                "train_seed": 42,
                "temporal_stride": 2,
                "hypothesis": "T=3 stride=2 matches T=5 temporal span at 60% compute"
            }
        },
        {
            "trial_id": "expl_06_highres640",
            "config": {
                "backbone": "mobilenet_v3_small",
                "input_size": 640,
                "temporal_T": 5,
                "sampler": "window_uniform",
                "head": "current_direct_bbox",
                "lr": 3e-4,
                "batch_size": 8,
                "epochs": 5,
                "total_epochs": 30,
                "train_seed": 42,
                "hypothesis": "640px improves small-object localization precision"
            }
        },
        {
            "trial_id": "expl_07_objectness_aux",
            "config": {
                "backbone": "mobilenet_v3_small",
                "input_size": 224,
                "temporal_T": 5,
                "sampler": "window_uniform",
                "head": "objectness_aux_head",
                "lr": 1e-3,
                "batch_size": 24,
                "epochs": 5,
                "total_epochs": 30,
                "train_seed": 42,
                "hypothesis": "Objectness aux head reduces no_response errors"
            }
        },
        {
            "trial_id": "expl_08_videobal_sampler",
            "config": {
                "backbone": "mobilenet_v3_small",
                "input_size": 224,
                "temporal_T": 5,
                "sampler": "video_balanced",
                "head": "current_direct_bbox",
                "lr": 1e-3,
                "batch_size": 24,
                "epochs": 5,
                "total_epochs": 30,
                "train_seed": 42,
                "hypothesis": "Video-balanced sampling prevents large-video dominance"
            }
        },
        {
            "trial_id": "expl_09_freeze_warmup",
            "config": {
                "backbone": "mobilenet_v3_small",
                "input_size": 224,
                "temporal_T": 5,
                "sampler": "window_uniform",
                "head": "current_direct_bbox",
                "lr": 1e-3,
                "batch_size": 24,
                "epochs": 5,
                "total_epochs": 30,
                "train_seed": 42,
                "freeze_backbone_epochs": 5,
                "hypothesis": "Freeze backbone for 5 epochs lets head adapt first"
            }
        },
        {
            "trial_id": "expl_10_convnext_tiny",
            "config": {
                "backbone": "convnext_tiny",
                "input_size": 224,
                "temporal_T": 5,
                "sampler": "window_uniform",
                "head": "current_direct_bbox",
                "lr": 1e-3,
                "batch_size": 16,
                "epochs": 5,
                "total_epochs": 30,
                "train_seed": 42,
                "hypothesis": "ConvNeXt Tiny (28M) establishes upper bound on backbone capacity for VCOD"
            }
        },
        {
            "trial_id": "expl_11_shufflenet_v2",
            "config": {
                "backbone": "shufflenet_v2_x1_5",
                "input_size": 224,
                "temporal_T": 5,
                "sampler": "window_uniform",
                "head": "current_direct_bbox",
                "lr": 1e-3,
                "batch_size": 24,
                "epochs": 5,
                "total_epochs": 30,
                "train_seed": 42,
                "hypothesis": "ShuffleNet-V2 x1.5 for ultra-light deployment tier"
            }
        },
        {
            "trial_id": "expl_12_giou_center_warmup",
            "config": {
                "backbone": "mobilenet_v3_small",
                "input_size": 224,
                "temporal_T": 5,
                "sampler": "window_uniform",
                "head": "current_direct_bbox",
                "lr": 1e-3,
                "batch_size": 24,
                "epochs": 5,
                "total_epochs": 30,
                "train_seed": 42,
                "warmup_epochs": 3,
                "loss_weights": {"smooth_l1": 2, "giou": 1, "center": 0.5},
                "hypothesis": "Multi-component loss (L1+GIoU+center) + warmup improves bbox quality"
            }
        },
    ]

    results = []
    passed = 0
    failed = 0
    skipped = []

    for t in trials:
        tid = t["trial_id"]
        cfg = t["config"]
        backbone = cfg["backbone"]
        head = cfg["head"]

        # Check if backbone/head needs implementation
        needs_impl = []
        if backbone not in ("mobilenet_v3_small", "mobilenet_v3_large",
                           "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
                           "convnext_tiny", "shufflenet_v2_x1_5"):
            needs_impl.append(f"backbone={backbone}")
        if head not in ("current_direct_bbox", "objectness_aux_head"):
            needs_impl.append(f"head={head}")

        if needs_impl:
            print(f"  [{tid}] SKIP — needs implementation: {', '.join(needs_impl)}")
            skipped.append({"trial_id": tid, "reason": needs_impl})
            continue

        print(f"  [{tid}] backbone={backbone} sz={cfg['input_size']} T={cfg['temporal_T']} "
              f"head={head} lr={cfg['lr']} ...", end=" ", flush=True)

        r = l0_screen_one(tid, cfg)
        results.append(r)

        if r["passed"]:
            passed += 1
            print("PASS")
            # Print key metrics
            ch = r["checks"]
            print(f"         params={ch.get('params',{}).get('count','?'):,}  "
                  f"loss={ch.get('one_batch_loss',{}).get('loss_value','?'):.4f}  "
                  f"mem={ch.get('peak_memory',{}).get('peak_gib','?'):.2f} GiB")
        else:
            failed += 1
            print("FAIL")
            for check_name, check_val in r["checks"].items():
                if not check_val.get("passed", False):
                    print(f"         FAIL [{check_name}]: {check_val.get('error', check_val)}")

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  L0 SCREEN SUMMARY: {passed} PASS, {failed} FAIL, {len(skipped)} SKIP (needs impl)")
    print("=" * 72)

    if skipped:
        print()
        print("  Skipped (needs implementation):")
        for s in skipped:
            print(f"    - {s['trial_id']}: {', '.join(s['reason'])}")

    # ── Save results ─────────────────────────────────────────────────────
    output_dir = os.path.join(PROJECT_ROOT, "local_runs", "autoresearch", "_l0_screens")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"l0_screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "skipped": skipped,
                   "summary": {"passed": passed, "failed": failed, "skipped": len(skipped)}},
                  f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
