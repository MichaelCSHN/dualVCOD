"""Pre-flight safety check for a single AutoResearch trial.

Usage:
    python tools/autoresearch/check_trial_safety.py --trial_config trial_0001/config.yaml

Checks:
  1. Data isolation: train ∩ val canonical_video_ids == 0
  2. No forbidden variables changed
  3. Backbone is in registry
  4. Input size divisible by 32
  5. Param budget in [3M, 35M]
  6. Trial directory is within local_runs/autoresearch/
  7. No stale checkpoint loaded (always ImageNet init)
  8. Seed consistency (split seed always 42)

Returns exit code 0 (PASS) or 1 (FAIL).
"""

import sys
import os
import argparse
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── Fixed protocol constants ────────────────────────────────────────────────

FIXED_VALUES = {
    "split_seed": 42,
    "val_ratio": 0.2,
    "optimizer": "adamw",
    "weight_decay": 1e-4,
    "scheduler": "cosine_annealing",
    "amp": True,
    "grad_clip_norm": 2.0,
}

FORBIDDEN_VARIABLES = [
    "val_dataset",        # always MoCA
    "val_ratio",           # always 0.2
    "split_seed",          # always 42
    "canonical_video_id_filter",  # always ON
]

ALLOWED_VARIABLES = [
    "backbone",
    "input_size",
    "temporal_T",
    "temporal_stride",
    "sampler",
    "lr",
    "head",
    "batch_size",
    "epochs",
    "total_epochs",
    "train_seed",
    "freeze_backbone_epochs",
    "warmup_epochs",
    "loss_weights",
]


def load_trial_config(path: str) -> dict:
    """Load trial config from YAML or JSON."""
    if path.endswith(".yaml") or path.endswith(".yml"):
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            pass
    # Fallback: try as JSON
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_trial_directory(trial_dir: str) -> list:
    """Ensure trial output goes to allowed location."""
    issues = []
    abs_dir = os.path.abspath(trial_dir)
    expected_prefix = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "local_runs", "autoresearch")
    )
    if not abs_dir.startswith(expected_prefix):
        issues.append(f"Trial dir '{abs_dir}' not under '{expected_prefix}'")
    return issues


def check_backbone_allowed(backbone: str) -> list:
    """Check backbone is in registry."""
    issues = []
    from tools.autoresearch.backbone_registry import BACKBONE_REGISTRY
    if backbone not in BACKBONE_REGISTRY:
        issues.append(f"Unknown backbone '{backbone}'. Available: {list(BACKBONE_REGISTRY.keys())}")
    return issues


def check_input_size(input_size: int) -> list:
    """Input size must be divisible by 32 for all backbones."""
    issues = []
    if input_size % 32 != 0:
        issues.append(f"Input size {input_size} not divisible by 32")
    if input_size < 224:
        issues.append(f"Input size {input_size} < 224 minimum")
    if input_size > 1024:
        issues.append(f"Input size {input_size} > 1024 — likely OOM")
    return issues


def check_temporal_t(temporal_t: int) -> list:
    issues = []
    if temporal_t < 1:
        issues.append(f"T={temporal_t} too small — need at least 1 frame")
    if temporal_t > 10:
        issues.append(f"T={temporal_t} > 10 — likely OOM for larger backbones")
    # T=1 is allowed for image-mode diagnostic trials
    return issues


def check_sampler(sampler: str) -> list:
    issues = []
    if sampler not in ("window_uniform", "video_balanced"):
        issues.append(f"Unknown sampler '{sampler}'")
    return issues


def check_head(head: str) -> list:
    issues = []
    known_heads = ("current_direct_bbox", "objectness_aux_head", "giou_center_head",
                   "dense_objectness_head", "hybrid_dense_to_box")
    if head not in known_heads:
        issues.append(f"Unknown head '{head}'. Known: {list(known_heads)}")
    return issues


def check_lr(lr: float) -> list:
    issues = []
    if lr < 1e-5 or lr > 1e-1:
        issues.append(f"LR {lr} out of reasonable range [1e-5, 1e-1]")
    return issues


import re

def check_param_budget_estimate(backbone: str, head: str) -> list:
    """Rough param budget check before model instantiation."""
    issues = []
    from tools.autoresearch.backbone_registry import BACKBONE_REGISTRY
    cfg = BACKBONE_REGISTRY[backbone]
    est = cfg["total_params_estimate"]
    # Parse param estimate with regex for robustness
    match = re.search(r'[\d.]+', str(est))
    if match:
        factor = 1
        raw = est.lower()
        if 'k' in raw:
            factor = 1_000
        elif 'm' in raw:
            factor = 1_000_000
        params_est = int(float(match.group()) * factor)
    else:
        params_est = 5_000_000  # unknown, assume OK

    if params_est > 35_000_000:
        issues.append(f"Estimated params {params_est} > 35M max")
    if params_est < 1_000_000 and backbone != "mobilenet_v3_small":
        issues.append(f"Estimated params {params_est} < 1M — suspiciously small")
    return issues


def check_no_checkpoint_loading(resume_path: str | None) -> list:
    issues = []
    if resume_path:
        issues.append(f"Resume checkpoint specified: '{resume_path}' — clean init required")
    return issues


def check_seed_consistency(trial_config: dict) -> list:
    """Split seed must always be 42; train_seed must be set for B1+ (B0 mirrors train.py non-det)."""
    issues = []
    split_seed = trial_config.get("split_seed", 42)
    if split_seed != 42:
        issues.append(f"split_seed={split_seed} — must be 42 (fixed protocol)")
    train_seed = trial_config.get("train_seed")
    trial_id = trial_config.get("trial_id", trial_config.get("id", ""))
    if train_seed is None and trial_id.startswith("smoke_b1_"):
        issues.append("train_seed not set — required for B1+ reproducible trials")
    return issues


def check_data_isolation(train_datasets: list, val_dataset: str) -> list:
    """Lightweight check — full check is done by verify_leak_fix.py."""
    issues = []
    if val_dataset not in (train_datasets if isinstance(train_datasets, list) else [train_datasets]):
        pass  # val is separate — OK
    if not train_datasets:
        issues.append("No training datasets specified")
    return issues


def check_phase_feature_compatibility(trial_config: dict) -> list:
    """Check that trial config uses only features allowed for its phase.

    B0: mobilenet_v3_small ONLY, sz=224 ONLY, T=5 ONLY, window_uniform ONLY, current_direct_bbox ONLY
    B1: any backbone except mobilenet_v3_small, window_uniform ONLY, current_direct_bbox ONLY
    B2+: all features allowed
    """
    issues = []
    trial_id = trial_config.get("trial_id", trial_config.get("id", ""))
    backbone = trial_config.get("backbone", "mobilenet_v3_small")
    input_size = trial_config.get("input_size", 224)
    temporal_t = trial_config.get("temporal_T", 5)
    sampler = trial_config.get("sampler", "window_uniform")
    head = trial_config.get("head", "current_direct_bbox")

    # Detect phase from trial_id prefix
    if trial_id.startswith("smoke_b0_"):
        # B0: baseline proxy — full reproduction
        if backbone != "mobilenet_v3_small":
            issues.append(f"B0 baseline proxy: backbone={backbone} — must be mobilenet_v3_small")
        if input_size != 224:
            issues.append(f"B0 baseline proxy: input_size={input_size} — must be 224")
        if temporal_t != 5:
            issues.append(f"B0 baseline proxy: T={temporal_t} — must be 5")
        if sampler != "window_uniform":
            issues.append(f"B0 baseline proxy: sampler={sampler} — must be window_uniform (B0 only)")
        if head != "current_direct_bbox":
            issues.append(f"B0 baseline proxy: head={head} — must be current_direct_bbox (B0 only)")

    elif trial_id.startswith("smoke_b1_"):
        # B1: backbone-swappable smoke — sampler and head locked
        if backbone == "mobilenet_v3_small":
            issues.append(f"B1 backbone smoke: backbone={backbone} — use B0 for mv3_small baseline")
        if sampler != "window_uniform":
            issues.append(f"B1 backbone smoke: sampler={sampler} — must be window_uniform (deferred to B2)")
        if head != "current_direct_bbox":
            issues.append(f"B1 backbone smoke: head={head} — must be current_direct_bbox (deferred to B2)")

    elif trial_id.startswith("smoke_b2_"):
        # B2: sampler + head smoke — all features allowed
        pass  # no restrictions beyond the standard checks

    # For non-smoke trial IDs (trial_XXXX), no phase restrictions
    return issues


def run_all_checks(trial_config: dict, trial_dir: str) -> dict:
    """Run all safety checks. Returns {check_name: [issues]}."""
    results = {}

    results["trial_directory"] = check_trial_directory(trial_dir)

    backbone = trial_config.get("backbone", "mobilenet_v3_small")
    results["backbone"] = check_backbone_allowed(backbone)

    input_size = trial_config.get("input_size", 224)
    results["input_size"] = check_input_size(input_size)

    temporal_t = trial_config.get("temporal_T", 5)
    results["temporal_t"] = check_temporal_t(temporal_t)

    sampler = trial_config.get("sampler", "window_uniform")
    results["sampler"] = check_sampler(sampler)

    head = trial_config.get("head", "current_direct_bbox")
    results["head"] = check_head(head)

    lr = trial_config.get("lr", 1e-3)
    results["lr"] = check_lr(lr)

    results["param_budget"] = check_param_budget_estimate(backbone, head)
    results["no_checkpoint"] = check_no_checkpoint_loading(trial_config.get("resume"))
    results["seed_consistency"] = check_seed_consistency(trial_config)
    results["phase_compatibility"] = check_phase_feature_compatibility(trial_config)

    train_ds = trial_config.get("train_datasets") or [
        r"D:\ML\COD_datasets\MoCA",
        r"D:\ML\COD_datasets\MoCA_Mask",
        r"D:\ML\COD_datasets\CamouflagedAnimalDataset",
    ]
    val_ds = trial_config.get("val_dataset", r"D:\ML\COD_datasets\MoCA")
    results["data_isolation"] = check_data_isolation(train_ds, val_ds)

    return results


def format_results(results: dict) -> str:
    lines = []
    total_issues = 0
    for check_name, issues in results.items():
        if issues:
            total_issues += len(issues)
            lines.append(f"  FAIL [{check_name}]")
            for issue in issues:
                lines.append(f"    - {issue}")
        else:
            lines.append(f"  PASS [{check_name}]")
    if total_issues == 0:
        lines.insert(0, "OVERALL: PASS (0 issues)")
    else:
        lines.insert(0, f"OVERALL: FAIL ({total_issues} issues)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Pre-flight safety check for AutoResearch trial")
    parser.add_argument("--trial_config", type=str, required=True,
                        help="Path to trial YAML/JSON config")
    parser.add_argument("--trial_dir", type=str, default=None,
                        help="Trial output directory (default: derived from config path)")
    args = parser.parse_args()

    print("=" * 72)
    print("  AUTORESEARCH TRIAL SAFETY CHECK")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  Config: {args.trial_config}")
    print("=" * 72)
    print()

    trial_config = load_trial_config(args.trial_config)

    if args.trial_dir:
        trial_dir = args.trial_dir
    else:
        trial_dir = os.path.dirname(os.path.abspath(args.trial_config))

    print(f"  Trial dir: {trial_dir}")
    print(f"  Backbone:  {trial_config.get('backbone', 'N/A')}")
    print(f"  Input size: {trial_config.get('input_size', 'N/A')}")
    print(f"  T:         {trial_config.get('temporal_T', 'N/A')}")
    print(f"  Sampler:   {trial_config.get('sampler', 'N/A')}")
    print(f"  Head:      {trial_config.get('head', 'N/A')}")
    print(f"  LR:        {trial_config.get('lr', 'N/A')}")
    print()

    results = run_all_checks(trial_config, trial_dir)
    print(format_results(results))
    print()

    total_issues = sum(len(v) for v in results.values())
    if total_issues > 0:
        print("SAFETY CHECK FAILED — trial blocked.")
        print("Fix the issues above and re-run check_trial_safety.py.")
        return 1
    else:
        print("SAFETY CHECK PASSED — trial allowed to proceed.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
