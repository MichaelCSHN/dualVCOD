"""Score completed AutoResearch trials and rank them for promotion.

Usage:
    python tools/autoresearch/score_trials.py
    python tools/autoresearch/score_trials.py --output rankings.json

Enhanced scoring (Revision 1):
  base = 1.0 × val_miou + 0.3 × val_recall_at_0_5 + 0.2 × val_recall_at_0_3
  area_penalty = abs(ln(global_area_ratio)) × 0.05
  empty_penalty = empty_pred_rate × 0.50
  instability_penalty = 0.02 if miou_std_last_3 > 0.03 else 0.0
  composite = base - area_penalty - empty_penalty - instability_penalty

  Tiebreaker (in order): fewer params, higher FPS, higher mIoU.

Hard reject rules (staged):
  Always-fire: OOM, NaN loss, data leak, params > 35M
  After 5 epochs: weak if mIoU < center_prior + 0.03; hard reject if loss diverging
  After 12 epochs: hard reject if mIoU < baseline - 0.03 or R@0.5 < baseline - 0.05
  Universal: empty_pred_rate > 2%, global_area_ratio < 0.4 or > 2.5
"""

import sys
import os
import json
import csv
import math
import argparse
from glob import glob
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

AUTORESEARCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "local_runs", "autoresearch"
)

# Baseline values from Phase 1.5 clean retraining
BASELINE_MIOU = 0.2861
BASELINE_R_0_5 = 0.1978
BASELINE_R_0_3 = 0.4141
CENTER_PRIOR_MIOU = 0.2017

# Scoring weights
PRIMARY_WEIGHT = 1.0
SECONDARY_R05_WEIGHT = 0.3
SECONDARY_R03_WEIGHT = 0.2
AREA_PENALTY_FACTOR = 0.05
EMPTY_PENALTY_FACTOR = 0.50
INSTABILITY_PENALTY = 0.02
INSTABILITY_STD_THRESHOLD = 0.03


def load_trial_metadata(trial_dir: str) -> dict | None:
    meta_path = os.path.join(trial_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def compute_composite_score(meta: dict) -> float:
    """Enhanced composite score with penalties."""
    miou = meta.get("final_val_miou_fp32", 0)
    r05 = meta.get("final_val_recall_at_0_5", 0)
    r03 = meta.get("final_val_recall_at_0_3", 0)

    base = PRIMARY_WEIGHT * miou + SECONDARY_R05_WEIGHT * r05 + SECONDARY_R03_WEIGHT * r03

    # Area penalty: abs(log(global_area_ratio)) penalizes systematic over/under-sizing
    gar = meta.get("global_area_ratio", 1.0)
    if gar > 0:
        area_penalty = abs(math.log(gar)) * AREA_PENALTY_FACTOR
    else:
        area_penalty = 0.10  # degenerate — max penalty

    # Empty penalty: dead predictions = model not engaging
    empty_rate = meta.get("empty_pred_rate", 0)
    empty_penalty = empty_rate * EMPTY_PENALTY_FACTOR

    # Instability penalty: high std in last 3 epochs = not converged
    miou_std = meta.get("miou_std_last_3_epochs", 0)
    instability_penalty = INSTABILITY_PENALTY if miou_std > INSTABILITY_STD_THRESHOLD else 0.0

    composite = base - area_penalty - empty_penalty - instability_penalty

    # Store breakdown for transparency
    meta["_score_breakdown"] = {
        "base": round(base, 4),
        "area_penalty": round(area_penalty, 4),
        "empty_penalty": round(empty_penalty, 4),
        "instability_penalty": round(instability_penalty, 4),
        "composite": round(composite, 4),
    }

    return round(composite, 4)


def check_hard_reject(meta: dict) -> tuple:
    """Staged hard reject check. Returns (rejected: bool, weak: bool, reasons: list)."""
    reasons = []
    weak = False

    # ── Always-fire ───────────────────────────────────────────────────
    if meta.get("status") == "failed":
        reasons.append(f"trial_failed: {meta.get('reason', 'unknown')}")
        return True, False, reasons
    if meta.get("status") == "blocked":
        reasons.append("trial_blocked_by_safety_check")
        return True, False, reasons

    miou = meta.get("final_val_miou_fp32", 0)
    r05 = meta.get("final_val_recall_at_0_5", 0)
    params = meta.get("total_params", 0)
    epochs = meta.get("epochs", 0)
    empty_rate = meta.get("empty_pred_rate", 0)
    global_area_ratio = meta.get("global_area_ratio", 1.0)
    train_losses = meta.get("train_losses", [])

    # OOM / NaN / data leak / param budget — always-fire
    if meta.get("reason", "").startswith("oom"):
        reasons.append("oom")
    if params > 35_000_000:
        reasons.append(f"params={params:,} > 35M budget")

    # NaN or zero miou with zero best
    best_miou = meta.get("best_val_miou", 0)
    if miou <= 0.001 and best_miou <= 0.001 and epochs > 0:
        reasons.append("zero_miou — model not learning")
        return True, False, reasons

    # ── After 5 epochs ────────────────────────────────────────────────
    if epochs >= 5:
        # Diverging loss check
        if len(train_losses) >= 5 and train_losses[0] > 0:
            if train_losses[-1] > 2.0 * train_losses[0]:
                reasons.append(f"diverging_loss: epoch{epochs}={train_losses[-1]:.4f} > 2× epoch1={train_losses[0]:.4f}")

        # Weak check: mIoU barely above center prior
        if miou < CENTER_PRIOR_MIOU + 0.03:
            weak = True

    # ── After 12 epochs ───────────────────────────────────────────────
    if epochs >= 12:
        if miou < BASELINE_MIOU - 0.03:
            reasons.append(f"miou={miou:.4f} < baseline_miou({BASELINE_MIOU:.4f}) - 0.03")
        if r05 < BASELINE_R_0_5 - 0.05:
            reasons.append(f"r05={r05:.4f} < baseline_r05({BASELINE_R_0_5:.4f}) - 0.05")

    # ── Universal structural rules ────────────────────────────────────
    if empty_rate > 0.02:
        reasons.append(f"empty_pred_rate={empty_rate:.4f} > 2%")

    if global_area_ratio < 0.4:
        reasons.append(f"global_area_ratio={global_area_ratio:.4f} < 0.4 — systematic under-sizing")
    elif global_area_ratio > 2.5:
        reasons.append(f"global_area_ratio={global_area_ratio:.4f} > 2.5 — systematic over-sizing")

    rejected = len(reasons) > 0
    return rejected, weak, reasons


def score_all_trials(autoresearch_dir: str = AUTORESEARCH_DIR) -> list:
    """Scan all trial directories, score, rank, and return sorted list."""
    trial_dirs = sorted(glob(os.path.join(autoresearch_dir, "trial_*")))
    trial_dirs += sorted(glob(os.path.join(autoresearch_dir, "smoke_*")))

    scored = []
    for td in trial_dirs:
        meta = load_trial_metadata(td)
        if meta is None:
            continue

        trial_id = meta.get("trial_id", os.path.basename(td))

        rejected, weak, reject_reasons = check_hard_reject(meta)
        composite = compute_composite_score(meta) if not rejected else 0.0

        entry = {
            "trial_id": trial_id,
            "trial_dir": td,
            "status": meta.get("status", "unknown"),
            "backbone": meta.get("backbone", "?"),
            "input_size": meta.get("input_size", "?"),
            "temporal_T": meta.get("temporal_T_train", "?"),
            "eval_T": meta.get("eval_T_primary", "?"),
            "sampler": meta.get("sampler", "?"),
            "head": meta.get("head", "?"),
            "lr": meta.get("lr", "?"),
            "epochs": meta.get("epochs", "?"),
            "train_seed": meta.get("train_seed", "?"),
            "total_params": meta.get("total_params", 0),
            "val_miou": meta.get("final_val_miou_fp32", 0),
            "val_recall_at_0_5": meta.get("final_val_recall_at_0_5", 0),
            "val_recall_at_0_3": meta.get("final_val_recall_at_0_3", 0),
            "diag_miou_T5": meta.get("diag_val_miou_T5"),
            "global_area_ratio": meta.get("global_area_ratio", 1.0),
            "mean_sample_area_ratio": meta.get("mean_sample_area_ratio", 1.0),
            "empty_pred_rate": meta.get("empty_pred_rate", 0),
            "empty_pred_count": meta.get("empty_pred_count", 0),
            "miou_std_last_3": meta.get("miou_std_last_3_epochs", 0),
            "inference_fps": meta.get("inference_fps", 0),
            "gpu_mem_gib": meta.get("gpu_mem_gib", 0),
            "train_time_s": meta.get("total_train_time_s", 0),
            "composite_score": composite,
            "score_breakdown": meta.get("_score_breakdown", {}),
            "hard_rejected": rejected,
            "marked_weak": weak,
            "reject_reasons": reject_reasons,
        }
        scored.append(entry)

    # Sort: non-rejected first by composite descending
    scored.sort(key=lambda x: (
        x["hard_rejected"],
        not x["marked_weak"],
        -x["composite_score"],
        x["total_params"],     # tiebreaker: fewer params
        -x["inference_fps"],   # tiebreaker: higher FPS
        -x["val_miou"],        # tiebreaker: higher mIoU
    ))

    # Assign ranks (weak trials ranked below non-weak)
    rank = 0
    for entry in scored:
        if not entry["hard_rejected"]:
            rank += 1
            entry["rank"] = rank
        else:
            entry["rank"] = None

    return scored


def write_rankings(scored: list, output_dir: str = AUTORESEARCH_DIR):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "rankings.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "baseline_miou": BASELINE_MIOU,
            "baseline_recall_at_0_5": BASELINE_R_0_5,
            "baseline_recall_at_0_3": BASELINE_R_0_3,
            "center_prior_miou": CENTER_PRIOR_MIOU,
            "scoring": {
                "composite_formula": (
                    "base - area_penalty - empty_penalty - instability_penalty"
                ),
                "base_formula": "1.0*miou + 0.3*r05 + 0.2*r03",
                "area_penalty_formula": "abs(ln(global_area_ratio)) * 0.05",
                "empty_penalty_formula": "empty_pred_rate * 0.50",
                "instability_penalty_formula": "0.02 if std(miou_last_3) > 0.03 else 0.0",
                "area_metrics": ["global_area_ratio", "mean_sample_area_ratio"],
            },
            "hard_reject_rules": [
                "oom", "nan_loss", "data_leak", "params > 35M",
                "5ep: diverging_loss (loss[last] > 2*loss[1])",
                "12ep: miou < baseline - 0.03",
                "12ep: r05 < baseline_r05 - 0.05",
                "universal: empty_pred_rate > 2%",
                "universal: global_area_ratio < 0.4 or > 2.5",
            ],
            "weak_mark_rule": "5ep: miou < center_prior + 0.03",
            "total_trials": len(scored),
            "passed_trials": sum(1 for s in scored if not s["hard_rejected"]),
            "weak_trials": sum(1 for s in scored if s.get("marked_weak")),
            "rejected_trials": sum(1 for s in scored if s["hard_rejected"]),
            "rankings": scored,
        }, f, indent=2)
    print(f"  Rankings JSON: {json_path}")

    csv_path = os.path.join(output_dir, "rankings.csv")
    fieldnames = [
        "rank", "trial_id", "backbone", "input_size", "temporal_T", "sampler",
        "head", "lr", "epochs", "total_params", "val_miou", "val_recall_at_0_5",
        "val_recall_at_0_3", "global_area_ratio", "mean_sample_area_ratio",
        "empty_pred_rate", "miou_std_last_3", "composite_score", "inference_fps",
        "hard_rejected", "marked_weak", "reject_reasons",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for entry in scored:
            row = dict(entry)
            row["reject_reasons"] = "; ".join(entry["reject_reasons"])
            writer.writerow(row)
    print(f"  Rankings CSV:  {csv_path}")


def print_summary(scored: list):
    print()
    print("=" * 72)
    print("  TRIAL RANKINGS")
    print("=" * 72)
    header = (f"  {'Rk':>3s} {'Trial ID':<22s} {'Backbone':<20s} {'Sz':>4s} "
              f"{'mIoU':>7s} {'R@0.5':>7s} {'GAR':>6s} {'Empty%':>6s} {'Score':>7s} {'Status'}")
    print(header)
    print("  " + "-" * 70)

    for entry in scored:
        if entry["hard_rejected"]:
            status = f"REJECTED ({'; '.join(entry['reject_reasons'][:1])})"
        elif entry.get("marked_weak"):
            status = f"rank #{entry['rank']} [WEAK]"
        else:
            status = f"rank #{entry['rank']}"
        gar = entry.get("global_area_ratio", 1.0)
        empty = entry.get("empty_pred_rate", 0)
        line = (f"  {str(entry['rank'] or '—'):>3s} "
                f"{entry['trial_id']:<22s} "
                f"{entry['backbone']:<20s} "
                f"{entry['input_size']:>4d} "
                f"{entry['val_miou']:7.4f} "
                f"{entry['val_recall_at_0_5']:7.4f} "
                f"{gar:6.3f} "
                f"{empty*100:5.1f}% "
                f"{entry['composite_score']:7.4f} "
                f"{status}")
        print(line)

    print()
    passed = [s for s in scored if not s["hard_rejected"]]
    weak = [s for s in passed if s.get("marked_weak")]
    strong = [s for s in passed if not s.get("marked_weak")]
    rejected = [s for s in scored if s["hard_rejected"]]

    print(f"  Total: {len(scored)} trials "
          f"({len(strong)} passed strong, {len(weak)} weak, {len(rejected)} rejected)")

    if strong:
        best = strong[0]
        print(f"  Best:  {best['trial_id']} — mIoU={best['val_miou']:.4f}, "
              f"score={best['composite_score']:.4f}")
        delta = best["val_miou"] - BASELINE_MIOU
        print(f"  Delta vs baseline ({BASELINE_MIOU:.4f}): {delta:+.4f}")
        if best.get("score_breakdown"):
            sb = best["score_breakdown"]
            print(f"  Score breakdown: base={sb['base']:.4f} "
                  f"area_pen={sb['area_penalty']:.4f} "
                  f"empty_pen={sb['empty_penalty']:.4f} "
                  f"instab_pen={sb['instability_penalty']:.4f}")

    if weak:
        print(f"  Weak trials (mIoU < center_prior + 0.03 = {CENTER_PRIOR_MIOU + 0.03:.4f}):")
        for w in weak:
            print(f"    {w['trial_id']}: mIoU={w['val_miou']:.4f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Score and rank AutoResearch trials")
    parser.add_argument("--autoresearch_dir", type=str, default=AUTORESEARCH_DIR)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 72)
    print("  AUTORESEARCH TRIAL SCORING (Revision 1)")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  Dir:  {args.autoresearch_dir}")
    print(f"  Baseline mIoU: {BASELINE_MIOU:.4f}  Center prior: {CENTER_PRIOR_MIOU:.4f}")
    print("=" * 72)

    if not os.path.isdir(args.autoresearch_dir):
        print(f"  Directory not found: {args.autoresearch_dir}")
        print(f"  No trials scored. Run run_trial.py first.")
        return 0

    scored = score_all_trials(args.autoresearch_dir)

    if not scored:
        print("  No completed trials found.")
        return 0

    write_rankings(scored, args.output or args.autoresearch_dir)
    print_summary(scored)

    # Promotion candidates
    strong = [s for s in scored if not s["hard_rejected"] and not s.get("marked_weak")]
    print("  Promotion candidates (strong, ranked):")
    n_promote = min(8, len(strong))
    for i, entry in enumerate(strong[:n_promote]):
        print(f"    {i+1}. {entry['trial_id']} — {entry['backbone']} "
              f"mIoU={entry['val_miou']:.4f} score={entry['composite_score']:.4f}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
