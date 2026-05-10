"""Aggregate scored trials across phases and generate summary report.

Usage:
    python tools/autoresearch/aggregate_trials.py                          # all trials
    python tools/autoresearch/aggregate_trials.py --phase 1_6_C            # single phase
    python tools/autoresearch/aggregate_trials.py --output report.md       # custom output

Generates:
  1. Summary table across all finished trials
  2. Per-dimension trend analysis (backbone, input_size, T, sampler, head, lr)
  3. Pareto frontier table (mIoU vs params vs FPS)
  4. Promotion recommendations for next phase
  5. Hard-reject breakdown with reasons
"""

import sys
import os
import json
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

AUTORESEARCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "local_runs", "autoresearch"
)
REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "reports"
)

BASELINE_MIOU = 0.2861
BASELINE_R_0_5 = 0.1978
BASELINE_R_0_3 = 0.4141
BASELINE_PARAMS = 1_400_000
BASELINE_BACKBONE = "mobilenet_v3_small"
CENTER_PRIOR_MIOU = 0.2017


def load_rankings(autoresearch_dir: str) -> dict:
    """Load rankings.json."""
    path = os.path.join(autoresearch_dir, "rankings.json")
    if not os.path.isfile(path):
        return {"rankings": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dimension_trend_analysis(scored: list) -> dict:
    """For each categorical dimension, compute mean mIoU per value."""
    dims = ["backbone", "input_size", "temporal_T", "sampler", "head", "lr"]
    trends = {}
    for dim in dims:
        buckets = defaultdict(list)
        for entry in scored:
            if entry["hard_rejected"]:
                continue  # skip rejected for trend analysis
            val = entry.get(dim)
            if val is not None:
                buckets[str(val)].append(entry["val_miou"])
        trends[dim] = {
            str(k): {
                "count": len(v),
                "mean_miou": round(float(np.mean(v)), 4) if v else 0,
                "std_miou": round(float(np.std(v)), 4) if len(v) > 1 else 0,
                "max_miou": round(float(np.max(v)), 4) if v else 0,
            }
            for k, v in sorted(buckets.items())
        }
    return trends


def pareto_frontier(scored: list) -> list:
    """Identify Pareto-optimal trials on (mIoU, params, FPS).

    A trial dominates another if it's better on all three axes.
    We report the Pareto frontier (non-dominated set).
    """
    passed = [s for s in scored if not s["hard_rejected"]]
    if not passed:
        return []

    # Extract vectors: higher mIoU is better, lower params is better, higher FPS is better
    # Normalize: params -> 1/params (higher is better, like FPS and mIoU)
    vectors = []
    for s in passed:
        miou = s["val_miou"]
        params_inv = 1.0 / max(s["total_params"], 1)
        fps = s["inference_fps"]
        vectors.append((miou, params_inv, fps))

    n = len(passed)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            vi, vj = vectors[i], vectors[j]
            # j dominates i if all axes >= and at least one >
            if (vj[0] >= vi[0] and vj[1] >= vi[1] and vj[2] >= vi[2] and
                    (vj[0] > vi[0] or vj[1] > vi[1] or vj[2] > vi[2])):
                dominated[i] = True
                break

    pareto = [passed[i] for i in range(n) if not dominated[i]]
    # Sort by mIoU descending
    pareto.sort(key=lambda x: -x["val_miou"])
    return pareto


def generate_report(scored: list, trends: dict, pareto: list, phase: str = "all") -> str:
    """Generate Markdown summary report."""
    passed = [s for s in scored if not s["hard_rejected"]]
    strong = [s for s in passed if not s.get("marked_weak")]
    weak = [s for s in passed if s.get("marked_weak")]
    rejected = [s for s in scored if s["hard_rejected"]]

    lines = []
    w = lines.append

    w(f"# AutoResearch Phase 1.6 Aggregate Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w("")
    w(f"**Phase filter**: {phase}")
    w(f"**Trials scored**: {len(scored)} "
      f"({len(strong)} strong, {len(weak)} weak, {len(rejected)} rejected)")
    w(f"**Baseline**: mIoU={BASELINE_MIOU:.4f}, R@0.5={BASELINE_R_0_5:.4f}, "
      f"params={BASELINE_PARAMS:,} ({BASELINE_BACKBONE})")
    w(f"**Center prior mIoU**: {CENTER_PRIOR_MIOU:.4f}")
    w("")

    # ── Top trials ────────────────────────────────────────────────────
    w("## 1. Top Trials by Composite Score")
    w("")
    w("| Rank | Trial ID | Backbone | Sz | T | mIoU | R@0.5 | GAR | Empty% | Score | Params | FPS | Flag |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for entry in passed[:15]:
        flags = []
        if entry.get("marked_weak"):
            flags.append("WEAK")
        gar = entry.get("global_area_ratio", 1.0)
        empty = entry.get("empty_pred_rate", 0)
        w(f"| {entry.get('rank', '—')} | {entry['trial_id']} | {entry['backbone']} | "
          f"{entry['input_size']} | {entry['temporal_T']} | "
          f"{entry['val_miou']:.4f} | {entry['val_recall_at_0_5']:.4f} | "
          f"{gar:.3f} | {empty*100:.1f}% | "
          f"{entry['composite_score']:.4f} | {entry['total_params']:,} | "
          f"{entry.get('inference_fps', '—')} | {' '.join(flags)} |")
    w("")

    # ── Baseline delta ────────────────────────────────────────────────
    if strong or passed:
        best = strong[0] if strong else passed[0]
        delta_miou = best["val_miou"] - BASELINE_MIOU
        delta_r05 = best["val_recall_at_0_5"] - BASELINE_R_0_5
        w("## 2. Best Trial vs Baseline")
        w("")
        w(f"- **Best trial**: {best['trial_id']} ({best['backbone']}, {best['input_size']}px, T={best['temporal_T']})")
        w(f"- **mIoU**: {best['val_miou']:.4f} (baseline {BASELINE_MIOU:.4f}, Δ={delta_miou:+.4f})")
        w(f"- **R@0.5**: {best['val_recall_at_0_5']:.4f} (baseline {BASELINE_R_0_5:.4f}, Δ={delta_r05:+.4f})")
        w(f"- **Global area ratio**: {best.get('global_area_ratio', '—')}")
        w(f"- **Mean sample area ratio**: {best.get('mean_sample_area_ratio', '—')}")
        w(f"- **Empty pred rate**: {best.get('empty_pred_rate', 0)*100:.1f}%")
        w(f"- **Params**: {best['total_params']:,} (baseline {BASELINE_PARAMS:,})")
        w(f"- **Inference FPS**: {best.get('inference_fps', '—')}")
        if best.get("score_breakdown"):
            sb = best["score_breakdown"]
            w(f"- **Score breakdown**: base={sb['base']:.4f} area_pen={sb['area_penalty']:.4f} empty_pen={sb['empty_penalty']:.4f} instab_pen={sb['instability_penalty']:.4f}")
        w("")

    # ── Dimension trends ──────────────────────────────────────────────
    w("## 3. Dimension Trend Analysis")
    w("")
    for dim, buckets in trends.items():
        if not buckets:
            continue
        w(f"### {dim}")
        w("")
        w("| Value | Trials | Mean mIoU | Std | Max mIoU |")
        w("|---|---|---|---|---|")
        for val, stats in sorted(buckets.items()):
            w(f"| {val} | {stats['count']} | {stats['mean_miou']:.4f} | "
              f"{stats['std_miou']:.4f} | {stats['max_miou']:.4f} |")
        w("")

    # ── Pareto frontier ───────────────────────────────────────────────
    w("## 4. Pareto Frontier (mIoU vs Params vs FPS)")
    w("")
    w("| Trial ID | Backbone | Sz | mIoU | Params | FPS | GAR | Empty% |")
    w("|---|---|---|---|---|---|---|---|")
    for entry in pareto:
        gar = entry.get("global_area_ratio", 1.0)
        empty = entry.get("empty_pred_rate", 0)
        w(f"| {entry['trial_id']} | {entry['backbone']} | {entry['input_size']} | "
          f"{entry['val_miou']:.4f} | {entry['total_params']:,} | "
          f"{entry.get('inference_fps', '—')} | {gar:.3f} | {empty*100:.1f}% |")
    w("")

    # ── Hard reject breakdown ─────────────────────────────────────────
    if rejected:
        w("## 5. Hard Reject Breakdown")
        w("")
        reject_counts = defaultdict(list)
        for entry in rejected:
            for reason in entry.get("reject_reasons", ["unknown"]):
                reject_counts[reason].append(entry["trial_id"])
        w("| Reason | Count | Trial IDs |")
        w("|---|---|---|")
        for reason, tids in sorted(reject_counts.items(), key=lambda x: -len(x[1])):
            w(f"| {reason} | {len(tids)} | {', '.join(tids[:5])}{'...' if len(tids) > 5 else ''} |")
        w("")

    # ── Promotion recommendations ─────────────────────────────────────
    w("## 6. Promotion Recommendations")
    w("")
    strong = [s for s in passed if not s.get("marked_weak")]
    top_n = min(8, len(strong))
    if top_n > 0:
        w(f"**Promote to next phase** (top {top_n} strong by composite score):")
        w("")
        for i, entry in enumerate(strong[:top_n]):
            w(f"{i+1}. `{entry['trial_id']}` — {entry['backbone']}, "
              f"sz={entry['input_size']}, T={entry['temporal_T']}, "
              f"mIoU={entry['val_miou']:.4f}, score={entry['composite_score']:.4f}")
    else:
        w("No trials eligible for promotion.")
    w("")

    w("---")
    w(f"*Report generated at {datetime.now().isoformat()}*")
    w(f"*Scoring: composite = base - area_pen - empty_pen - instab_pen | base = mIoU + 0.3×R@0.5 + 0.2×R@0.3*")
    w(f"*Area metrics: global_area_ratio = mean(pred_area)/mean(gt_area); mean_sample_area_ratio = mean(pred_area/gt_area)*")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Aggregate AutoResearch trial results")
    parser.add_argument("--autoresearch_dir", type=str, default=AUTORESEARCH_DIR,
                        help="Path to local_runs/autoresearch/")
    parser.add_argument("--phase", type=str, default="all",
                        help="Phase label to filter by")
    parser.add_argument("--output", type=str, default=None,
                        help="Output report path (default: reports/report_aggregate_yyyymmddhhmm.md)")
    args = parser.parse_args()

    print("=" * 72)
    print("  AUTORESEARCH AGGREGATE REPORT")
    print(f"  Time:  {datetime.now().isoformat()}")
    print(f"  Phase: {args.phase}")
    print("=" * 72)

    rankings = load_rankings(args.autoresearch_dir)
    scored = rankings.get("rankings", [])

    if not scored:
        print("  No scored trials found. Run score_trials.py first.")
        return 0

    print(f"  Loaded {len(scored)} scored trials")
    print()

    # Run analysis
    trends = dimension_trend_analysis(scored)
    pareto = pareto_frontier(scored)
    report = generate_report(scored, trends, pareto, args.phase)

    # Write report
    if args.output:
        report_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        report_path = os.path.join(REPORTS_DIR, f"report_aggregate_{timestamp}.md")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"  Aggregate report: {report_path}")
    print()

    # Print dimension trends summary
    print("  Dimension trends:")
    for dim, buckets in trends.items():
        best_val = max(buckets.items(), key=lambda x: x[1]["mean_miou"]) if buckets else None
        if best_val:
            print(f"    {dim:16s}: best={best_val[0]:20s}  mean_mIoU={best_val[1]['mean_miou']:.4f}")

    print()
    print("=" * 72)
    print("  AGGREGATION COMPLETE")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
