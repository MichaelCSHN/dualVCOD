"""Standalone batch runner for speedtest grid.

Usage (from PowerShell):
    python tools/autoresearch/run_speedtest_batch.py

Runs all 9 grid configs sequentially, collects timing profiles,
and prints a summary table. Results saved to each trial's directory.
"""
import subprocess, sys, os, time, json, glob
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRIAL_SCRIPT = os.path.join(PROJECT_ROOT, "tools", "autoresearch", "run_trial_minimal.py")

CONFIGS = [
    ("_speedtest_bs32_nw0", "configs/autoresearch/_speedtest_bs32_nw0.json"),
    ("_speedtest_bs32_nw2", "configs/autoresearch/_speedtest_bs32_nw2.json"),
    ("_speedtest_bs32_nw4", "configs/autoresearch/_speedtest_bs32_nw4.json"),
    ("_speedtest_bs48_nw0", "configs/autoresearch/_speedtest_bs48_nw0.json"),
    ("_speedtest_bs48_nw2", "configs/autoresearch/_speedtest_bs48_nw2.json"),
    ("_speedtest_bs48_nw4", "configs/autoresearch/_speedtest_bs48_nw4.json"),
    ("_speedtest_bs64_nw0", "configs/autoresearch/_speedtest_bs64_nw0.json"),
    ("_speedtest_bs64_nw2", "configs/autoresearch/_speedtest_bs64_nw2.json"),
    ("_speedtest_bs64_nw4", "configs/autoresearch/_speedtest_bs64_nw4.json"),
]


def collect_results():
    base = os.path.join(PROJECT_ROOT, "local_runs", "autoresearch")
    dirs = sorted(glob.glob(os.path.join(base, "_speedtest_bs*")))
    results = []
    for d in dirs:
        meta_path = os.path.join(d, "metadata.json")
        profiles_path = os.path.join(d, "profiles.jsonl")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            m = json.load(f)
        profiles = []
        if os.path.exists(profiles_path):
            with open(profiles_path) as f:
                for line in f:
                    profiles.append(json.loads(line.strip()))
        n = len(profiles)
        avg_data = sum(p["data_time_s"] for p in profiles) / n if n else 0
        avg_fwd = sum(p["forward_time_s"] for p in profiles) / n if n else 0
        avg_bwd = sum(p["backward_time_s"] for p in profiles) / n if n else 0
        avg_epoch = sum(p["total_time_s"] for p in profiles) / n if n else 0
        avg_batch = sum(p["data_time_s"] + p["forward_time_s"] + p["backward_time_s"]
                        for p in profiles) / sum(p.get("n_batches", 1) for p in profiles) if n else 0
        results.append({
            "trial_id": m["trial_id"],
            "bs": m.get("train_batch_size", "?"),
            "nw": m.get("num_workers", "?"),
            "status": m["status"],
            "best_val_miou": m.get("best_val_miou", 0),
            "gpu_mem_gib": m.get("gpu_mem_gib", 0),
            "total_time_s": m.get("total_train_time_s", 0),
            "avg_epoch_s": round(avg_epoch, 1),
            "avg_data_s": round(avg_data, 1),
            "avg_fwd_s": round(avg_fwd, 1),
            "avg_bwd_s": round(avg_bwd, 1),
            "avg_batch_s": round(avg_batch, 3),
        })
    return results


def print_summary(results):
    if not results:
        print("No completed results yet.")
        return
    print(f"\n{'Config':<24s} {'bs':>4s} {'nw':>3s} {'status':>12s} {'mIoU':>7s} "
          f"{'epoch_s':>8s} {'data_s':>8s} {'fwd_s':>7s} {'bwd_s':>7s} "
          f"{'batch_s':>8s} {'GPU_GiB':>7s}")
    print("-" * 110)
    for r in sorted(results, key=lambda x: (x["bs"], x["nw"])):
        print(f"{r['trial_id']:<24s} {r['bs']:>4} {r['nw']:>3} "
              f"{r['status']:>12s} {r['best_val_miou'] or 0:>7.4f} "
              f"{r['avg_epoch_s']:>8.1f} {r['avg_data_s']:>8.1f} "
              f"{r['avg_fwd_s']:>7.1f} {r['avg_bwd_s']:>7.1f} "
              f"{r['avg_batch_s']:>8.3f} {r['gpu_mem_gib'] or 0:>7.2f}")


def main():
    os.chdir(PROJECT_ROOT)
    total = len(CONFIGS)
    completed = 0
    start_all = time.time()

    print("=" * 80)
    print(f"  GRID SPEEDTEST: {total} configs (batch_size x num_workers)")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    for idx, (trial_id, cfg_rel) in enumerate(CONFIGS, 1):
        elapsed_m = (time.time() - start_all) / 60
        print(f"\n--- [{idx}/{total}] {trial_id} (elapsed: {elapsed_m:.1f}m) ---")

        cfg_path = os.path.join(PROJECT_ROOT, cfg_rel)
        start_run = time.time()

        result = subprocess.run(
            [sys.executable, TRIAL_SCRIPT, "--trial_id", trial_id, "--config", cfg_path],
            capture_output=False,
            cwd=PROJECT_ROOT,
        )

        run_time = time.time() - start_run
        if result.returncode != 0:
            print(f"  WARNING: {trial_id} exited with code {result.returncode} after {run_time:.0f}s")
        else:
            completed += 1
            print(f"  OK: {trial_id} completed in {run_time:.0f}s")

        # Print incremental summary
        results = collect_results()
        if len(results) >= 1:
            print_summary(results)

    total_m = (time.time() - start_all) / 60
    print(f"\n{'=' * 80}")
    print(f"  GRID COMPLETE: {completed}/{total} succeeded in {total_m:.1f}m")
    print(f"{'=' * 80}")

    # Final summary
    results = collect_results()
    print_summary(results)

    # Save summary JSON
    summary_path = os.path.join(PROJECT_ROOT, "local_runs", "autoresearch",
                                "_speedtest_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
