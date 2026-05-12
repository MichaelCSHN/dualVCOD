"""Collect speedtest results from all _speedtest_* trial directories."""
import os, json, glob

def main():
    base = r"D:\dualvcod\local_runs\autoresearch"
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
        # Average epoch timing from profiles
        avg_data = sum(p["data_time_s"] for p in profiles) / len(profiles) if profiles else 0
        avg_fwd = sum(p["forward_time_s"] for p in profiles) / len(profiles) if profiles else 0
        avg_bwd = sum(p["backward_time_s"] for p in profiles) / len(profiles) if profiles else 0
        avg_epoch = sum(p["total_time_s"] for p in profiles) / len(profiles) if profiles else 0
        results.append({
            "trial_id": m["trial_id"],
            "train_batch_size": m["train_batch_size"],
            "num_workers": m.get("num_workers", "?"),
            "status": m["status"],
            "best_val_miou": m.get("best_val_miou"),
            "gpu_mem_gib": m.get("gpu_mem_gib"),
            "total_train_time_s": m.get("total_train_time_s"),
            "avg_epoch_s": round(avg_epoch, 1),
            "avg_data_s": round(avg_data, 1),
            "avg_fwd_s": round(avg_fwd, 1),
            "avg_bwd_s": round(avg_bwd, 1),
            "n_epochs": len(profiles),
        })

    # Print table
    print(f"{'Config':<24s} {'bs':>4s} {'nw':>3s} {'status':>12s} {'mIoU':>7s} {'epoch_s':>8s} {'data_s':>8s} {'fwd_s':>7s} {'bwd_s':>7s} {'GPU_GiB':>7s}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: (x["train_batch_size"], x["num_workers"])):
        print(f"{r['trial_id']:<24s} {r['train_batch_size']:>4d} {r['num_workers']:>3d} "
              f"{r['status']:>12s} {r['best_val_miou'] or 0:>7.4f} {r['avg_epoch_s']:>8.1f} "
              f"{r['avg_data_s']:>8.1f} {r['avg_fwd_s']:>7.1f} {r['avg_bwd_s']:>7.1f} "
              f"{r['gpu_mem_gib'] or 0:>7.2f}")

if __name__ == "__main__":
    main()
