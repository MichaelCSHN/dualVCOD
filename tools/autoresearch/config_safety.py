"""CUDA environment setup + GPU preflight + config validation.

Must be called BEFORE any CUDA operations (torch.cuda.init, model creation).
Ensures the GPU is free, healthy, and properly configured.
"""

import os, sys, time, subprocess, json
from datetime import datetime


def setup_cuda_allocator(expandable: bool = True):
    """Configure CUDA allocator before torch imports CUDA.

    Must run before any `import torch` or CUDA API call.
    max_split_size_mb: prevents large-block fragmentation OOM
    expandable_segments: PyTorch >= 2.4, reduces fragmentation
    """
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        conf = "max_split_size_mb:256"
        if expandable:
            conf += ",expandable_segments:True"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf


def preflight_gpu(min_free_gib: float = 8.0) -> dict:
    """Check GPU health and availability before starting a trial.

    Returns dict with status and diagnostics. Raises RuntimeError on fatal issues.
    """
    import torch

    result = {
        "timestamp": datetime.now().isoformat(),
        "status": "ok",
        "checks": {},
        "warnings": [],
    }

    # 1. CUDA available
    cuda_available = torch.cuda.is_available()
    result["checks"]["cuda_available"] = cuda_available
    if not cuda_available:
        result["status"] = "fatal"
        result["checks"]["error"] = "CUDA not available — cannot run GPU trials"
        return result

    device = torch.cuda.current_device()
    result["checks"]["device"] = torch.cuda.get_device_name(device)

    # 2. GPU memory
    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    free_mem = total_mem - reserved
    result["checks"]["total_mem_gib"] = round(total_mem, 1)
    result["checks"]["reserved_gib"] = round(reserved, 2)
    result["checks"]["allocated_gib"] = round(allocated, 2)
    result["checks"]["free_mem_gib"] = round(free_mem, 1)

    if reserved > 1.0:
        result["warnings"].append(f"GPU has {reserved:.1f} GiB reserved — another process may be running")

    if free_mem < min_free_gib:
        result["status"] = "fatal"
        result["checks"]["error"] = f"Only {free_mem:.1f} GiB free (need {min_free_gib:.1f})"

    # 3. GPU compute capability (sm_86 = RTX 30 series, sm_89 = RTX 40 series)
    cc_major = torch.cuda.get_device_capability(device)[0]
    cc_minor = torch.cuda.get_device_capability(device)[1]
    result["checks"]["compute_capability"] = f"{cc_major}.{cc_minor}"

    # 4. Check nvidia-smi for other processes
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            timeout=10, text=True
        ).strip()
        other_procs = []
        my_pid = os.getpid()
        for line in smi_out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                pid_str, name, mem_str = parts[0], parts[1], parts[2]
                try:
                    pid = int(pid_str)
                    mem = float(mem_str)
                    if pid != my_pid and mem > 100:  # ignore self and tiny allocations
                        other_procs.append({"pid": pid, "name": name, "mem_mib": int(mem)})
                except ValueError:
                    pass
        result["checks"]["other_cuda_processes"] = other_procs
        if other_procs:
            result["warnings"].append(f"Other CUDA processes detected: {other_procs}")
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        result["checks"]["other_cuda_processes"] = "nvidia-smi unavailable"

    # 5. CUDA allocator config
    result["checks"]["cuda_alloc_conf"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(not set)")

    # 6. GPU temperature (quick check — not blocking)
    try:
        smi_temp = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            timeout=10, text=True
        ).strip()
        temp = int(smi_temp)
        result["checks"]["gpu_temp_c"] = temp
        if temp > 85:
            result["warnings"].append(f"GPU temperature {temp}°C — may throttle")
    except Exception:
        pass

    return result


def validate_trial_config(config: dict) -> list:
    """Validate trial config for safety and sanity. Returns list of issues."""
    issues = []

    # Required keys
    for key in ["backbone", "input_size", "temporal_T", "head", "lr", "epochs"]:
        if key not in config:
            issues.append(f"Missing required key: {key}")

    # Sanity bounds
    if config.get("input_size", 224) > 640:
        issues.append(f"input_size={config['input_size']} is unusually large")
    if config.get("temporal_T", 5) > 15:
        issues.append(f"temporal_T={config['temporal_T']} is unusually large")
    if config.get("epochs", 5) > 100:
        issues.append(f"epochs={config['epochs']} is unusually large")
    if config.get("lr", 0.001) > 0.1:
        issues.append(f"lr={config['lr']} is unusually high")
    if config.get("num_workers", 0) > 8:
        issues.append(f"num_workers={config['num_workers']} may cause CPU oversubscription")

    # Batch size sanity
    train_bs = config.get("train_batch_size", config.get("batch_size", 16))
    eval_bs = config.get("eval_batch_size", train_bs * 2)
    if train_bs > 128:
        issues.append(f"train_batch_size={train_bs} is unusually large")
    if eval_bs > 256:
        issues.append(f"eval_batch_size={eval_bs} is unusually large")

    # Loss weights sanity
    lw = config.get("loss_weights", {})
    for k in ["smooth_l1", "giou", "center", "log_wh", "objectness"]:
        if k in lw and lw[k] > 10.0:
            issues.append(f"loss_weights.{k}={lw[k]} is unusually high")

    return issues
