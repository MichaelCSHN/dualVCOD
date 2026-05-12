# B1 Backbone-Only Smoke — Interim Status (GPU Crash)

**Time:** 2026-05-07 ~21:10 CST

## Completed Trials

| Trial | Backbone | Best val_mIoU | B1 Gate (0.2661) | Hard Reject | Status |
|---|---|---|---|---|---|
| smoke_b1_effb0 | efficientnet_b0 | **0.1708** (ep 1) | FAIL | **YES** (NaN ep 3-5) | Rejected |
| smoke_b1_mnv3large | mobilenet_v3_large | **0.2594** (ep 4) | FAIL (-0.0067) | No | Completed |
| smoke_b1_effb1 | efficientnet_b1 | N/A | N/A | N/A | **GPU crash** |

## Detailed: smoke_b1_effb0
- Train loss: 1.134 → 1.136 → NaN → NaN → NaN
- LR: 1e-3. Gradient explosion at epoch 3 despite clip_norm=2.0
- FPS: 26.3 | GPU: 1.79 GiB | Params: 4,587,456
- Verdict: **HARD REJECT** — NaN loss, EfficientNet at 512px/lr=1e-3 is unstable

## Detailed: smoke_b1_mnv3large
- Epoch trajectory: 0.2395 → 0.2362 → 0.2461 → **0.2594** → 0.2462
- Missed B1 gate by **0.0067**
- Train/val gap at best: 0.4994 - 0.2594 = 0.2400 (severe overfitting)
- Global area ratio: 1.17 (ok) | Mean sample ratio: 12.22 (extreme skew)
- FPS: 46.7 | GPU: 1.81 GiB | Params: 3,505,780
- Verdict: **FAILED GATE** — strongest performer but overfitting at 512px limits validation performance

## Current Blocker: GPU Lost State
- nvidia-smi: "GPU is lost. Reboot the system to recover this GPU"
- RTX 4090: State OK in Device Manager but inaccessible via CUDA
- RTX 2060: Error state in Device Manager
- Cause: CUDA OOM during efficientnet_b1 model init (7.1M params, `.to(DEVICE)`)
- Required action: **System reboot**

## After Reboot — smoke_b1_effb1
Config ready at `local_runs/autoresearch/smoke_b1_effb1/trial_config.json`:
- backbone: efficientnet_b1, input_size: 512, lr: 3e-4 (lower to avoid NaN)
- batch_size: 12, epochs: 5, train_seed: 42

Command:
```
python -u tools/autoresearch/run_trial.py --trial_id smoke_b1_effb1 --config local_runs/autoresearch/smoke_b1_effb1/trial_config.json
```

Note: The OOM during `.to(DEVICE)` with only 7.1M params is suspicious — likely caused by GPU memory fragmentation from prior runs, not the model size itself. A clean boot should resolve this.
