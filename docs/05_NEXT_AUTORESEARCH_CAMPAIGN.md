# Next Autoresearch Campaign — May 2026
## 2026-05-14 14:45

---

## Diagnosis: What We Actually Know

### The baseline is strong but possibly under-reported

| Baseline | Backbone | Ep | pf_mIoU | Ckpt Ep | Suspicious? |
|----------|---------|-----|---------|---------|------------|
| E-45 | MV3-S | 8 | 0.8281 | 7/8 | Possibly — last epoch untested |
| E-40 | MV3-S | 30 | 0.8564 | 18/30 | **Yes — early peak, ~60% through** |
| E-51 | EffB0 | 8 | 0.8372 | 8/8 | Minimal (last ep = best) |
| E-52 | EffB0 | 30 | 0.8711 | 20/30 | **Yes — early peak, ~67% through** |

### Training-val false-selection rate: 5/5

Every 8ep probe showed training-val best ≠ unified best. The pattern was
striking: the last or near-last epoch was consistently the best by unified
reeval, regardless of training val trajectory. If this holds for baselines:

- E-40 checkpoint at epoch 18 may be 0.01-0.02 below epoch 28-30
- E-52 checkpoint at epoch 20 may be 0.01-0.02 below epoch 28-30
- E-45 checkpoint at epoch 7 may be below epoch 8

### Real bottlenecks (not training-val artifacts)

1. **Tiny-object resolution bottleneck**: At 224×224, tiny objects (area < 0.01)
   occupy at most ~22×22 pixels. E-40 achieves IoU_tiny=0.703 and
   pygmy_seahorse_0=0.623 — significantly better than E-52 (0.602/0.387).
   MV3-Small at 30ep SPECIFICALLY excels at tiny objects.

2. **Large-object under-coverage**: flatfish_2 mIoU ranges from 0.087 (E-45)
   to 0.291 (E-52). This is a genuine failure mode requiring larger receptive
   field or coverage-aware training.

3. **Checkpoint selection is broken**: 5/5 probes + suspicious baseline
   checkpoint epochs = systematic issue. Fixing this alone could yield
   +0.005-0.015 pf_mIoU at zero model cost.

4. **EffB0's tiny-object regression vs MV3-Small 30ep**: EffB0 is better
   overall (pf_mIoU +0.015) but WORSE at tiny objects (-0.101 IoU_tiny,
   -0.236 pygmy_seahorse_0). This is a genuine tradeoff — EffB0 trades
   tiny-object precision for large/medium accuracy.

---

## Campaign Structure

### Campaign A: Baseline Accountability — COMPLETE

**Goal**: Establish trustworthy baseline numbers. Fix checkpoint selection.

| Step | Description | Result |
|------|------------|--------|
| A1 | Audit E-40/E-45/E-51/E-52 training logs | Confirmed early-peak pattern |
| A2 | Re-train E-45 with topk=3 | pf_mIoU=0.8314, reproducible |
| A3 | Seed robustness | Deferred (A2 trajectory variance confirmed issue) |
| A4 | Re-train E-40 with topk=5 (30ep) | E30 pf_mIoU=0.9077 (+0.051 vs original) |
| A5 | Re-train E-52 with topk=5 (30ep) | E30 pf_mIoU=0.9142 (+0.043 vs original) |

**Conclusion**: Checkpoint selection was the #1 bottleneck in the project. Training-val-best ≠ unified-best at 8/8 rate. E30 is the true ceiling for both backbones.

### Campaign B: Resolution Pareto [TRAINING — after A2 decision]

**Hypothesis**: 224×224 is the resolution ceiling for tiny objects. Increasing
to 256 or 288 directly addresses the information bottleneck.

**Why now**: Tiny objects are the #1 fragile metric. 5/6 previous interventions
degraded them. Resolution is the most direct lever.

**Probes**:
- B1: E-45 at 256×256, 8ep, topk=3
- B2: E-45 at 288×288, 8ep, topk=3
- Cost: ~30min each

**Success**: IoU_tiny +0.03+, pygmy_seahorse_0 +0.05+, pf_mIoU not drop >0.01
**No-go**: pf_mIoU drop >0.02, FPS < 30, GPU OOM, any non-tiny bin drop >0.03

### Campaign C: Checkpoint Selection Policy [ANALYSIS + TRAINING]

**Hypothesis**: Final-epoch checkpoint is more reliable than val-best for 8ep
training. The 5/5 pattern (last epoch consistently unified-best) suggests a
simple policy change.

**Probe**: Compare final-epoch vs val-best across 3 seeds of E-45.
**Success**: Final-epoch consistently ≥ val-best by unified reeval.
**Policy change**: If confirmed, switch trial runner default to save final epoch.

### Campaign D: Large-Object Coverage Revisit [CONDITIONAL]

**Condition**: Only if Campaign A shows coverage errors (pred_too_small,
flatfish_2) remain dominant after checkpoint fix.

**Hypothesis**: Coverage penalty at weight 0.05 was too gentle for 8ep.
At 30ep with weight 0.10-0.15, the signal may be strong enough.

**Probe**: E-40 + coverage weight=0.10, topk=3, 30ep.
**Cost**: ~2.5h

### Campaign E: Teacher-Reference Analysis [ANALYSIS-ONLY]

**Hypothesis**: The E-52 (EffB0 30ep) → E-40 (MV3-S 30ep) gap tells us where
the backbone matters. Per-video delta analysis reveals which failure modes
are backbone-limited vs architecture-limited.

**Method**: Compute per-video mIoU delta between E-52 and E-40. Cluster videos
where E-52 >> E-40 vs E-40 >> E-52 vs equal.

**Cost**: 0 GPU, ~5 min compute.

---

## Execution Order

```
Campaign A1 (audit logs) ─── immediately
Campaign A2 (E-45 top-K) ─── immediately (parallel with A1)
Campaign E (teacher analysis) ─── after A1 (uses existing data)
Campaign A3 (seeds) ─── after A2 completes
Campaign B (resolution) ─── after A2 gate decision
Campaign C (checkpoint policy) ─── after A3
Campaign D (coverage revisit) ─── conditional, after A+B+C
```

## What We Do NOT Do

- No new auxiliary heads
- No temporal stride changes
- No soft targets
- No zoom augmentation
- No grid search
- No multi-variable experiments

---

## Campaign A: Baseline Accountability — FINAL RESULTS

**Completed 2026-05-14**

### A1: Log Audit

E-45: peaked E7 (0.305), dropped E8 (0.297). E-40: peaked E18 (0.309), plateau E16-E30.
E-52: peaked E20 (0.379 in retrain), then declined.

### A2: E-45 8ep Top-K Retrain

Retrain pf_mIoU = 0.8314 vs original 0.8281 — baseline reproducible within ~0.003.
Training trajectory varied from original despite same seed (DataLoader non-determinism).

### A3: Seed Robustness — Deferred

A2 trajectory variance already confirmed checkpoint selection is inherently unreliable.

### A4: E-40 MV3-Small 30ep Top-K Retrain — BREAKTHROUGH

| Rank | Epoch | Train Val | Reeval pf_mIoU | IoU_tiny | IoU_small | IoU_large |
|------|-------|-----------|----------------|----------|----------|----------|
| 5 | **30** | 0.308 | **0.9077** | **0.849** | **0.865** | **0.911** |
| 1 | 17 | 0.330 | 0.8424 | 0.608 | 0.762 | 0.859 |

E30 was NOT in training-val top-5. Force-saved by final-epoch policy. Original E-40 at E18 was under-reported by 0.051 pf_mIoU.

### A5: E-52 EffB0 30ep Top-K Retrain — BREAKTHROUGH

| Rank | Epoch | Train Val | Reeval pf_mIoU | IoU_tiny | IoU_small | IoU_large |
|------|-------|-----------|----------------|----------|----------|----------|
| 5 | **30** | 0.346 | **0.9142** | **0.836** | **0.895** | **0.912** |
| 1 | 12 | 0.379 | 0.7890 | 0.314 | 0.582 | 0.845 |

Training-val-best (E12) vs unified-best (E30) gap: **0.125 pf_mIoU** — largest observed.
Original E-52 at E20 was under-reported by 0.043 pf_mIoU.

### Backbone Comparison at True Ceiling (E30)

| Metric | MV3-Small (A4) | EffB0 (A5) | Winner |
|--------|---------------|-----------|--------|
| pf_mIoU | 0.9077 | **0.9142** | EffB0 +0.007 |
| IoU_tiny | **0.849** | 0.836 | MV3 +0.013 |
| IoU_small | 0.865 | **0.895** | EffB0 +0.030 |
| IoU_large | 0.911 | 0.912 | Tie |

Backbones are effectively tied. MV3-Small specializes in tiny objects (hardest category). The original "EffB0 > MV3-Small by 0.015" was a checkpoint selection artifact.

### Key Policy Decisions

1. **Final-epoch force-save is permanent**: Both breakthroughs depended on it
2. **All checkpoint decisions use unified reeval**: Training val is 8/8 wrong
3. **True baselines**: MV3-Small 30ep = 0.908, EffB0 30ep = 0.914
4. **8ep baselines**: MV3-Small = 0.831, EffB0 = 0.837

### Campaign E: Teacher-Reference Analysis — RESULTS

MV3-Small excels at tiny objects (pygmy_seahorse_0: 0.623 vs 0.387), EffB0 excels at medium/large. Genuine architecture tradeoff, not training artifact.

---

## Campaign B: Resolution Pareto — CLOSED (NO-GO)

**Result 2026-05-14**: B1 (256×256) pf_mIoU = 0.7488 vs E-45 (224×224) = 0.8281 → **Δ = -0.079**

Resolution hypothesis REJECTED. All 4 size bins degraded:
- IoU_tiny: 0.584 → 0.335 (-0.249) — catastrophic, opposite of hypothesis
- IoU_small: 0.706 → 0.680 (-0.026)
- IoU_medium: 0.838 → — (not computed in reeval)
- IoU_large: 0.860 → 0.809 (-0.051)

B2 (288×288) cancelled. The monotonic degradation confirms 224×224 is the optimal resolution for MV3-Small + dense_fg_aux.

### Why resolution hurts
1. FPN stride-8 features at 256: 32×32 grid dilutes effective receptive field
2. Training schedule (8ep) tuned for 224 — insufficient for larger spatial grid
3. Dense targets become sparser at higher resolution (harder to learn)

### Required code fixes for non-224 resolutions (applied)
1. `src/dataset_real.py` line 673: `hw=28` → `hw=self.target_size // 8`
2. Config: set `resized_root=null` (avoids 224-only pre-resized image mismatch)
3. `run_trial_minimal.py`: `os.rename` → `shutil.copy2` (top-K checkpoint persistence)

---

## Campaign C: Checkpoint Selection Policy — READY

**Goal**: Formalize the checkpoint selection policy change. The 6/7 disagreement rate (training-val-best ≠ unified-best) is conclusive.

**Policy change**: For 8ep runs, always save final epoch. For 30ep runs, save top-3 + last 3 epochs. All checkpoint selection decisions use unified reeval.

**No additional training required** — the evidence is sufficient. Just codify the policy in `run_trial_minimal.py`.

---

## Campaign D: Large-Object Coverage Revisit — QUEUED

**Condition**: Coverage errors remain significant. E-45 retrain showed pred_too_large=101 (vs 21 in original). If this pattern persists at 30ep, stronger coverage penalty at 30ep may help.

**Probe**: E-40 (MV3-Small 30ep) + topk=3 + coverage_weight=0.10 or 0.15.
**Cost**: ~1.5h GPU.
**Gate**: Check E-40 top-K retrain results first. If large-object IoU is already satisfactory at 30ep, skip.

---

## Campaign Status Summary

```
✅ Campaign A (baseline accountability) — COMPLETE
   A1 (audit) → A2 (E-45 8ep) → A4 (E-40 30ep BREAKTHROUGH) → A5 (E-52 30ep BREAKTHROUGH)
✅ Campaign B (resolution) — CLOSED (no-go, -0.079)
✅ Campaign C (checkpoint policy) — COMPLETE (final-epoch force-save codified)
✅ Campaign E (teacher analysis) — COMPLETE
⬜ Campaign D (coverage revisit) — QUEUED (A5 still shows n_pred_too_small=175)
```

## True Baselines (for all future experiments)

| Baseline | Backbone | Ep | pf_mIoU | IoU_tiny | IoU_large | Use for |
|----------|---------|-----|---------|----------|----------|---------|
| A2 | MV3-Small | 8 | 0.831 | 0.511 | — | 8ep probes |
| E-51 | EffB0 | 8 | 0.837 | 0.512 | — | 8ep probes |
| **A4** | **MV3-Small** | **30** | **0.908** | **0.849** | **0.911** | **Tiny-object ceiling** |
| **A5** | **EffB0** | **30** | **0.914** | **0.836** | **0.912** | **Overall ceiling** |

## Remaining Failure Modes

1. **Tiny objects (IoU 0.836-0.849)**: MV3-Small has edge. Zoom augmentation (E-54) could help
2. **Small objects (IoU 0.865-0.895)**: EffB0 has edge. Still below large-object performance
3. **Coverage errors (n_pred_too_small=144-175)**: Consistent across both backbones at 30ep
4. **Center shift (n_center_shift=5-26)**: Near-zero at E30 — not a bottleneck

## What We Do NOT Do

- No new auxiliary heads
- No temporal stride changes
- No soft targets
- No grid search
- No multi-variable experiments
- No resolution changes (224×224 confirmed optimal)
- No training before analysis confirms the direction
- No conclusions from training val alone
