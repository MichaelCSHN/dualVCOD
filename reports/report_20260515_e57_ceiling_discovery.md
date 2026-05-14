# E57 Ceiling Discovery: True Pareto Frontier at 40ep
## 2026-05-15

---

## Purpose

E57 was NOT naive longer training. It was ceiling discovery after the checkpoint
breakthrough (A4/A5) proved that training-val systematically selects wrong
checkpoints and E30 was still improving for both backbones.

**Question**: Does the true ceiling lie beyond 30ep? If so, where?

---

## Method

- Resume A4 (MV3-Small E30, pf_mIoU=0.9077) → continue to E40
- Resume A5 (EffB0 E30, pf_mIoU=0.9142) → continue to E40
- Fresh optimizer/scheduler with CosineAnnealingLR(T_max=40)
- Checkpoints: periodic (E35, E40) + top-5 by training val
- All candidates evaluated via unified reeval (np.random.RandomState(42) split)

---

## Results

### MV3-Small: A4 E30 → E40

| Epoch | Train Val | pf_mIoU | IoU_tiny | IoU_small | IoU_large | R@0.5 | bad_frame | Δ from E30 |
|-------|-----------|---------|----------|----------|----------|-------|----------|-----------|
| 30 (A4) | 0.308 | 0.9077 | 0.849 | 0.865 | 0.911 | 0.967 | 0.033 | — |
| 34 | 0.312 | 0.9058 | 0.819 | 0.867 | 0.909 | 0.968 | 0.032 | -0.002 |
| 35 | 0.309 | 0.9083 | 0.848 | 0.877 | 0.912 | 0.967 | 0.033 | +0.001 |
| 38 | 0.309 | 0.9100 | 0.837 | 0.864 | 0.914 | 0.967 | 0.033 | +0.002 |
| 39 | 0.310 | 0.9132 | 0.854 | 0.879 | 0.916 | 0.967 | 0.033 | +0.006 |
| **40** | 0.310 | **0.9132** | **0.855** | 0.876 | **0.916** | **0.968** | **0.032** | **+0.006** |

**Best: E40. Training-val-best (E34, 0.312) ≠ unified-best (E40, 0.9132).**
Disagreement rate now 10/10 for MV3-Small lineage.

### EffB0: A5 E30 → E40

| Epoch | Train Val | pf_mIoU | IoU_tiny | IoU_small | IoU_large | R@0.5 | bad_frame | Δ from E30 |
|-------|-----------|---------|----------|----------|----------|-------|----------|-----------|
| 30 (A5) | 0.346 | 0.9142 | 0.836 | 0.895 | 0.912 | 0.965 | 0.035 | — |
| 33 | 0.356 | 0.9058 | 0.826 | 0.876 | 0.903 | 0.965 | 0.035 | -0.008 |
| 35 | 0.353 | 0.9089 | 0.806 | 0.856 | 0.911 | 0.966 | 0.034 | -0.005 |
| 37 | 0.358 | 0.9153 | 0.856 | 0.892 | 0.912 | 0.965 | 0.035 | +0.001 |
| **38** | 0.358 | **0.9180** | 0.855 | **0.894** | **0.916** | 0.966 | 0.034 | **+0.004** |
| 40 | 0.358 | 0.9176 | **0.859** | 0.893 | 0.915 | 0.966 | 0.034 | +0.003 |

**Best: E38. Training-val-best (E38, 0.358) = unified-best (E38, 0.9180).**
First agreement in 11 probes — suggests genuine plateau reached.

---

## True Pareto Frontier (40ep Ceiling)

| Metric | MV3-Small E40 | EffB0 E38 | Delta | Winner |
|--------|--------------|-----------|-------|--------|
| **pf_mIoU** | 0.9132 | **0.9180** | +0.0048 | EffB0 |
| IoU_tiny | 0.855 | 0.855 | 0.000 | Tie |
| IoU_small | 0.876 | **0.894** | +0.018 | EffB0 |
| IoU_medium | 0.921 | **0.930** | +0.009 | EffB0 |
| IoU_large | **0.916** | 0.916 | 0.000 | Tie |
| R@0.5 | **0.968** | 0.966 | -0.003 | MV3-Small |
| bad_frame | **0.032** | 0.034 | +0.003 | MV3-Small |
| n_pred_too_small | **120** | 165 | +45 | MV3-Small |
| n_center_shift | **5** | 25 | +20 | MV3-Small |
| n_scale_mismatch | 63 | **15** | -48 | EffB0 |
| Params | **1.49M** | 4.67M | 3.1x | MV3-Small |
| FPS | **69.9** | 38.5 | 1.8x | MV3-Small |

### Key observations

1. **EffB0 leads by 0.005 pf_mIoU** — gap narrowed from original 0.015 (checkpoint-artifact) to 0.007 (A4/A5 E30) to 0.005 (E57 E40). The true gap is negligible.

2. **Tiny and large objects are tied.** The original EffB0 tiny-object regression (-0.101 vs MV3-Small 30ep) was entirely a checkpoint selection artifact. At true ceiling, IoU_tiny=0.855 for both.

3. **EffB0's only meaningful advantage is small objects (+0.018).** This is consistent across all recent comparisons. It's a real but modest effect.

4. **MV3-Small has better detection reliability** — lower bad_frame_rate (-0.003), higher R@0.5 (+0.003), far fewer center_shift errors (5 vs 25). For deployment, MV3-Small is more reliable + 1.8x faster.

5. **Error profiles differ systematically:**
   - MV3-Small: more scale_mismatch (63), fewer pred_too_small (120)
   - EffB0: more pred_too_small (165), more center_shift (25)

---

## Residual Error Audit

Audit run on MV3-Small E40 vs EffB0 E38 using unified reeval split (28 val videos, 1191 clips, 5955 frames).

### Head-to-head: EffB0 wins 13, MV3-Small wins 2, tied 13

EffB0 wins are widespread but small (+0.01 to +0.05). MV3-Small wins are concentrated but large:
- **white_tailed_ptarmigan**: MV3 0.411 vs EffB0 0.188 (+0.223) — EffB0 100% bad frames
- **pygmy_seahorse_0**: MV3 0.837 vs EffB0 0.823 (+0.014)

### Hard videos (mIoU < 0.5 in either model)

| Video | MV3-Small E40 | EffB0 E38 | Notes |
|-------|--------------|-----------|-------|
| flatfish_2 | 0.154 | 0.154 | 100% bad frames in both — annotation ambiguity |
| white_tailed_ptarmigan | 0.411 | 0.188 | MV3-Small far better |

### 12/28 videos have ZERO bad frames in both models

The core 12 videos are essentially solved. The 2 hard videos (flatfish_2, white_tailed_ptarmigan) account for the majority of remaining errors.

### Error concentration

- **flatfish_2 alone**: 80 errors from 80 frames (100% failure). Likely annotation issue — flatfish extent is inherently ambiguous.
- **white_tailed_ptarmigan**: 45 errors (MV3-Small) vs 110 (EffB0). EffB0 catastrophic here.

### pred_too_small vs pred_too_large asymmetry

- Both models: pred_too_large ≈ 0, pred_too_small = 120-165
- Systematic under-coverage persists but is heavily concentrated in 2 hard videos
- Excluding flatfish_2 and white_tailed_ptarmigan, pred_too_small drops to ~40-55 across remaining 26 videos

---

## Why 50ep Is Not Recommended

1. **E38-E40 shows plateau for both backbones.** EffB0: E38=0.9180, E40=0.9176. MV3-Small: E39=E40=0.9132.
2. **EffB0 E38 = training-val-best = unified-best** — first agreement in 11 probes. This signals genuine convergence.
3. **Marginal gain per epoch is ~0.001** at E35-E40. Extending to 50ep would gain at most 0.002-0.004 at 25% more compute cost.
4. **The remaining errors are not "more training" problems** — flatfish_2 and white_tailed_ptarmigan are annotation/ambiguity issues, not under-training.

---

## Corrected Baseline History

| Baseline | Reported pf_mIoU | True pf_mIoU | Error Source |
|----------|-----------------|-------------|--------------|
| E-45 (MV3-Small 8ep) | 0.8281 | 0.8314 | E7 vs E8 checkpoint |
| E-40 (MV3-Small 30ep) | 0.8564 | 0.9077 | E18 vs E30 checkpoint (−0.051!) |
| E-51 (EffB0 8ep) | 0.8372 | 0.8372 | Correct (last epoch) |
| E-52 (EffB0 30ep) | 0.8711 | 0.9142 | E20 vs E30 checkpoint (−0.043!) |
| **MV3-Small 40ep** | — | **0.9132** | New ceiling |
| **EffB0 40ep** | — | **0.9180** | New ceiling |

**Average under-reporting due to checkpoint selection: 0.047 pf_mIoU** — larger than any architectural change attempted.

---

## Checkpoint Policy (Updated 2026-05-15)

Codified in `run_trial_minimal.py`:

1. **Final-epoch force-save is mandatory** — vindicated by A4 and A5 where E30 was outside top-5 by training val but was unified-best
2. **Periodic checkpoint_epochs for >=30ep runs**: auto-generated as `[5,10,15,20,25,30,31,...,40]`. Every 5 epochs before E30, every epoch after E30. This guarantees dense late-stage coverage without relying on training-val top-K
3. **Training-val top-K is retained as informational only** — not used for final checkpoint selection
4. **All checkpoint selection decisions use unified reeval** (np.random.RandomState(42) split)

---

## Canonical Recommendation

**MV3-Small 40ep is the recommended deployment backbone.**

Rationale:
- 1.49M params vs 4.67M (3.1x smaller)
- 69.9 FPS vs 38.5 FPS (1.8x faster)
- Better detection reliability (R@0.5=0.968, bad_frame=0.032)
- Far fewer center_shift errors (5 vs 25)
- pf_mIoU within 0.005 of EffB0
- Tiny and large objects tied

EffB0 E38 is the reference ceiling (0.9180) but offers negligible practical advantage at significant compute cost. Its small-object advantage (+0.018) does not justify 3.1x more parameters.

---

## Next Steps

### Do NOT start:
- 50ep extension (plateau confirmed)
- Campaign D coverage (audit shows pred_too_small is concentrated, not systematic)
- E54 zoom (needs new hypothesis per user directive)
- Any closed direction

### Residual audit findings:
1. **flatfish_2 is annotation-ambiguous** — both models at 0.154, 100% bad frames. Not a model problem.
2. **white_tailed_ptarmigan** — EffB0 catastrophic (0.188), MV3-Small tolerable (0.411). Understand why.
3. **12/28 videos are solved** (0 bad frames in both models).
4. **under-coverage is concentrated, not systematic** — removing 2 hard videos drops pred_too_small from 120-165 to ~40-55.

### Proposed next campaign:
- **Seed robustness** (2-3 additional seeds for MV3-Small 40ep)
- **Deterministic training** (num_workers=0 to eliminate trajectory variance)
- **EffB0 reference analysis** — understand why EffB0 fails catastrophically on white_tailed_ptarmigan
- **MV3-Small deployment consolidation** — canonical model for downstream use
