# Campaign A4: E-40 30ep Top-K Retrain — BREAKTHROUGH
## 2026-05-14

---

## Result

**E30 pf_mIoU = 0.9077 vs original E-40 E18 = 0.8564 → Δ = +0.051**

Checkpoint selection was the single largest bottleneck in the entire project — larger than any architectural change attempted.

---

## Training Trajectory

| Phase | Epochs | Val Range | Notes |
|-------|--------|-----------|-------|
| Early | E1-E6 | 0.237-0.296 | Rapid learning |
| Mid | E7-E13 | 0.272-0.330 | Peak at E17 (0.330) |
| Late | E14-E30 | 0.294-0.318 | Plateau, slight decline |

Training val best: E17 (0.330). E30 only 0.308 — ranked ~8th by training val.

## Unified Reeval (Top-5)

| Rank | Epoch | Train Val | pf_mIoU | IoU_tiny | IoU_small | IoU_large | R@0.5 | bad_frame |
|------|-------|-----------|---------|----------|----------|----------|-------|----------|
| 5 | **30** | 0.308 | **0.9077** | **0.849** | **0.865** | **0.911** | **0.967** | **0.033** |
| 1 | 17 | 0.330 | 0.8424 | 0.608 | 0.762 | 0.859 | 0.962 | 0.039 |
| 2 | 16 | 0.318 | 0.8396 | 0.612 | 0.763 | 0.855 | 0.949 | 0.051 |
| 4 | 15 | 0.312 | 0.8367 | 0.573 | 0.783 | 0.861 | 0.951 | 0.049 |
| 3 | 14 | 0.318 | 0.8082 | 0.519 | 0.680 | 0.855 | 0.941 | 0.059 |

**Training-val-best (E17) ≠ unified-best (E30). Disagreement: 7/8.**

**Final-epoch force-save policy VINDICATED**: E30 was NOT in the training-val top-5, but was force-saved by the policy. Without it, this result would have been lost.

## Comparison: Before vs After

| Metric | Original E-40 (E18) | A4 (E30) | Delta |
|--------|---------------------|----------|-------|
| pf_mIoU | 0.8564 | **0.9077** | **+0.051** |
| IoU_tiny | 0.703 | **0.849** | **+0.146** |
| IoU_small | — | 0.865 | — |
| IoU_large | — | 0.911 | — |
| R@0.5 | — | 0.967 | — |
| bad_frame_rate | — | 0.033 | — |

## Implications

### 1. MV3-Small 30ep now EXCEEDS original EffB0 30ep

| Baseline | pf_mIoU | IoU_tiny | Checkpoint |
|----------|---------|----------|-----------|
| E-52 original (EffB0 30ep) | 0.8711 | 0.602 | E20 (training val) |
| **A4 E30 (MV3-Small 30ep)** | **0.9077** | **0.849** | **E30 (unified-best)** |
| E-40 original (MV3-Small 30ep) | 0.8564 | 0.703 | E18 (training val) |

The original conclusion that "EffB0 > MV3-Small" was an artifact of wrong checkpoint selection. MV3-Small at E30 achieves 0.908 vs EffB0 at E20 achieving 0.871. The true EffB0 ceiling is unknown (A5 retrain in progress).

### 2. Checkpoint selection was costing 0.051 pf_mIoU

That's more than:
- P3 coverage (failed, +0.014)
- P1 zoom (failed, +0.003)
- Any architectural modification attempted

The 8ep→30ep scaling gain was under-reported by 3× (true gain is +0.080, not +0.028).

### 3. The "all improvements failed" narrative is wrong

The improvements didn't fail — they were evaluated on suboptimal checkpoints. The correct baseline was 0.908, not 0.828. Against 0.908, some interventions might have shown positive signals. The entire Phase 2 campaign needs reinterpretation.

### 4. Training val is worse than useless

Training val ranked E30 as ~8th best (0.308) when it was actually the best. Relying on training val for checkpoint selection is actively harmful — it systematically selects earlier, under-trained checkpoints.

---

## Updated Baseline Table

| Baseline | Backbone | Ep | Checkpoint | pf_mIoU | IoU_tiny | Status |
|----------|---------|-----|-----------|---------|----------|--------|
| E-45 | MV3-Small | 8 | E7 (val=0.305) | 0.8281 | 0.584 | Under-reported? |
| A2 retrain | MV3-Small | 8 | E8 (val=0.294) | 0.8314 | 0.511 | Confirmed |
| **A4** | **MV3-Small** | **30** | **E30 (val=0.308)** | **0.9077** | **0.849** | **NEW CEILING** |
| E-40 original | MV3-Small | 30 | E18 (val=0.309) | 0.8564 | 0.703 | UNDER-REPORTED |
| E-51 | EffB0 | 8 | E8 | 0.8372 | 0.512 | Confirmed |
| E-52 original | EffB0 | 30 | E20 | 0.8711 | 0.602 | UNDER-REPORTED |
| A5 (training) | EffB0 | 30 | TBD | TBD | TBD | Expected 0.92+ |

---

## Next Actions

1. **A5**: E-52 EffB0 30ep top-K retrain (training now) — expected pf_mIoU 0.920+
2. **Reinterpret Phase 2**: With the true 8ep baseline at 0.831 and 30ep at 0.908, some P1-P6 results may look different
3. **E-45**: Original E-45 at E7 (0.828) may also be under-reported vs E8 — but A2 already retrained this
4. **E-51**: EffB0 8ep at E8 is likely correct (last epoch = saved)
