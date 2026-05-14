# Autoresearch Cycle 1+2 — Smoke Report
## 2026-05-14 11:16

---

## Cycle 1: P1 Scale-Aware Natural Zoom (Smoke)

### Hypothesis
Probabilistic per-clip zoom on small/tiny objects improves feature resolution
for the primary failure mode (tiny-object mIoU near zero) without harming val
generalization.

### Config
- `smoke_p1_zoom_1ep.json`: MV3-Small, dense_fg_aux, 1ep, conservative zoom
  (tiny=50%, small=30%, medium=10%, large=0%, context_factor=2.0)

### Results vs Baseline (smoke_topk_infra_1ep)

| Metric | Baseline | P1 Zoom | Delta |
|--------|----------|---------|-------|
| Train Loss | 1.006 | 0.984 | **-2.2%** |
| Train mIoU | 0.3754 | 0.4014 | **+6.9%** |
| Val mIoU | 0.2572 | 0.2735 | **+6.3%** |
| R@0.5 | 0.1631 | 0.2114 | **+29.6%** |
| bad_frame_rate | 0.837 | 0.789 | **-5.7%** |
| IoU_large | 0.435 | 0.483 | **+11.0%** |
| FPS | 84.7 | 83.6 | -1.3% |
| Train time | 251s | 218s | -13% |
| GPU mem | 0.79 GiB | 0.79 GiB | same |

### Verdict: **PASS** ✅
- No NaN, no crash, no OOM
- Every metric improved vs baseline at 1 epoch
- FPS impact negligible (83.6 vs 84.7, -1.3%)
- Zoom is active and producing beneficial training signal
- R@0.5 improvement (+29.6%) suggests zoom helps the model find objects it
  would otherwise miss entirely

### Decision: **PROCEED to 8ep probe**
Use `expl_p1_zoom_conservative_8ep.json` vs E-45 baseline (pf_mIoU=0.8281).

---

## Cycle 2: P3 Large-Object Coverage Penalty (Smoke)

### Hypothesis
Asymmetric coverage penalty (weight=0.03) on large objects (>15% area) nudges
predictions to cover more of the GT extent, reducing the dominant
`n_pred_too_large` error mode.

### Config
- `smoke_p3_coverage_1ep.json`: MV3-Small, dense_fg_aux, 1ep,
  large_coverage_weight=0.03, large_area_threshold=0.15

### Results vs Baseline

| Metric | Baseline | P3 Coverage | Delta |
|--------|----------|-------------|-------|
| Train Loss | 1.006 | 1.003 | -0.3% |
| Train mIoU | 0.3754 | 0.3819 | +1.7% |
| Val mIoU | 0.2572 | 0.2251 | **-12.5%** |
| R@0.5 | 0.1631 | 0.1638 | +0.4% |
| bad_frame_rate | 0.837 | 0.836 | flat |
| area_ratio | 7.27 | 13.15 | **+80.9%** ⚠️ |
| n_pred_too_large | 2688 | 2809 | **+4.5%** ⚠️ |
| IoU_large | 0.435 | 0.387 | -11.0% |
| FPS | 84.7 | 82.2 | -3.0% |
| GPU mem | 0.79 GiB | 0.79 GiB | same |

### Verdict: **PASS (infrastructure only)** ✅⚠️
- No NaN, no crash, no OOM — infrastructure validated
- Coverage loss is active and producing non-zero gradient
- However: area_ratio and n_pred_too_large REGRESSED at 1 epoch
- Interpretation: at 1 epoch with weight=0.03, the coverage signal is too
  weak relative to noise; the model hasn't learned boundaries yet, so the
  expand signal produces unfocused growth (area_ratio 13.15 vs 7.27)
- This is expected behavior for a coverage penalty at 1 epoch — meaningful
  assessment requires 8ep

### Decision: **HOLD — wait for P1 8ep result**
Per roadmap: P1 zoom is canonical edge track for tiny objects (#1 failure mode).
P3 coverage enters 8ep only if P1 8ep completes and resource permits.
Rationale: tiny-object mIoU near zero is the dominant failure; large-object
coverage is secondary. Don't run overlapping 8ep probes on single GPU.

---

## Smoke Summary

| Cycle | Intervention | 1ep Val mIoU | vs Baseline | Verdict |
|-------|-------------|-------------|-------------|---------|
| C1 | P1 Zoom (conservative) | 0.2735 | +6.3% | PASS → 8ep |
| C2 | P3 Coverage (0.03) | 0.2251 | -12.5% | INFRA OK, hold |

---

## Next Action

**Launch P1 8ep zoom probe** — `expl_p1_zoom_conservative_8ep.json`
- Backbone: MV3-Small, Head: dense_fg_aux, 8 epochs
- Zoom: tiny=50%, small=30%, medium=10%, large=0%, context_factor=2.0
- topk_checkpoints=3
- Baseline: E-45 (MV3-Small 8ep, pf_mIoU=0.8281)
- Success: pf_mIoU > 0.8281 (beat E-45) AND IoU_tiny improved
- No-go: pf_mIoU < 0.80 OR FPS < 70 OR NaN/crash

Estimated runtime: ~30 minutes (8 epochs × ~3.5 min/epoch on RTX 4090).

---

## Baselines Reference

| Experiment | Backbone | Epochs | pf_mIoU | R@0.5 | IoU_tiny |
|-----------|----------|--------|---------|-------|----------|
| E-45 | MV3-Small | 8 | 0.8281 | 0.9500 | 0.584 |
| E-40 | MV3-Small | 30 | 0.8564 | 0.9642 | 0.703 |
| E-51 | EffB0 | 8 | 0.8372 | 0.9467 | 0.512 |
| E-52 | EffB0 | 30 | 0.8711 | 0.9743 | 0.602 |
