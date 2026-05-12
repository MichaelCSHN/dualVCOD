# Phase 1.6-R Autonomous Controlled AutoResearch Redesign

**Generated:** 2026-05-07 22:20 CST
**Status:** Design only — no training executed, no commit, no push
**Predecessor:** Phase 1.6-B0/B1 (archived as partial smoke evidence)

---

## 0. CUDA Sanity Check (Post-Reboot)

| Check | Result |
|---|---|
| nvidia-smi | RTX 4090, 24 GB, Driver 591.86, CUDA 13.1 |
| GPU Memory | 0 MiB / 24564 MiB |
| GPU Temp | 38°C, P8 idle |
| Processes | None |
| torch.cuda.is_available() | True |
| torch memory allocated | 0 |
| torch memory reserved | 0 |
| Residual Python training processes | None |

**Verdict: Environment clean and ready. No stale GPU state.**

---

## 1. Why Not Resume the Unfinished B1 (effb1)

### 1.1 What B1 Completed

| Trial | Backbone | Epochs | Best mIoU | Status |
|---|---|---|---|---|
| smoke_b1_effb0 | efficientnet_b0, 512px, lr=1e-3 | 5 (NaN @ ep3-5) | 0.1708 (ep1) | Hard reject — NaN |
| smoke_b1_mnv3large | mobilenet_v3_large, 512px, lr=1e-3 | 5 | 0.2594 (ep4) | Failed gate (-0.0067), overfitting |
| smoke_b1_effb1 | efficientnet_b1, 512px, lr=3e-4 | 0 | N/A | GPU lost / infrastructure failure |

### 1.2 Why Not Resume

1. **The old B1/B2 gate framework is architecturally wrong for the task.** It was designed as a linear promotion pipeline (B0→B1→B2→C→D→E→F) — a sequential gating structure appropriate for validating a pre-defined search space, not for discovering surprise improvements. The user's goal has shifted from "validate these 6 variables" to "find the best GreenVCOD within deployable budgets."

2. **The effb1 config is a child of the old framework.** Its lr=3e-4, sz=512, T=5 were chosen defensively within a 6-variable prison. Resuming it would produce one more data point in a space we are now abandoning.

3. **The GPU crash created a clean break point.** Rather than mechanically completing a stale plan, use the restart to redesign the search.

4. **The partial findings are sufficient as priors** (see Section 2). They tell us what we need to know about lr sensitivity, EfficientNet stability, and overfitting risks at 512px. Running effb1 at 3e-4 would not add decisive new information.

### 1.3 What Happens to B1 Artifacts

- `local_runs/autoresearch/smoke_b1_effb0/` — preserved as evidence of NaN/gradient explosion
- `local_runs/autoresearch/smoke_b1_mnv3large/` — preserved as near-threshold evidence
- `local_runs/autoresearch/smoke_b1_effb1/` — config exists, trial never ran; mark as infrastructure failure
- All checkpoints preserved, not deleted or moved

---

## 2. B1 Partial Findings as Priors

### 2.1 Finding 1: EfficientNet + lr=1e-3 = Numerical Instability (effb0)

- effb0 at 512px, lr=1e-3 produced NaN at epoch 3 despite grad_clip=2.0
- Train loss: 1.134 → 1.136 → NaN → NaN → NaN
- This is **not** a model quality failure — it's a numerical stability failure
- **Prior for Phase 1.6-R:** EfficientNet backbones require lower LR, warmup, or AMP diagnostic. Do not pair EN with lr≥1e-3 without stability safeguards.

### 2.2 Finding 2: MV3-Large Near-Threshold with Severe Overfitting (mv3large)

- Best mIoU=0.2594 at epoch 4, then dropped to 0.2462 at epoch 5
- Missed the old B1 gate (0.2661) by only 0.0067
- Train/val gap: 0.4994 - 0.2594 = 0.2400 (severe overfitting)
- Global area ratio: 1.17 (reasonable) but mean_sample_area_ratio: 12.22 (extreme skew)
- **Prior for Phase 1.6-R:** MV3-Large has potential (near-threshold at epoch 4) but the 512px + lr=1e-3 + 5-epoch window may not be the right setting. Longer training + lower LR + warmup could allow recovery. This is a `late_bloomer_candidate`.

### 2.3 Finding 3: Clean Baseline Shows Late Recovery (train.py 30-epoch)

- Epoch 5: mIoU=0.2437 (worse than epoch 3's 0.2674)
- Epoch 12: mIoU=0.2814 — recovery begins
- Epoch 16: mIoU=0.2862 — first peak
- Epoch 25: mIoU=0.2862 — second peak
- **Critical prior:** The plateau→recovery pattern means 3-5 epoch smoke CANNOT judge a model's potential. Multi-fidelity evaluation with explicit recovery probes is mandatory.

### 2.4 Finding 4: GPU Lost Not a Model Quality Signal (effb1)

- effb1 OOM during `.to(DEVICE)` with only 7.1M params — this was GPU memory fragmentation from prior runs, not the model
- **Prior for Phase 1.6-R:** Infrastructure failures must be distinguished from model failures. effb1 gets no model quality conclusion from this event.

### 2.5 Summary of Priors

| Prior | Source | Implication |
|---|---|---|
| EN + lr≥1e-3 → NaN risk | effb0 | Lower LR, warmup, AMP diagnostic for EfficientNet |
| MV3-Large near-threshold | mv3large | Late bloomer candidate, needs longer training |
| Plateau→recovery at epoch 12+ | Clean baseline | 3-5 epoch smoke insufficient for go/no-go |
| 512px increases overfitting | mv3large | Higher resolution needs stronger regularization |
| GPU fragmentation possible | effb1 | Always verify GPU state before large model init |
| Clean baseline mIoU=0.2861 is lower bound | train.py 30ep | This is a reference, not a ceiling |

---

## 3. The New Search Universe

### 3.1 A. Backbone / Capacity

| Tier | Param Range | Candidates | Purpose |
|---|---|---|---|
| Nano | 1–5M | mobilenet_v3_small, mobilenet_v3_large, shufflenet_v2_x1_5 | Deployment-constrained baseline |
| Light | 5–15M | efficientnet_b0, efficientnet_b1, resnet18, regnet_y_400mf, fastvit_t8 | Primary exploration tier |
| Compact | 15–35M | efficientnet_b2, efficientnet_b3, resnet34, regnet_y_800mf, convnext_nano | Higher-capacity exploration |
| Upper Diagnostic | 35–80M | convnext_tiny, efficientnet_b4, resnet50 | Upper-bound only, not deployable |

**Rule:** 1.4M is a lower-bound reference, NOT a target. Do not exclude backbones solely because they exceed 1.4M params.

### 3.2 B. Input Resolution

| Resolution | Tier | GPU Budget Multiplier vs 224 |
|---|---|---|
| 224 | Baseline | 1× |
| 320 | Light | ~2× |
| 384 | Light | ~3× |
| 448 | Compact | ~4× |
| 512 | Compact | ~5× |
| 576 | Compact | ~6.5× |
| 640 | Heavy | ~8× |
| 672 | Heavy | ~9× |
| 768 | Diagnostic only | ~12× |

Excluded: <224 (too small), >768 (diagnostic only).

### 3.3 C. Temporal Setting

| Parameter | Values |
|---|---|
| T (frames) | 1, 3, 5, 7, 9 |
| Stride | 1, 2, 3 |
| Stride jitter | [1,2], [1,3], [1,4] |
| Pooling | mean, max, learned attention, depthwise 1D conv, gated fusion, light cross-frame attention |

Primary eval T = training T. Variable-T diagnostic run separately.

### 3.4 D. Head / Supervision

| Head Type | Training Only? | Deployment |
|---|---|---|
| Current direct bbox | No | Bbox-only |
| Center + WH head | No | Bbox-only |
| Objectness auxiliary | Yes (aux loss) | Bbox-only |
| Dense objectness map | Yes (aux loss) | Bbox-only |
| Mask-like auxiliary | Yes (aux loss) | Bbox-only |
| Bbox from objectness map | Yes | Bbox-only |
| Hybrid dense-to-box | Yes | Bbox-only |

All heads must produce bbox-only output at deployment. Training can use heavier supervision.

### 3.5 E. Loss Functions

| Loss | Weight Range | Purpose |
|---|---|---|
| L1 / SmoothL1 | 0.5–5 | Primary bbox regression |
| GIoU / DIoU / CIoU | 0–2 | IoU-aware bbox loss |
| Center loss | 0–1 | Center point accuracy |
| Size / log-WH loss | 0–1 | Scale-aware size regression |
| Area-ratio penalty | 0–0.5 | Prevent systematic size bias |
| Objectness BCE / Focal | 0.05–1 | Foreground/background discrimination |
| Temporal consistency | 0–0.5 | Frame-to-frame smoothness |

### 3.6 F. Optimizer / Schedule

| Parameter | Values |
|---|---|
| LR | 1e-4, 3e-4, 6e-4, 1e-3, 2e-3 |
| Warmup epochs | 0, 2, 5 |
| Scheduler | cosine, cosine+warmup, onecycle |
| Weight decay | 1e-5, 1e-4, 3e-4, 1e-3 |
| Grad clip | 0.5, 1, 2, 5 |
| EMA | on/off |
| Freeze backbone | 0, 3, 5 epochs |
| AMP | on/off (diagnostic if NaN appears) |

### 3.7 G. Sampling / Source Mixture

| Sampler | Description |
|---|---|
| window_uniform | Current: large videos dominate |
| video_balanced | Each video ~equal windows per epoch |
| source_balanced | MoCA / MoCA_Mask / CAD weighted equally |
| dedup_by_canonical_id | Remove duplicate windows across sources |

Source mixture ratios must be recorded per trial.

### 3.8 H. Augmentation

| Augmentation | Purpose | Caution |
|---|---|---|
| Color jitter (sweep strength) | Generalization | Camouflage cues may be color/texture-based |
| Brightness/contrast/saturation/hue | Illumination invariance | May destroy camouflage signal |
| Blur | Motion robustness | |
| Noise | Sensor noise robustness | |
| Horizontal flip | Spatial invariance | Safe |
| Scale jitter | Size invariance | |
| Mild crop | Position invariance | |
| Temporal dropout | Temporal robustness | Drop random frames |
| Temporal jitter | Frame timing robustness | |

**Key insight:** Camouflage tasks use color and texture as both cues AND overfitting sources. Do not default to "more augmentation = better."

### 3.9 I. Runtime / Deployment Recording

Every candidate must record: params, FLOPs/MACs (if computable), FPS, peak GPU memory, input size, T, ONNX/TensorRT feasibility (if quick test possible), bbox-only deployment feasibility.

---

## 4. Physical Laws (Absolute Invariants)

These are the 15 physical laws. They cannot be violated.

| # | Law | Rationale |
|---|---|---|
| 1 | Clean train/val isolation must not be changed | Foundation of all metric validity |
| 2 | No MoCA val or same-source val videos in training | Data leak invalidates all results |
| 3 | canonical_video_id filtering must remain enabled | Cross-dataset leakage prevention |
| 4 | Never load verified_candidate_baseline.pth | Pre-leak-fix checkpoint, permanently invalid |
| 5 | Never use legacy/non-clean checkpoint for init, distillation, or teacher | Contamination risk |
| 6 | Never modify evaluator semantics and claim metrics are comparable | Metric integrity |
| 7 | Never substitute diagnostic metrics for primary metrics | Ranking validity |
| 8 | All trials must record complete metadata | Audit trail |
| 9 | All failed trials must record failure reason | Negative results are data |
| 10 | No automatic git commit | Human oversight on repo changes |
| 11 | No automatic git push | Human oversight on remote changes |
| 12 | Never commit checkpoints, local_runs, outputs, logs, raw data | Repo hygiene |
| 13 | Never delete, move, or overwrite any existing checkpoint | Data preservation |
| 14 | Report all trials — successes, failures, and anomalies | Scientific integrity |
| 15 | All results must be reproducible and auditable | Trust |

These are the only hard constraints. Everything else is explorable.

---

## 5. Multi-Fidelity Reward System

The core problem: full 30-epoch evaluation is too expensive for exploration (>2 GPU-hours per trial). 3-5 epoch mIoU is unreliable (clean baseline hit plateau at epoch 5, recovered at epoch 12). Solution: multi-fidelity evaluation.

### Level 0: Zero-Train Static Screen

**Cost:** seconds
**GPU:** dummy forward only

Checks:
- Param count within declared tier
- FLOPs/MACs estimation
- Dummy forward produces valid bbox shape
- One-batch finite loss
- Peak memory estimate from dummy pass
- model.to(cuda) succeeds

**Purpose:** Birth eligibility only. Does not judge performance.

### Level 1: 1-Epoch Viability Probe

**Cost:** ~4 minutes
**GPU:** 1 epoch training + val

Records:
- Loss decreases? (binary)
- NaN / OOM? (binary)
- Empty prediction rate
- global_area_ratio
- mean_sample_area_ratio
- Val mIoU > zero / random / center prior? (binary)
- R@0.3 has signal? (binary)
- Grad norm distribution
- AMP stability

**Purpose:** Exclude physically dead configs. Does NOT judge quality.

### Level 2: 3–5 Epoch Learning Dynamics Probe

**Cost:** ~12-20 minutes
**GPU:** 3-5 epoch training + val

Records:
- best_mIoU_5
- AUC_mIoU_1_to_5 (area under mIoU curve)
- slope_last3 (linear fit to epochs 3-5)
- recovery_potential (slope_last3 / best_mIoU_5)
- train_loss_drop (loss[1] - loss[5]) / loss[1]
- train-val gap at best epoch
- R@0.3, R@0.5 at best epoch
- global_area_ratio, mean_sample_area_ratio
- Empty rate
- FPS, GPU memory

**Critical rule:** If mIoU at epoch 5 is modest BUT slope_last3 > 0, loss is dropping, and area ratio is reasonable, mark as `late_bloomer_candidate` — do NOT kill.

### Level 3: 12-Epoch Recovery Probe

**Cost:** ~50-80 minutes
**GPU:** 12 epoch training + val

This is the first meaningful promotion gate. Clean baseline showed recovery starting at epoch 12.

Records:
- score_12 (composite at epoch 12)
- Δ from epoch 5 to epoch 12
- last3 stability (std of epochs 10-12)
- val mIoU, R@0.5, R@0.3 at epoch 12
- Size calibration (area ratios)
- no_response / error type distribution

**Promotion threshold:** mIoU_12 > baseline_mIoU - 0.02 AND recovery evidence.

### Level 4: 30-Epoch Confirmation

**Cost:** ~2-2.5 hours
**GPU:** Full 30-epoch training

Only for top candidates that passed Level 3.

Requirements: Clean re-eval, controls (center prior, shuffled, random), error analysis, FPS/params/memory confirmation, deployment feasibility check.

### Level 5: Multi-Seed / Held-Out Confirmation

**Cost:** ~2-2.5 hours per seed
**GPU:** Full 30-epoch with different seeds

Only for finalists. Not exploration — this is evidence generation.

### Cost Reduction Summary

| Old Approach | New Approach | Savings |
|---|---|---|
| 3-5 epoch smoke decides go/no-go | Level 1 viability → Level 2 dynamics → Level 3 recovery | Early kill of dead configs, promote only promising ones |
| 24 trials at 5 epochs each | Most trials stop at Level 1 or 2 | ~70-90% GPU savings on dead configs |
| Single metric (mIoU) at fixed epoch | Multi-fidelity trajectory analysis | Catches late bloomers, kills early losers |
| Linear promotion pipeline | Branching explorer/exploiter | Parallel exploration, focused exploitation |

---

## 6. Research Value Objective Function

The goal is NOT to maximize mIoU. The goal is to maximize research value.

```
research_value = expected_improvement + novelty_bonus + information_gain - compute_cost - risk_penalty
```

### Component Definitions

**expected_improvement** (0–1 scale):
- Estimated mIoU gain over clean baseline (0.2861), normalized
- Informed by backbone capacity, resolution, temporal setting
- 0 = no expected gain, 1 = breakthrough expected

**novelty_bonus** (0–0.3):
- Explores new structure, mechanism, or combination not previously tested
- 0 = pure exploitation of known-good config
- 0.3 = entirely novel head/loss/temporal architecture

**information_gain** (0–0.4):
- Answers a key question even if performance is not top-tier
- Examples: "Does T=1 work at all?" → high info gain
- "What happens at 768px?" → moderate info gain
- 0 = confirmatory only, no new question answered

**compute_cost** (0–0.5):
- Estimated GPU-hours normalized against budget
- 0 = cheap (Level 0/1), 0.5 = expensive (Level 4/5)

**risk_penalty** (0–0.3):
- Protocol risk (data leak potential, metric incomparability)
- Implementation complexity
- Reproducibility risk
- 0 = safe, 0.3 = high risk

### Expected Information Gain per Trial

Every trial proposal MUST state its expected information gain explicitly. Examples:

| Trial Concept | Expected Information Gain |
|---|---|
| T=1 image-mode diagnostic | Answers: "Is temporal information necessary at all?" |
| 640px high-res probe | Answers: "Does resolution improve small-object localization?" |
| Objectness auxiliary head | Answers: "Can auxiliary supervision reduce no_response errors?" |
| Freeze backbone warmup | Answers: "Does backbone warmup improve late recovery?" |
| efficientnet_b2 at 384px | Answers: "What is the capacity ceiling for EfficientNet family?" |
| Cosine+warmup vs onecycle | Answers: "Does LR schedule affect plateau→recovery timing?" |

---

## 7. Death Conditions

### 7.1 Physical Death (Immediate Stop)

- Data leak detected (canonical_video_id overlap > 0)
- Legacy checkpoint loaded (verified_candidate_baseline.pth or any pre-leak-fix)
- Repeated NaN loss (≥2 occurrences after one LR reduction fallback)
- Repeated OOM after one fallback (batch_size halved, resolution reduced)
- Evaluator broken (produces invalid metrics or crashes)
- Oracle GT sanity fails (GT bboxes produce mIoU < 0.95)
- Metadata missing (trial started without recording config)
- Model output invalid bbox > 10% of predictions
- CUDA/GPU lost (hardware failure, not model failure)

### 7.2 Biological Death (Alive but Cannot Be Promoted)

- After 12 epochs: mIoU < center_prior + 0.03 (0.2317) — model barely beating spatial prior
- Empty prediction rate > 5% — model not engaging
- global_area_ratio < 0.2 or > 5.0 — extreme systematic bias
- Train loss not decreasing (slope_last3_loss ≥ 0)
- R@0.3 < 0.1 and R@0.5 < 0.05 after 12 epochs — complete recall failure
- Train-val gap > 0.40 with zero recovery trend — severe overfitting without hope

**Biological death does NOT delete the trial.** Results are preserved for analysis.

### 7.3 Ecological Death (Wrong Niche)

- Performance is competitive but params/FPS/memory cost is 3×+ higher than a similarly-performing smaller model
- Model does not fit its declared deployment tier (e.g., claims "light" but needs 35M params)
- Cannot export to ONNX/TensorRT or deployment risk too high
- Can serve as upper diagnostic but NOT as deployable candidate

---

## 8. Explorer / Exploiter Dual Loop

### 8.1 Explorer Loop

**Purpose:** Discover surprise improvements and generate information.

**Rules per round:**
- Propose 8–16 trials per round
- ≥30% novelty trials (new mechanisms, surprising combinations)
- ≥30% exploitation trials (refine known-good directions)
- ≥20% diagnostic trials (answer specific questions)
- ≥20% cheap probes (Level 0/1 only)
- Every trial must have an explicit hypothesis
- Conservative designs allowed but not required
- Goal: information gain and surprise discovery

### 8.2 Exploiter Loop

**Purpose:** Confirm and refine Explorer discoveries.

**Actions:**
- Take high-potential Explorer findings to Level 3 (12-epoch recovery probe)
- Take Level 3 passers to Level 4 (30-epoch confirmation)
- Take Level 4 passers to Level 5 (multi-seed confirmation)
- Maintain Pareto frontier across (mIoU, params, FPS, memory)
- Cross-trial analysis: what works, what doesn't, why

### 8.3 Interaction

```
Explorer (Levels 0-2) ── discovers ──▶ Exploiter (Levels 3-5)
                                          │
          ◀────────── feeds back ─────────┘
          (what worked, what didn't, new hypotheses)
```

Each Explorer round informs the next. Exploiter results narrow the search or open new directions.

---

## 9. First 12 Explorer Trial Proposals

### Trial E-01: Lower LR + Warmup Late Recovery

| Field | Value |
|---|---|
| trial_id | `expl_01_lowlr_warmup_mv3small` |
| category | exploitation |
| hypothesis | Clean baseline plateau→recovery suggests LR is decaying too fast. Lower LR + warmup allows smoother convergence and earlier recovery. |
| changed | lr=3e-4, warmup=5, scheduler=cosine+warmup |
| fixed | backbone=mobilenet_v3_small, sz=224, T=5, sampler=window_uniform, head=current_direct_bbox |
| expected mechanism | Warmup prevents early overfitting; lower LR allows longer productive training |
| level | 2 (5-epoch dynamics probe) |
| GPU cost | ~15 min |
| info gain | Answers: "Is the plateau→recovery pattern LR-dependent?" |
| risk | Low — same backbone as baseline, only LR/schedule changed |
| fallback | If NaN: disable AMP, halve LR to 1.5e-4 |
| promotion condition | slope_last3 > 0 AND mIoU trajectory above baseline at same epochs |
| death condition | Biological: loss not decreasing |
| targets | late recovery, stability |

### Trial E-02: MobileNetV3-Large Extended Recovery

| Field | Value |
|---|---|
| trial_id | `expl_02_mnv3large_recovery` |
| category | exploitation |
| hypothesis | B1 mv3large at 512px/lr=1e-3 overfit severely. At 224-320px with lr=3e-4 + warmup, the same backbone may avoid overfitting and show recovery. |
| changed | backbone=mobilenet_v3_large, sz=320, lr=3e-4, warmup=5 |
| fixed | T=5, sampler=window_uniform, head=current_direct_bbox |
| expected mechanism | Moderate resolution reduces overfitting surface; warmup + lower LR stabilize training |
| level | 2 (5-epoch dynamics probe) |
| GPU cost | ~20 min |
| info gain | Answers: "Can MV3-Large perform better than baseline with better optimization?" |
| risk | Medium — B1 version overfit; this is a do-over with different settings |
| fallback | If overfitting persists: sz=224 |
| promotion condition | mIoU_5 > baseline mIoU_5 (0.2437) AND slope_last3 > 0 |
| death condition | Physical: NaN; Biological: train-val gap > 0.35 with no recovery trend |
| targets | late recovery, deployment efficiency (3.5M params) |

### Trial E-03: EfficientNet-B0 Stability Diagnostic

| Field | Value |
|---|---|
| trial_id | `expl_03_effb0_stable` |
| category | diagnostic |
| hypothesis | B1 effb0 produced NaN at lr=1e-3. Testing with lr=1e-4 + AMP off (FP32) isolates whether the instability is LR, AMP, or backbone architecture. |
| changed | backbone=efficientnet_b0, sz=320, lr=1e-4, AMP=off, warmup=5 |
| fixed | T=5, sampler=window_uniform, head=current_direct_bbox |
| expected mechanism | Conservative LR + FP32 eliminates numerical instability; if still NaN → EfficientNet architecture incompatible with current loss/head |
| level | 1 (1-epoch viability probe, then extend to Level 2 if stable) |
| GPU cost | ~5-20 min |
| info gain | Answers: "Is EfficientNet instability fixable, or should EN family be excluded?" |
| risk | Low — designed to be maximally stable |
| fallback | If NaN at 1e-4: mark EN family as high-risk, deprioritize |
| promotion condition | No NaN, loss decreasing after epoch 1 |
| death condition | Physical: NaN at lr=1e-4 |
| targets | stability diagnostic |

### Trial E-04: T=1 Image-Mode Diagnostic

| Field | Value |
|---|---|
| trial_id | `expl_04_t1_imagemode` |
| category | diagnostic |
| hypothesis | If temporal information is necessary for the task, T=1 should perform near or below center prior (0.2017). If T=1 performs well, the temporal module is not the primary performance driver and spatial features dominate. |
| changed | T=1 (training + eval), temporal_neck bypassed (identity) |
| fixed | backbone=mobilenet_v3_small, sz=224, lr=1e-3, sampler=window_uniform, head=current_direct_bbox |
| expected mechanism | Without temporal context, model must rely on single-frame spatial features — a pure image-mode baseline |
| level | 2 (5-epoch dynamics probe) |
| GPU cost | ~10 min |
| info gain | Answers: "What is the spatial-only performance ceiling? Is temporal gain real?" |
| risk | Low |
| fallback | N/A — simple config |
| promotion condition | N/A — diagnostic, not promoted |
| death condition | Biological only: irrelevant as diagnostic |
| targets | temporal gain quantification, lower-bound establishment |

### Trial E-05: T=3 Temporal Efficiency

| Field | Value |
|---|---|
| trial_id | `expl_05_t3_efficiency` |
| category | exploitation |
| hypothesis | T=3 with stride=2 provides similar temporal context at 60% of the compute cost of T=5. If performance is close, T=3 is the more deployable option. |
| changed | T=3, stride=2 |
| fixed | backbone=mobilenet_v3_small, sz=224, lr=1e-3, sampler=window_uniform, head=current_direct_bbox |
| expected mechanism | 3 frames with stride 2 span similar physical time as 5 frames stride 1, but with lower compute |
| level | 2 (5-epoch dynamics probe) |
| GPU cost | ~8 min |
| info gain | Answers: "Can T=3 match T=5 performance at lower cost?" |
| risk | Low |
| fallback | If performance drops > 20% vs baseline: confirm T=5 is necessary |
| promotion condition | mIoU_5 within 0.02 of baseline T=5 mIoU_5 |
| death condition | Biological: mIoU < 0.18 after 5 epochs |
| targets | deployment efficiency, temporal efficiency |

### Trial E-06: High-Resolution 640px Localization

| Field | Value |
|---|---|
| trial_id | `expl_06_highres640` |
| category | exploitation |
| hypothesis | Small camouflaged objects may benefit from higher resolution. 640px provides 8× more pixels than 224px, potentially improving bbox precision for small targets. |
| changed | sz=640, lr=3e-4 (compensate for larger feature maps), batch_size=8 (GPU memory) |
| fixed | backbone=mobilenet_v3_small, T=5, sampler=window_uniform, head=current_direct_bbox |
| expected mechanism | Higher resolution preserves fine spatial details needed for precise bbox regression |
| level | 2 (5-epoch dynamics probe) |
| GPU cost | ~35 min |
| info gain | Answers: "Does resolution improve localization of small camouflaged objects?" |
| risk | Medium — higher GPU memory, slower training; may overfit on texture |
| fallback | If OOM: sz=512, batch_size=6 |
| promotion condition | mIoU_5 > baseline mIoU_5 AND improved R@0.5 (better localization) |
| death condition | Physical: OOM at 512px too; Biological: extreme overfitting |
| targets | box_too_small, no_response, localization precision |

### Trial E-07: Objectness Auxiliary Head

| Field | Value |
|---|---|
| trial_id | `expl_07_objectness_aux` |
| category | novelty |
| hypothesis | A parallel objectness head predicting "is there an object here?" provides auxiliary supervision that may reduce no_response errors and improve R@0.3/R@0.5. |
| changed | head=objectness_aux_head (bbox head + parallel objectness MLP with BCE loss, weight=0.1) |
| fixed | backbone=mobilenet_v3_small, sz=224, T=5, lr=1e-3, sampler=window_uniform |
| expected mechanism | Objectness signal helps the backbone learn foreground/background discrimination, reducing cases where model predicts empty or all-background |
| level | 2 (5-epoch dynamics probe) |
| GPU cost | ~15 min |
| info gain | Answers: "Can auxiliary objectness supervision reduce no_response errors?" |
| risk | Medium — new head implementation; may interfere with bbox regression if weight too high |
| fallback | If bbox loss diverges: reduce objectness weight to 0.05 |
| promotion condition | R@0.3 improvement > 0.03 over baseline at same epoch |
| death condition | Biological: bbox mIoU drops > 0.05 vs baseline at same epoch |
| targets | no_response, recall improvement |

### Trial E-08: Video-Balanced Sampler

| Field | Value |
|---|---|
| trial_id | `expl_08_videobal_sampler` |
| category | exploitation |
| hypothesis | window_uniform sampler lets large videos (many windows) dominate training. video_balanced ensures each video contributes equally, potentially improving generalization to rare/short videos. |
| changed | sampler=video_balanced |
| fixed | backbone=mobilenet_v3_small, sz=224, T=5, lr=1e-3, head=current_direct_bbox |
| expected mechanism | Balanced sampling prevents large-video overfitting and improves performance on under-represented videos |
| level | 2 (5-epoch dynamics probe) |
| GPU cost | ~15 min |
| info gain | Answers: "Does balanced sampling improve validation mIoU by reducing large-video bias?" |
| risk | Low — sampler change only |
| fallback | If performance drops: sampler is not the bottleneck |
| promotion condition | mIoU_5 > baseline mIoU_5 OR reduced train-val gap |
| death condition | Biological: mIoU significantly below baseline |
| targets | generalization, overfitting reduction |

### Trial E-09: Freeze Backbone Warmup

| Field | Value |
|---|---|
| trial_id | `expl_09_freeze_warmup` |
| category | exploitation |
| hypothesis | Freezing the pretrained backbone for the first 3-5 epochs lets the temporal neck and bbox head adapt before fine-tuning the backbone. This may reduce early overfitting and improve late recovery. |
| changed | freeze_backbone=5 (epochs 1-5 frozen, unfreeze at epoch 6), lr=1e-3, warmup=0 (unfreeze acts as implicit warmup) |
| fixed | backbone=mobilenet_v3_small, sz=224, T=5, sampler=window_uniform, head=current_direct_bbox |
| expected mechanism | Backbone features are stable during early training; head and temporal neck learn first, then backbone fine-tunes |
| level | 2 (5-epoch dynamics probe, but needs >5 epochs to see unfreeze effect) → promote to Level 3 if dynamics look promising |
| GPU cost | ~15 min (Level 2), ~60 min (Level 3) |
| info gain | Answers: "Does staged training (head first, then backbone) improve convergence?" |
| risk | Medium — 5 epochs may be insufficient to see the unfreeze benefit |
| fallback | If head-only loss plateaus before epoch 5: unfreeze earlier (epoch 3) |
| promotion condition | Training dynamics show head convergence in frozen phase + backbone adaptation after unfreeze |
| death condition | Biological: no learning in frozen phase |
| targets | late recovery, training stability |

### Trial E-10: ConvNeXt Nano Upper Diagnostic

| Field | Value |
|---|---|
| trial_id | `expl_10_convnext_nano` |
| category | novelty |
| hypothesis | ConvNeXt Nano (~15M params) uses a modern CNN design with depthwise convs and inverted bottlenecks. It may provide better feature quality than MobileNet/EfficientNet families at similar param count. |
| changed | backbone=convnext_nano (needs registry entry), sz=320, lr=3e-4, warmup=2 |
| fixed | T=5, sampler=window_uniform, head=current_direct_bbox |
| expected mechanism | ConvNeXt's modern design (LayerNorm, GELU, larger kernels) may capture better spatial features for camouflage |
| level | 1 → 2 (viability probe first, extend if stable) |
| GPU cost | ~5-20 min |
| info gain | Answers: "Do modern CNN architectures outperform MobileNet/EfficientNet for this task?" |
| risk | Medium-High — new backbone, untested with current FPN/temporal setup; registry implementation needed |
| fallback | If registry entry fails or output shape mismatch: mark as implementation-deferred |
| promotion condition | Level 1 passes AND loss decreases |
| death condition | Physical: NaN/OOM; Ecological: >20M params if performance ≤ baseline |
| targets | upper-bound discovery, architectural novelty |

### Trial E-11: ShuffleNet-V2 Deployment Efficiency

| Field | Value |
|---|---|
| trial_id | `expl_11_shufflenet_v2` |
| category | exploitation |
| hypothesis | ShuffleNet-V2 x1.5 (~3.5M params) with channel shuffle operations may provide competitive performance at very low compute cost, targeting the most deployment-constrained tier. |
| changed | backbone=shufflenet_v2_x1_5 (needs registry entry), sz=224, lr=1e-3 |
| fixed | T=5, sampler=window_uniform, head=current_direct_bbox |
| expected mechanism | Channel shuffle provides efficient feature mixing at low param count |
| level | 2 (5-epoch dynamics probe) |
| GPU cost | ~12 min |
| info gain | Answers: "What is the Pareto frontier for ultra-light models?" |
| risk | Medium — needs registry entry; channel shuffle may interact oddly with FPN |
| fallback | If registry fails: skip, note as implementation cost |
| promotion condition | mIoU_5 within 0.03 of baseline at 1/3 the params |
| death condition | Biological: mIoU < 0.18 |
| targets | deployment efficiency, Pareto frontier |

### Trial E-12: Surprising Combination — GIoU + Center Loss + Warmup

| Field | Value |
|---|---|
| trial_id | `expl_12_giou_center_warmup` |
| category | novelty |
| hypothesis | The baseline uses pure L1 loss. Adding GIoU (IoU-aware) + center loss (position-aware) may provide better gradient signal for bbox regression, especially for small objects where L1 is insensitive to small shifts. Combined with warmup, this could yield smoother convergence. |
| changed | loss=L1(weight=2) + GIoU(weight=1) + center_loss(weight=0.5), warmup=3, lr=1e-3 |
| fixed | backbone=mobilenet_v3_small, sz=224, T=5, sampler=window_uniform, head=current_direct_bbox |
| expected mechanism | Multi-component loss provides richer gradient: L1 for global fit, GIoU for overlap quality, center for position accuracy |
| level | 2 (5-epoch dynamics probe) |
| GPU cost | ~15 min |
| info gain | Answers: "Can a richer loss function improve bbox quality without changing architecture?" |
| risk | Medium — multi-loss balancing; may need weight tuning |
| fallback | If one loss component dominates: adjust weights or simplify to L1+GIoU only |
| promotion condition | mIoU_5 > baseline mIoU_5 OR R@0.5 improvement |
| death condition | Biological: loss not converging |
| targets | box_too_small, box_too_large, localization precision |

---

## 10. Trial Coverage Map

| Trial | Category | Backbone | Resolution | Temporal | Head/Loss | Key Target |
|---|---|---|---|---|---|---|
| E-01 | exploitation | mv3_small | 224 | T=5 | L1 | late recovery, LR |
| E-02 | exploitation | mv3_large | 320 | T=5 | L1 | late recovery, overfitting |
| E-03 | diagnostic | effb0 | 320 | T=5 | L1 | stability diagnostic |
| E-04 | diagnostic | mv3_small | 224 | T=1 | L1 | temporal gain |
| E-05 | exploitation | mv3_small | 224 | T=3 | L1 | temporal efficiency |
| E-06 | exploitation | mv3_small | 640 | T=5 | L1 | high-res localization |
| E-07 | novelty | mv3_small | 224 | T=5 | objness aux | no_response |
| E-08 | exploitation | mv3_small | 224 | T=5 | L1, vidbal | generalization |
| E-09 | exploitation | mv3_small | 224 | T=5 | L1, freeze | recovery |
| E-10 | novelty | convnext_nano | 320 | T=5 | L1 | architectural novelty |
| E-11 | exploitation | shufflenet_v2 | 224 | T=5 | L1 | deployment efficiency |
| E-12 | novelty | mv3_small | 224 | T=5 | GIoU+center | loss innovation |

**Category distribution:** 2 novelty (17%), 7 exploitation (58%), 2 diagnostic (17%), 1 cheap probe (8%)
*Note: E-03 starts as Level 1 (cheap probe), E-10 starts as Level 1.*

The 12 trials cover:
1. ✅ Lower LR / warmup / late recovery (E-01)
2. ✅ MobileNetV3-Large extended recovery (E-02)
3. ✅ EfficientNet stability with lower LR / AMP diagnostic (E-03)
4. ✅ T=1 image-mode diagnostic (E-04)
5. ✅ T=3 vs T=5 temporal efficiency (E-05)
6. ✅ High-res localization (E-06, 640px)
7. ✅ Objectness auxiliary / dense supervision idea (E-07)
8. ✅ Video-balanced sampler (E-08)
9. ✅ Freeze backbone warmup (E-09)
10. ✅ Upper diagnostic / higher-capacity candidate (E-10, ConvNeXt Nano)
11. ✅ Deployment-efficiency candidate (E-11, ShuffleNet-V2)
12. ✅ Surprising combination not in original 6-variable search (E-12, GIoU+center)

---

## 11. How Evaluation Cost Is Reduced

| Mechanism | Old Cost | New Cost | How |
|---|---|---|---|
| Level 0 static screen | N/A | seconds | Eliminate broken configs before any GPU time |
| Level 1 viability probe | 3 epochs (~12 min) | 1 epoch (~4 min) | Kill dead configs at 1/3 the cost |
| Level 2 dynamics probe | Full 30 epochs | 5 epochs (~20 min) | Early promotion only for promising configs |
| Late bloomer detection | Missed (killed at 5ep) | Flagged for Level 3 | Saves re-discovering viable configs |
| Explorer/Exploiter split | All trials same depth | 80% stop at Level 1-2 | GPU budget concentrated on top candidates |

**Estimated GPU budget for first 12 Explorer trials:** ~2.5-4 hours total (all at Level 1-2).
Compare to old plan: 24 trials × 5 epochs = ~16 hours.

---

## 12. Can We Enter First Explorer Level 0/1 Cheap Probes?

### Prerequisites Check

| Prerequisite | Status |
|---|---|
| CUDA environment clean | ✅ RTX 4090, 0 MiB, no residual processes |
| Clean baseline established | ✅ clean_seed42_best_miou.pth, mIoU=0.2861 |
| Data isolation verified | ✅ canonical_video_id filtering active |
| B1 partial findings archived | ✅ This report |
| Old B1/B2 gate flow stopped | ✅ Not resuming |
| Trial infrastructure ready | ✅ run_trial.py functional (verified in B0) |
| Backbone registry functional | ✅ 5 backbones registered and probed |
| No commits, no pushes | ✅ |

### Recommended First Action

**Level 0 static screens for all 12 Explorer trials:**
- Param count verification
- Dummy forward shape check
- One-batch finite loss check
- Peak memory estimate
- model.to(cuda) success

This takes < 2 minutes total and catches any implementation errors before GPU training.

**Then, Level 1 viability probes for the fastest/lowest-risk trials:**
- E-04 (T=1 diagnostic) — fastest, lowest risk
- E-05 (T=3 efficiency) — fast, low risk
- E-03 (effb0 stable) — critical diagnostic for the entire EN family

### Readiness: YES, with user approval

The infrastructure is verified (B0 baseline proxy proved run_trial.py works). The search space is defined. The 12 trial proposals are designed. Level 0 screens are trivial and risk-free.

---

## 13. Items Requiring User Approval

1. **Confirm stop of old B1/B2 gate flow** — do not resume effb1, do not start B2
2. **Confirm Phase 1.6-R framework** — Explorer/Exploiter dual loop, multi-fidelity reward, research value objective
3. **Approve first 12 Explorer trial proposals** — with or without modifications
4. **Approve Level 0 static screens** — immediate, no training, seconds of GPU time
5. **Approve Level 1 viability probes** — for top-3 lowest-risk trials (E-04, E-05, E-03)
6. **Implementation prerequisites for new backbones:**
   - E-10 (convnext_nano): needs registry entry + stage slice verification — implementation effort ~30 min
   - E-11 (shufflenet_v2): needs registry entry — implementation effort ~20 min
   - E-07 (objectness_aux_head): needs ObjectnessHead class + loss modification — implementation effort ~40 min
   - E-12 (GIoU+center loss): needs multi-component loss — implementation effort ~30 min
7. **Any additional Explorer trial ideas** the user wants to add

---

## 14. Final Answers

### Have we escaped the baseline parameter prison?

**Yes.** The original search space was 6 categorical variables (backbone, input_size=512/640, T=3/5, sampler=2, head=2, lr=3e-4/1e-3) — a 128-combination grid. The new search universe includes:
- 12+ backbones across 4 tiers
- 8 resolutions from 224 to 768
- 6 temporal settings (T=1/3/5/7/9 + stride + pooling variants)
- 7 head/supervision variants
- 9 loss functions with continuous weight ranges
- 5 LR values + warmup + 3 schedulers + EMA + freeze
- 4 sampler variants + source mixtures
- 8 augmentation types

Total search space: effectively infinite (continuous weights × categorical × combinatorial). The baseline parameters are the exploration origin, not the prison walls. Every parameter in the baseline can be changed.

### Have we defined a search universe that is broad but not absurd?

**Yes.** The universe is bounded by:
- Physical laws (15 invariants) — prevents dangerous exploration
- Death conditions (3 categories) — kills dead ends automatically
- Multi-fidelity evaluation — explores broadly at low cost, commits deeply only for winners
- Deployment tiers — constrains param/FPS/memory to real-world feasibility
- Upper diagnostic tier — allows architectural exploration without deployment promises

The universe is wide enough to discover surprises (novelty trials, surprising combinations) but structured enough to not waste GPU time on obviously broken configs.

### Have we defined evaluation that is fast and effective?

**Yes.** The 5-level multi-fidelity system:
- Level 0: seconds (eliminate broken)
- Level 1: ~4 min (eliminate dead)
- Level 2: ~20 min (detect late bloomers, promote promising)
- Level 3: ~60 min (first real quality signal, recovery-aware)
- Level 4: ~2.5 hrs (confirmation, only for top candidates)
- Level 5: ~2.5 hrs/seed (evidence, only for finalists)

The key insight is that Level 2 does NOT kill models with modest mIoU — it looks at slope, loss trajectory, and area calibration to flag `late_bloomer_candidates`. This directly addresses the clean baseline's plateau→recovery pattern.

### Do we recommend entering Explorer Level 0/1 cheap probes?

**Yes, pending user approval.** The environment is clean. The infrastructure is verified. The proposals are designed. Level 0 screens are zero-risk. Level 1 probes for E-04, E-05, and E-03 are low-cost (~5 min each) and answer critical diagnostic questions before committing to the full 12-trial round.

---

*Report generated at 2026-05-07T22:20 CST*
*Phase 1.6-R Autonomous Controlled AutoResearch Redesign*
*Design only — no training, no commit, no push*
*B1 partial findings archived as smoke evidence*
*Next: user approval for Level 0 screens and first Level 1 probes*
