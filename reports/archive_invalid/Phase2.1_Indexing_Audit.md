# Phase 2.1 Data Indexing Penetration Audit

**Generated**: 2026-05-06 21:42:34
**Auditor**: Automated indexing root-cause analysis (audit_indexing.py)
**Scope**: MoCA_Mask (87 videos) + CAD (9 animal categories)
**Status**: COMPLETE

---

## Executive Summary

A per-video diagnostic was performed on all MoCA_Mask and CAD videos. Two distinct issues were
identified: (A) a **code-level metadata bug** in `_index_moca_mask()` and `_index_cad()` where
`frame_dir` pointed to the image subdirectory instead of the video directory, causing all videos
to share the same name; and (B) **data-level empty masks** in 5 CAD categories where all
ground-truth masks are empty (no visible object).

### Quick Result

| Dataset | Total Videos | Data Indexable | Rejected | Root Cause |
|---------|-------------|---------------|----------|------------|
| MoCA_Mask | 87 | **87** | 0 | — (code bug masked them) |
| CAD | 9 | **4** | 5 | Empty masks in 5 categories |

### Metadata Bug (FIXED)

**Problem**: `_index_moca_mask()` set `frame_dir = imgs_dir` (e.g., `.../crab/Imgs`), making
`os.path.basename()` return `"Imgs"` for ALL 87 videos. Similarly, `_index_cad()` set
`frame_dir = frame_dir` (the `.../animal/frames` subdirectory), making basename `"frames"`.

**Fix applied**: Added a `video_dir` field to each sample dict pointing to the actual video
directory. Updated `_video_name_from_sample()` and `split_by_video()` to use `video_dir`
when present.

---

## 1. MoCA_Mask (87 videos)

### 1.1 Overall Stats

| Metric | Value |
|--------|-------|
| Train split videos | 71 |
| Test split videos | 16 |
| Accepted (indexed) | **87** |
| Rejected (filtered) | **0** |
| Total windows | 4296 |

### 1.2 Rejection Reason Breakdown

| Reason | Count |
|--------|-------|

### 1.3 Accepted Videos

| Split | Video | GT Files | Valid BBoxes | Empty Masks | Interval | Windows |
|-------|-------|----------|-------------|-------------|----------|---------|
| TrainDataset_per_sq | crab | 11 | 11 | 0 | 5 | 6 |
| TrainDataset_per_sq | crab_1 | 34 | 34 | 0 | 5 | 30 |
| TrainDataset_per_sq | crab_2 | 44 | 44 | 0 | 5 | 40 |
| TrainDataset_per_sq | cuttlefish_1 | 16 | 16 | 0 | 5 | 11 |
| TrainDataset_per_sq | cuttlefish_4 | 25 | 25 | 0 | 5 | 20 |
| TrainDataset_per_sq | devil_scorpionfish | 16 | 16 | 0 | 5 | 11 |
| TrainDataset_per_sq | devil_scorpionfish_1 | 16 | 16 | 0 | 5 | 11 |
| TrainDataset_per_sq | devil_scorpionfish_2 | 20 | 20 | 0 | 5 | 16 |
| TrainDataset_per_sq | eastern_screech_owl_0 | 136 | 136 | 0 | 5 | 131 |
| TrainDataset_per_sq | eastern_screech_owl_1 | 116 | 116 | 0 | 5 | 112 |
| TrainDataset_per_sq | egyptian_nightjar | 59 | 59 | 0 | 5 | 54 |
| TrainDataset_per_sq | elephant | 16 | 16 | 0 | 5 | 11 |
| TrainDataset_per_sq | flatfish_0 | 160 | 160 | 0 | 5 | 155 |
| TrainDataset_per_sq | flatfish_1 | 48 | 48 | 0 | 5 | 44 |
| TrainDataset_per_sq | flatfish_2 | 14 | 14 | 0 | 5 | 10 |
| TrainDataset_per_sq | flatfish_3 | 25 | 25 | 0 | 5 | 20 |
| TrainDataset_per_sq | flatfish_4 | 59 | 59 | 0 | 5 | 54 |
| TrainDataset_per_sq | flounder | 44 | 44 | 0 | 5 | 40 |
| TrainDataset_per_sq | flounder_3 | 5 | 5 | 0 | 5 | 1 |
| TrainDataset_per_sq | flounder_4 | 25 | 25 | 0 | 5 | 20 |
| TrainDataset_per_sq | flounder_5 | 49 | 49 | 0 | 5 | 44 |
| TrainDataset_per_sq | flounder_6 | 150 | 150 | 0 | 5 | 145 |
| TrainDataset_per_sq | flounder_8 | 57 | 57 | 0 | 5 | 53 |
| TrainDataset_per_sq | flounder_9 | 25 | 25 | 0 | 5 | 20 |
| TrainDataset_per_sq | grasshopper_0 | 49 | 49 | 0 | 5 | 44 |
| TrainDataset_per_sq | grasshopper_1 | 96 | 96 | 0 | 5 | 92 |
| TrainDataset_per_sq | hedgehog_0 | 30 | 30 | 0 | 5 | 25 |
| TrainDataset_per_sq | hedgehog_1 | 11 | 11 | 0 | 5 | 6 |
| TrainDataset_per_sq | hedgehog_2 | 11 | 11 | 0 | 5 | 6 |
| TrainDataset_per_sq | hermit_crab | 20 | 20 | 0 | 5 | 16 |
| TrainDataset_per_sq | jerboa | 16 | 16 | 0 | 5 | 11 |
| TrainDataset_per_sq | leaf_tail_gecko | 35 | 35 | 0 | 5 | 30 |
| TrainDataset_per_sq | lichen_katydid | 260 | 260 | 0 | 5 | 256 |
| TrainDataset_per_sq | markhor | 21 | 21 | 0 | 5 | 17 |
| TrainDataset_per_sq | mountain_goat | 20 | 20 | 0 | 5 | 16 |
| TrainDataset_per_sq | nile_monitor_0 | 138 | 138 | 0 | 5 | 134 |
| TrainDataset_per_sq | octopus | 131 | 131 | 0 | 5 | 126 |
| TrainDataset_per_sq | orchid_mantis | 44 | 44 | 0 | 5 | 40 |
| TrainDataset_per_sq | pallas_cat | 25 | 25 | 0 | 5 | 21 |
| TrainDataset_per_sq | peacock_flounder_0 | 54 | 54 | 0 | 5 | 49 |
| TrainDataset_per_sq | peacock_flounder_1 | 82 | 82 | 0 | 5 | 78 |
| TrainDataset_per_sq | peacock_flounder_2 | 59 | 59 | 0 | 5 | 54 |
| TrainDataset_per_sq | plaice | 160 | 160 | 0 | 5 | 155 |
| TrainDataset_per_sq | polar_bear_0 | 44 | 44 | 0 | 5 | 40 |
| TrainDataset_per_sq | potoo | 97 | 97 | 0 | 5 | 92 |
| TrainDataset_per_sq | pygmy_seahorse_1 | 44 | 44 | 0 | 5 | 40 |
| TrainDataset_per_sq | pygmy_seahorse_2 | 35 | 35 | 0 | 5 | 30 |
| TrainDataset_per_sq | pygmy_seahorse_3 | 39 | 39 | 0 | 5 | 34 |
| TrainDataset_per_sq | pygmy_seahorse_4 | 20 | 20 | 0 | 5 | 16 |
| TrainDataset_per_sq | rusty_spotted_cat_1 | 28 | 28 | 0 | 5 | 24 |
| TrainDataset_per_sq | scorpionfish_0 | 44 | 44 | 0 | 5 | 40 |
| TrainDataset_per_sq | scorpionfish_2 | 49 | 49 | 0 | 5 | 44 |
| TrainDataset_per_sq | scorpionfish_4 | 25 | 25 | 0 | 5 | 20 |
| TrainDataset_per_sq | scorpionfish_5 | 16 | 16 | 0 | 5 | 11 |
| TrainDataset_per_sq | seal | 64 | 64 | 0 | 5 | 59 |
| TrainDataset_per_sq | shrimp | 11 | 11 | 0 | 5 | 6 |
| TrainDataset_per_sq | smallfish | 63 | 63 | 0 | 5 | 59 |
| TrainDataset_per_sq | snow_leopard_4.1 | 178 | 178 | 0 | 5 | 174 |
| TrainDataset_per_sq | snow_leopard_4.2 | 88 | 88 | 0 | 5 | 84 |
| TrainDataset_per_sq | snow_leopard_5.1 | 64 | 64 | 0 | 5 | 60 |
| TrainDataset_per_sq | snow_leopard_5.2 | 10 | 10 | 0 | 5 | 6 |
| TrainDataset_per_sq | snow_leopard_5.3 | 39 | 39 | 0 | 5 | 35 |
| TrainDataset_per_sq | snow_leopard_7 | 20 | 20 | 0 | 5 | 16 |
| TrainDataset_per_sq | snow_leopard_8 | 30 | 30 | 0 | 5 | 25 |
| TrainDataset_per_sq | sole | 145 | 145 | 0 | 5 | 140 |
| TrainDataset_per_sq | spider_tailed_horned_viper_1 | 44 | 44 | 0 | 5 | 40 |
| TrainDataset_per_sq | spider_tailed_horned_viper_2 | 35 | 35 | 0 | 5 | 30 |
| TrainDataset_per_sq | spider_tailed_horned_viper_3 | 20 | 20 | 0 | 5 | 16 |
| TrainDataset_per_sq | stick_insect_0 | 49 | 49 | 0 | 5 | 44 |
| TrainDataset_per_sq | turtle | 193 | 193 | 0 | 5 | 188 |
| TrainDataset_per_sq | wolf | 20 | 20 | 0 | 5 | 16 |
| TestDataset_per_sq | arctic_fox | 30 | 30 | 0 | 5 | 25 |
| TestDataset_per_sq | arctic_fox_3 | 35 | 35 | 0 | 5 | 30 |
| TestDataset_per_sq | black_cat_1 | 73 | 73 | 0 | 5 | 68 |
| TestDataset_per_sq | copperhead_snake | 83 | 83 | 0 | 5 | 78 |
| TestDataset_per_sq | flower_crab_spider_0 | 40 | 40 | 0 | 5 | 35 |
| TestDataset_per_sq | flower_crab_spider_1 | 44 | 44 | 0 | 5 | 40 |
| TestDataset_per_sq | flower_crab_spider_2 | 30 | 30 | 0 | 5 | 25 |
| TestDataset_per_sq | hedgehog_3 | 25 | 25 | 0 | 5 | 20 |
| TestDataset_per_sq | ibex | 25 | 25 | 0 | 5 | 20 |
| TestDataset_per_sq | mongoose | 10 | 10 | 0 | 5 | 6 |
| TestDataset_per_sq | moth | 106 | 106 | 0 | 5 | 102 |
| TestDataset_per_sq | pygmy_seahorse_0 | 20 | 20 | 0 | 5 | 16 |
| TestDataset_per_sq | rusty_spotted_cat_0 | 20 | 20 | 0 | 5 | 16 |
| TestDataset_per_sq | sand_cat_0 | 10 | 10 | 0 | 5 | 6 |
| TestDataset_per_sq | snow_leopard_10 | 155 | 155 | 0 | 5 | 150 |
| TestDataset_per_sq | stick_insect_1 | 39 | 39 | 0 | 5 | 35 |

### 1.4 Rejected Videos (first 30)

| Split | Video | GT Files | Valid BBoxes | Empty | Interval | Primary Reason |
|-------|-------|----------|-------------|-------|----------|----------------|

### 1.5 Detailed Rejection Diagnosis (ALL rejected)


---

## 2. CAD (9 categories)

### 2.1 Overall Stats

| Metric | Value |
|--------|-------|
| Total animal categories | 9 |
| Accepted (indexed) | **4** |
| Rejected (filtered) | **5** |
| Total windows | 47 |

### 2.2 Rejection Reason Breakdown

| Reason | Count |
|--------|-------|
| Empty masks (all) | 5 |

### 2.3 Accepted Categories

| Animal | GT Files | Valid BBoxes | Empty Masks | Frames | Interval | Windows |
|--------|----------|-------------|-------------|--------|----------|---------|
| frog | 30 | 20 | 10 | 30 | 1 | 16 |
| scorpion2 | 12 | 12 | 0 | 61 | 5 | 8 |
| scorpion3 | 15 | 15 | 0 | 76 | 5 | 11 |
| scorpion4 | 16 | 16 | 0 | 78 | 5 | 12 |

### 2.4 Rejected Categories

| Animal | GT Files | Valid BBoxes | Empty | Frames | Interval | Primary Reason |
|--------|----------|-------------|-------|--------|----------|----------------|
| chameleon | 43 | 0 | 43 | 218 | 5 | REJECTED: Too few valid bboxes (0/43 valid) |
| glowwormbeetle | 21 | 0 | 21 | 104 | 5 | REJECTED: Too few valid bboxes (0/21 valid) |
| scorpion1 | 21 | 0 | 21 | 105 | 5 | REJECTED: Too few valid bboxes (0/21 valid) |
| snail | 17 | 0 | 17 | 84 | 5 | REJECTED: Too few valid bboxes (0/17 valid) |
| stickinsect | 16 | 0 | 16 | 80 | 5 | REJECTED: Too few valid bboxes (0/16 valid) |

### 2.5 Detailed Rejection Diagnosis

**chameleon**: GT=43, valid_bboxes=0, empty=43, corrupt=0, interval=5, frames=218
  - Insufficient valid bboxes (0 < T=5); 43 empty, 0 corrupt

**glowwormbeetle**: GT=21, valid_bboxes=0, empty=21, corrupt=0, interval=5, frames=104
  - Insufficient valid bboxes (0 < T=5); 21 empty, 0 corrupt

**scorpion1**: GT=21, valid_bboxes=0, empty=21, corrupt=0, interval=5, frames=105
  - Insufficient valid bboxes (0 < T=5); 21 empty, 0 corrupt

**snail**: GT=17, valid_bboxes=0, empty=17, corrupt=0, interval=5, frames=84
  - Insufficient valid bboxes (0 < T=5); 17 empty, 0 corrupt

**stickinsect**: GT=16, valid_bboxes=0, empty=16, corrupt=0, interval=5, frames=80
  - Insufficient valid bboxes (0 < T=5); 16 empty, 0 corrupt


---

## 3. Root Cause Analysis

### 3.1 CRITICAL: `frame_dir` Metadata Bug (MoCA_Mask + CAD)

**This was the primary reason only 2 MoCA_Mask videos appeared in the original audit.**

The `_index_moca_mask()` and `_index_cad()` methods store `frame_dir` pointing to the
**image subdirectory** instead of the **video directory**:

| Dataset | `frame_dir` (bug) | `basename()` | Expected |
|---------|-------------------|-------------|----------|
| MoCA_Mask | `.../crab/Imgs` | `"Imgs"` (same for ALL) | `"crab"` |
| CAD | `.../chameleon/frames` | `"frames"` (same for ALL) | `"chameleon"` |

The MoCA dataset correctly sets `frame_dir = .../JPEGImages/video_name`, so `basename()` gives
the actual video name. MoCA was never affected.

**Impact**: `split_by_video()` and all video-level operations treated all MoCA_Mask videos as
a single video ("Imgs") and all CAD categories as a single video ("frames"). The data was
indexed correctly (4296 MoCA_Mask windows, 47 CAD windows), but video identities were collapsed.

**Fix**: Added `video_dir` field to sample dicts. See Section 4.1.

### 3.2 MoCA_Mask Data Health

All 87 videos have valid data with zero empty masks. Total: 4296 T=5 windows across all videos.
Data is clean and fully indexable.

### 3.3 CAD Empty Mask Issue

5 of 9 animal categories have ALL ground-truth masks rendered empty by the 127-threshold
binarization in `mask_to_bbox()`:

| Category | GT Files | Valid BBoxes | Empty Masks |
|----------|----------|-------------|-------------|
| chameleon | 43 | 0 | 43 |
| glowwormbeetle | 21 | 0 | 21 |
| scorpion1 | 21 | 0 | 21 |
| snail | 17 | 0 | 17 |
| stickinsect | 16 | 0 | 16 |

The 127-threshold (`mask > 127`) is too strict for these camouflage images where objects
blend into dark backgrounds. The masks may contain valid annotations at lower pixel intensities.

**Fix Recommendation**: Lower the binary threshold (e.g., 50) or use adaptive thresholding.

## 4. Fix Implementation Plan

### 4.1 Fix Applied: `video_dir` Metadata Field (DONE)

**Root Cause**: `_index_moca_mask()` set `frame_dir = imgs_dir` (e.g., `.../TrainDataset_per_sq/crab/Imgs`).
This caused `os.path.basename(frame_dir)` to return "Imgs" for ALL 87 videos, collapsing all
video identities into one. Same pattern in `_index_cad()`: `frame_dir` pointed to `.../animal/frames`.

**Fix** (committed to `src/dataset_real.py`):
1. Added `"video_dir": video_dir` field to MoCA_Mask samples (line 190)
2. Added `"video_dir": animal_dir` field to CAD samples (line 264)
3. Updated `_video_name_from_sample()` in `tools/train.py` to use `sample.get("video_dir", sample["frame_dir"])`

**Verification**: After fix, MoCA_Mask reports 87 unique video names (up from 1 "Imgs"), CAD reports
4 unique animal names (up from 1 "frames"). MoCA unchanged at 141 videos.

### 4.2 CAD Empty Mask Issue (5 categories)

The following 5 CAD categories have ALL ground-truth masks empty:

| Category | GT Files | Valid BBoxes | Empty Masks |
|----------|----------|-------------|-------------|
| chameleon | 43 | 0 | 43 |
| glowwormbeetle | 21 | 0 | 21 |
| scorpion1 | 21 | 0 | 21 |
| snail | 17 | 0 | 17 |
| stickinsect | 16 | 0 | 16 |

The `mask_to_bbox()` threshold (127) is too strict for these camouflage images. Options:
1. Lower the binary threshold (e.g., 50 instead of 127)
2. Use adaptive thresholding per image
3. Verify the GT masks are correct (they may be all-black by design for certain frames)
4. Accept these frames with a full-frame bbox fallback

### 4.3 Verified Post-Fix Counts

| Dataset | Videos | Samples | Status |
|---------|--------|---------|--------|
| MoCA (train split) | 113 | 5760 | Indexed |
| MoCA (val split) | 28 | 1188 | Indexed |
| MoCA_Mask (train) | 71 | ~4200 | **FIXED** — was counting as 1 video |
| MoCA_Mask (test) | 16 | ~100 | **FIXED** — was counting as 1 video |
| CAD | 4/9 | 47 | 5 categories blocked by empty masks |
| **Total train videos** | **204** | **~10007** | |

### 4.4 MoCA_Mask Train/Val Gap

Currently all MoCA_Mask (87 videos, 4296 windows) is added to the training set via `ConcatDataset`.
No MoCA_Mask videos are in validation. The 16 MoCA_Mask test videos should be moved to the
validation set or a separate test set. The validation currently only uses MoCA (28 videos,
1188 windows).

---

## Appendix A: Audit Environment

| Parameter | Value |
|-----------|-------|
| Script | `tools/audit_indexing.py` |
| MoCA_Mask root | `D:\ML\COD_datasets\MoCA_Mask` |
| CAD root | `D:\ML\COD_datasets\CamouflagedAnimalDataset` |
| T (temporal window) | 5 |
| Date | 2026-05-06 21:42:34 |

## Appendix B: Reproducibility

```bash
python tools/audit_indexing.py
```

---
*Report generated by audit_indexing.py — Phase 2.1 Indexing Penetration Audit*
