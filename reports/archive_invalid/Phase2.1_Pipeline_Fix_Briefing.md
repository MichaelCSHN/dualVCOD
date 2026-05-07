# Phase 2.1 数据管道修复 & Dataloader 预检简报

**生成时间**: 2026-05-06
**状态**: COMPLETE — 训练尚未启动 (CEO halt order respected)
**范围**: CAD (9 categories) + MoCA_Mask (87 videos) + MoCA (141 videos)

---

## 阶段一：数据管道修复

### 修复 A: CAD 掩码阈值 — `mask_to_bbox()` (src/dataset_real.py:28)

**问题**: 5 个 CAD 类别 (chameleon, glowwormbeetle, scorpion1, snail, stickinsect) 的 GT 掩码使用稀疏像素值 (1-2) 而非标准二值 (0/255)。原阈值 `mask > 127` 和 `mask > 10` 均无法提取有效 BBox。

**修复**: 将阈值从 `> 10` 改为 `> 0`，同时兼容二值掩码 (0/255) 和稀疏掩码 (0/1/2)。

**验证结果** — 全部 9 个 CAD 类别成功索引:

| 类别 | GT文件 | 有效BBox | 空掩码 | 状态 |
|------|--------|----------|--------|------|
| chameleon | 43 | 43 | 0 | OK |
| frog | 30 | 20 | 10 | OK |
| glowwormbeetle | 21 | 21 | 0 | OK |
| scorpion1 | 21 | 21 | 0 | OK |
| scorpion2 | 12 | 12 | 0 | OK |
| scorpion3 | 15 | 15 | 0 | OK |
| scorpion4 | 16 | 16 | 0 | OK |
| snail | 17 | 17 | 0 | OK |
| stickinsect | 16 | 16 | 0 | OK |

### 修复 B: MoCA_Mask 测试集硬隔离 (src/dataset_real.py:154)

**问题**: `_index_moca_mask()` 遍历 `["TrainDataset_per_sq", "TestDataset_per_sq"]`，将 16 个测试视频混入训练集。

**修复**: 循环限制为 `["TrainDataset_per_sq"]`，TestDataset_per_sq 的 16 个视频完全排除。

**验证结果**: 训练索引中零 TestDataset_per_sq 视频 (71/71 训练视频索引, 0/16 测试视频泄露)。

---

## 阶段一结果：修复后数据分布

| 数据集 | 视频数 | 窗口数 | 用途 |
|--------|--------|--------|------|
| MoCA | 141 | 6,948 | 训练 (113 train / 28 val via split) |
| MoCA_Mask | 71 | 3,624 | 训练 (仅 TrainDataset_per_sq) |
| CAD | 9 | 145 | 训练 (全部 9 类别) |
| **Joint Train** | **221** | **10,717** | |
| **Validation** | **28** | **1,188** | MoCA 20% video-level split |

### MoCA_Mask TestDataset_per_sq (16 视频) — 已隔离，未进入任何 Loader

---

## 阶段二：Dataloader 预检

### 脚本: `tools/check_dataloader.py`

- 独立运行，无 CUDA 依赖
- 加载 Joint Train (MoCA + MoCA_Mask + CAD)，shuffle=True
- 运行 8 个 batch，batch_size=16, T=5
- 检查项目: 张量形状、NaN/Inf、BBox 范围 [0,1]、全零 BBox、x1<x2 / y1<y2

### 预检结果: 全部通过

```
  batch   0 | frames (16,5,3,224,224) | bboxes (16,5,4) | range [0.0059, 0.9995] | OK
  batch   1 | frames (16,5,3,224,224) | bboxes (16,5,4) | range [0.0000, 0.9178] | OK
  batch   2 | frames (16,5,3,224,224) | bboxes (16,5,4) | range [0.0000, 1.0000] | OK
  batch   3 | frames (16,5,3,224,224) | bboxes (16,5,4) | range [0.0020, 0.9995] | OK
  batch   4 | frames (16,5,3,224,224) | bboxes (16,5,4) | range [0.0000, 0.9992] | OK
  batch   5 | frames (16,5,3,224,224) | bboxes (16,5,4) | range [0.0039, 0.9000] | OK
  batch   6 | frames (16,5,3,224,224) | bboxes (16,5,4) | range [0.1016, 0.9986] | OK
  batch   7 | frames (16,5,3,224,224) | bboxes (16,5,4) | range [0.0000, 0.9998] | OK

  Shape consistency : PASS — All batches (16, 5, 3, 224, 224)
  [PASS] All checks passed — 8 batches clean
```

| 检查项 | 预期 | 实际 | 结果 |
|--------|------|------|------|
| frames.shape | (B,5,3,224,224) | (16,5,3,224,224) | PASS |
| bboxes.shape | (B,5,4) | (16,5,4) | PASS |
| BBox NaN | 0 | 0 | PASS |
| BBox Inf | 0 | 0 | PASS |
| BBox OOB (<0 or >1) | 0 | 0 | PASS |
| BBox x1<x2, y1<y2 | True | True | PASS |
| 全零 BBox | 0 | 0 | PASS |
| 形状一致性 (8 batches) | 一致 | 一致 | PASS |

---

## 变更文件清单

| 文件 | 变更 | 行号 |
|------|------|------|
| `src/dataset_real.py` | `mask > 127` → `mask > 10` → `mask > 0` | L28 |
| `src/dataset_real.py` | MoCA_Mask 循环: 移除 `"TestDataset_per_sq"` | L154 |
| `tools/check_dataloader.py` | **新建** — Dataloader 预检脚本 | — |
| `tools/verify_pipeline_fix.py` | **新建** — 修复验证脚本 | — |

---

## 结论

数据管道修复完成，所有检查通过。训练数据从 221 个视频 (10,717 窗口) 构成，验证集 28 个 MoCA 视频 (1,188 窗口)。MoCA_Mask TestDataset_per_sq (16 视频) 已硬隔离。

**训练尚未启动** — 等待 CEO 批准后进行 Phase 2.1 正式训练。

---

*简报由数据管道修复 & Dataloader 预检流程自动生成*
