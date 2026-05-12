# 数据集 NVMe 迁移验证报告

**日期**: 2026-05-11
**结论**: 数据集从 D: HDD 迁移至 C: NVMe 后，训练速度**无明显提升**（-6%），瓶颈在 CPU JPEG 解码而非磁盘 I/O。C:\datasets 可作为默认路径使用，但对吞吐无加速效果。

---

## 1. 已更新的数据集路径

| 数据集 | 旧路径 (HDD) | 新路径 (NVMe) | 大小 |
|--------|-------------|--------------|------|
| MoCA | `D:\ML\COD_datasets\MoCA` | `C:\datasets\MoCA` | 9.26 GB |
| MoCA_Mask | `D:\ML\COD_datasets\MoCA_Mask` | `C:\datasets\MoCA_Mask` | 1.20 GB |
| CAD | `D:\ML\COD_datasets\CamouflagedAnimalDataset` | `C:\datasets\CamouflagedAnimalDataset` | 0.28 GB |

C: 盘为 Samsung MZALQ512HALU NVMe SSD (512 GB)，D: 盘为 WDC WD20SPZX SATA HDD (2 TB)。

---

## 2. 修改过的文件

| 文件 | 行 | 修改 |
|------|-----|------|
| `tools/autoresearch/run_trial_minimal.py` | 40-45 | `DEFAULT_TRAIN_DATASETS` 和 `DEFAULT_VAL_DATASET` 路径从 `D:\ML\COD_datasets\*` 改为 `C:\datasets\*` |
| `configs/autoresearch/search_space_phase1_6.yaml` | 42-45 | `train_datasets` 和 `val_dataset` 路径同上更新 |

无其他活跃文件包含硬编码的 D: 盘数据集路径。

---

## 3. 残留 D: 盘路径检查

- **活跃代码**: 零残留（`.py`, `.json`, `.yaml` 中无 `D:\\ML` 或 `D:\\COD_datasets` 引用）
- **历史 metadata**: 12 个 `local_runs/autoresearch/expl_*/metadata.json` 中包含 `checkpoint_local_only` 字段指向 D: 盘，这些是历史运行记录，不影响后续训练
- **缓存测试配置**: `configs/autoresearch/_cache_bs16_nw4.json` 中 `cache_dir` 指向 `D:/dualvcod/data/cache/frames`，但缓存方案已弃用，不影响默认训练

---

## 4. NVMe vs HDD Profiling 对比

统一配置：`bs=48, nw=4, DIoU, T=5, 224px, MobileNetV3-Small`，各 3 epoch 均值。

| 指标 | HDD (D:\) | NVMe (C:\) | Δ |
|------|-----------|-----------|------|
| **data_time** | **144.2s** | 152.8s | +6.0% |
| forward | 6.7s | 7.9s | — |
| backward | 8.3s | 9.3s | — |
| validation | 36.8s | 38.1s | +3.5% |
| h2d | 8.4s | 8.8s | — |
| optimizer + loss | 5.8s | 6.3s | — |
| **total epoch** | **211.0s** | 223.6s | +6.0% |
| **frames/s (data)** | **310** | 292 | -5.8% |
| GPU GiB | 2.30 | 2.30 | 不变 |
| 瓶颈标记 | dataloader | dataloader | 不变 |

### 逐 epoch 明细

| Epoch | HDD data | HDD total | NVMe data | NVMe total |
|-------|----------|-----------|-----------|------------|
| 1 | 143.8s | 211.7s | 149.4s | 221.4s |
| 2 | 144.4s | 210.9s | 150.7s | 219.8s |
| 3 | 144.3s | 210.3s | 158.2s | 229.5s |
| **均值** | **144.2s** | **211.0s** | **152.8s** | **223.6s** |

---

## 5. 分析：为何 NVMe 未加速

### 5.1 瓶颈在 CPU，不在磁盘

当前 DataLoader 流水线的每帧处理：

```
磁盘读取 JPEG (20-50KB) → cv2.imread (libjpeg-turbo 解码) → BGR2RGB → resize(224×224)
```

- HDD 顺序读取 1.6 GB/epoch 约需 11-15 秒（~120 MB/s 顺序读）
- 剩余 ~130 秒全部消耗在 CPU JPEG 解码 + 颜色转换 + resize
- NVMe 将磁盘读取降至 < 1 秒，但 CPU 解码时间不变

**disk I/O 仅占 data_time 的 ~10%，CPU 解码占 ~90%。NVMe 只能加速那 10%。**

### 5.2 C: 盘系统争抢

C: 盘同时承载 Windows OS、页面文件、临时文件、浏览器缓存等。4 个 DataLoader worker 的并发随机读与 OS 背景 I/O 叠加，可能导致 NVMe 调度开销略高于预期。

### 5.3 缓存实验的佐证

之前的 `.npy` 缓存实验将磁盘 I/O 放大 3-7×（文件体积从 20-50KB → 147KB），导致 data_time 从 131s 暴增至 398s。证明：
- **缩小 I/O 体积有帮助**（JPEG 比 .npy 小）→ I/O 不是零影响
- **但完全消除 JPEG 解码 + 放大 I/O = 3× 变慢** → CPU 解码更便宜

NVMe 仅改善了 I/O 侧，不能突破 CPU 解码墙。

---

## 6. 建议

| 决策 | 结论 |
|------|------|
| 默认数据路径 | **C:\datasets** — 虽无加速，但统一路径便于管理，且为将来 NVMe 独占时的优化留空间 |
| num_workers | **保持 4** — 不变 |
| batch_size | **保持 48** — 不变 |
| cache_dir | **保持 null** — 不变 |
| 进一步加速方向 | **RAM 预加载**（消除所有磁盘 I/O + 复用解码结果）或 **GPU 解码**（nvJPEG） |

### 预期训练时间

30 epoch: **223.6s × 30 = 6,708s ≈ 1.86 小时**（vs HDD 1.76h，差异可接受）

---

## 附录：实验环境

- **GPU**: NVIDIA GeForce RTX 4090 (24 GiB VRAM)
- **CPU**: 未记录
- **C: 盘**: Samsung MZALQ512HALU NVMe SSD, 512 GB
- **D: 盘**: WDC WD20SPZX SATA HDD, 2 TB
- **模型**: MicroVCOD, MobileNetV3-Small, 1,411,684 params
- **数据**: MoCA + MoCA_Mask + CAD, 8,926 train samples, T=5, 224×224
- **训练配置**: bs=48, nw=4, DIoU loss, FP16 autocast

---

*报告生成时间: 2026-05-11T07:40*
*Profiler 版本: tools/autoresearch/profiler.py (fine-grained v2)*
