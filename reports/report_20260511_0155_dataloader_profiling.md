# DataLoader 吞吐优化实验报告

**日期**: 2026-05-11
**结论**: 最优训练配置为 **bs=48, nw=4**，30 epoch 预计 1.76h（相比 E-31 的 2.6h 节省 32%）。DataLoader 为绝对瓶颈，frame-level .npy 缓存反致性能恶化 3×。

---

## 1. batch_size=32 速度异常分析

### 现象
E-31 使用 bs=16, nw=2 时 30 epoch 耗时 9403s (2.6h)，约 313s/epoch。将 bs 提升至 32 后，预期训练时间减半（批次数量从 558 降至 279），实际却**未获加速**，反而 batch 耗时从 0.57s 上升至 1.31s，总时间增至 418s/epoch。

### 根因
**DataLoader 为绝对瓶颈，batch_size 增大不减少数据加载总量**。每 epoch 固定加载 44,640 帧图像，无论 batch_size 如何，cv2.imread + BGR2RGB + resize 的总帧数不变。增大 bs 仅减少 GPU forward/backward 次数（从 558 次降至 279 次），但 GPU 计算仅占总时间的 9-15%，节省的 ~20s 被 batch 内部更多帧的加载等待完全抵消。

| 指标 | bs=16, nw=2 | bs=32, nw=2 | 变化 |
|------|------------|------------|------|
| 每 epoch 帧数 | 44,640 | 44,640 | 不变 |
| 批次数 | 558 | 279 | -50% |
| 每 batch 帧数 | 80 | 160 | +100% |
| data_time | 255s | 318s | **+25%** |
| GPU compute | 54s | 37s | -31% |
| 总 epoch 时间 | 357s | 418s | **+17%** |

结论：**增大 batch_size 反致吞吐下降**，因为每 batch 内 2 个 worker 需加载更多帧，worker 间并行度不足以补偿。

---

## 2. batch_size × num_workers 吞吐对比

### 2.1 Phase 1: 固定 nw=2 的 bs 扫描（1 epoch）

5 组配置各跑 1 epoch，统一使用 DIoU loss, T=5, MobileNetV3-Small, RTX 4090。

| 配置 | bs | nw | Batches | Epoch(s) | Data(s) | Data% | GPU(s) | Batch(s) | **Frames/s** | mIoU | GPU GiB |
|------|----|----|---------|----------|---------|-------|--------|----------|-------------|------|---------|
| prof_bs16_nw0 | 16 | 0 | 558 | 634 | 509 | 80.3% | 56.2 | 1.14 | 87.7 | 0.212 | 0.79 |
| prof_bs16_nw2 | 16 | 2 | 558 | 357 | 255 | 71.5% | 53.8 | 0.57 | 174.8 | 0.263 | 0.79 |
| **prof_bs16_nw4** | **16** | **4** | **558** | **254** | **131** | **51.5%** | 72.0 | **0.38** | **341.3** | 0.215 | 0.79 |
| prof_bs24_nw2 | 24 | 2 | 372 | 355 | 269 | 76.0% | 36.9 | 0.85 | 165.7 | 0.220 | 1.17 |
| prof_bs32_nw2 | 32 | 2 | 279 | 418 | 318 | 76.2% | 36.9 | 1.31 | 140.3 | 0.225 | 1.55 |

**Phase 1 结论**：nw=2 时增大 bs 反降低吞吐，batch 帧数增长超过 GPU 节省。

### 2.2 Phase 2: nw=4 下的 bs 扩展测试（3 epoch）

在确认 nw=4 为最优 worker 数后，进一步测试 bs=32/48/64 在 nw=4 下的表现。以下数据为 3 epoch 均值。

| 配置 | bs | nw | Batches | Epoch(s) | Data(s) | Data% | GPU*(s) | Batch(s) | **Frames/s** | GPU GiB |
|------|----|----|---------|----------|---------|-------|---------|----------|-------------|---------|
| speed_bs16_nw4 | 16 | 4 | 558 | 254 | 131 | 51.5% | 72.0 | 0.38 | 341 | 0.79 |
| speed_bs32_nw4 | 32 | 4 | 279 | 250 | 159 | 63.4% | 36.1 | 0.73 | 281 | 1.55 |
| **speed_bs48_nw4** | **48** | **4** | **186** | **211** | **144** | **68.3%** | **20.8** | **0.93** | **310** | **2.30** |
| speed_bs64_nw4 | 64 | 4 | 140 | 212 | 147 | 69.4% | 17.1 | 1.24 | 304 | 3.06 |

> GPU* = fwd + bwd + loss + optimizer (不含 h2d 和 validation)

### 关键发现

1. **num_workers 是主导因子**：nw=4 (341 fps) ≈ 2× nw=2 (175 fps) ≈ 4× nw=0 (88 fps)
2. **nw=4 解锁 bs 扩展收益**：nw=2 时增大 bs 反降吞吐，但 nw=4 下 bs=48/64 的 data_time 仅从 131s 微增至 144-147s，而 GPU compute 从 72s 骤降至 21-17s，**总 epoch 时间从 254s 降至 211s**（-17%）
3. **bs=48 为甜点**：211s/epoch，与 bs=64 持平（212s），GPU 内存仅 2.30 GiB（bs=64 需 3.06 GiB），性价比最高
4. **mIoU 单 epoch 波动大**（0.212-0.263），不可作为选型依据

---

## 3. Profiling 各阶段耗时占比

### 3.1 bs=16, nw=4（1 epoch profiling，总时间 254.1s）

| 阶段 | 时间 (s) | 占比 | 说明 |
|------|----------|------|------|
| **data_time** | **130.8** | **51.5%** | DataLoader 等待：cv2.imread + BGR2RGB + resize |
| forward | 29.6 | 11.7% | 模型前向传播 |
| backward | 33.9 | 13.3% | 反向传播 + unscale + clip_grad |
| validation | 41.4 | 16.3% | 验证集评估 |
| h2d | 9.0 | 3.6% | CPU→GPU 数据传输 |
| optimizer | 5.5 | 2.2% | scaler.step() + scaler.update() |
| loss | 3.0 | 1.2% | criterion 损失计算 |

```
data_time  ██████████████████████████████████████████████████ 51.5%
forward    ████████████ 11.7%
backward   █████████████ 13.3%
validation ████████████████ 16.3%
h2d        ████ 3.6%
opt+loss   ███ 3.4%
```

GPU compute: 72.0s (28.3%) | GPU 等效利用率: **~28%**

### 3.2 bs=48, nw=4（3 epoch 均值，总时间 211.0s）

| 阶段 | 时间 (s) | 占比 | 说明 |
|------|----------|------|------|
| **data_time** | **144.2** | **68.3%** | DataLoader 等待（占比上升因 GPU compute 大幅下降） |
| forward | 6.7 | 3.2% | 批次数从 558→186，fwd 累计时间锐减 |
| backward | 8.3 | 3.9% | 同上 |
| validation | 36.8 | 17.4% | 验证集固定开销 |
| h2d | 8.4 | 4.0% | 总数据量不变，h2d 基本持平 |
| optimizer | 5.2 | 2.5% | 优化器步数从 558→186，累计时间下降 |
| loss | 0.6 | 0.3% | |

```
data_time  ██████████████████████████████████████████████████████████████████ 68.3%
forward    ███ 3.2%
backward   ████ 3.9%
validation █████████████████ 17.4%
h2d        ████ 4.0%
opt+loss   ███ 2.8%
```

GPU compute: 20.8s (9.9%) | GPU 等效利用率: **~10%**

### 3.3 对比分析

| 指标 | bs=16, nw=4 | bs=48, nw=4 | 变化 |
|------|------------|------------|------|
| 批次数 | 558 | 186 | -67% |
| data_time | 130.8s | 144.2s | +10% |
| GPU compute (fwd+bwd+opt+loss) | 72.0s | 20.8s | **-71%** |
| 总 epoch 时间 | 254.1s | 211.0s | **-17%** |
| data_time 占比 | 51.5% | 68.3% | — |
| GPU 利用率 | ~28% | ~10% | — |

核心洞察：**nw=4 下 data_time 基本恒定（130-147s），增大 bs 的核心收益是减少 GPU forward/backward/optimizer 的累计调用次数**（558→186→140 次），每次调用有固定开销（kernel launch、sync 等）。bs=48 比 bs=16 节省 51.2s GPU compute，仅多花 13.4s data_time，净收益 37.8s/epoch。

data_time 占比从 51.5% 升至 68.3% **不是退化**——是因为 GPU compute 缩减幅度远超 data_time 增幅，总时间反而下降。瓶颈仍是 DataLoader，但绝对值已在可接受范围。

---

## 4. DataLoader 确认为主要瓶颈

### 证据链

| 证据 | 数据 |
|------|------|
| 最优配置 data_time% | 51.5% (nw=4) |
| 最差配置 data_time% | 80.3% (nw=0) |
| GPU 等效利用率 | ~28% |
| 瓶颈标记 | 所有 5 配置均标记为 "dataloader" |
| frames/s (data) | 88-341 fps（远低于 GPU 推理 FPS 88.5） |

### 瓶颈根因

每 epoch 需处理 44,640 帧，每帧执行：
```
cv2.imread(JPEG, 20-50KB) → cv2.cvtColor(BGR→RGB) → cv2.resize(→224×224)
```
- **磁盘 I/O**: 44,640 × ~35KB(avg JPEG) ≈ 1.6 GB/epoch
- **JPEG 解码**: CPU 密集型，cv2.imread 内部调用 libjpeg-turbo
- **颜色转换 + 缩放**: CPU 操作，每次约 0.1-0.2ms

在 nw=4 下，4 个 worker 进程并行执行上述流水线，达到 341 fps 的数据吞吐。瓶颈已从纯 CPU 转向 **磁盘随机读取 + CPU 解码混合**。

---

## 5. 缓存机制实现方案与代码位置

### 5.1 设计思路

预解码每帧图像（cv2.imread → BGR2RGB → resize → target_size），缓存为 `.npy` 格式的 uint8 RGB 数组，后续访问直接 `np.load()` 替代 cv2 流水线。

### 5.2 代码位置

| 文件 | 修改内容 |
|------|---------|
| `src/dataset_real.py:9` | 新增 `import hashlib` |
| `src/dataset_real.py:84-88` | `__init__` 新增 `cache_dir` 参数 |
| `src/dataset_real.py:281-311` | 新增 `_cache_path_for()` 和 `_load_or_decode()` 方法 |
| `src/dataset_real.py:347-348` | `__getitem__` 中替换 inline cv2 解码为 `_load_or_decode()` |
| `tools/autoresearch/run_trial_minimal.py:148` | 从 config 读取 `cache_dir` |
| `tools/autoresearch/run_trial_minimal.py:184,199,242` | 传递 `cache_dir` 给 3 处 Dataset 构造 |
| `configs/autoresearch/_cache_bs16_nw4.json` | 缓存测试配置 (cache_dir="D:/dualvcod/data/cache/frames") |

### 5.3 核心实现

```python
def _load_or_decode(self, fpath):
    if self.cache_dir:
        cache_path = self._cache_path_for(fpath)  # MD5-based filename
        if os.path.exists(cache_path):
            return np.load(cache_path)  # Cache hit: fast I/O read
    
    # Cache miss: full decode pipeline
    img = cv2.imread(fpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (self.target_size, self.target_size))
    
    if self.cache_dir:
        # Atomic write via temp file + os.replace
        tmp_path = cache_path + ".tmp"
        np.save(tmp_path, img)
        os.replace(tmp_path, cache_path)
    
    return img
```

### 5.4 缓存效果

| 指标 | No-Cache (nw=4) | Cache Epoch 2 (read) | 变化 |
|------|----------------|---------------------|------|
| data_time | 130.8s | 398.1s | **+204% (恶化)** |
| total epoch | 254.1s | 526.8s | **+107%** |
| frames/s | 341.3 | 112.1 | **-67%** |

**缓存未达预期效果，反致吞吐下降 3 倍。** 原因：

1. **I/O 膨胀**: `.npy` 文件 (147KB/帧) 比 JPEG 源文件 (20-50KB) 大 3-7 倍，每 epoch I/O 量从 ~1.6GB 增至 ~6.6GB
2. **小文件随机读**: 每 batch 读取 80 个独立 `.npy` 文件（共 11.5MB），4 worker 并发产生严重磁盘随机 I/O
3. **通道瓶颈确认**: 当前流水线瓶颈在**磁盘 I/O 带宽**而非 CPU JPEG 解码，cv2.imread 已高度优化

### 5.5 可能的改进方向（未实施）

- **RAM 预加载**: 将全部 44,640 帧 (~6.6GB) 预加载至 CPU RAM，消除每 epoch 的磁盘 I/O（需要 ≥32GB 系统内存）
- **LMDB/Memmap**: 使用内存映射数据库，利用 OS 页缓存
- **Batch-level 缓存**: 缓存整个 batch 的 tensor 而非单帧，减少文件数和随机 I/O
- **更快的存储**: NVMe SSD 可提供 3-7 GB/s 顺序读，显著缓解 I/O 瓶颈

---

## 6. 推荐后续默认训练配置

### 推荐配置

```json
{
  "batch_size": 48,
  "train_batch_size": 48,
  "eval_batch_size": 64,
  "num_workers": 4,
  "loss_weights": {
    "smooth_l1": 1.0,
    "giou": 1.0,
    "center": 0.0,
    "use_diou": true
  }
}
```

### 理由

| 参数 | 选择 | 依据 |
|------|------|------|
| batch_size | **48** | 比 bs=16 节省 71% GPU compute（72s→21s），总 epoch 缩短 17%（254s→211s）；与 bs=64 持平但 GPU 内存更低（2.30 vs 3.06 GiB） |
| num_workers | **4** | 相比 nw=2 提升 95% 数据吞吐，是吞吐的第一主导因子 |
| eval_batch_size | 64 | 验证集较小 (1188 样本)，64 已足够 |
| cache_dir | **null** | 帧级 .npy 缓存恶化性能 3×（I/O 膨胀），不建议启用 |

### 预期训练时间

30 epoch 预计耗时：**211s/epoch × 30 = 6,330s ≈ 1.76 小时**（比 E-31 的 2.6h 节省约 32%）。

### 吞吐量参考

| 配置 | Frames/s | Epoch(s) | 30ep 预计 | vs E-31 |
|------|----------|----------|-----------|---------|
| E-31 (bs=16, nw=2) | 175 | 313 | 2.6h | baseline |
| bs=16, nw=4 | 341 | 254 | 2.1h | -19% |
| **bs=48, nw=4** | **310** | **211** | **1.76h** | **-32%** |
| bs=64, nw=4 | 304 | 212 | 1.77h | -32% |

---

## 附录：实验环境

- **GPU**: NVIDIA GeForce RTX 4090 (24 GiB VRAM)
- **CPU**: 未记录（推测 ≥8 核）
- **存储**: 推测 SATA SSD 或 HDD（cache 写入速度仅 ~23 MB/s 表明非 NVMe）
- **PyTorch**: CUDA available, autocast FP16
- **模型**: MicroVCOD, MobileNetV3-Small backbone, 1,411,684 params
- **数据**: MoCA + MoCA_Mask + CAD, 8,926 train samples, T=5 temporal frames, 224×224 input

---

*报告自动生成时间: 2026-05-11T01:55*
*Profiler 版本: tools/autoresearch/profiler.py (fine-grained v2)*
