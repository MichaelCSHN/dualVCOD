# dualVCOD: 核心宪章与当前状态 (CEO/CTO Directives)

> **核心原则**：以模型运转为中心，快速迭代，拒绝流程性空转。单张 RTX 4090 必须能跑通全流程。部署端只认 BBox，不需要实例级分割。
>
> **核心对标**：[arXiv:2501.10914v1] Section 3 (GreenVCOD) — 仅借鉴 temporal window 概念。本项目为独立 PyTorch 实现，命名为 **MicroVCOD**。零代码复用，纯梯度下降训练。

## 1. 项目最终目标
构建一个面向单模/双模（RGB/LWIR）视频流输入的轻量级不显著目标视频检测（VCOD）模型。
*   **硬件约束**：单卡 RTX 4090 训练，边缘设备（如无人机端）推理。
*   **网络输出**：直接输出视频序列每一帧的 BBox（边界框）。不追求像素级 Mask 分割。

## 2. 三阶段路线（当前锁定在 Phase 1）
*   **Phase 1 (进行中)**：实现单模 RGB 视频序列的轻量化 BBox 检测 MVP。核心为独立设计的时空邻域 (TN) 模块用于 BBox 预测。
*   **方法命名**：**MicroVCOD** (Micro Video Camouflaged Object Detection)。
*   **已知限制 (2026-05-07)**：TN 模块为 order-invariant（时序无序），forward 与 reversed 结果完全一致。GlobalAvgPool 消除了时序位置信息——现阶段实际为 multi-frame pooling。
*   **Phase 2 (规划中)**：扩展为可见光-长波红外 (RGB-LWIR) 双模视频检测。
*   **Phase 3 (规划中)**：进一步引入端侧跟踪 (Tracking) 逻辑，优化连续帧的抖动和资源消耗。

## 3. CTO 划定的“红线” (Non-Goals)
在未收到明确指令前，严禁进行以下操作：
1. 严禁计算显式光流 (Optical Flow) —— 坚持轻量化高效理念。
2. 严禁引入笨重的 3D 卷积或大规模 Video Transformer。
3. 严禁输出和评估 Mask（掩码） —— 所有评估指标必须围绕 BBox 展开。

## 4. 架构选型基准
*   **输入**：$T$ 帧连续图像（例如 $T=3$ 或 $5$）。
*   **核心模块**：轻量级主干网络提取空间特征 + multi-frame pooling（当前 order-invariant）用于空间融合。
*   **输出头**：时序融合特征 -> Objectness + BBox Regression。