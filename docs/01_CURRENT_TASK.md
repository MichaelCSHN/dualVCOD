# 当前任务：M1-M2 MicroVCOD Baseline & 环境搭建

## 1. 任务目标
完成项目“破冰”：建立正确的 Conda 视频开发环境，并跑通第一版基于多帧输入的极简 BBox 检测和评估流水线。

## 2. 执行范围 (Must Do)
1.  **环境准备 (Critical)**：建立 conda 环境 `dualvcod`。请直接从现有的 `dualcod-cu` 环境克隆（执行 `conda create --name dualvcod --clone dualcod-cu`），以复用 PyTorch 和 CUDA 依赖。后续所有开发必须在该环境内进行。
2.  **数据流构建**：编写轻量级的视频帧读取器（Dataloader 雏形），能够读取短视频片段（如 5 帧滑动窗口）并转换为 Tensor。
3.  **假数据评估器**：编写 `eval/eval_video_bbox.py`，实现多帧时序下的 BBox 评估指标（如 `mean bbox IoU`, `Recall@0.5`）。
4.  **基准测试桩 (Stub)**：编写极简 MicroVCOD_Lite Model，接收 $(B, T, C, H, W)$ 格式的输入，输出对应帧的 BBox 坐标，证明整个数据前向流程和评估管线畅通。

## 3. 测试与验证 (Acceptance Criteria)
不需要真实 MoCA 或 CAD 数据集，请使用随机生成的 numpy array (Synthetic Data) 模拟视频帧完成 Smoke Test：
*   Dummy 模型能够吃进多帧张量，且不报显存或维度错误。
*   评估脚本能够正确对比预测出的 BBox 和 GT BBox 序列，输出平均 IoU 和 FPS。

## 4. 产出要求
完成后，在 `reports/` 生成名为 `phase1.1-yyyymmddhhmm.md` 的简短报告，列出新增的脚本清单、环境创建确认及 Dummy 测试的运行结果。等待下一步正式实现 TN 模块的指令。