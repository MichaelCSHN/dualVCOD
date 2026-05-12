# dualVCOD / MicroVCOD 项目上下文压缩包

**生成时间**：2026-05-07  
**用途**：用于另起新对话时快速恢复项目上下文。  
**安全说明**：本文不包含任何 API Key、Token、数据集真实文件、模型权重或私密凭据。

---

## 0. 项目当前身份与治理原则

本项目现在由 ChatGPT 作为“项目 CEO/CTO”参与决策与审查，Claude Code / DeepSeek / Gemini 类工具仅作为执行器使用。  
Claude 不再拥有独立判断权，尤其不能自行宣布：

- `SAFE TO TRAIN`
- `PAPER READY`
- `FINAL VERDICT`
- `REVIEWER-READY`
- `SOTA`
- `ROBUST`
- `VALID RESULT`

所有关键结论必须由 ChatGPT 审查后确认。

### 当前最高优先级

> 建立一个可审计、无数据泄漏、可复现、轻量级 bbox-only Video Camouflaged Object Detection / Localization 项目。

项目方向不再是“复现原文 GreenVCOD”，而是：

> **MicroVCOD：轻量级 bbox-level 视频伪装目标定位原型 / baseline。**

---

## 1. 项目定位

### 1.1 当前项目名称

推荐使用：

```text
MicroVCOD
```

全称可写为：

```text
Micro Video Camouflaged Object Detection / Localization
```

或论文方向写成：

```text
Lightweight BBox-level Video Camouflaged Object Localization
```

### 1.2 不建议继续使用的表述

不建议继续把方法直接称为：

```text
GreenVCOD
```

原因：原文已有《Green Video Camouflaged Object Detection》/ GreenVCOD，继续用同名容易造成方法归属与抄袭误解。

### 1.3 与原文 GreenVCOD 的关系

原文 GreenVCOD 的核心是：

- Video Camouflaged Object Detection；
- image-level COD + temporal refinement；
- Green Learning / XGBoost / prediction map refinement；
- mask / prediction-map style 输出；
- 与 bbox-only neural detector 不是同一实现范式。

本项目与原文关系应表述为：

> 本项目受 VCOD 中多帧/时序邻域思想启发，但不是原文 GreenVCOD 的复现，不复用其代码；本项目采用 PyTorch + MobileNetV3/FPN + bbox head 的轻量神经网络实现，目标是 bbox-only 视频伪装目标定位。

### 1.4 “Temporal Neighborhood” 的处理

“Temporal Neighborhood” 不是原文 GreenVCOD 首创概念。  
视频理解中利用邻近帧/时间窗口是通用思想。

本项目当前模块实测更接近：

```text
multi-frame pooling / multi-frame visual aggregation
```

而不是严格的：

```text
temporal dynamics modeling
```

原因：red-team 发现 forward 与 reversed temporal order 几乎一致，说明当前模块对时间方向不敏感。

论文或报告中应避免使用：

```text
true temporal dynamics
motion-aware temporal modeling
temporal lock-on proof
```

更稳妥表述：

```text
lightweight multi-frame aggregation
multi-frame visual consistency
bbox-level multi-frame pooling
```

---

## 2. 当前 GitHub 仓库

仓库地址：

```text
https://github.com/MichaelCSHN/dualVCOD
```

后续 ChatGPT 只能检查 GitHub 远程仓库，无法看到 Claude 本地未 push 的修改。因此需要严格控制 commit/push 节奏。

### 2.1 Git 节奏原则

本地可以小步 commit，但远程 push 必须对应可审查里程碑。

推荐分支：

```text
main                         # 稳定状态
dev/clean-leak-fix            # 数据泄漏修复与验证
dev/smoke-train               # 1 epoch smoke training
dev/clean-retrain             # full clean retraining
dev/baselines                 # baseline 实验
dev/paper-integrity           # 论文审计材料
```

### 2.2 每次 push 前强制要求

Claude 必须：

1. `git status`
2. 运行 repo safety audit
3. 确认没有数据集、权重、API key、大文件
4. 生成 `reports/report_yyyymmddhhmm.md`
5. 报告：
   - 当前分支；
   - commit hash；
   - 修改文件列表；
   - 是否涉及训练；
   - 是否涉及数据 split；
   - leak verification 是否通过；
   - repo safety audit 是否通过；
   - 是否存在未提交/未 push 修改。

### 2.3 禁止事项

- 禁止 `git push --force`
- 禁止提交数据集
- 禁止提交 checkpoints / `.pth` / `.pt` / `.ckpt`
- 禁止提交 `.env`、API key、token
- 禁止提交 `D:\ML\`、`D:\dualVCOD\`、`C:\Users\...` 等敏感本地路径，除非是内部诊断摘要
- 禁止把旧 `mIoU=0.8705` 当作有效结果宣传

---

## 3. 重大历史：0.8705 指标已作废

### 3.1 旧结果

Phase 2.1 曾报告：

```text
mIoU = 0.8705
Recall@0.5 = 0.9340
```

该结果来自：

```text
best_greenvcod_box_miou.pth
```

### 3.2 red-team 审查结论

该结果已被判定为无效：

> MoCA validation 28 videos / 1,188 temporal windows 全部进入 training ConcatDataset，validation videos 收到了梯度更新。

这属于致命数据泄漏。

### 3.3 当前处理原则

- 不得使用旧 leaked checkpoint；
- 不得从旧 checkpoint resume；
- 不得把旧 visual audit 图表当作有效证据；
- 可将旧结果保留在 `reports/archive_invalid/` 作为问题追溯；
- README 中必须明确说明 `mIoU=0.8705` 无效；
- 论文中如需提及，只能作为“泄漏审计案例”，不是有效结果。

---

## 4. 已发现并修复的数据泄漏问题

### 4.1 第一层：MoCA 内部泄漏

原 bug：

```text
split_by_video() 生成了 train_idx / val_idx
但 train_idx 没有用于训练集
joint_train_ds 包含 full MoCA
```

已修复目标：

```text
MoCA train-only = Subset(MoCA, train_idx)
MoCA val = Subset(MoCA, val_idx)
```

最新报告显示：

```text
MoCA train canonical_video_ids: 113
MoCA val canonical_video_ids: 28
MoCA internal overlap: 0
```

### 4.2 第二层：MoCA_Mask 与 MoCA Val 同源视频泄漏

关键认知：

> 泄漏判断看视觉内容，不看 annotation format。

因此：

```text
MoCA bbox 与 MoCA_Mask mask 若包含同源视频/同源帧，仍属于泄漏风险。
```

已发现 MoCA_Mask 与 MoCA Val 有 13 个同源视频重叠，并已要求排除：

```text
cuttlefish_1
cuttlefish_4
flatfish_1
flatfish_2
flounder_3
flounder_6
leaf_tail_gecko
peacock_flounder_0
polar_bear_0
rusty_spotted_cat_1
seal
sole
spider_tailed_horned_viper_2
```

最新 `verify_leak_fix.py` 报告显示：

```text
MoCA_Mask total canonical_video_ids (unfiltered): 71
Raw overlap with MoCA Val: 13
MoCA_Mask excluded by filter: 13
MoCA_Mask retained: 58
```

### 4.3 第三层：CAD 与 MoCA Val

最新报告显示：

```text
CAD total canonical_video_ids: 9
CAD vs MoCA Val overlap: 0
CAD excluded: 0
```

### 4.4 当前 leak verification 状态

最新 canonical-video-level 检查：

```text
MoCA internal overlap: 0
MoCA_Mask vs MoCA Val: 13 detected and excluded
CAD vs MoCA Val: 0
JointTrain vs Val post-filter canonical_video_id overlap: 0
```

阶段性结论：

```text
Canonical-video-level leak check: PASS
Known MoCA/MoCA_Mask cross-dataset leakage: RESOLVED
Safe to start controlled smoke training: YES
Publication-grade leakage audit: NOT YET
```

---

## 5. 尚未完成的泄漏审计

虽然 canonical_video_id 检查已通过，但还不能称为论文级最终无泄漏审计。仍需补：

```text
1. train_frame_paths ∩ val_frame_paths = empty
2. train_image_hashes ∩ val_image_hashes = empty
3. perceptual hash overlap check, if feasible
4. final DataLoader manifest export
5. confirm train.py and verify_leak_fix.py use exactly the same filtering logic
```

如果 hash 检查暂时做不到，报告必须写：

```text
hash_check = pending
```

不得写：

```text
publication-grade leak audit complete
```

---

## 6. 当前训练许可状态

### 6.1 可以做

可以允许：

```text
1 epoch smoke training
```

目的仅限：

- 检查 CUDA / 4090 正常；
- 检查 loss 是否下降；
- 检查 DataLoader 与 filtering logic 是否生效；
- 检查 checkpoint 命名不覆盖旧 leaked checkpoint；
- 检查报告生成流程。

### 6.2 不可以做

暂不允许：

```text
full clean retraining
paper assets generation
IR fusion
正式论文结果写作
宣传有效性能
```

full retraining 之前必须确认：

```text
train.py 实际使用 filtered MoCA_Mask
verify_leak_fix.py 与 train.py filtering logic 一致
smoke training 通过
训练前后 leak verification 均通过
```

---

## 7. 环境状态

Claude CLI / Claude Code 现已可运行在：

```text
conda env: dualvcod
cwd: D:\dualVCOD
Python: D:\anaconda3\envs\dualvcod\python.exe
Python version: 3.10.20
PyTorch: 2.6.0+cu124
torch.cuda.is_available(): True
GPU: NVIDIA GeForce RTX 4090
VRAM: ~24 GB
CUDA version in PyTorch: 12.4
NVIDIA driver: 591.86
```

### 7.1 Claude CLI 连接 DeepSeek 的经验

曾遇到的问题：

- VS Code Claude 插件旧 GLM 5.1 / BigModel 配置污染；
- `.claude` 与 `.claude.json` 残留；
- `ANTHROPIC_AUTH_TOKEN` 与 `ANTHROPIC_API_KEY` 冲突；
- `deepseek-v4-pro[1m]` 与 `deepseek-v4-pro` 模型名混乱；
- `sk-ant` Anthropic key 与 DeepSeek key 混淆。

最后可行方案以用户实测为准。当前已可使用 Claude CLI 运行环境诊断，并识别 4090。

### 7.2 关键安全提醒

不要把 API key 写进 repo。  
不要把本地 `.ps1` 启动脚本提交。  
如需保存启动脚本，命名为：

```text
start_claude_deepseek.local.ps1
```

并在 `.gitignore` 中加入：

```gitignore
*.local.ps1
.env
*.env
```

---

## 8. Claude 执行安全约束

以后给 Claude 的工单必须包含安全约束。固定原则如下：

### 8.1 禁止事项

1. 禁止擅自启动训练；
2. 禁止擅自删除重要文件；
3. 禁止提交 API key / token / `.env`；
4. 禁止提交数据集和模型权重；
5. 禁止 force push；
6. 禁止自行宣布论文级结论；
7. 禁止把不同标注格式当作“非泄漏”理由；
8. 禁止使用旧 `mIoU=0.8705` 作为有效结果；
9. 禁止进入 IR fusion，除非明确允许；
10. 禁止生成 paper assets，除非明确允许。

### 8.2 每次改代码前

必须：

```text
git status
说明将修改哪些文件
不得进行无关重构
```

### 8.3 每次训练前

必须确认：

```text
MoCA train ∩ MoCA val = empty
JointTrain canonical_video_id ∩ Val canonical_video_id = empty
train frame paths ∩ val frame paths = empty
train image hashes ∩ val image hashes = empty, or hash_check = pending
```

### 8.4 每次 push 前

必须：

```text
repo safety audit
检查 staged files
确认无数据集/权重/key/大文件
生成 reports/report_yyyymmddhhmm.md
```

### 8.5 每次异常

必须：

```text
停止执行
报告错误
不得反复试错
不得自行换模型、换路径、换数据集继续跑
```

---

## 9. 下一阶段建议路线

### Stage A — 当前状态确认

目标：

```text
确保 GitHub repo 上有最新 leak fix 与 verify report
```

动作：

1. Claude commit/push 最新 `verify_leak_fix.py` 与报告；
2. ChatGPT 检查 GitHub repo；
3. 不训练。

### Stage B — smoke training

目标：

```text
1 epoch smoke training only
```

动作：

1. 训练前运行 verify_leak_fix.py；
2. 确认 train.py 使用 filtered MoCA_Mask；
3. 从 clean init / ImageNet pretrained 开始；
4. 不加载旧 leaked checkpoint；
5. 运行 1 epoch；
6. 生成 smoke report；
7. 训练后再次运行 verify_leak_fix.py；
8. commit/push 到 `dev/smoke-train`；
9. ChatGPT 检查 GitHub。

### Stage C — full clean retraining

条件：

```text
Stage B 通过
```

动作：

1. 固定 config / seed / manifest；
2. full clean retrain；
3. 记录 run registry；
4. 不 push checkpoint；
5. push reports / metrics / config / manifest hash；
6. ChatGPT 审查。

### Stage D — baseline registry

必须补：

```text
center prior
train-set mean box
previous-frame GT propagation
first-frame GT propagation
single-frame model
T=1 repeated-frame model
T=5 forward
T=5 reversed
temporal shuffled
non-overlapping clips
unique-frame evaluation
per-video average
```

原因：旧 red-team 发现 GT propagation baseline 很强，甚至在泄漏模型上高于模型本身。

### Stage E — paper_integrity

建立：

```text
paper_integrity/
  00_ai_use_statement.md
  01_human_responsibility_statement.md
  02_development_timeline.md
  03_data_split_manifest_summary.md
  04_leak_audit_history.md
  05_training_run_registry_template.csv
  06_baseline_registry_template.csv
  07_known_limitations.md
  08_reproducibility_checklist.md
```

---

## 10. AI 参与开发与论文署名

ChatGPT / Claude / Gemini / DeepSeek 不能作为论文作者署名。  
如果投稿，应在 Acknowledgement / AI Use Statement 中披露 AI-assisted tools 的使用。

推荐声明：

```text
The authors used AI-assisted tools, including ChatGPT and Claude Code, for code drafting, debugging support, experiment-log summarization, and manuscript language refinement. All experimental designs, data splits, code changes, results, interpretations, and final manuscript claims were reviewed and approved by the human authors, who take full responsibility for the work.
```

项目内部原则：

```text
AI-assisted development, human-supervised scientific validation.
```

---

## 11. 当前已知风险

### 11.1 数据风险

- canonical_video_id 检查已过；
- path/hash 检查仍需补；
- train.py 与 verify_leak_fix.py filtering logic 必须保持一致；
- training-side duplicate canonical_video_ids 需要解释：
  - MoCA train-only 113
  - MoCA_Mask filtered 58
  - CAD 9
  - component sum = 180
  - union unique = 126
  - 需要列出 training-side duplicates，并确认不在 Val。

### 11.2 模型风险

- 当前模块更接近 multi-frame pooling；
- 不应声称 true temporal dynamics；
- reversed order baseline 必须保留；
- T=1 vs T=5 必须保留。

### 11.3 评价风险

- GT propagation baseline 很强；
- 如果模型不显著超过 propagation / prior baseline，论文主张要降级；
- bbox-level mIoU 不能和 mask-level VCOD mIoU 直接比较；
- clip-level sliding window 可能重复统计，需要 unique-frame / non-overlap / per-video 评估。

### 11.4 论文风险

- 不能包装旧结果；
- 不能声称 SOTA；
- 不能隐瞒 AI-assisted development；
- 必须保留 data split manifest 与 leak audit history。

---

## 12. 给 Claude 的下一条建议工单

```text
【执行安全约束｜必须遵守】
不得 full train，不得生成 paper assets，不得进入 IR fusion，不得使用旧 leaked checkpoint，不得自行宣布最终结论。

当前任务：准备 smoke training 前的最终代码与数据链路确认。

请执行：

1. git status，报告当前分支和未提交文件。
2. 检查 tools/train.py 是否实际使用与 verify_leak_fix.py 相同的 filtering logic：
   - MoCA 使用 train_idx；
   - MoCA_Mask 排除 13 个 MoCA Val 同源视频；
   - CAD 做 canonical_video_id 检查；
   - 不允许 full MoCA_Mask 直接进入 joint_train_ds。
3. 导出最终 training manifest summary：
   - MoCA train videos；
   - MoCA val videos；
   - MoCA_Mask excluded videos；
   - MoCA_Mask retained videos；
   - CAD retained/excluded；
   - JointTrain canonical_video_id union；
   - Val canonical_video_id set；
   - JointTrain ∩ Val。
4. 解释为什么 component unique sum = 180，但 JointTrain union unique = 126；
   - 列出 training-side duplicate canonical_video_ids；
   - 确认这些 duplicate ids 不在 Val。
5. 增加或运行 frame path overlap check：
   - train_frame_paths ∩ val_frame_paths = empty。
6. 如果 hash check 暂未实现，报告 hash_check = pending。
7. 不启动 full training。最多只准备 1 epoch smoke training 命令，不执行，除非收到明确许可。
8. 在 reports 目录生成 "report_yyyymmddhhmm.md"（须使用 UTF-8 编码），汇报上述结果。
9. commit 到 dev/smoke-train-prep 或 dev/clean-leak-fix，并 push。
10. push 后暂停，等待外部审查。
```

---

## 13. 关键一句话

当前项目不是“高分模型已经成功”，而是：

> **一个经历过数据泄漏审计、正在转向 clean retraining 的轻量级 bbox-only VCOD 项目。它的潜力在于把视频伪装目标定位任务做干净、做轻、做可复现，而不是依赖旧的虚高指标。**
