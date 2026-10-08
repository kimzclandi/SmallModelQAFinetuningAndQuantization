# CE/KL 参数梯度诊断：先区分损失数值与更新方向

上一轮同条件三臂训练未验证门控蒸馏相对 gold-SFT 的质量优势。完整 KL 三个 seed 均在 242/242 步发生总梯度裁剪，门控组为 238/239/233 步；这些记录没有拆出 CE 与 KL 的参数梯度，所以不能由标量 loss 占比或裁剪频次推断方向冲突、梯度主导或质量退化原因。

本轮新增只用 TRAIN 的诊断入口。在完全相同的模型参数上，分别计算加权 CE、完整 KL、门控 KL 及两个合成目标的真实 autograd 梯度，记录每个 LoRA 参数块的范数、内积、余弦和梯度线性组合重建误差。该功能不执行优化器更新，不生成答案，不读取已消耗的留出集。

## 执行前固定的范围

[冻结协议](../configs/ce-kl-gradient-v1.json)规定：从已有 242 条 TRAIN 中，在可答、不可答两类分别按固定 ID 哈希顺序取前 8 条，共 16 条。三个固定 seed，每个 seed 检查初始化与 gold/full-KD/gated-KD 三个已保存末态，共 12 个参数状态、192 个 probe。

主诊断是初始化时的 48 个 probe；末态的 144 个 probe 仅作补充。16 道题来自 12 个不同文段，并在多个 seed/状态重复，不能当成 192 个独立样本，也不能把不同末态之间的梯度差异单独归因为门控。

- 固定 CPU/FP32、8 线程、T=2；参数状态测量使用 eval 模式，已有 LoRA dropout 为 0。
- 使用已有且身份绑定的 teacher cache 和 adapter，不重新生成或训练。
- 加权梯度分别为 `g_CE = 0.5 ∇CE`、`g_full = 2 ∇KL_full`、`g_gate = 2 ∇KL_gate`，门控分母始终是全部监督 token 数。
- 这里的 CE 是混合目标中的 0.5 倍 CE 分量，不是权重为 1 的 gold-only 训练目标；范数大小不得混用这两个分母。
- 分别用合成目标的独立 autograd 结果对照 `g_CE + g_full`、`g_CE + g_gate`；预先固定 `atol=1e-5, rtol=1e-4`，这只是 FP32 数值契约，不是质量门槛。
- 梯度余弦遇零向量记为 null，不能当成通过。LoRA B 初始化为零时，A 的梯度为零是预期情况。
- 总预算 20 分钟、进程 RSS 16 GiB、输出 256 MiB；逐阶段/样本检查，不是对阻塞调用的硬实时中断。失败保存新目录，禁止覆盖或选择性重跑。

FP16 teacher log-probability 保留原质量总和 `m = sum(q_teacher_cached)`，没有在本轮悄悄重归一化。对学生 logits 的未加权 KL 梯度为 `(m × p_student_T − q_teacher_cached) / T`，再按监督 token 数归约；解析参考必须保留 m，不能假设它严格为 1。

## 公开记录与可复算边界

原计划为所有 probe 保留各参数块的 3×3 Gram 矩阵、梯度摘要、层/模块聚合和重建误差。另按执行前固定规则，保留每个状态第一道 TRAIN 题、第 0 层 q_proj 的 LoRA B 参数块的五份真实梯度，计划共 12 个小型 NPZ 文件（本次在首 probe 失败，仅实际写入 1 个）；不保存模型权重或 teacher 完整词表张量。

离线 CI 可重算全部 Gram 汇总，并从这些公开块的真实向量重算局部内积、摘要及组合关系。其余参数块只有 Gram 和摘要，不能宣称已独立重演它们的完整 autograd；整个模型计算的复验需要执行者已有、身份一致的本地模型和 cache。前后参数摘要用于检查该诊断没有更新模型。

## 判断方式与限制

两个事前声明的描述性问题是：初始化的 48 个 probe 中，完整 KL 加权梯度范数大于 CE 的比例是否严格过半；teacher top-1 一致性门控是否在所有这些 probe 上消除了负内积。任何负内积都是后者的反例；存在零向量时须单列未定义，不能直接给出肯定结论。

这些问题描述固定 TRAIN 样本和固定参数状态上的梯度几何。它们不是显著性检验，不代表历史 AdamW 更新轨迹，也不证明对留出集质量的因果影响。诊断结论不能改写上一轮“未通过采用门槛”的质量结论，不产生新的压缩比或速度结果。

本轮设计、实现、执行和证据整理由 Codex/AI 辅助完成；Qwen、PyTorch、PEFT、LoRA 与 KL 方法归其上游作者。新代码与协议须先提交，再执行真实模型诊断；本次真实执行结果见下方。

## 实际执行：首个 probe 正确性门槛失败

协议与代码先提交于 `e63d128bbfec22377f16db96249d610b9eb55471`，随后于 2026-10-09（新加坡时间）执行一次 CPU 诊断。首次初始化状态、seed 20261009、第一条固定 TRAIN 输入即触发数值门槛；程序保存记录后退出，没有执行剩余 191 个 probe。

实际是 **1 个失败 probe，0 个通过 probe，0 个完整状态**，不是完成 192 组实验。原始目录逐字节保存于 [reports/ce-kl-gradient-v1](../reports/ce-kl-gradient-v1/)，包括 [run](../reports/ce-kl-gradient-v1/run.json)、[失败 probe](../reports/ce-kl-gradient-v1/probes/000.json)、[状态前后摘要](../reports/ce-kl-gradient-v1/states/initial-20261009.json)及冻结源码。

| 检查 | 实际结果 |
|---|---|
| 旧版 full/gated 合成 loss 对照 | 完全一致，最大绝对误差 0 |
| CE logit 梯度 vs 独立 FP64 公式 | 通过预先固定门槛 |
| full/gated KL logit 梯度 | 各 1 元素超差，最大绝对误差 5.5133e-5 |
| 两个合成目标的 logit 梯度 | 各 1 元素超差，最大绝对误差 5.5080e-5 |
| 分别求导后相加 vs 合成目标求导 | 第 0 层 v_proj LoRA B 两目标各 6 元素、第 6 层各 2 元素超差；最大绝对误差分别 3.1829e-5、1.6864e-5 |
| 参数状态、输入及源码身份 | 参数前后摘要相同；输入及源码完成运行前校验，失败路径未执行最终全量输入检查 |

logit 失败位置是监督 token 位置 2、词表索引 198。3 个监督位置的 teacher top-1 均等于 gold，所以本 probe 的完整和门控 KL 完全相同，不能据此比较门控效果。

该失败说明本次 FP32 梯度观测未满足预先声明的数值契约；**不等于已证明旧训练错误、PyTorch 缺陷或质量退化的根因**。没有放宽容差、挑换输入、重跑模型或更改 teacher 概率质量。预先声明的两个描述性假设均未完成检验，不引用失败 probe 的范数/余弦作为机制结论。

总运行记录约 4.52 秒，最后一次阶段检查记录的进程高水位 RSS 为 3,334,144,000 字节；失败后没有再次采样，不能代表本次执行的最终 RSS 峰值。它不是吞吐、峰值设备内存或性能结论。训练更新、自由生成、留出集预测均为 0。

### 公开证据的缺口

预先选择公开的原始数组是第 0 层 **q_proj** LoRA B，这个块通过检查。失败的 **v_proj** 和 logit 张量只保存了误差统计与参数梯度摘要，未保存其完整原始数组，因此本次离线 CI **不能重演失败坐标或独立确定根因**。离线验收核对失败回执、冻结来源与前后状态，并从实际公开的 q_proj 向量重算该通过块；验收成功只表示失败归档一致，不表示诊断正确性通过。

完整运行验收器 [verify_gradient_diagnostics.py](../scripts/verify_gradient_diagnostics.py) 面向未来完整归档，本次真实失败目录应被它拒绝。专用 [verify_gradient_failure.py](../scripts/verify_gradient_failure.py) 明确返回 `diagnostic_complete=false`。这两个状态不可混用。

```sh
python scripts/verify_gradient_failure.py
python -m pytest -q tests/test_gradient_diagnostics.py tests/test_verify_gradient_diagnostics.py tests/test_verify_gradient_failure.py
```

### 复现实测命令

以下命令要求已有、哈希一致的本地模型、teacher cache、九组 adapter 与固定依赖；不自动下载。三个路径变量分别指本地 HF hub 缓存、已有教师缓存和已有训练输出根目录。必须选择不存在的新输出目录，禁止覆盖本次原始记录；如修改实现或数值协议，需要新版本并先提交。

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m qa_lab.gradient_diagnostics \
  --protocol configs/ce-kl-gradient-v1.json \
  --cache "$TEACHER_CACHE" --training-root "$TRAINING_ROOT" \
  --artifact data/coverage-v3-gold242 --data-dir data/complexity-v1 \
  --student-config configs/teacher-gated-confirmation-v1/student.json \
  --cache-dir "$MODEL_HUB_CACHE" --output work/new-gradient-reproduction
```

实际依赖为 PyTorch 2.8.0、Transformers 4.56.2、PEFT 0.17.1、NumPy 2.5.3，Python 3.12。下一项必要改进是先完善失败张量留存，再以独立版本定位数值差异；本次没有宣称修复或新的压缩收益。
