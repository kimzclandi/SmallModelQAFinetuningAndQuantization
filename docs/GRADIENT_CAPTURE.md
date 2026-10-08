# 梯度失败留证与同输入重放

旧 [梯度诊断](CE_KL_GRADIENT_DIAGNOSTICS.md) 在首个 TRAIN probe 未通过数值门槛，却只归档了预选的、通过检查的 q_proj 梯度。失败的 logit 和 v_proj 数组没有保留，错误摘要不能替代原值重放。

新入口在每次 autograd 返回后、执行数值比较前，原子保存该目标的 logit 导数、FP64 解析参考和全部 96 个 LoRA 参数块的导数。输入另存独立本地目录；中途异常保留已完成阶段。状态分别记录捕获是否完整、模型与输入身份是否仍一致、原数值门槛是否通过。捕获成功不会把旧失败改成成功。

## 固定范围

[协议与第二次执行版本](../configs/ce-kl-gradient-capture-v1-exec2.json) 固定旧研究首个 TRAIN 输入、seed 20261009、初始化参数、CPU/FP32、8 线程及原 `atol=1e-5, rtol=1e-4`。只执行一个 probe，五个目标为加权 CE、full KL、gated KL 及两个合成目标。无优化器更新、自由生成或留出集读取，不恢复 192-probe 研究。预算为 120 秒、进程 RSS 16 GiB、两目录共 128 MiB，按阶段检查而非阻塞调用的硬中断。

源码与协议在执行前提交，运行保存该提交与逐文件 SHA256。沿用既有模型和 teacher cache 的身份校验，并在数值门槛失败后仍执行最终模型、参数、输入和源码审计。两个输出目录必须是仓库 `work/` 下不存在的独立目录。

## 复验与公开边界

- [NumPy 验证器](../scripts/verify_gradient_capture.py) 不加载模型：从完整本地数组重算 logit 门槛、参数梯度哈希、Gram、零向量和原 FP32 组合检查；额外 FP64 加法只用于区分最后一步加法舍入，不替换原门槛。
- [同输入重放](../scripts/replay_private_gradient_capture.py) 使用保存的 logits 与 teacher log-probabilities：复查 FP32 导数逐位相同，并对照 FP64 autograd、独立公式及固定的中间算术分解。该脚本必须与捕获时已提交源码一致。
- **完整真实张量全部留在本地 `work/`。** CE 导数可能暴露目标 token，KL 导数携带教师分布信息；将其命名为“导数”并不构成匿名化。PR 只发布实现、合成故障测试和审查后的数值摘要。公共 CI 的合成验证不能声称重放了未公开的真实数组。
- 本地原始张量可验证本次单输入差异；不恢复旧 v1 未保存的数组，不证明历史训练质量退化原因，也不提供加速结果或个人独立掌握证据。实现与实验由 Codex/AI 辅助完成。

## 本地命令

以下环境变量指已有且哈希匹配的资源，不会下载、转换模型或重新生成 cache。代码和协议须先提交；输出必须新建，禁止覆盖和选择性重跑。

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m qa_lab.gradient_capture \
  --protocol configs/ce-kl-gradient-capture-v1-exec2.json \
  --cache "$TEACHER_CACHE" --training-root "$TRAINING_ROOT" \
  --artifact data/coverage-v3-gold242 --data-dir data/complexity-v1 \
  --student-config configs/teacher-gated-confirmation-v1/student.json \
  --cache-dir "$MODEL_HUB_CACHE" \
  --output work/gradient-capture-new --private-output work/gradient-inputs-new
python scripts/verify_gradient_capture.py --root work/gradient-capture-new
python scripts/replay_private_gradient_capture.py \
  --capture work/gradient-capture-new --private-input work/gradient-inputs-new/inputs.npz \
  --output work/gradient-replay-new.json
```

## 实际执行与结果

首次启动基于 `8cac7d6`，但最后一项验收器重构在冻结提交后写入，准确源码检查在协议阶段拒绝执行：**0 次模型加载、0 个 probe**。保留[失败收据](../reports/ce-kl-gradient-capture-v1/execution-01-rejected.json)，没有覆盖目录。第二版先提交于 `d925dcc3e55c55b939e101dde0fd83c3de0b3171`，再完成唯一一次真实模型首 probe 捕获。[执行摘要](../reports/ce-kl-gradient-capture-v1/execution-02-summary.json)记录全部最终身份检查通过、原数值门槛仍失败。

本地 NumPy 验证器复算 **96 块 × 5 目标、2,703,360 个参数梯度值**，以及 **2,279,040 个 logit 导数与对应 FP64 参考**。旧 v1 保存的全部参数摘要、梯度哈希、Gram、loss 和比较统计与本次逐项相同；这重现了旧记录中的差异，但没有补造旧时未保存的数组。[本地完整重放的标量摘要](../reports/ce-kl-gradient-capture-v1/numpy-replay-summary.json)：

| 检查 | 观察 |
|---|---|
| full/gated KL 及两个合成目标的 logit 梯度 | 各 1 元素失败，零基坐标 `[2,198]`；full KL 最大绝对误差 `5.51326e-5` |
| v_proj LoRA B 梯度组合 | 第 0/6 层各目标分别 6/2 元素失败，与旧记录相同 |
| 仅将最后的两个 FP32 向量相加改用 FP64 复算 | 上述 6/2 失败仍存在，不能仅归因于最后一次向量加法舍入 |
| 参数、源码、输入与模型文件最终检查 | 通过；没有参数更新或梯度缓冲累积 |

[同输入 logit 数值分解](../reports/ce-kl-gradient-capture-v1/same-input-replay-summary.json)只读取保存数组，没有再次加载模型：

- 五目标 FP32 autograd 导数与捕获值全部逐位一致；FP64 解析参考也逐位一致。
- 同输入 FP64 目标/autograd 与独立公式五组均通过，最大绝对误差约 `1.62e-14`。
- 单独使用 FP32 teacher exp 的混合公式最大误差 `3.13e-9`；连同 FP32 log-softmax 前向值重建的混合公式最大误差 `7.99e-6`，均通过原门槛。
- 原 FP32 反向导数与“已观测反向输入 + 已观测前向值、再用 FP64 代数重建”的差异最大为 `6.31e-5`，仍有 1 个失败元素。因此这些记录将排查范围缩小到该 logit 目标的数值反向计算链路；没有锁定具体 kernel、归约或 exp 实现为唯一根因。混合公式是数值分解，不是替代训练实现，也未解释参数 VJP 全链路或留出集质量。

捕获耗时约 5.91 秒，最终阶段记录的进程高水位 RSS 为 3,663,167,488 字节，两本地目录约 42.4 MB。这些数值只用于预算核验，**不是性能基准或设备显存结果**。

## 可复核层级

完整真实张量保存在本机 `work/ce-kl-gradient-capture-v1-exec2` 与 `work/ce-kl-gradient-capture-v1-inputs-exec2`。公开目录仅含审查过的标量结果与[本地文件哈希清单](../reports/ce-kl-gradient-capture-v1/local-artifact-manifest.json)，没有输入、模型权重或真实梯度值。哈希和执行收据本身不证明未公开张量的真实性；外部执行者需持有相同本地数据才能完整重放。

`python scripts/verify_gradient_capture_summary.py` 在 CI 检查公开摘要一致性、范围与原失败状态；合成测试验证完整捕获/重放的拒绝路径。**公共 CI 没有重放本次真实数组。** 完整本地重放也未独立重新计算旧标量 loss，一致性仅由运行收据支持。原 192-probe 研究仍未完成，训练质量和加速结论均不变。
