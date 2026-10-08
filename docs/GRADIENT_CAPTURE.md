# 梯度失败留证与同输入重放

旧 [梯度诊断](CE_KL_GRADIENT_DIAGNOSTICS.md) 在首个 TRAIN probe 未通过数值门槛，却只归档了预选的、通过检查的 q_proj 梯度。失败的 logit 和 v_proj 数组没有保留，错误摘要不能替代原值重放。

新入口在每次 autograd 返回后、执行数值比较前，原子保存该目标的 logit 导数、FP64 解析参考和全部 96 个 LoRA 参数块的导数。输入另存独立本地目录；中途异常保留已完成阶段。状态分别记录捕获是否完整、模型与输入身份是否仍一致、原数值门槛是否通过。捕获成功不会把旧失败改成成功。

## 固定范围

[新协议](../configs/ce-kl-gradient-capture-v1.json) 固定旧研究首个 TRAIN 输入、seed 20261009、初始化参数、CPU/FP32、8 线程及原 `atol=1e-5, rtol=1e-4`。只执行一个 probe，五个目标为加权 CE、full KL、gated KL 及两个合成目标。无优化器更新、自由生成或留出集读取，不恢复 192-probe 研究。预算为 120 秒、进程 RSS 16 GiB、两目录共 128 MiB，按阶段检查而非阻塞调用的硬中断。

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
  --protocol configs/ce-kl-gradient-capture-v1.json \
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

真实执行结果在完成验证后另附；原冻结实验与失败记录保持不变。
