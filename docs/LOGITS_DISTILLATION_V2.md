# Logits distillation v2：242条训练数据

## 预注册目标

本实验在不改动v1证据的前提下，把训练覆盖从24条扩大到冻结的全部242条TRAIN数据。硬目标来自公开gold答案；Qwen2.5-1.5B教师仅在相同答案token位置提供完整词表soft distribution。dev/test答案不参与缓存或训练。

训练目标保持为 `0.5 * CE + 0.5 * T^2 * KL`，`T=2`。三个seed各训练242步，即每条训练样本恰好曝光一次。历史`coverage-v3/gold242`是匹配控制组：训练ID、seed、优化步数、LoRA配置和学习率相同，区别是控制组只有gold hard CE。

## 完成结果

三个seed的74题dev整体EM分别为58.11%、59.46%和63.51%，均值为**60.36%±2.81%**；有答案EM均值为46.46%。24条v1 logits方案为46.40%±7.44%，说明扩大训练覆盖后均值提高、seed波动下降。

匹配的gold242 hard-CE-only控制组为64.86%±1.35%，仍高于v2 logits方案。当前结果因此不能证明soft targets带来额外收益。完整词表float16缓存包含242条记录、占374.69 MiB，CPU生成70.14秒；每个seed训练监督1289 tokens。完整逐题证据见[冻结报告](../reports/logits-distillation-v2/RESULTS.md)。

## 解释边界

- 74题dev已被历史实验多次使用，只能作为开发证据，不是独立确认集。
- 不依据结果选择seed、调整温度、损失权重或追加训练。
- 当前协议不运行外部集；若开发结果值得继续，需另建未使用的评测协议。
- 完整词表float16缓存的磁盘占用和生成时间需要报告，这是方法成本而不是实现细节。
- gold teacher forcing回答的是“在参考答案前缀下，教师下一token分布能否帮助学生”，不同于v1对教师自由生成响应的蒸馏。

## 固定入口

以下保留历史执行命令。当前代码另加[模型文件身份预检](MODEL_IDENTITY.md)：`cache/train` 可用 `--model-cache-dir` 指向已有 HF hub；若新训练复用缺少身份回执的旧缓存，须显式传入 `--allow-legacy-unbound-cache`，并保留教师字节身份未绑定的标记。历史离线验收不要求新字段，也不因此获得追溯认证。任何新的训练仍需新协议和输出目录。

```bash
HF_HUB_OFFLINE=1 .venv-ci/bin/python -m qa_lab.logits_distillation cache \
  --artifact data/coverage-v3-gold242 \
  --teacher-config configs/teacher-study-v2/teacher.json \
  --teacher-device cpu \
  --objective configs/logits-distillation-v2/objective.json \
  --output work/logits-distillation-v2-cache-cpu

HF_HUB_OFFLINE=1 .venv-ci/bin/python -m qa_lab.logits_distillation train \
  --cache work/logits-distillation-v2-cache-cpu \
  --student-device cpu \
  --train-config configs/logits-distillation-v2/train-20260918.json \
  --objective configs/logits-distillation-v2/objective.json \
  --output work/logits-distillation-v2-20260918-cpu
```

另外两个seed仅替换配置和全新输出目录。训练后只对冻结dev运行`qa_lab.inference`，随后一次性汇总三seed及其相对同seed gold242控制组的逐题变化。上述命令已经执行；复现时必须使用新输出目录，不能覆盖冻结结果。
