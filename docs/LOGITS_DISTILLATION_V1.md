# Logits distillation v1

## 目标与边界

这一增量把已有“教师最终文本作为标签”的响应级蒸馏，扩展为答案 token 位置上的 soft-target 蒸馏。教师为固定 revision 的 Qwen2.5-1.5B-Instruct，学生为 Qwen2.5-0.5B-Instruct；训练目标为：

`L = 0.5 * CE(student, teacher_response_tokens) + 0.5 * T^2 * KL(P_teacher^T || P_student^T)`，其中 `T=2`。

KL 使用每个受监督位置上的完整词表分布。因果对齐为 `logits[:, :-1]` 对 `labels[:, 1:]`；prompt token 的 label 为 `-100`，不参与 CE 或 KL。教师 log-probability 以 float16 缓存，训练时转回 float32，因此属于完整词表但有限精度的分布蒸馏。

固定实验已在CPU完成三seed训练和dev评测；结果没有通过继续门槛，详见[完整负结果](../reports/logits-distillation-v1/RESULTS.md)。不得声称 logits 蒸馏提升了 EM/F1。

## 为什么先缓存教师分布

教师和学生无需同时驻留设备，减少统一内存峰值；同一份哈希绑定的教师分布可供三个学生 seed 复用。缓存只覆盖24个训练 ID 的回答 token 位置，不读取 dev/test 标签。缓存会验证：

- 教师、学生 tokenizer 的完整词表映射和 special-token 配置一致；
- response-distillation artifact 的来源、方法和哈希未改变；
- 每条缓存文件的 SHA-256 未改变；
- cache temperature、学生 prompt/config 与训练配置一致。

## 运行

先按 README 准备 PyTorch 环境和两套本地模型。新输出只能写入新的 `work/` 目录。

先运行受保护的单题、单步 smoke。缓存和训练记录都会标记为 `smoke_complete`，完整训练入口会拒绝读取该缓存：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m qa_lab.logits_distillation cache \
  --artifact data/teacher-study-v2-prompted \
  --teacher-config configs/teacher-study-v2/teacher.json \
  --teacher-device cpu \
  --objective configs/logits-distillation-v1/objective.json \
  --limit 1 --output work/logits-distillation-v1-smoke-cache

HF_HUB_OFFLINE=1 .venv/bin/python -m qa_lab.logits_distillation train \
  --cache work/logits-distillation-v1-smoke-cache \
  --student-device cpu \
  --train-config configs/logits-distillation-v1/train-smoke.json \
  --objective configs/logits-distillation-v1/objective.json \
  --smoke --output work/logits-distillation-v1-smoke-train
```

smoke 通过后才生成完整缓存并运行固定实验：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m qa_lab.logits_distillation cache \
  --artifact data/teacher-study-v2-prompted \
  --teacher-config configs/teacher-study-v2/teacher.json \
  --objective configs/logits-distillation-v1/objective.json \
  --output work/logits-distillation-v1-cache

HF_HUB_OFFLINE=1 .venv/bin/python -m qa_lab.logits_distillation train \
  --cache work/logits-distillation-v1-cache \
  --train-config configs/logits-distillation-v1/train-20260918.json \
  --objective configs/logits-distillation-v1/objective.json \
  --output work/logits-distillation-v1-20260918
```

另外两个 seed 使用相应配置和全新输出目录。训练完成后，沿用 `qa_lab.inference` 只评估 dev；在三个 seed 全部完成前不做方法结论。历史 gold/response 两臂可作上下文对照，但监督 token 数不同，不能声称严格等计算量。

## 验收点

- 相同 teacher/student 分布的 KL 约为0；
- 单元测试锁定 causal shift，防止把 token 自身位置当成预测位置；
- steps 日志分别记录 total loss、hard CE、soft KL、监督 token 数和梯度范数；
- training.json 记录 objective、模型配置、cache manifest hash、耗时和依赖版本；
- dev/test 推理与训练解耦，训练 loss 不作为质量提升证据。
