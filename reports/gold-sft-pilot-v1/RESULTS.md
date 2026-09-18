# 普通监督微调 pilot：整体提高，但有答案题回归，拒绝采用

本轮仅使用公开 SQuAD 的原始 gold 标签；没有使用用户提交的 GPT 回答训练。训练前已提交配置与 gate，见 preregistration.json（Git be5089f）。

| 开发集指标（74题） | 原始学生 | gold-SFT |
|---|---:|---:|
| 整体 EM | 25.68% | 55.41% |
| 整体 token F1 | 30.13% | 59.18% |
| 有答案 EM（33题） | 57.58% | 45.45% |
| 无答案正确拒答（41题） | 0.00% | 63.41% |
| 格式合规 | 95.95% | 95.95% |

逐题配对：修复 26 题，退化 4 题。完整 ID 列表及重算分数见 comparison.json；具体输入/输出见 paired-changes.jsonl。

**预登记 gate：REJECT。** 虽然总体 EM 从 19/74 提高到 41/74，但有答案题从 19/33 降到 15/33；不满足“整体提高且有答案 EM 不下降”。整体 EM 与始终拒答对照同为 41/74，不能仅凭总分宣布优化成功。

## 输入、机制、输出与限制

输入为固定 pilot 的 24 道 train 题，gold 可回答/不可回答各12题。LoRA 更新 q_proj/v_proj，rank=8，alpha=16；lr=1e-4，batch=1，48步，两次遍历同一固定顺序，seed=20260918，FP32/MPS。监督 prompt 以外的回答与结束标记，共264个监督 token。

输出是保存的 LoRA adapter、逐步 loss/gradient 日志，以及未训练过的74道 dev 的完整预测。adapter 权重只存本地 checkpoints，不进 Git；可通过固定配置重新训练。机制是答案级交叉熵监督，不是教师蒸馏或 RLHF。

结果说明：该小样本 SFT 学会更多拒答，但也开始拒答部分可回答问题。观察到的是行为权衡；未通过消融确认具体根因。候选不作为新的默认模型，不根据此次 dev 结果在本轮追加调参。

局限：单seed、24题、264监督token，小规模pilot；无显著性或跨领域结论。本轮仅评估dev，没有新增test推理；不能写成测试集提升。尚无teacher-SFT，所以不能比较教师数据与gold标签的训练价值。自然输出长度不同，不从本次时延差异推断推理加速。

## 复现

```bash
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 .venv/bin/python -m qa_lab.train \
  --config configs/gold-sft-pilot-v1.json \
  --selection data/teacher-pilot-v1/manifest.json \
  --output checkpoints/gold-sft-reproduction-01
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 .venv/bin/python -m qa_lab.inference \
  --adapter checkpoints/gold-sft-reproduction-01 --splits dev \
  --output work/gold-sft-reproduction-01
PYTHONPATH=. .venv/bin/python scripts/compare_gold_pilot.py \
  --output work/gold-pilot-comparison-recomputed.json
```

最后一条命令重算已冻结的实验，而不是比较新训练运行。运行版本、数据与源码hash在training.json/eval-dev/run.json中；源码快照在source/。
