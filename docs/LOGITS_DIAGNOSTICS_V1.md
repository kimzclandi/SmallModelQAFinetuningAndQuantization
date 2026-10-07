# v2 蒸馏失败的独立诊断

2026-10-08，Codex 辅助实现与分析。分析已冻结的 242 条 TRAIN 教师缓存、3×242 步日志与已复用的 74 题 dev；没有重新训练、生成预测、调参或外部确认。

## 能从记录中确认什么

完整缓存哈希逐条校验后，对 1,289 个监督位置分析 T=2 教师分布：top-1 与 gold token 一致率 **88.52%**，gold 概率均值 **0.6266**，熵均值 **3.4770 nats**。
这来自 gold teacher-forcing 的 TRAIN token，包含容易预测的 token 和终止 token，不等于自由生成答案准确率或教师适用性通过。
float16 log-probability 的概率和最大偏差约 **0.0003990**；诊断在 float64 中重新归一化，原训练计算不变。
未保存舍入前的完整 logits，不能仅据此量化原始 float32 与 float16 KL 的全部差异，也不能证明数值误差导致性能失败。

| seed | 有答案正确数：KD / gold（33题） | 无答案正确数：KD / gold（41题） | 修复 / 回归 | KL 在标量总损失中的占比 |
|---|---:|---:|---:|---:|
| 20260918 | 14 / 12 | 29 / 35 | 6 / 10 | 76.54% |
| 20260919 | 16 / 12 | 28 / 36 | 7 / 11 | 74.74% |
| 20260920 | 16 / 17 | 31 / 32 | 6 / 8 | 74.95% |

相对 gold-SFT，无答案正确数三组均较低；有答案两组增加、一组减少。这里有拒答行为取舍的线索，尚不能证明教师或 KL 权重是根因。
`0.5 CE + 0.5 T² KL` 在 T=2 时是 `0.5 CE + 2 KL`。
表中占比按各步加权标量损失求和计算；**不是梯度范数占比、参数更新占比或“KL 太强”的因果证据**。
日志中的总损失重构最大误差为 2.39e-7 以下，支持记录与目标函数一致，不能排除所有训练实现错误。

## 修正因果表述

v1→v2 同时改变训练数据、步数和目标策略（教师自由生成目标→gold teacher forcing）。
旧冻结报告中的“覆盖扩大有效”应理解为观察性结果，不能作为数据量单因素因果结论。
原报告原样保留，以本说明限定解释。当前依然是 dev EM 60.36%±2.81%，低于匹配 gold 的 64.86%±1.35%。
若后续要辨别原因，需要新的事先注册对照；本次没有根据这份 dev 分析改变 T、权重或训练配置。

## 代码、记录与核验

- [诊断程序](../scripts/diagnose_logits_v2.py)：因果位移、缓存哈希、教师分布、逐题对照与损失重构。
- [逐 token 统计](../reports/logits-diagnostics-v1/teacher-token-statistics.json)、[汇总及逐 seed 转移](../reports/logits-diagnostics-v1/summary.json)、[缓存与源码来源](../reports/logits-diagnostics-v1/provenance.json)。
- 完整教师缓存仍保留本机；公开统计可重算汇总，但独立验证统计提取需提供匹配哈希的原缓存，不能把公开统计当成已公开全部原始张量。

```bash
python scripts/diagnose_logits_v2.py verify --output reports/logits-diagnostics-v1
# 有原始本地缓存时重新提取，输出必须是不存在的新目录：
python scripts/diagnose_logits_v2.py run --cache /path/to/frozen-cache --output work/new-diagnosis
```

离线 CI 重算统计与逐题对照；不读取教师张量、不训练、不证明模型质量通过。
