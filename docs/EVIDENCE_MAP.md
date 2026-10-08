# 核心表述与可核验证据

所有链接相对当前分支。2026-10-09 核验：[PR #13](https://github.com/kimzclandi/SmallModelQAFinetuningAndQuantization/pull/13) 已合并，v2、诊断、缓存和 checkpoint 身份预检均已进入默认 `main`。本轮同 CPU 三臂与新文段留出评估由 [PR #14](https://github.com/kimzclandi/SmallModelQAFinetuningAndQuantization/pull/14) 提供，合并前需通过该 PR 核验。历史报告、失败记录、协议与原始预测保持原样。

| 表述 | 实现／协议 | 报告／原始记录 |
|---|---|---|
| TRAIN-only 梯度诊断首 probe 数值门槛失败，0 通过 / 1 失败；无质量结论 | [实现](../qa_lab/gradient_diagnostics.py)、[冻结协议](../configs/ce-kl-gradient-v1.json) | [结果与缺口](CE_KL_GRADIENT_DIAGNOSTICS.md)、[原始失败](../reports/ce-kl-gradient-v1/)、[失败回执核验](../scripts/verify_gradient_failure.py) |
| answer-only LoRA、响应监督 | [训练](../qa_lab/train.py)、[目标数据](../qa_lab/closure_data.py) | [首轮含失败结果](../reports/closure-v1/RESULTS.md) |
| 同 CPU 三臂×三 seed，256题新文段留出；门控与gold均51.82%，主差0.00pp，未通过采用门槛 | [统一训练](../qa_lab/confirmation_training.py)、[冻结协议](../configs/teacher-gated-confirmation-v1/protocol.json)、[数据排除](../scripts/prepare_confirmation_data.py)、[预测锁定](../qa_lab/confirmation_study.py) | [结果与边界](TEACHER_GATED_CONFIRMATION_RESULTS.md)、[全部记录](../reports/teacher-gated-confirmation-v1/)、[离线重算](../scripts/verify_confirmation.py) |
| 242条完整词表soft-target蒸馏，固定T=2、CE/KL各0.5 | [实现](../qa_lab/logits_distillation.py)、[固定协议](../configs/logits-distillation-v2/protocol.json)、[目标函数](../configs/logits-distillation-v2/objective.json) | [v2报告](../reports/logits-distillation-v2/RESULTS.md)、[汇总](../reports/logits-distillation-v2/summary.json)、[缓存清单](../reports/logits-distillation-v2/cache-manifest.json)、[逐seed记录](../reports/logits-distillation-v2/) |
| v2 dev EM 60.36%±2.81%，低于匹配gold 64.86%±1.35% | [v2离线核验](../scripts/verify_logits_distillation_v2.py)、[gold控制](../reports/coverage-v3/RESULTS.md) | [gold原始记录](../reports/coverage-v3/)、[v1负结果](../reports/logits-distillation-v1/RESULTS.md) |
| Q8 988.10→525.05 MB；decode 266.21→313.86 tokens/s；dev少对1/74 | [MLX实现](../qa_lab/mlx_experiment.py)、[协议](QUANTIZATION_V4.md) | [报告及Q4失败](../reports/quantization-v4/RESULTS.md)、[原始记录](../reports/quantization-v4/) |
| 240候选、75接收、37盲审样本 | [selection profile](../qa_lab/selection_profile.py)、[盲审](../qa_lab/blind_review.py) | [QC规则与证据](SYNTHETIC_QC.md)、[盲审协议](BLIND_REVIEW.md)、[未填写的审阅包](maintenance/2026-09-22-depth/evidence/review-bundle/reviewer/) |

±为三个seed的样本标准差，不是置信区间。v2在gold答案前缀上teacher forcing；v1使用教师生成目标，且步数不同。因此扩大覆盖后观察到更高dev EM，但不能将全部差异归因于数据量，更不能推出soft targets胜过匹配gold-SFT。74题dev已经复用，未做外部确认，也没有结果后调参。完整 logits 缓存与模型权重不随仓库分发，缓存清单记录来源、规模与哈希。

无参考权限时保持review；无独立人工标注时语义结论保持unknown。盲审样本生成不是完成人工盲审。个人项目采用AI辅助实现与执行，归属见[贡献说明](../CONTRIBUTIONS.md)。

## 只读验收

```sh
python scripts/acceptance.py --output work/new-evidence-acceptance
python .github/scripts/check_readmes.py
python scripts/verify_confirmation.py
git diff --check
```

统一入口运行完整pytest与冻结证据核验，输出到新的work目录并检查冻结树未变；不训练、不推理、不改实验门槛。[CI定义](../.github/workflows/tests.yml)。CI通过只证明这些检查通过，不证明模型质量或部署就绪。

[新增冻结蒸馏诊断](LOGITS_DIAGNOSTICS_V1.md)。与旧冻结实验分开保存，不替换历史结果。

[缓存训练前身份校验与搬迁入口](LOGITS_CACHE_INTEGRITY.md)：修复源内容与缓存 ID/token/mask 核对缺口，不重新训练、不改变历史指标。

[Checkpoint 文件身份预检](MODEL_IDENTITY.md)：新 logits 缓存/训练核验本地模型实际文件，并绑定同一加载目录；历史 teacher 身份缺口显式保留，不追认旧实验。
