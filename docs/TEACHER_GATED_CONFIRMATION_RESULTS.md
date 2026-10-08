# 教师一致性门控蒸馏：完整结果与未通过采用门槛的结论

**本轮未验证门控 KD 相对同条件 gold-SFT 的质量收益，候选不采用。** 在 256 题、三个固定训练 seed 上，gold 与 gated_kd 平均 EM 均为 **51.82%**；主比较 gated_kd − gold 为 **0.00 个百分点**，文章聚类配对 bootstrap 95% 区间为 **[−2.82, +2.64] 个百分点**。区间包含零，不能将均值相同解释为已经证明等效。完整数值、逐题分数、全部 bootstrap 样本及门槛判断见 [scores.json](../reports/teacher-gated-confirmation-v1/scores.json)。执行完成与候选质量通过是两个独立状态。

本轮只检验一个事前确定的候选，未依据新留出集结果改目标、门槛、输入或选择最佳 seed。原有 v1/v2 蒸馏及量化证据保持原样；本报告不替代历史结果，也不追加新的量化、速度或部署结论。

**比较方法与训练约束。** 学生为固定 revision 的 Qwen2.5-0.5B-Instruct，教师为 Qwen2.5-1.5B-Instruct。三组均使用同一个 CPU、float32、eager、8 线程训练入口；242 条冻结 TRAIN、每个 seed 242 步，LoRA r=8、alpha=16、dropout=0，目标模块 q_proj/v_proj，学习率 1e−4，梯度范数裁剪阈值 1。每次均从同一未微调学生开始；同 seed 的初始可训练参数摘要、训练顺序摘要和缓存摘要完全相同。[冻结协议](../configs/teacher-gated-confirmation-v1/protocol.json) · [训练实现](../reports/teacher-gated-confirmation-v1/frozen-source/qa_lab/confirmation_training.py) · [全部训练记录](../reports/teacher-gated-confirmation-v1/training)

| 组 | 每步目标 |
|---|---|
| gold | 全部答案 token 的平均 CE，权重 1 |
| full_kd | 0.5 CE + 0.5 T² × 全部监督位置 KL 之和 / N，T=2 |
| gated_kd | 0.5 CE + 0.5 T² × 教师 argmax 与 gold 相同位置的 KL 之和 / N，T=2 |

N 始终是该步的全部监督 token 数，包含 EOS；门控组不按保留位置数重新归一化。新缓存完整包含 242 条 TRAIN、1,289 个监督位置、151,936 维完整词表的 FP16 log-probability；训练转为 FP32，未重新归一化缓存分布。教师 top-1 与 gold 一致的位置为 1,141/1,289（88.52%），148 个位置不一致；此数值是 teacher-forced TRAIN token 一致率，不是教师自由生成质量。新缓存具有教师和学生文件身份回执；旧缓存未被追认或覆盖。[缓存公开回执](../reports/teacher-gated-confirmation-v1/teacher-cache.receipt.json)

门控同时改变 KL 的位置选择与有效总量。本轮没有额外的“统一降低 KL 权重”控制，因此不能将观察到的差异单独归因为剔除了错误教师信息。历史 v2 full-KD 使用 CPU，旧 gold242 使用 MPS，且计时范围不同；本轮重新执行三组一致控制，不从旧训练耗时推导速度比。

**全部九次结果。** 每个运行评测相同的 256 题，其中可回答与不可回答各 128 题。表中 EM、F1 和格式有效率单位均为 %；最后一列为训练阶段发生梯度裁剪的步数，分母均为 242。展示值保留两位小数，实际门槛按未舍入数值判断。

| 组 | seed | 整体 EM | 整体 F1 | 可答 EM | 不可答 EM | 格式有效率 | 裁剪步数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gold | 20261009 | 56.64 | 62.54 | 39.84 | 73.44 | 97.66 | 173 |
| gold | 20261010 | 51.17 | 58.28 | 51.56 | 50.78 | 93.36 | 174 |
| gold | 20261011 | 47.66 | 54.81 | 43.75 | 51.56 | 94.53 | 163 |
| full_kd | 20261009 | 43.36 | 50.52 | 52.34 | 34.38 | 90.23 | 242 |
| full_kd | 20261010 | 45.70 | 52.40 | 58.59 | 32.81 | 89.45 | 242 |
| full_kd | 20261011 | 45.31 | 52.78 | 50.78 | 39.84 | 88.67 | 242 |
| gated_kd | 20261009 | 55.08 | 60.71 | 45.31 | 64.84 | 93.36 | 238 |
| gated_kd | 20261010 | 49.61 | 56.51 | 53.12 | 46.09 | 89.84 | 239 |
| gated_kd | 20261011 | 50.78 | 57.15 | 39.84 | 61.72 | 91.02 | 233 |

三个 seed 的均值及样本标准差如下；标准差描述这三个训练结果的波动，不是置信区间。分组 EM 与格式有效率列为三个 seed 的均值。

| 组 | 整体 EM 均值 ± SD | 整体 F1 均值 ± SD | 可答 EM | 不可答 EM | 格式有效率 |
|---|---:|---:|---:|---:|---:|
| gold | 51.82 ± 4.53 | 58.54 ± 3.87 | 45.05 | 58.59 | 95.18 |
| full_kd | 44.79 ± 1.26 | 51.90 ± 1.21 | 53.91 | 35.68 | 89.45 |
| gated_kd | 51.82 ± 2.88 | 58.13 ± 2.26 | 46.09 | 57.55 | 91.41 |

full_kd 是预先声明的次要对照：其平均 EM 比 gold 低 7.03 个百分点；可答 EM 高 8.85 个百分点，而不可答 EM 低 22.92 个百分点。这是回答／拒答表现的观察差异，尚无根因确认。gated_kd 高于 full_kd 的描述性结果不能替代预先声明的 gold 主控制组。裁剪频次也只是保存的训练统计，不能证明 KL 梯度主导或裁剪导致质量变化。

**主比较的采用门槛逐项保留。**

| 冻结条件 | 实际 gated_kd − gold 结果 | 判定 |
|---|---|---|
| 平均 EM 提升至少 3 个百分点 | 0.00 个百分点 | 未通过 |
| 95% 区间下界大于 0 | 下界 −2.82 个百分点 | 未通过 |
| 至少 2/3 seed 的 EM 改善 | −1.5625、−1.5625、+3.125 个百分点；仅 1/3 改善 | 未通过 |
| 可答平均 EM 下降不超过 2 个百分点 | +1.04 个百分点 | 通过 |
| 不可答平均 EM 下降不超过 2 个百分点 | −1.04 个百分点 | 通过 |
| 每个候选 seed 格式有效率至少 99% | 93.36%、89.84%、91.02% | 未通过 |

在此可答／不可答各占一半的构造集上，始终输出 `NO_ANSWER` 的 EM 为 **50%**。gold 与 gated_kd 的平均值仅高出该朴素参照 1.82 个百分点；本轮没有针对它另行完成显著性或生产可用性验证，不能仅凭整体 EM 略高于 50% 宣称具备可靠问答能力。该均衡采样比例不代表真实请求分布。

EM/F1 使用本项目的 SQuAD-style 归一化实现，并非官方 scorer 榜单成绩；`NO_ANSWER` 拒答须严格匹配。格式检查则要求输出为原文精确子串或 `NO_ANSWER`。因此，大小写、标点或冠词变化可能通过归一化 EM，却不符合原文片段格式：full_kd 三个 seed 分别有 3/6/7 条、gated_kd 有 4/5/4 条 `EM=1` 但 `format_valid=false`；gold 为 0/0/0。将三个 seed 的同题运行记录相加，gold 的 `categories.correct` 与 EM 分子均为 398；full_kd 分别为 328 与 344（差 16），gated_kd 为 385 与 398（差 13）。这些是 768 条配对运行记录的计数，不是 768 道独立题。失败类别优先标记格式问题，所以 `categories.correct` 不必等于 EM 正确数。两种口径均保留，不在看到结果后改评分器。[冻结评分实现](../reports/teacher-gated-confirmation-v1/frozen-source/qa_lab/metrics.py) · [逐题预测](../reports/teacher-gated-confirmation-v1/predictions)

**独立性和区间的范围。** 本轮从已审计项目实验中未使用的文段抽取 128 个文段、256 题，最终来自 25 篇此前项目已涉及的文章。选择排除了已知 QA ID、归一化文段及相同／近重复问题，并整篇排除了准备阶段意外显示过摘要的五篇文章。这是新的项目文段留出集，不是新文章、未见主题、官方隐藏测试或已证明预训练无污染的数据。[数据清单与排除记录](../reports/teacher-gated-confirmation-v1/cohort/manifest.json)

执行流程先完成并锁定九个 adapter，再完成并锁定九份无标签输入预测，最后解析参考标签评分；公开时间戳及哈希可审查这一顺序，但不构成对抗式或加密盲法证明。[模型锁](../reports/teacher-gated-confirmation-v1/execution-lock.json) · [预测锁](../reports/teacher-gated-confirmation-v1/prediction-lock.json)

主区间先对每道题的三个 seed 配对 EM 差取平均，再以文章为簇有放回抽样 10,000 次，保留簇内全部题目，按题加权取 95% percentile 区间。它条件于这三个固定 seed 及此已观察文章集合，只描述该重采样设计的题目／文章抽样变化，不估计完整训练 seed 不确定性，也不证明跨新文章泛化。重复评测三个 seed 不能当作三倍独立题量。结果已观察后，这份留出集已被消耗；今后依据它修改方法属于开发。

**执行失败、版本和成本。**

代码及协议先提交为 [`dff77b115431bbf8e76113d7dd97712bd98ca1f3`](https://github.com/kimzclandi/SmallModelQAFinetuningAndQuantization/commit/dff77b115431bbf8e76113d7dd97712bd98ca1f3)。首次执行在约 **0.0156 秒**后因沙箱拒绝 RSS 监控中的子进程枚举而报 `PermissionError`；完成阶段为 0，尚未创建 teacher cache，没有训练、预测或评分结果。失败记录与原日志保留，属于执行权限故障，不是模型质量失败。[首次失败](../reports/teacher-gated-confirmation-v1/attempts/execution-01/study.json) · [首次日志](../reports/teacher-gated-confirmation-v1/attempts/execution-01/teacher-cache.log)

第二次执行前新增 [execution-02.json](../configs/teacher-gated-confirmation-v1/execution-02.json)，提交为 [`d2f0af8a5087b1b1d0874dd2f1a1f4c8665614a1`](https://github.com/kimzclandi/SmallModelQAFinetuningAndQuantization/commit/d2f0af8a5087b1b1d0874dd2f1a1f4c8665614a1)。两提交之间只增加该执行说明；训练、评分、选择器代码、协议、模型、输入选择规则、seed 和门槛相同。取得本机执行权限后使用新的 `-exec2` 输出目录，未覆盖首次失败，也没有根据质量结果重跑。

exec2 在 Apple M4 Max 上完成 1 次新 cache、9 次训练和 9 次串行离线预测，共 19 个阶段、2,178 个训练步和 2,304 条预测。PyTorch 2.8.0、Transformers 4.56.2、PEFT 0.17.1；CPU/FP32。总 wall time **1,416.25 秒（23.60 分钟）**，阶段监控记录的最高采样进程 RSS 为 **8,466,399,232 bytes**，均在冻结预算内。此处为资源成本核算，采样 RSS 不是模型权重大小或设备峰值显存；不同阶段的耗时不用于计算加速倍率。[完整执行收据](../reports/teacher-gated-confirmation-v1/study.json)

**公开可复算的内容与限制。** 归档包括 87 个受 manifest 约束的文件：冻结源码／协议／TRAIN 身份、数据和标签、全部九份逐题预测、评分结果、九组逐步 loss 与梯度裁剪记录、模型／adapter 身份摘要及首次失败。完整模型权重、九个 adapter 权重、teacher 完整 logits 张量及本地完整运行日志没有上传；公开缓存回执移除了私有 `artifact_path`，保留原 manifest SHA、每条缓存文件 SHA 及所有缓存文件合计字节数，因此不能从脱敏回执重构原 manifest 字节。[归档 manifest](../reports/teacher-gated-confirmation-v1/archive-manifest.json)

离线 verifier 核验文件集合与 SHA、来源绑定、运行顺序、所有控制组和门槛，重新计算公开预测的评分与 bootstrap。它不重新训练、推理或重演未上传的缓存张量及 adapter 更新；初始参数、未公开权重和缓存字节身份只能核对收据，不能在 CPU CI 内独立重测。本轮本机另完成缓存张量和 adapter 文件检查，但该检查不能因公开摘要而升级为第三方完整重演。训练 loss／clip 算术采用 verifier 明示的浮点容差；`family_macro_em` 仅为描述性指标，允许 1e−15 绝对误差处理冻结实现的集合累加顺序，主比较、bootstrap、采用门槛与其余评分字段按原值复算。

从仓库根目录运行下列离线核验，不需要模型权重或新推理：

```bash
python scripts/verify_confirmation.py \
  --root reports/teacher-gated-confirmation-v1
python -m pytest -q \
  tests/test_confirmation_data.py \
  tests/test_confirmation_scoring.py \
  tests/test_confirmation_study.py \
  tests/test_confirmation_training.py \
  tests/test_verify_confirmation.py
```

测试依赖沿用仓库锁定文件；训练张量测试需要 CPU PyTorch 2.8.0，未安装时会显式跳过，不能将这种跳过写成训练契约已通过。CI 的 `cache-contract` job 安装该版本并强制执行训练张量测试。

下面记录实际使用的完整执行路径；`/local/path` 是执行者已有文件的位置，不是下载入口。代码和协议先提交后才执行，完整运行会真实训练／推理并消耗预算；已有输出目录拒绝覆盖，不能用重跑选择更好的结果。

```bash
python scripts/prepare_confirmation_data.py \
  --source /local/path/squad-dev-v2.0.json \
  --output work/teacher-gated-cohort-v1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m qa_lab.confirmation_study run \
  --cohort work/teacher-gated-cohort-v1 \
  --cache-dir /local/path/huggingface/hub \
  --output work/teacher-gated-confirmation-v1-exec2
```

**贡献边界。** 维护者给出项目目标、授权和事实约束；本轮实验设计、实现、命令执行、审阅、证据整理及文档由 Codex/AI 辅助完成。Qwen 模型、SQuAD/Wikipedia 数据与 PyTorch、Transformers、PEFT 框架及 LoRA/KL 基础方法归属上游，不声称原创模型、训练框架或门控蒸馏算法。[AI 辅助说明](../CONTRIBUTIONS.md) · [第三方组件](../THIRD_PARTY.md) · [数据许可](../DATA_LICENSE.md)

三个组均是同规模 0.5B 学生；本轮检验 soft targets 是否带来额外质量收益，没有测得新的模型压缩比，没有教师同集质量保持对照，也没有量化、推理加速、CUDA/Ascend、多卡或生产部署证据。
