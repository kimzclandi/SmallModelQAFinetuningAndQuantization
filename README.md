# 小模型问答蒸馏、微调与量化

**简体中文** | [English](README.en.md)

可审计的小模型抽取式问答实验：数据隔离 → 原始基线 → gold-SFT与响应蒸馏 → 数据覆盖对照 → 同框架量化 → 冻结候选的新来源验证。输入是文段和问题，输出最短原文答案或严格 `NO_ANSWER`；不包含检索或闭卷知识问答。

[![offline-integrity](https://github.com/kimzclandi/SmallModelQAFinetuningAndQuantization/actions/workflows/tests.yml/badge.svg)](https://github.com/kimzclandi/SmallModelQAFinetuningAndQuantization/actions/workflows/tests.yml)

**历史五轮结论：训练候选均未通过采用门槛；Q8通过英文dev压缩筛选及96题中文新来源的质量保持检查，尚未获得业务部署验证。** 所有质量数字来自保存的逐条预测，保留拒答基线、失败样例和回归。实际蒸馏使用本地Qwen教师；手工GPT候选仅作审计，未进入训练。

## 同条件三臂与新文段留出评估

新一轮在同一 CPU/FP32 训练循环中比较 gold-SFT、完整 KL、教师与 gold 一致时才施加 KL，三个固定 seed 各 242 步。对排除既有实验文段与近重复问题后的 **128 文段、256 题**完成 2,304 条预测；全部模型与预测锁定后才评分。

| 方法 | 留出集归一化 EM，三 seed 均值±样本标准差 |
|---|---|
| gold-SFT | 51.82%±4.53% |
| 完整 KL | 44.79%±1.26% |
| 教师一致性门控 KL | 51.82%±2.88% |

**门控相对 gold 的主比较差值为 0.00 个百分点，文章聚类 95% 区间 [-2.82,+2.64]，未通过采用门槛。** 门控格式有效率为 91.41%，gold 为 95.18%；只有 1/3 seed 的 EM 更高。始终拒答基线为 50%。完整 KL 是次要对照，不能用相对它的改善替代主比较。

25 篇文章均有项目接触记录，本轮只支持审计范围内的**新文段留出**，不支持文章级泛化、预训练无污染或部署质量。未据此调参重跑。[完整结果与失败分析](docs/TEACHER_GATED_CONFIRMATION_RESULTS.md) · [执行前协议](docs/TEACHER_GATED_CONFIRMATION_V1.md) · [原始预测与记录](reports/teacher-gated-confirmation-v1/) · [离线重算](scripts/verify_confirmation.py)

## 蒸馏失败诊断

新增 [TRAIN-only CE/KL 参数梯度诊断](docs/CE_KL_GRADIENT_DIAGNOSTICS.md)：代码和协议先提交，再实际执行。首个 probe 的 logit 解析参考及两个 LoRA 参数块重建超差，程序早停并保存失败；实际 0 个通过、1 个失败，未完成原计划 192 个。未放宽门槛、重跑训练或读取留出集，不形成梯度机制、质量或性能结论。[原始失败归档](reports/ce-kl-gradient-v1/) · [失败回执核验](scripts/verify_gradient_failure.py)。

对现有教师缓存、训练日志和逐题对照新增诊断，无重新训练或调参。无答案题三组均低于 gold-SFT；KL 约占标量总损失的75%–77%，但不是梯度占比或因果结论。[分析与可重算记录](docs/LOGITS_DIAGNOSTICS_V1.md)。

训练入口另增加[缓存血缘与 token/mask 预检](docs/LOGITS_CACHE_INTEGRITY.md)：重新绑定冻结 TRAIN 内容、完整 ID 和 tokenizer 身份，支持同内容 artifact 搬迁；属于工程完整性修复，不产生新质量结果。

logits 缓存/训练入口增加[checkpoint 文件身份预检](docs/MODEL_IDENTITY.md)：按已登记模型与 revision 流式核验本地权重、配置和 tokenizer，并加载同一个已验证目录。旧缓存的新训练需显式承认教师字节身份未绑定；不修改或追认历史实验。

## 242条完整词表 logits 蒸馏对照

在保留24条v1负结果的基础上，使用全部242条冻结TRAIN数据完成新一轮gold teacher-forcing蒸馏。三个固定seed各训练242步；训练ID、seed、步数、LoRA配置和学习率与历史gold242 hard-CE-only控制组匹配。

扩大数据后，74题dev整体EM由v1的46.40%±7.44%提高到**60.36%±2.81%**，但仍低于匹配gold242控制组的64.86%±1.35%。完整词表float16缓存为374.69 MiB。扩大覆盖后的结果改善，但v1→v2同时改变训练步数与目标来源，不能单独归因于样本数量；soft targets仍未超过匹配gold SFT。dev已复用，本轮未运行外部集，也未结果后调参。[完整结果](reports/logits-distillation-v2/RESULTS.md) · [协议与命令](docs/LOGITS_DISTILLATION_V2.md)

## 核心表述与证据入口

[代码、逐题记录、固定协议与核验命令](docs/EVIDENCE_MAP.md)。其中242条v2是当前开发结果；24条v1及其他历史负结果完整保留。CI验证工程契约与保存证据，不代表独立模型质量确认或生产部署。

## 24条完整词表 logits 蒸馏负结果

新增答案 token 位置上的 `hard-label CE + T²·KL` 训练入口，教师完整词表 log-probability 先写入哈希绑定缓存，三个学生 seed 可复用而无需教师、学生同时驻留设备。实现显式校验 tokenizer 映射、causal shift、prompt mask、temperature、artifact 来源和逐文件哈希。

固定 `T=2`、CE/KL各0.5，在CPU完成3个seed、每组48步及74题dev评测：整体EM为50.00% / 51.35% / 37.84%，均值46.40%；有答案EM均为42.42%。没有超过历史gold-SFT与response-distillation均值，且未通过历史有答案继续门槛，因此未运行外部集，也未看结果后调参。[完整负结果](reports/logits-distillation-v1/RESULTS.md) · [方法与命令](docs/LOGITS_DISTILLATION_V1.md)

## 项目沿革（2026-09-20 补记）

根据维护者对本地开发过程的说明，相关早期工作约于 2026 年 6 月开始在本地开展，之后集中整理并上传 GitHub。该月份是早期工作的近似起点，不表示当前全部功能和实验在当时已完成。后续实现、实验与维护保留各自的实际版本及运行日期。

## 新增：中文数据质量对照（2026-09-20）

完成9次LoRA训练、576步，并对未训练基线和全部9个适配器做一次96题留出评估。同题将教师目标替换为参考答案后，平均严格EM从17.36%升至24.31%，差+6.94个百分点，配对95%区间[+1.04,+13.54]；F1差值区间仍跨零。使用了训练参考标签，不能称为无标注自动筛选；1.5B/3B自动核验器均未通过挑战门槛。

**外部补充：DRCD 96题上，random_gold与random_teacher的平均EM均为57.29%，差值95%区间[-5.21,+4.86]个百分点，未复现CMRC上的正向主比较。** [外部评测与边界](docs/EXTERNAL_DRCD.md)

[结果、失败案例与限制](reports/quality-study-20260920/RESULTS.md) · [离线检查 / 两题真实推理 / 重新训练](docs/QUALITY_STUDY_RELEASE.md) · [完整训练入口重跑](docs/ENTRYPOINT_RETRAIN.md)

## 可追溯候选质检（2026-09-22）

新增统一离线入口，逐条输出接收／拒收／待复核、参考权限、来源与规则版本。实际检查240个历史真实教师候选：有训练参考时，英文原提示24条为8/11/5，新提示24条为8/12/4，中文192条为59/40/93（依次为接收/拒收/待复核）。无参考权限时不自动接收合法片段；格式匹配不等于语义正确。近重复只触发复核，原始候选和冻结证据不变。

```bash
python scripts/synthetic_qc.py --preset chinese --label-permission train_reference --output work/qc-01
python scripts/verify_synthetic_qc.py
```

中文接收59条与已有参考筛选训练组的输入、目标逐条一致，可复用历史对照；本次未新增训练或测得语义准确率。英文接收样本的有答案占比由原提示62.5%变为新提示25%，不能仅看接收总数。全部规则、逐条证据、训练复用限制及复现命令见[质检说明](docs/SYNTHETIC_QC.md)。

## 历史五轮实验与证据

| 实验 | 主要结果 | 结论与证据 |
|---|---|---|
| 1. 基线、gold-SFT、响应蒸馏与一次数据修订 | 原始学生test EM24.51%；gold-SFT51.96%；蒸馏36.27%；增补数据48.04% | 三个候选dev有答案EM均退化，未过门槛。test始终拒答基线52.94%。[完整结果](reports/closure-v1/RESULTS.md) |
| 2. 教师提示 × 三seed | gold / 原教师 / 新提示教师dev EM均值55.41% / 33.78% / 55.86% | 新提示改善总分但降低有答案能力；均未通过三seed门槛，未新增test推理。[结果](reports/teacher-study-v2/RESULTS.md) · [方法](docs/TEACHER_STUDY_V2.md) |
| 3. 24题重复与242题覆盖对照 | 两篇新文章64题，三seed整体EM均值36.46% → 44.27% | 同242步，未控制token/epoch/类别分布；仍低于始终拒答50%，dev门槛未过。[结果](reports/coverage-v3/RESULTS.md) · [方法](docs/COVERAGE_V3.md) |
| 4. 同MLX FP16 / Q4 / Q8 | Q8权重988.10 → 525.05MB；dev少答对1/74；固定工作量decode266.21 → 313.86tokens/s | Q8通过预设dev压缩门槛，Q4未通过；单机速度不是通用加速承诺。[结果与回归](reports/quantization-v4/RESULTS.md) · [方法](docs/QUANTIZATION_V4.md) |
| 5. 冻结Q8的中文新来源检查 | FP16和Q8严格EM均为44/96；字符LCS F1为72.68% / 72.12% | 正确ID集合相同，输出并非逐条无损。全部有答案，未验证中文拒答。[结果](reports/chinese-v5/RESULTS.md) · [方法](docs/CHINESE_V5.md) |

第一轮4-bit量化在同MLX框架下将test EM从24.51%降至14.71%，因此不能只凭文件缩小与解码变快采用。第四轮重新同时测量三种精度；不跨轮次或跨框架拼接速度比。

## 数据、模型与适用范围

- 英文主数据来自SQuAD2.0公开dev中的Computational_complexity_theory：418题、48个文段家族，train/dev/test为242/74/102；近重复家族合并后切分。项目test不是官方隐藏测试。
- 第三轮新增Packet_switching和Prime_number共64题；仍来自同一公开SQuAD文件。第五轮CMRC2018公开dev每篇文章最多一题，共96题。新来源不等于模型预训练未见。
- 学生为固定revision的Qwen2.5-0.5B-Instruct；教师为Qwen2.5-1.5B-Instruct，本地FP32生成。baseline指未接受本项目训练。模型版本、数据hash、代码快照与配置随运行记录保存。
- 历史实测为Apple M4 Max、48GiB统一内存、40核GPU、macOS27.0。PyTorch/MPS用于训练，独立MLX环境用于量化。未验证CUDA/Ascend训练、生产负载或外部业务泛化。
- 三seed描述训练波动，不是三个独立测试集；dev已多轮使用。中文指标含明确命名的自定义指标，不能称CMRC官方成绩。后续选择新方法需要新协议与独立数据。

## 快速开始：离线证据验收

从仓库根目录运行，Python3.12。此入口无需模型、GPU或API key；首次安装依赖需要网络，之后验收离线运行。

```bash
uv venv .venv-ci --python 3.12
uv pip install --python .venv-ci/bin/python -r requirements-ci.lock.txt
.venv-ci/bin/python scripts/acceptance.py --output work/acceptance-01
```

入口执行测试、离线核验、gold对照与教师审计；日志只写新的`work/`子目录，前后校验`reports/data/configs`。缺失冻结汇总直接失败，不补写；禁止`python -O`。Linux CI使用同一入口，**不下载模型、不训练、不重新推理**。历史工程验收含51项测试；最新执行以Actions记录为准。

## 实际模型推理与训练复现

以下入口实际加载学生并执行两题dev推理。首次下载约1GB；建议至少8GiB可用内存、5GB磁盘用于少量学生推理，最低配置未实测。完整训练/量化历史实测使用48GiB Mac。

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.lock.txt
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
.venv/bin/python scripts/download_model.py
HF_HUB_OFFLINE=1 .venv/bin/python -m qa_lab.inference \
  --device cpu --splits dev --limit 2 --output work/smoke-01
```

MPS推理将`--device cpu`换成`--device mps`；不可用会报错，不静默回退。已有输出拒绝覆盖。实际训练、教师生成和MLX量化见[完整复现手册](docs/REPRODUCE_CLOSURE.md)。PyTorch与MLX使用独立锁与环境；MLX锁面向macOS arm64。锁文件固定版本但不含wheel哈希，不是跨平台供应链锁。

同机新环境CPU冒烟曾实际运行，尚无异机模型推理/训练复现。冻结的`reports/`、`data/`、`configs/`及源码快照不可覆写；新实验使用隔离目录。

## 实现与导航

| 入口 | 作用 |
|---|---|
| `qa_lab/data.py` | 来源hash、去重、近重复家族与固定切分 |
| `qa_lab/inference.py`、`metrics.py` | 输入白名单、真实推理、ID完整覆盖、EM/F1/拒答/格式 |
| `qa_lab/train.py`、`train_artifact.py`、`closure_data.py` | LoRA、answer-only loss、本地教师数据与训练边界 |
| `qa_lab/logits_distillation.py` | 回答 token 的完整词表 teacher cache、温度 KL 与 CE 混合训练 |
| `qa_lab/mlx_experiment.py` | 同框架转换、质量比较与固定工作量测速 |
| `scripts/acceptance.py`、`scripts/verify_*.py` | 测试与保存证据的只读重算 |

[实验协议](docs/PROTOCOL.md) · [研究范围](docs/ROADMAP.md) · [历史工程验收](docs/RELEASE_AUDIT.md) · [手工候选审计流程](docs/TEACHER_CANDIDATE_AUDIT.md) · [AI辅助与贡献](CONTRIBUTIONS.md) · [数据许可](DATA_LICENSE.md) · [组件归属](THIRD_PARTY.md)

`reports/`保留各轮原始报告及其当时状态；其中“未公开”“尚无线上CI”等是发布前历史快照。当前代码、许可数据与记录已公开，当前CI见页首。模型权重、缓存与环境目录不随仓库分发。

[2026-09-19 工程维护与验证边界](docs/maintenance/2026-09-19/README.md)

[2026-09-21 工程维护与验证](docs/maintenance/2026-09-21/README.md)

[2026-09-22 implementation and verification](docs/maintenance/2026-09-22/README.md)

[2026-09-22 detail review and regression fixes](docs/maintenance/2026-09-22-detail/README.md)

Further review: [2026-09-22 evidence and export hardening](docs/maintenance/2026-09-22-readiness/README.md).

2026-09-22 deeper evaluation: [盲审与错误分析](docs/BLIND_REVIEW.md).
