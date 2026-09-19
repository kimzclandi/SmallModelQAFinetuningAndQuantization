# Domain QA Lab

可审计的小模型抽取式问答实验：数据隔离 → 原始基线 → gold-SFT与响应蒸馏 → 数据覆盖对照 → 同框架量化 → 冻结候选的新来源验证。输入是文段和问题，输出最短原文答案或严格 `NO_ANSWER`；不包含检索或闭卷知识问答。

[![offline-integrity](https://github.com/kimzclandi/domain-qa-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/kimzclandi/domain-qa-lab/actions/workflows/tests.yml)

**历史五轮结论：训练候选均未通过采用门槛；Q8通过英文dev压缩筛选及96题中文新来源的质量保持检查，尚未获得业务部署验证。** 所有质量数字来自保存的逐条预测，保留拒答基线、失败样例和回归。实际蒸馏使用本地Qwen教师；手工GPT候选仅作审计，未进入训练。

## 新增：中文数据质量对照（2026-09-20）

完成9次LoRA训练、576步，并对未训练基线和全部9个适配器做一次96题留出评估。同题将教师目标替换为参考答案后，平均严格EM从17.36%升至24.31%，差+6.94个百分点，配对95%区间[+1.04,+13.54]；F1差值区间仍跨零。使用了训练参考标签，不能称为无标注自动筛选；1.5B/3B自动核验器均未通过挑战门槛。

[结果、失败案例与限制](reports/quality-study-20260920/RESULTS.md) · [离线检查 / 两题真实推理 / 重新训练](docs/QUALITY_STUDY_RELEASE.md)

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

入口执行测试、六套离线核验、gold对照与教师审计；日志只写新的`work/`子目录，前后校验`reports/data/configs`。缺失冻结汇总直接失败，不补写；禁止`python -O`。Linux CI使用同一入口，**不下载模型、不训练、不重新推理**。历史工程验收含51项测试；最新执行以Actions记录为准。

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
| `qa_lab/mlx_experiment.py` | 同框架转换、质量比较与固定工作量测速 |
| `scripts/acceptance.py`、`scripts/verify_*.py` | 测试与保存证据的只读重算 |

[实验协议](docs/PROTOCOL.md) · [研究范围](docs/ROADMAP.md) · [历史工程验收](docs/RELEASE_AUDIT.md) · [手工候选审计流程](docs/TEACHER_CANDIDATE_AUDIT.md) · [AI辅助与贡献](CONTRIBUTIONS.md) · [数据许可](DATA_LICENSE.md) · [组件归属](THIRD_PARTY.md)

`reports/`保留各轮原始报告及其当时状态；其中“未公开”“尚无线上CI”等是发布前历史快照。当前代码、许可数据与记录已公开，当前CI见页首。模型权重、缓存与环境目录不随仓库分发。

[2026-09-19 工程维护与验证边界](docs/maintenance/2026-09-19/README.md)
