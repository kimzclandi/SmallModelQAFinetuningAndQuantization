# Domain QA Lab · 小模型领域问答优化实验室

一个实际运行过的「原始基线 → 教师响应蒸馏 → 权重量化 → 失败分析与一次数据改进」实验项目。面向模型训练、推理优化与实验设计的面试讲解。**四阶段最小闭环已运行；已有训练候选未通过采用门槛，不宣称优化成功或生产可用。**

本地仓库，尚未公开发布；没有调用付费API或上传数据。GPT手工候选只作审计，实际蒸馏使用本地开源教师。

## 任务与资源

输入英文计算复杂性文段和问题，输出最短原文答案或严格 `NO_ANSWER`。不含检索、不做闭卷知识问答。

- 数据：[SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/) 的 Computational_complexity_theory，CC BY-SA4.0。418题、48文段/家族，train/dev/test=242/74/102。先近重复家族合并再切分。
- 学生：[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，Apache-2.0，固定revision；baseline指未接受本项目训练。
- 教师：[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，Apache-2.0，固定revision，本地FP32生成。
- 实测设备：Apple M4 Max、48GiB统一内存、40核GPU；macOS27.0。PyTorch/MPS用于训练，独立MLX环境用于公平量化对照。

本项目的test是公开SQuAD dev内部重新切出的holdout，**不是官方隐藏测试**。同文章语义相关性、公开benchmark预训练污染仍是限制。

## 实际结果

所有数字均来自保存的逐条预测。[完整报告](reports/closure-v1/RESULTS.md)解释口径、对照与负结果。

| 方法 | 训练题/步数 | dev EM | test EM | test token F1 |
|---|---:|---:|---:|---:|
| 原始学生 | 0/0 | 25.68% | 24.51% | 31.56% |
| gold-SFT | 24/48 | 55.41% | 51.96% | 54.33% |
| 本地教师响应蒸馏 | 24/48 | 32.43% | 36.27% | 41.14% |
| 增补有答案训练题 | 48/48 | 50.00% | 48.04% | 50.25% |

**总分提高不等于达到目标。** 三个候选都使dev有答案EM低于原学生，未通过提前固定的门槛。始终拒答的test EM为52.94%；不能隐藏这一简单对照。蒸馏教师只有9/24条与gold完全匹配；格式/内容错误未被事后筛除。一轮数据修订相对gold-SFT在dev/test均只修复1题、退化5题。

量化比较使用**同一原始学生、同一MLX框架**，FP16与4-bit权重量化（group64，KV保持浮点）：

| 指标 | MLX FP16 | MLX 4-bit |
|---|---:|---:|
| test EM | 24.51% | 14.71% |
| 权重文件，decimal MB | 988.10 | 278.06 |
| 固定工作量decode tokens/s | 255.70 | 370.17 |

测速固定8个train输入、每个强制生成32tokens，按FP16/Q4/Q4/FP16串行运行，表内为每精度两次运行的中位数。约1.45×的此处解码加速伴随9.80个百分点EM损失；没有宣称低比特普遍加速。TTFT、RSS、MLX内存和逐题时延见报告，统一内存口径不能相加。不用PyTorch与MLX跨框架速度差证明量化收益。

## 第二轮：教师提示与三seed对照（仅dev）

追加了同24题、同48步的三组×三个seed真实训练，只改变教师提示方案，未新增test推理。

| 标签来源 | dev EM均值 ± 样本标准差 | dev有答案EM均值 |
|---|---:|---:|
| 标准答案SFT | 55.41% ± 2.70个百分点 | 42.42% |
| 原教师蒸馏 | 33.78% ± 2.34个百分点 | 51.52% |
| 新提示词教师蒸馏 | 55.86% ± 2.06个百分点 | 38.38% |

新教师总分提高伴随回答能力退化，三种方法都未通过三seed门槛。新方案平均总EM仅比gold高0.45个百分点，不能据此宣称蒸馏优于监督微调。始终拒答的dev EM是55.41%。三seed不是三个独立测试集，±不是置信区间；同48步也不等于监督token预算相同。

[第二轮完整证据](reports/teacher-study-v2/RESULTS.md) · [机制、复现与十道自测题](docs/TEACHER_STUDY_V2.md)。本地31项测试通过，GitHub线上CI尚未运行。

## 第三轮：扩大覆盖与跨文章检查

使用标准答案，固定242步，比较24题重复训练与全部242题训练，各3个seed。开发集决定先提交冻结，再对两篇未参与项目训练的文章（Packet_switching、Prime_number）共64题评估；未重跑旧test。

| 方法 | 新文章整体EM | 新文章有答案EM |
|---|---:|---:|
| 原始学生 | 23.44% | 46.88% |
| 24题重复训练，3seed均值 | 36.46% | 38.54% |
| 242题训练，3seed均值 | 44.27% | 47.92% |

扩展组整体EM在三个seed中均高于24题对照，两篇文章各自的均值也提高，但总EM仍低于始终拒答的50%；开发集预设门槛也未通过。新文章仍属同一公开SQuAD文件，不能声称预训练未见或跨业务泛化。固定步数不控制token、epoch和样本分布。

[第三轮完整结果](reports/coverage-v3/RESULTS.md) · [实验机制、复现与自测](docs/COVERAGE_V3.md)。本地33项测试通过；所有历史证据保留，GitHub线上CI未运行。

## 第四轮：8-bit压缩筛选（仅dev）

同一FP16原始学生导出、同MLX实现，三种精度重新运行；测速按轮换顺序各3次，固定同8个输入与32输出token。

| 指标 | FP16 | affine Q4 | affine Q8 |
|---|---:|---:|---:|
| dev EM（74题） | 25.68% | 21.62% | 24.32% |
| 权重 / MB | 988.10 | 278.06 | 525.05 |
| decode tokens/s（本轮中位数） | 266.21 | 390.87 | 313.86 |

Q8权重减少46.86%、解码速度约提高17.90%，dev少答对1题，通过预先固定的压缩门槛。它是**本地dev压缩候选**，不代表独立测试已验证、无损量化或QA任务可部署；Q4仍因质量损失未通过。没有重跑旧test或跨文章holdout。

[完整证据与回归样例](reports/quantization-v4/RESULTS.md) · [量化机制、复现与自测](docs/QUANTIZATION_V4.md)。本地36项测试通过，线上CI未运行。

## 第五轮：中文新来源验证

冻结第四轮已选Q8及FP16参考后，在CMRC2018公开dev抽取96篇文章各一题；预先固定中文提示和指标，未训练或按结果重采样。

| 指标（96题） | FP16 | Q8 |
|---|---:|---:|
| 严格EM | 45.83%（44/96） | 45.83%（44/96） |
| 自定义字符LCS F1 | 72.68% | 72.12% |

Q8通过本轮预登记的质量保持门槛，严格正确集合相同。这支持此中文样本上的压缩保真，**不表示逐条输出无损、CMRC官方成绩或企业业务达标**。全部题目有答案，拒答能力未验证；公开数据仍可能有预训练污染。没有把第四轮英文测速当成中文加速证据。

[完整结果与数据审计](reports/chinese-v5/RESULTS.md) · [指标、隔离复现与面试题](docs/CHINESE_V5.md)。本地44项测试通过；两题隔离复现入口已实际运行，异机验证尚未完成。

## 快速开始：小规模验证

从仓库根目录运行，Python3.12。首次下载免费学生约1GB。推荐至少8GiB可用内存、5GB磁盘做学生验证（最低配置未实测）；完整本地教师/训练/量化在48GiB Mac实测，建议预留10GB磁盘。CPU可用于少量推理；本仓未提供CUDA/Ascend验证。

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.lock.txt
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
.venv/bin/python scripts/download_model.py
.venv/bin/python -m pytest -q
PYTHONPATH=. .venv/bin/python scripts/verify_artifacts.py
PYTHONPATH=. .venv/bin/python scripts/verify_closure.py
PYTHONPATH=. .venv/bin/python scripts/verify_teacher_study.py
PYTHONPATH=. .venv/bin/python scripts/verify_coverage.py
PYTHONPATH=. .venv/bin/python scripts/verify_quantization.py
PYTHONPATH=. .venv/bin/python scripts/verify_chinese.py
HF_HUB_OFFLINE=1 .venv/bin/python -m qa_lab.inference \
  --device cpu --splits dev --limit 2 --output work/smoke-01
```

MPS推理把 `--device cpu` 换为 `--device mps`。受限沙箱可能看不到Metal；GPU不可用会报错，不静默回退。日常验证仅跑dev。已有输出拒绝覆盖，换新目录保留历史证据。

**完整训练/蒸馏/量化复现见 [复现手册](docs/REPRODUCE_CLOSURE.md)**，先隔离输出、再安装运行。两套依赖锁分别为 requirements.lock.txt 和 requirements-mlx.lock.txt。教师不需要API key。

## 代码与证据链

| 文件/目录 | 作用 |
|---|---|
| qa_lab/data.py | 来源hash、精确去重、近似家族、固定切分 |
| qa_lab/inference.py / metrics.py | 输入白名单、真实推理、完整ID覆盖、EM/F1/拒答/格式 |
| qa_lab/train.py | gold-SFT与训练入口冒烟 |
| qa_lab/closure_data.py / train_artifact.py | 本地教师采集、响应蒸馏数据、训练边界、数据修订 |
| qa_lab/mlx_experiment.py | 本地转换、同框架量化、质量与固定工作量测速 |
| qa_lab/teacher_io.py / teacher_audit.py | 用户手工回答采集与审计，不参与实际蒸馏 |
| configs/、data/ | 固定配置、许可数据、来源与样例 |
| reports/baseline-v1/、gold-sft-pilot-v1/ | 冻结的早期基线/对照证据 |
| reports/closure-v1/ | 教师原始失败、FP32修复、训练、dev选择、test、量化、源码快照 |
| tests/、scripts/verify_*.py | 必要逻辑测试与离线证据重算 |

证据链：模型revision/文件hash＋数据hash＋代码hash＋配置 → 逐条预测 → 逐条评分 → 汇总。训练adapter和模型权重仅在本地被忽略的目录；不提交缓存、密钥或大权重。

## 学习与面试

- [阶段一协议](docs/PROTOCOL.md)：指标、泄漏、性能口径、测试集使用。
- [原知识手册](docs/INTERVIEW.md)：领域地图、五个思维模型、三个争议及主动回忆。
- [闭环面试讲解](docs/CLOSURE_INTERVIEW.md)：最新版30秒/3分钟回答、故障与技术追问。
- [完整结果](reports/closure-v1/RESULTS.md)：教师失败、蒸馏对照、量化取舍和修复回归。
- [贡献边界](CONTRIBUTIONS.md)、[数据许可](DATA_LICENSE.md)、[组件归属](THIRD_PARTY.md)、[状态与缺口](docs/ROADMAP.md)。

## 验证范围与未完成事项

已完成同机独立环境CPU冒烟、数据重建、训练/adapter加载、真实模型评估与证据核验。CI定义已准备，尚未在GitHub运行。第一轮只有单seed，第二轮补了三seed；第三轮扩展至242题并完成两篇新文章评测，但仍是小规模公开数据且dev被重复使用。尚无中文业务评测、更多独立来源验证、充分的教师质量验证或在目标业务与硬件上通过验证的质量-效率方案；Q8已通过英文dev压缩筛选和96题中文新来源检查，但未达到业务部署验证。未做生产部署、RLHF、KV量化或自动数据飞轮；GitHub公开发布仍需用户明确授权。
