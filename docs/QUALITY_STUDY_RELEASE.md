# 中文数据质量对照：公开复现入口

2026-09-20新增实验。输入是公开CMRC2018文段、问题及参考答案；学生输出原文答案片段或NO_ANSWER。方法比较教师自动核验、使用参考标签的选样，以及同题监督目标替换。最终留出评估只使用已冻结模型；不回填历史结果，不声称业务效果。

## 读结果与证据

[结果与限制](../reports/quality-study-20260920/RESULTS.md) · [原始数据与预测清单](../reports/quality-study-20260920/release-manifest.json)

- `calibration-04`、`score-calibration-04`：1.5B核验器的三种提示、标签似然差和失败验证。
- `fresh-05`：192 train /64 dev /96 holdout、1.5B原始教师输出、逐条原子缓存与故障恢复收据。
- `teacher-3b-06`：更大候选在96条合成挑战和64题dev上的失败结果。
- `supervised-07`：三组×三seed，共576训练更新；目标、步骤、token计数、预测和评分。
- `holdout-08`：未训练基线和9个固定适配器的一次96题评估，全部结果保留。此留出集现已消耗，后续方法不得再把它当新验证。

各阶段协议描述当时边界，早期记录中的“未推理holdout”并不否定后续holdout-08。更早的pilot-01/semantic-02/holdout-03探索方法文档保留作为过程背景，其原始本地交付包不包含在本次公开包中；当前结果均可从上述公开文件独立重算。模型与适配器权重不分发，保留训练时记录的hash和本地重载检查；公开CI不验证被排除的权重。

## 1. 最小离线验收

Python3.12，无模型、GPU或API key。首次安装需要网络，之后运行离线。

```bash
uv venv .venv-ci --python 3.12
uv pip install --python .venv-ci/bin/python -r requirements-ci.lock.txt
.venv-ci/bin/python scripts/verify_quality_release.py
.venv-ci/bin/python scripts/acceptance.py --output work/acceptance-quality-01
```

第一个入口重算新实验的分组、指标、阈值和决策；第二个还执行全仓测试及历史证据检查。所有记录文件的清单和SHA256必须一致，多出、缺少或损坏文件都会报错。离线通过只证明保存证据自洽，不代表重新训练或模型收益。

## 2. 真实两题CPU推理

使用锁定推理环境与固定0.5B模型。首次模型下载约1GB；无付费API。沿用已有缓存时设置HF_HOME为相应目录。

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.lock.txt
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
.venv/bin/python scripts/download_model.py
HF_HUB_OFFLINE=1 .venv/bin/python scripts/reproduce_quality_release.py smoke \
  --output work/quality-smoke-01
```

加载模型前先校验公开清单和全部文件hash；缺失、多出或被修改的文件会在创建输出前失败。输出`predictions.jsonl`和`receipt.json`；已有目标目录直接失败，reports/data/configs及其符号链接别名不能作为输出位置。每题完成后立即写入预测；可捕获的异常会保留已完成行，将receipt标为failed并重新抛出异常。SIGKILL或断电不保证更新最终状态，也不提供自动恢复训练。这只是公开dev前两题的真实推理，不是完整质量评测。

## 3. 重新训练全部对照

需可用MPS与上述模型缓存；实际运行环境为48GiB统一内存Mac，其他硬件未验证。使用同一锁定环境，无需下载3B教师，因为原始1.5B教师输出和训练分组已保存。

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/reproduce_quality_release.py train \
  --output work/quality-retrain-01
```

复用公开的固定三组输入，重新产生协议时间、当前代码hash、9个LoRA适配器、576步日志及64题dev预测，最后对新输出重算。总预算30分钟在样本/更新边界检查；不是挂起内核的强制中断。损失、梯度异常直接报错。训练中断后保留部分产物，不支持优化器检查点恢复；用新的输出目录重试，不能删失败记录后冒充一次完成。

命令不会对holdout自动推理，也不会修改reports。底层训练函数已实际完成上述9次训练；新增包装入口的目录与协议连接另作隔离测试，未声称再次完整重训。浮点训练跨平台不保证逐token一致；本地保存/重载检查与异机复现分开报告。

## 限制与归属

全部自然问答都有答案，无法评估中文无答案拒答。数据是同一公开CMRC2018语料的新切分，不是新外部来源或预训练未见证明。主要比较仅替换同题目标，仍伴随目标长度/监督token变化；选样比较还混合题目难度和覆盖。合成挑战不能代表自然问题总体。

源数据归属与CC BY-SA4.0见公开包ATTRIBUTION.md和SOURCE_LICENSE.txt；基础模型、框架与AI辅助开发见[贡献说明](../CONTRIBUTIONS.md)。不声称提出新的蒸馏算法或训练框架。

## 发布前验收记录

本次在新建Python3.12环境安装CI锁后，92项测试与全部10个验收检查通过，前后冻结文件hash不变；公开两题CPU推理入口实际完成。重新训练包装器的目录、协议、输入hash和拒绝覆盖行为已测试，底层9次完整训练与适配器重载另有原始记录。公开CI仍只做保存证据重算，不代表异机模型执行。约3.6MB公开证据不含权重、本机缓存路径或访问令牌。

2026-09-20追加入口可靠性修复：前置完整性核验、冻结目录保护、逐条预测保存与异常状态记录；覆盖输入篡改、符号链接和第二次生成失败的回归测试。未改变模型、训练参数、数据切分或已保存指标。
