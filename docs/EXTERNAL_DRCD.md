# 外部 DRCD 评测

2026-09-20新增。评测协议在推理前冻结于提交`5b3f3d1`，不覆盖CMRC历史记录。输入为DRCD官方dev的96个不同维基条目各1题，保持繁体原文；固定0.5B基线和此前三组×三个种子的全部9个适配器，不重新训练或挑选种子。

DRCD由Delta Research Center整理并采用CC BY-SA 3.0。原始版本、下载地址、输入hash、筛选记录和既有数据hash见[selection.json](../reports/external-drcd-20260920/selection.json)，许可见[归属](../reports/external-drcd-20260920/ATTRIBUTION.md)。这是新的标注数据集，但仍与CMRC共享维基百科来源，且有繁简文字差异；不是独立新业务领域，也无法证明预训练未见。

## 预先固定的协议

- 候选仅按固定ID哈希排序，不查看模型输出。每条验证参考答案offset，不截断输入；只纳入prompt不超过768 token、参考答案不超过47 token的题目，结果不覆盖长文本。
- OpenCC 0.1.7 t2s只用于去重，不修改推理或评分文本。与全部CMRC train/dev及本项目历史JSONL输入核对归一化标题、完全相同问题、上下文字符5gram Jaccard≥0.7。选中样本之间同样隔离条目、问题及近重复上下文。这是明确的启发式，不是语义去重保证。
- 主比较：三个种子的random_gold相对random_teacher的平均strict EM差；按96条目配对bootstrap，10000次、seed20260920。95%区间下界>0才支持本轮正向主比较。其余比较及F1为描述性，无多重比较校正。
- 沿用非官方strict EM和字符LCS F1，不冒充官方DRCD榜单分数。全部题目可回答；另报NO_ANSWER比例、格式失败和长度上限输出，不能据此评估不可回答问题。
- 协议、模型hash、全部种子和48-token生成预算先冻结；首次预测即消耗该评测集，后续不得据此调参后重复称独立验证。bootstrap条件于既定训练子集和种子，不含重新抽取训练数据的不确定性。

## 运行

在锁定Python环境下，最小离线重算：

```bash
.venv-ci/bin/python scripts/verify_external_drcd.py
```

真实推理需要按[公开训练入口](QUALITY_STUDY_RELEASE.md)先产生9个本地适配器。新适配器的hash不保证与历史完全一致，必须新建协议及输出目录，不能修改冻结协议来绕过核验。原运行命令为：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/external_drcd.py run \
  --out /path/to/new-frozen-evaluation \
  --adapters /path/to/retrain/supervised-07/run
```

准备入口`external_drcd.py prepare --help`需要DRCD官方dev、完整CMRC train/dev和本地适配器，并额外安装`opencc-python-reimplemented==0.1.7`。本次原始DRCD约2.2MB，无新模型下载或训练，MPS推理预算1800秒。报告保留逐条预测和失败输出，不缩小分母。CI仅重算已保存证据，不运行模型或验证本地权重。

## 实际结果

| 固定模型组 | 平均strict EM | 平均字符LCS F1 |
|---|---:|---:|
| baseline | 56.25% | 73.06% |
| oracle_selected | 56.94% | 74.52% |
| random_teacher | 57.29% | 74.55% |
| random_gold | 57.29% | 73.19% |

主比较EM差为0.00个百分点，95%区间[-5.21,+4.86]，未通过正向门槛。CMRC内部留出集上的正向EM结果没有在本次DRCD外部样本上得到支持；不能合并分母或只报告有利来源。原始浮点差约-5.8e-19，是求和精度，不是可解释的负效应。每组3个固定种子，基线1次；全部共960条预测，只有96道独立题。

[逐模型结果与配对区间](../reports/external-drcd-20260920/verification.json) · [原始预测清单](../reports/external-drcd-20260920/release-manifest.json)。该外部样本现已消耗，后续方法须使用另行冻结的数据。
