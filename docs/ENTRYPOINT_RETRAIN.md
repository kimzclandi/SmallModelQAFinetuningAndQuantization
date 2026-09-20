# 完整公开训练入口重跑

2026-09-20，通过提交`446299e118940aa274ec9404ed30fe2e7191ffef`的公开训练命令完成9次LoRA训练和dev推理。新增记录单独保存，原始实验不变。本次仅验证运行链路，没有更改方法、挑选种子或再次评估holdout。

## 输入与执行

固定0.5B模型、原始三组各59条训练样本、三个种子、每组每种子64次更新及64题dev。使用已有MPS机器与本地缓存，无下载或API调用。实际Python环境逐项符合`requirements.lock.txt`；不是新建环境或异机复现。训练命令和包版本见[environment.json](../reports/entrypoint-retrain-20260920/environment.json)。

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/reproduce_quality_release.py train \
  --output work/quality-retrain-02
```

必须使用不存在的输出目录。准备环境、模型缓存和失败行为见[复现手册](QUALITY_STUDY_RELEASE.md)。

## 实际结果

9次训练、576次更新完成；训练及dev推理用时413.25秒。这是单次运行耗时，不是性能优化对照。9个适配器在本地完成文件hash核验，不包含本轮独立重载推理测试。576条dev预测文本与原记录逐条一致，各次运行的EM/F1也一致。

| 训练组 | 原记录平均EM | 重跑平均EM | 原记录平均F1 | 重跑平均F1 |
|---|---:|---:|---:|---:|
| oracle_selected | 28.13% | 28.13% | 68.48% | 68.48% |
| random_teacher | 26.56% | 26.56% | 67.10% | 67.10% |
| random_gold | 33.33% | 33.33% | 71.76% | 71.76% |

每组为3个种子在相同64题上的均值；576条是9次运行的预测数，不是576道独立题目。完整逐运行对比见[comparison.json](../reports/entrypoint-retrain-20260920/comparison.json)，步骤日志与逐条预测见[文件清单](../reports/entrypoint-retrain-20260920/release-manifest.json)。

## 无模型离线核验

```bash
.venv-ci/bin/python scripts/verify_retraining.py
```

校验文件集合与hash、来源协议、训练次序、有限损失和梯度、token计数、预测配对、指标与新旧结果比较。已接入`acceptance.py`和CI。公开包不含模型/适配器权重，CI不运行训练，也不重验被排除的权重；[verification.json](../reports/entrypoint-retrain-20260920/verification.json)保留当时本地权重校验记录，离线重算明确关闭该检查。

## 边界

这里只补全公开入口的同机端到端运行证据。dev已被多轮使用，一致结果不是新的独立质量验证；历史holdout仍已消耗。参考答案监督、监督token不相等、同语料及全部可回答题目的限制均不变。真实外部数据验证和异机训练复现仍未完成。数据归属与许可随新证据包保留，适配器权重仅存本地。
