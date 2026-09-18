# 第三轮：训练覆盖度与跨文章验证

## 输入与输出

训练输入只来自第一轮已冻结的Computational_complexity_theory训练集。对照组使用原24题，扩展组使用全部242题，标签均为SQuAD标准答案，不用教师生成或新评测题。扩展组116道有答案、126道无答案，原24题组为12/12。两组都从同一Qwen2.5-0.5B-Instruct开始，以相同LoRA、学习率、242步和三个固定seed训练。输出包括6个本地adapter、逐步loss/监督token记录、6组74题dev预测，以及冻结后7组64题跨文章预测（原始模型+6个候选）。

## 为什么这样设计

前两轮显示教师质量不足，单纯增强拒答提示又损害有答案题能力。因此本轮先研究可靠标签的覆盖度，而不扩大低质量教师标签。

24题组训练242步，相当于每题10或11次曝光；242题组每题只出现一次。固定步数用于避免把额外优化步数误当成数据收益，但不控制epoch、样本长度、监督token数、类别比例或知识覆盖。这是“更广覆盖、较少重复”整套训练策略的对照，不能声称纯粹识别了样本数的因果作用。

## 新文章评测的边界

从同一官方SQuAD2.0公开dev文件选Packet_switching与Prime_number两篇文章，各16道有答案、16道无答案题，共64题、46个文段家族。选择顺序由固定hash决定，每个文段每类最多一题。按原阈值排除与旧train/dev/test近似的样本、以及新集合内部近似问题；同文段不同类别可以共存，family_id因此用于保留相关性。

算法读取旧test文本只用于去重隔离，没有进行旧test推理或按旧test分数选模型。新评测标签只用于预先固定的类别平衡和最终计分，不用于训练、教师提示、学习率或模型选择。

这些是项目级未训练文章，不是新的独立数据源。公开benchmark可能被模型预训练见过；两个文章也不足以覆盖计算机问答业务。50%无答案的人为平衡分布不等于真实线上分布，不应直接解释为业务准确率。数据来源与许可：[Stanford SQuAD官方页面](https://rajpurkar.github.io/SQuAD-explorer/)，CC BY-SA4.0。

## 防止测试集驱动迭代

1. 先固定文章、抽样、去重、两个训练方案和三个seed。
2. 只完成开发集评估；要求每个seed整体EM>19/74、有答案EM>=19/33。
3. 把dev决定与adapter hash提交Git，再记录freeze.json。
4. 对原学生及全部6个候选各评估一次新文章集，包括未过dev门槛的候选。
5. 如实报告改善/回归，不用新文章结果追加训练或挑seed。

完整实际结果见[第三轮报告](../reports/coverage-v3/RESULTS.md)。该报告的dev与external指标分开，不与旧test总分混用。

## 小规模复现

先按README安装环境并下载学生。全部本地MPS运行；无GPU时可做CPU两题推理，不能保证242步训练耗时可接受。新输出必须隔离，以下示例复现扩展组一个seed和dev验证，不默认反复访问新holdout。

```bash
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_OFFLINE=1
mkdir -p work
.venv/bin/python -m qa_lab.train_artifact \
  --artifact data/coverage-v3-gold242 \
  --config configs/coverage-v3/train-20260918.json \
  --output work/coverage-reproduction-01-adapter
.venv/bin/python -m qa_lab.inference \
  --adapter work/coverage-reproduction-01-adapter --splits dev \
  --output work/coverage-reproduction-01-dev
```

要比较24题控制组，仅将artifact改成 `data/teacher-study-v2-gold` 并换新输出目录。三个seed对应configs中的三份固定配置。原始执行命令保存在reports/coverage-v3/commands.jsonl。`scripts/coverage_study.py`保留原始分阶段入口，有已冻结输出时禁止覆盖。

离线证据重算不加载模型、不再进行测试推理：

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_coverage.py
```

要重建新文章数据，应在隔离checkout中将旧数据目录移入备份后运行prepare，而非覆盖原目录。官方原文件SHA256在configs/data.json中，prepare会强制核对。已有公开基准仍有预训练污染风险，重新抽样并不能消除。

## 面试：30秒与追问

“我发现24条标签的蒸馏容易把学生变成更倾向拒答的模型，于是增加标准答案覆盖，设置同242步的24题重复训练对照，并用3个seed观察波动。候选先按开发集切片门槛冻结，再在两篇未参与训练的文章上统一评估。我同时报告总分、有答案分数和逐题回归，不用测试结果继续挑模型。”结果与是否采用应引用实际报告，不提前填入提升。

核心追问：同样步数不代表同样训练信息量；同文章切分不代表跨文章泛化；新文章也不代表预训练未见；跨seed稳定不代表跨业务稳定。

## 主动回忆（不附答案）

1. 为什么这轮先扩大gold，而不是扩大教师生成数据？
2. 两组同242步，排除了什么混淆，又留下什么？
3. 24题重复10次和242题一次，各可能出现怎样的过拟合或欠拟合？
4. 为什么要按文章隔离，而不仅仅随机切分问题？
5. 新holdout有50%无答案题，始终拒答基线是多少？它能告诉你什么、不能告诉你什么？
6. 用新holdout类别标签做预登记平衡，与用它挑模型有什么区别？
7. 如果扩展组dev更好、跨文章更差，你会优先怀疑哪些机制？
8. 如果一篇文章提升、另一篇退化，总分应如何解释？
9. 为什么保留未过dev门槛候选的外部结果，仍不能按它重新选seed？
10. 将本实验迁移到企业知识库时，还需要怎样的数据隔离、业务指标与验收？
