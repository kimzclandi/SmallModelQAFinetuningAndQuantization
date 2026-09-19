# 使用参考标签的监督质量对照（2026-09-20）

当前公开入口见[数据质量实验发布说明](QUALITY_STUDY_RELEASE.md)。本文描述各阶段执行时的状态；其中“未使用holdout”仅限该阶段，最终holdout-08已按冻结协议评估一次。
在[3B教师筛查](TEACHER_3B_SCREEN.md)失败后，改做有明确监督信息的诊断实验。它回答：在固定学生和训练预算下，用参考答案修正训练目标能否改善开发集表现？它不是无需标注的自动质量筛选，也不是对失败核验器的继续使用。

## 输入与对照

输入是已保存的192条CMRC2018训练样本与1.5B教师原始输出。只在EOS结束、非拒答、答案为原文片段的候选池中取样；参考标签只来自train。

| 组 | 59条输入的来源 | 训练目标 | 匹配参考数 |
|---|---|---|---:|
| oracle_selected | 教师输出严格匹配任一训练参考的全部59题 | 对应教师输出 |59|
| random_teacher | 固定种子从候选池随机取59题 | 对应教师输出 |22|
| random_gold | 与random_teacher完全相同的题目与顺序 | 第一条参考答案 |59|

主要比较为random_gold − random_teacher：输入题目和每个seed的训练顺序相同，只替换目标。次要比较oracle_selected − random_teacher还混合了题目难度和覆盖差异，不能全部解释为答案质量效果。严格不匹配不一定表示语义错误；参考标签本身也可能不完整。

## 冻结训练协议

Qwen2.5-0.5B-Instruct固定revision；PyTorch/MPS FP32；LoRA只作用q_proj/v_proj，rank8、alpha16、dropout0；learning rate 1e-4。3组×3固定seed×64更新，共576更新。batch size1、answer-only loss、梯度裁剪1.0，非有限loss/梯度直接失败。各组更新次数相同，但输入/监督token数并不相同，实际token数随步骤记录。

开发集沿用64题，只作探索性分析。96题holdout未复制到本轮训练目录，也不执行推理。已查看过开发集，因此任何提升都不是独立验证；不挑选最好seed、不根据结果改训练参数。本轮为2026-09-20新增研究，不回填历史结果。

## 本轮实测结果

9次训练和576次更新全部完成，训练加开发集推理总用时405.83秒。固定三个seed的结果如下：

| 组 | 各seed严格答对数（分母均64） | 平均严格EM | 平均字符LCS F1 |
|---|---|---:|---:|
| oracle_selected |17 /20 /17|28.13%|68.48%|
| random_teacher |17 /18 /16|26.56%|67.10%|
| random_gold |22 /22 /20|33.33%|71.76%|

未训练学生的既有基线为17/64，F1为66.69%。参考答案替换相对同题教师目标，平均EM差+6.77个百分点，配对文章bootstrap 95%区间[-0.52,+15.10]；F1差+4.66个百分点，区间[-0.09,+9.77]。两者都跨零，且dev已经使用过，不能声称收益已被独立验证。

参考筛选相对随机教师的EM差仅+1.56个百分点，区间[-2.60,+6.25]。此结果也不支持“只保留教师已答对的题目一定更好”。筛选可能改变题目难度、覆盖和监督目标分布，不能把差异解释成纯粹的质量效应。

三个seed累计监督token分别为：oracle_selected 1659、random_teacher 1975、random_gold 2299。因此同题目标替换仍伴随目标长度/监督token变化，本实验不能单独分离这些机制。没有为了消除这一差异而在看到结果后更改训练预算。

## 运行与验证

`RUN_ROOT`是含fresh-05和semantic-02协议的外部本地证据目录；公开证据现位于`reports/quality-study-20260920/`，最小复现不再依赖外部目录。先准备再训练，已有目标目录拒绝覆盖。

```bash
.venv-ci/bin/python scripts/supervised_quality_control.py prepare "$RUN_ROOT"
HF_HUB_OFFLINE=1 .venv/bin/python scripts/supervised_quality_control.py run "$RUN_ROOT"
.venv-ci/bin/python scripts/verify_supervised_quality.py "$RUN_ROOT"
HF_HUB_OFFLINE=1 .venv/bin/python scripts/reload_supervised_adapters.py "$RUN_ROOT"
```

实际训练需要已缓存固定模型和requirements.lock.txt环境；离线核验不加载模型。核验器重新构建分组，检查缓存身份、步骤和抽样顺序、损失/梯度有限性、token累计、适配器hash、逐条预测覆盖与指标。重载检查对每个保存适配器重跑前两条dev，要求输出token IDs、文本、停止原因全部一致；它只验证保存/重载一致性，不证明质量提升。

主次比较使用按文章配对的bootstrap区间（每文章一题），先对固定三个seed求平均，再重采样64篇文章。区间只反映固定训练子集及seed条件下的样本不确定性，不包含抽样子集不确定性，也不是三个独立数据集。
