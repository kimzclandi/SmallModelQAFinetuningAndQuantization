# 盲审与错误分析（2026-09-22 后续改进）

问题：规则接收只证明规则通过，不能给出人工语义准确率。已有训练比较还受到问题选择、拒答比例与监督 token 预算影响。本次提供抽样和标注分析工具，不替人填写标签，也不新增训练。

## 输入、输出与机制

输入为通过 `verify_synthetic_qc.verify_run` 的历史 QC 目录。抽样单位是候选响应；按批次 × 决定 × answerability 分层，每层无放回抽取最多 3 条，固定 seed=20260922。排序后抽样使输入顺序不影响结果。240 条候选产生 37 条样本，同题不同提示的响应可以同时被抽中，不是 37 个独立问题。

`reviewer/` 只包含不透明 blind_id、文段、问题、原始响应和空标注模板。模型、参考答案、QC 决定、原因和分层映射保存在 `coordinator-only.json`。仅把 reviewer 子目录交给审阅者。这是流程层面的盲法；阅读源码、原始数据或协调文件的人可能识别来源，不能保证完全盲审。

```sh
python scripts/blind_review.py prepare \
  --run docs/synthetic-qc-20260922/evidence/english-original-train_reference \
  --run docs/synthetic-qc-20260922/evidence/english-prompted-train_reference \
  --run docs/synthetic-qc-20260922/evidence/chinese-train_reference \
  --output work/review-new
python scripts/blind_review.py analyze \
  --bundle work/review-new/coordinator-only.json \
  --annotations work/review-new/reviewer/annotations.json \
  --output work/review-pending
python -m pytest tests/test_blind_review.py -q
```

首次分析应为 `pending`，没有语义准确率。输出必须使用新的 work/ 子目录，不覆盖旧记录。

## 人工标注规则

判断响应是否依据文段正确回答问题。等义但与参考措辞不完全一致，不能仅因字符串差异判错。拒答只在文段不足以回答时成立。每条填写 correct / incorrect / uncertain；错误类别为 wrong_answer、unsupported、false_abstention、missed_abstention，问题歧义归入 uncertain/ambiguous_question。incorrect 与 uncertain 必须说明理由。

每位审阅者使用独立模板和匿名 alias，完成后本人声明 `origin=human_attested`。模型生成或测试标签只能用 `synthetic_fixture`。两人先独立审阅再比较；不要边看他人答案边填写。程序只能验证声明与结构，不能确认真实身份或独立性。

分析时可重复传 `--annotations`，最多两人。缺失、重复、未知 ID，错包、矛盾标签、重复审阅者会被拒绝。未完成的审阅者不产生质量比例。完整结果分层显示正确/错误/不确定计数及错误类别；不确定留在分母。不同抽样率不能直接合并成全体准确率。两人的分歧进入待裁决队列，不自动生成“共识真值”，也不提供把相关候选当独立样本的置信区间。

## 下一步训练设计：草案，尚未执行或最终预登记

假设：人工确认的目标修正，能在同一题目集合上改善回答质量，而不是单靠题目选择或更高拒答比例。完成审阅、处理分歧并冻结允许使用的训练集合后，再冻结样本量、排除规则、训练配置和预算。

保留已有等量随机教师、同题参考 SFT 与筛选教师对照，避免重复包装已有实验。新增对照优先比较同题的原教师目标与经人工确认的修正目标；若目标完全相同，应保留这个负结果而不训练。相同步数不等于相同监督 token 数；分别报告两者，必要时另设 token 预算匹配分析，不能宣称同时消除了所有混杂。

沿用已使用开发集只能做开发诊断；已查看的留出集不能称为新测试集。冻结新协议时明确三 seed、每组样本量、算力/时间预算、EM/F1、有答案/无答案子集和预设采用门槛。未具备人工结果与训练资源前不运行，不使用本次 37 条抽样宣称收益。

## 最小失败案例与限制

测试中删掉一条标注或用正确标签配错误类别，分析立即失败；空模板返回 pending；构造标签全正确仍显示 synthetic_test_only。测试证明工具契约，不证明人类判断正确。当前人工标注完成数为 0。
