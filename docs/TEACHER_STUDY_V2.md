# 教师提示与三随机种子对照

这是第二轮探索性开发集实验，不是新的独立测试结果。第一轮完整闭环的报告仍保留在 `reports/closure-v1/`，不能将本轮dev指标与第一轮test指标直接比较。

## 输入、输出与机制

输入是同一组24道训练题，以及三套回答标签：标准答案（gold）、原提示词教师回答、新提示词教师回答。新教师仍是固定版本Qwen2.5-1.5B-Instruct、FP32、greedy；只改变system prompt，加入四个人工编写的虚构示例，强调数量、否定和问题前提必须得到文段支持。

输出是9个学生LoRA训练记录和9组完整74题dev预测：三个标签来源分别运行种子20260918、20260919、20260920。学生推理提示词仍是原始提示词；新教师提示词没有带入学生推理。

机制是响应级蒸馏：把教师生成文本当作学生的训练目标，使用answer-only cross entropy。只有回答token和结束标记参与loss，输入文段与问题的label用-100屏蔽；batch维度为1，input_ids/labels形状为[1, 序列token数]。LoRA更新q_proj/v_proj的低秩参数，基础模型参数冻结。没有获取教师概率分布，也没有Logits KL或特征对齐。

三个配对seed控制初始化和训练样本shuffle：每个seed内三组读取相同24个ID、相同顺序、各48步，每题出现两次。不同标签长度导致监督token数量不同；不能声称等token预算。改变提示词同时改变多个具体措辞，因此这是整个提示方案的对照，不能分辨哪一句指令起效。

## 提前固定的边界

- 配置与协议在新教师生成、训练、评估之前提交：`1bacfca`。
- 不做gold正确性筛选；所有非空、EOS结束的教师回答保留。截断或空回答会阻止创建训练数据。
- 不因训练中间结果改提示词、改学习率、延长步数或追加seed。
- 只评估dev，不新增旧test推理。dev已经参与前一轮诊断，不再是独立确认样本。
- 一个方法只有三个seed都满足整体EM>19/74且有答案EM>=19/33，才可进入后续外部验证；这不等于生产可采用。
- 报告逐seed结果、均值、样本标准差、paired fixes/regressions。三个seed只描述本轮波动，不提供显著性或泛化保证。

结果见 [实际报告](../reports/teacher-study-v2/RESULTS.md)。源码快照保留了本轮实际执行版本；历史实验不被当前源码变化覆盖。

## 离线核验

```bash
.venv/bin/python -m pytest -q
PYTHONPATH=. .venv/bin/python scripts/verify_teacher_study.py
```

不加载模型，不重新推理，核对训练来源、配对顺序、模型和配置、adapter hash、逐条预测与指标。`scripts/teacher_study.py run` 是本次冻结执行入口，已有输出时会拒绝覆盖。

## 在新目录重跑小实验

沿用README的依赖安装和学生下载，再运行 `scripts/download_teacher.py` 下载免费教师。Mac MPS需要可访问Metal；无MPS请先用CPU做学生小样本冒烟，不默认9组训练可在CPU上快速完成。所有模型本地运行，无需GPT订阅或API。

从仓库根目录执行以下命令。先创建一个从未使用的目录，以下示例只重跑新提示词教师及一个学生seed；完整9组使用相同方法依次更换标签来源与三个固定seed。

```bash
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_OFFLINE=1
mkdir work/teacher-study-reproduction-01
.venv/bin/python -m qa_lab.closure_data teacher \
  --teacher-config configs/teacher-study-v2/teacher.json \
  --output work/teacher-study-reproduction-01/teacher
.venv/bin/python -m qa_lab.closure_data distill \
  --teacher-config configs/teacher-study-v2/teacher.json \
  --teacher-report work/teacher-study-reproduction-01/teacher \
  --output work/teacher-study-reproduction-01/labels
.venv/bin/python -m qa_lab.train_artifact \
  --artifact work/teacher-study-reproduction-01/labels \
  --config configs/teacher-study-v2/train-20260918.json \
  --output work/teacher-study-reproduction-01/adapter
.venv/bin/python -m qa_lab.inference \
  --adapter work/teacher-study-reproduction-01/adapter --splits dev \
  --output work/teacher-study-reproduction-01/dev
```

标准答案组可使用已有 `data/teacher-study-v2-gold`，原教师组可使用 `data/distilled-local-v1`。这两套都是保存的训练数据，不是读取测试标签。每个运行使用独立adapter与dev输出目录。想复现固定标签下的训练随机性，可直接使用已保存的 `data/teacher-study-v2-prompted`，避免重新生成教师标签引入额外变化。

## 一个需要真正理解的问题

教师训练标签EM提高一点，为什么学生开发集仍可能变差？先区分标签整体正确率、类别分布、错误类型、标签长度和学生能否学到这些目标。然后比较配对seed结果以及“修复哪些题、退化哪些题”。只观察teacher总EM或training loss，都不足以证明蒸馏有效。

## 主动回忆（先不看答案）

1. 教师只提供回答文本时，为什么不能把这次方法叫Logits蒸馏？
2. 为什么教师EM更高，学生EM仍可能更低？给出一个具体机制。
3. 三组同样训练48步，为什么仍不算等token预算？
4. 同一个seed内共享初始化和shuffle，排除了什么干扰，又没有排除什么？
5. 总EM提高而有答案EM降低，可能对应什么行为变化？
6. 新教师格式不合法但normalized EM得分为1，是否矛盾？评测应如何披露？
7. 如果只保留教师答对的题再与全量gold-SFT比较，会新增哪些混杂因素？
8. 三个seed结果一致是否就能证明跨文章泛化？还缺什么证据？
9. 旧test已经展示过，为什么本轮不继续拿它挑提示词？
10. 若你只有一小时额外计算预算，会优先改善教师、加训练步数还是收集新评测？说明你会先读哪些证据。

固定seed不保证跨设备、软件版本或所有算子逐位一致。本轮只声称记录并控制这些seed与样本顺序，参见 [PyTorch reproducibility说明](https://docs.pytorch.org/docs/2.8/notes/randomness.html)。
