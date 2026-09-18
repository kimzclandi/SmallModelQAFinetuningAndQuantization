# 用户提交教师回答：训练集小样本审计

教师标签由用户报告为 **GPT-5（Codex）**。这不是已核验的 API snapshot，也不能确认它对应 ChatGPT 界面的 Pro 模式。原始消息没有给出生成日期、采样参数、延迟或完整对话记录，均不推测。请求包不含 gold/dev/test；实际执行是否使用其他上下文未独立核验。

## 导入结果

- 请求与响应均为 24 个 ID，完全覆盖，无重复或额外 ID；全部来自原训练集。
- 收到的 JSON 有 13 处字面 `NO\_ANSWER`；`\_` 不是合法 JSON 转义。原始文本保留，仅在解析副本中将完整字符串 `"NO\_ANSWER"` 改为 `"NO_ANSWER"`。不能确定转义来自教师还是传输/粘贴。
- 修复后 24/24 符合“原文连续片段或 NO_ANSWER”的输出格式。
- 数据未进入训练器，没有按 gold 筛掉差回答，没有修改原始标签。

| 指标 | 结果 |
|---|---:|
| 样本数（训练题，不是 test） | 24 |
| 与 gold 的归一化完全匹配 | 17/24 = 70.83% |
| token F1 | 73.40% |
| 有答案题 EM | 7/12 = 58.33% |
| 不可回答题正确拒答 | 10/12 = 83.33% |
| 有答案题拒答 | 3/12 = 25.00% |
| 无答案题非拒答 | 2/12 = 16.67% |

这些是**提供的回答与本项目 gold 的一致率**，不是 GPT-5 的官方性能，也不能与学生 102 道 test 的 24.51% EM 直接比较。修复只针对收到文本的 JSON 传输格式，不改变既有 baseline 的严格拒答计分。

## 七条不完全匹配的审查

| ID | 教师回答 | gold | 观察与未决问题 |
|---|---|---|---|
| 56e1aba0e3433e1400423095 | other models of computation known to us today | an algorithm | 文段两处都讨论可计算性；教师选中了另一个有依据的表述，EM=0 不能单独证明语义错误。保留为答案范围分歧。 |
| 56e1ce08e3433e14004231a5 | NO_ANSWER | if every problem in C can be reduced to X / problem in C is harder than X | 问题把 class C 写成 problem C，且“conflict”含义不明确，参考答案本身不一致。标注/题意歧义待人工确认。 |
| 5ad542db5b96ef001a10abf5 | Cobham's thesis | 不可回答 | 原文说 polynomial，问题写 monoinomial；教师可能按熟悉概念自动纠错而忽略证据差异。是诊断假设，不是已确认根因。 |
| 56e1e9dfe3433e14004231fe | NO_ANSWER | polynomial time hierarchy / polynomial time | 原文是 NP-complete ⇒ hierarchy collapses；问题措辞容易读成反向蕴含。严格蕴含视角下拒答有可讨论理由，不改基准 gold。 |
| 56e1aff7cd28a01900c67a69 | a fixed set of rules | rules / a fixed set of rules to determine its future actions | 教师选取了包含核心答案的原文，属于边界不完全匹配；F1=0.6154，不等于事实错误。 |
| 5ad54a375b96ef001a10ac4a | DTIME(f(n)) | 不可回答 | 问题的 series of solutions 与原文 set of problems 有概念差别；自动指标视为无依据回答，是否属于可接受改写仍需独立审阅。 |
| 56e1b62ecd28a01900c67aa6 | NO_ANSWER | time | 段落围绕 time 定义，但没有明确作“most critical”排序；可能是教师过度严格，也可能是问题限定过强。 |

审查结论：不要把这 7 条全部命名为“教师幻觉”，也不要凭主观解释修改 gold 让教师得分更高。若将来做人工修订，应新建带理由和版本的数据，不修改本轮固定测试集。当前没有完成独立人工 adjudication。

## 使用范围核实与纠正

2026-09-18 核对 [OpenAI Terms of Use](https://openai.com/policies/terms-of-use/)（页面生效日期 2026-01-01），其中限制：**“Use Output to develop models that compete with OpenAI.”**

此前将“已有 Pro 可以提供教师候选回答”作为技术路径，不代表已经确认可用其回答训练 Qwen。本项目不能仅根据订阅状态推定该训练用途获得许可；这里不把个人学习项目一概判定为竞争用途，也不作确定法律结论。这份 GPT 候选当前仅用于输出审计，不进入 Qwen 训练。普通 gold-SFT 对照使用公开 SQuAD 标签，不依赖这份输出的训练权限。完整蒸馏可转用许可明确、可本地运行的开源教师，或在取得明确适用许可后再使用这批候选。

原始文本、解析副本与哈希位于 `data/teacher-received-v1/`。可重算命令：

```bash
.venv/bin/python -m qa_lab.teacher_audit \
  --raw data/teacher-received-v1/user-message.txt \
  --output work/teacher-audit-reproduction-01
```
