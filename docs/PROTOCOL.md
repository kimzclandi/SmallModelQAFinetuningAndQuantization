# 阶段一冻结协议

任务：输入一段英文计算复杂性材料和一个问题，输出最短原文答案片段；没有答案时输出严格的 `NO_ANSWER`。没有检索、工具调用或闭卷知识问答。输入白名单是 `QAInput(context, question)`；gold、is_impossible、split 不进入模型提示词。

## 数据与隔离

来源是官方 SQuAD 2.0 **dev** 文件中的单一文章。重新切出的 test 仅是本项目 holdout，不能作为官方 SQuAD 成绩，也不能排除原模型预训练见过 SQuAD。精确来源 hash 在配置中固定。

1. 规范化大小写、词边界，按 `(context, question)` 去精确重复。
2. 同文段的所有问题属于一个家族；文段词集合 Jaccard ≥0.70，或问题词集合 Jaccard ≥0.80，或问题 SequenceMatcher 相似度 ≥0.90，合并为连通分量。
3. 按 seed 与家族内最小 ID 的 SHA-256 排序，前 60% 家族训练、再 20% 开发、其余测试。整数取整导致样本数比例不同；不按实验结果平衡或重切。
4. 输出内容与 manifest 冻结；禁止覆盖。必要测试确认跨 split 不触发上述近似规则。

这排除了检测规则覆盖的近重复，不等于语义级无泄漏。不同段落仍可共享概念和事实。未来需新增整篇文章级外部测试；不替换本版本测试集。

## 预定对照与指标

- 基线：固定 revision 的原始 Instruct 学生；没有本项目训练，**并非从未训练的模型**。
- 无模型对照：始终 `NO_ANSWER`。用来揭示不可回答比例带来的分数。
- EM：英文冠词/标点/大小写标准化后的完全匹配，多个 gold 取最好。
- token F1：标准化词 token 的多重集合重叠；单题对多个 gold 分别计算后取最好，再 macro average。
- `NO_ANSWER` 严格识别；空输出/近似标记不获正确拒答分。无答案题非拒答均计质量 0。
- format validity：非空，且严格拒答或原文连续子串；与 EM/F1 分开，归一化相同也可能格式不合格。
- 分别报告 answerable/unanswerable、错误拒答率、无答案题非拒答率和 family-macro EM。非拒答率是风险代理，不是经人工确认的幻觉率。
- 自动错误标签是症状而非根因；知识缺失、指令误解不能仅凭输出自动确诊。

## 推理与性能

PyTorch 2.8.0、Transformers 4.56.2、FP32、eager attention、batch=1、greedy、KV cache 开启。最大输入 2048 tokens，超过即失败，禁止静默截断。最多生成 48 tokens（含 EOS）。配置固定在 `configs/baseline.json`。

两次非数据集的合成样本预热不计指标。TTFT 从 tokenization 和设备传输完成后的同步点开始，包含 prefill 和首 token 的 device 同步；decode tokens/s = 总的后续 token 数 / 总 decode 时长，含 EOS。另存包含编码/解码的端到端耗时。仅单进程串行测量；不是服务并发吞吐。若比较速度，应另加固定输入/固定输出长度性能实验，任务自然停止长度不同不能直接推断加速。

RSS 每 10 ms 采样；MPS allocator/driver 每 token 采样。报告 sampled maximum，不能称精确峰值；统一内存下这些数值不可相加。缓存分配器可能保留内存。模型文件存储和 FP32 参数内存是不同口径。

## 测试集使用纪律

阶段一固定一次完整 baseline；开发集用于失败分析与后续训练设计。测试集逐条预测保留供审计，不据此构造训练样本。后续阶段比较候选应先在 dev 冻结选择与配置，再按事先登记的实验阶段运行 test；记录每次访问，禁止根据 test 反复调参。

本轮候选只有 baseline；两步训练 smoke 不做 test 评估。计划阶段二为 baseline / 同训练预算 gold-SFT / response-distilled 三组；阶段三为同一权重、同框架浮点/量化；阶段四仅用开发集选一个错误类别进行一轮修复，报告 paired fixes/regressions。尚未运行的对照不填数字。

## 代码快照与执行环境修复

基线实际运行的核心代码冻结于 `reports/baseline-v1/source/qa_lab/`，与 run.json 的 source_sha256 对应。之后仅修复硬件名称查询遭沙箱拒绝时的异常处理，以及命令记录的本机路径脱敏；没有改数据、评分、提示词或解码。验证脚本核对冻结代码，当前代码用于新运行。CPU 冒烟首次因 sysctl 权限失败，保留失败日志并记录后续修复。基线 run.json 导出时只将 argv[0] 的绝对路径改为项目相对路径，原始元数据另存本地，变更 hash 记录在 reports/export-redaction.json。
