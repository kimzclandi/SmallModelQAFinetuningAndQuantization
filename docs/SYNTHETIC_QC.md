# 可追溯的抽取式候选质检（2026-09-22）

本次新增纯 Python 离线质检模块；使用历史真实教师候选执行新规则，没有新增教师生成、模型训练或人工语义复核。日期为 Asia/Singapore 本地日期，运行收据记录实际 UTC。原有 `data/`、`reports/`、`configs/` 字节不变。个人项目；不属于企业实习或实验室成果。

## 原有能力与本次范围

原有 `qa_lab.teacher_io` 检查手工候选 schema、ID 和原文片段格式；`quality_pilot.accept` 检查 EOS、非拒答与原文片段；`supervised_quality_control.groups` 用训练参考答案筛选并建立等量随机教师／同题参考目标对照。英文 `data.related` 和中文 `chinese.near` 已有家族与近重复隔离。直接复用这些定义与已有实验，不另引入自动标注模型。

新增 `qa_lab.quality_control.audit` 将结构检查、参考检查、重复与跨划分审计分开，输出 accept / reject / review；`scripts/synthetic_qc.py` 提供统一 CLI。独立输出保留原始候选、原始/输出响应、稳定原始样本 ID、候选位置及内容摘要 ID、来源行与文段家族、逐项检查、规则版本和全部原因。没有清理或改写响应：`transformations=[]`，前后内容相同。历史 provenance 缺项明确为 `unknown`。候选不是可靠推理链。

## 输入、权限与输出

仅需仓库规定的 Python 3.12；运行模块只用标准库及仓库自身模块。测试依赖沿用 `requirements-ci.lock.txt`，无需模型、GPU或API。

自定义输入是三个 JSONL 文件：

- `sources`：训练来源，必填非空字符串 `id, split, family_id, context, question`，其中 split 必须为 train。允许保留 `source_title` 等已有来源字段。许可为 `train_reference` 时可提供 `answers: list[str]`、`is_impossible: bool`；参考本身必须自洽且答案能定位到原文。损坏或缺失参考导致待复核。
- `candidates`：`id`、`prediction`（原始响应）、`stop_reason`，可选 `provenance` 中的 model_id、revision、prompt_version、generation_parameters、run_id。未知 ID、重复候选 ID、非字符串或空响应拒收；缺失停止信息待复核。
- `split-index`：严格只允许 `id, split, family_id, context, question` 五字段。包含 train/dev/test 或 holdout 的输入用于隔离检查，禁止参考答案及有无答案标签；同划分重复 ID 直接报错。

`--label-permission none` 的自定义来源若包含标签会报错；不会默默忽略越权字段。预置模式从旧文件投影掉标签。历史文件共置文段、问题与标签，适配器必须解析这些文件，但只把五个允许字段传入隔离检查，不查阅、统计或用留出标签制定规则。没有重新推理留出集，也没有进行模型选择。

```bash
python scripts/synthetic_qc.py \
  --sources work/input/train.jsonl --candidates work/input/candidates.jsonl \
  --split-index work/input/label-free-index.jsonl --language zh \
  --label-permission train_reference --output work/my-qc-01
```

输出必须是不存在的仓库 `work/` 子目录；冻结目录及其符号链接别名被拒绝。`raw-candidates.jsonl` 保留候选 JSON 值，原输入文件 SHA-256 绑定原始字节；`sources.jsonl`、`split-index.jsonl` 保存实际输入；`records.jsonl` 保存逐条决定；三个决策 JSONL 单独保存；`leakage.jsonl` 保存可追踪的跨划分配对；`summary.json`、`run.json`、`manifest.json` 保存统计、配置、实际时间、代码/来源/产物哈希。缺失候选 ID 单独列出，不伪造响应或默默补齐。批次源 schema/JSON 损坏会整体失败，记录级候选错误则逐条拒收。

## 规则与适用边界

| 检查 | 判定 | 不能推出的结论 |
|---|---|---|
| 非空字符串、无前后空白、严格 NO_ANSWER 或连续原文片段 | 不合法拒收；连续片段单独记录 | 片段能回答问题、是最短正确片段 |
| EOS 完成；响应不超出文段字符数（拒答除外） | 截断拒收；未知停止信息复核 | token 上限等价于语义完整；没有人为新增长度阈值 |
| 获授权且有效的训练参考 | 严格匹配可通过；拒答与可回答标签矛盾拒收；其他不匹配复核 | 参考答案穷尽所有语义正确表达，或独立测得语义准确率 |
| 没有参考权限 | 即使格式合法也复核 | NO_ANSWER 正确、自动规则已证明正确性 |
| 同 ID 多个候选 | 全部拒收，避免顺序决定保留谁 | 自动挑到最优候选 |
| 相同文段与问题、不同 ID | 全部复核；相同答案本身不算重复 | 共享 NO_ANSWER 的不同问题是重复题 |
| 跨划分相同 ID／家族／原始文段 | 阻断接收，列出对侧 ID/划分 | 原始数据被物理删除 |
| 跨划分近重复 | 待复核，保留原记录 | 阈值经过本次人工校准、可批量自动删除 |

近重复沿用既有阈值，未依据本次 holdout 或标签调参。英文：小写词归一化后文段词集合 Jaccard≥0.7，或问题 Jaccard≥0.8，或问题 SequenceMatcher≥0.9。中文：NFC、小写、去空白标点后，文段字符 5-gram Jaccard≥0.7，或问题 SequenceMatcher≥0.9。两者为启发式审计候选：常见问句可能误报、改写可能漏报；不保证语义隔离。当前两两扫描适用于几百条数据，不宣称大规模吞吐。

`checks.reference_match` 与 `checks.format_valid` 独立计算，语义栏始终为 `not_assessed`。`accept` 表示参考严格一致且工程门槛通过，不是人类确认；`review` 不等于错误，也不得自动作为训练集。本模块不提供训练器自动接收待复核项的入口。

## 本次真实运行结果

```bash
python scripts/synthetic_qc.py --preset english-original --label-permission train_reference --output work/qc-original-01
python scripts/synthetic_qc.py --preset english-prompted --label-permission train_reference --output work/qc-prompted-01
python scripts/synthetic_qc.py --preset chinese --label-permission train_reference --output work/qc-chinese-01
# 分别再以 --label-permission none 和新的输出目录运行无标签权限模式。
python scripts/verify_synthetic_qc.py
python scripts/acceptance.py --output work/acceptance-qc-01
```

[全部逐条记录与运行配置](synthetic-qc-20260922/evidence/)；原始候选来自 24 条英文原提示、同 24 题的新提示、192 条中文真实教师输出，共 240 个候选响应（不是 240 道独立题）；两种权限运行共 480 条审计记录，不是新增生成数量。

| 候选来源 | 权限 | 总量 | 接收 | 拒收 | 待复核 | 格式合法 |
|---|---|---:|---:|---:|---:|---:|
| 英文原提示 | 训练参考 | 24 | 8 | 11 | 5 | 22/24 |
| 英文新提示 | 训练参考 | 24 | 8 | 12 | 4 | 20/24 |
| 中文 | 训练参考 | 192 | 59 | 40 | 93 | 181/192 |
| 英文原提示 | 无标签 | 24 | 0 | 2 | 22 | 22/24 |
| 英文新提示 | 无标签 | 24 | 0 | 4 | 20 | 20/24 |
| 中文 | 无标签 | 192 | 0 | 18 | 174 | 181/192 |

三个来源均未发现既定规则下的跨划分泄漏；英文审计完整242/74/102输入，中文审计192/64/96输入，不只检查候选子集。重复检查也未发现候选重复。不能将零命中解释为没有一切形式的数据泄漏。

原因允许重叠，不能加总当作独立样本数：英文原提示格式失败2、拒答参考矛盾11、非拒答参考不匹配5；新提示分别4、10、6。中文格式失败11、未EOS完成7、拒答参考矛盾22、非拒答参考不匹配111。最后一项有18条同时有其他拒收原因，因此最终待复核为93。完整原因分布见各 summary。

英文两组原始都是12条有答案／12条无答案。原提示接收5有答案+3无答案，有答案占62.5%；新提示接收2有答案+6无答案，有答案占25%。新提示有答案接收率仅2/12，原提示为5/12；接收总数一样掩盖了分布差异。中文全部有答案，接收59/192，不能验证拒答能力。没有难度标注或人工复核，不能声称筛选没有偏向易题。

## 失败分析与质检有效性

针对空值／错误类型／解释或代码围栏／大小写／空白、错误原文片段、错误拒答、无参考权限、损坏参考、未知或重复ID、缺失候选、重复输入、跨split家族/文段/ID、英文及中文近重复、截断生成、输出覆盖/符号链接和证据篡改执行行为测试。测试使用AI辅助构造的已知用例，不是人工抽检自然问题，不将通过率当语义准确率。

真实失败例可在英文原提示记录中按 ID `56e1aba0e3433e1400423095` 查到：输出 `algorithm` 可在原文定位，而参考为 `an algorithm`，本模块待复核；原有英文 normalized EM 可能判对，两种指标不混用。新提示同 ID 输出 `NO_ANSWER`，与有答案参考矛盾，拒收。英文新提示大小写变化如 `Analysis of algorithms` 可违反严格原文约束，即使旧评测归一化后判对也不静默改写。

没有完成自然问题的独立语义复核，因此未测语义误接收/误拒收率。未来如要测量，应在通过、拒收、待复核及有无答案各层抽样，盲化规则决定，由独立复核者标注最短片段及可回答性并处理分歧，保留抽样概率与分母；该工作本次未执行。固定规则的参考一致性统计不能替代这项验证。

## 训练证据复用与未完成部分

逐条比对确认，本次中文接收的59条 **ID、文段、问题、目标响应全部等于** 历史 `supervised-07/oracle_selected.jsonl`。因此可复用已有相关训练证据；不是本次新训练，不为新模块宣称新增收益。`verify_synthetic_qc.py` 验证此身份等价，历史验收重算模型指标。

历史中文每组59条、3 seed、每次64步，都是同0.5B模型与64题dev协议。等量随机教师组是从“EOS完成、非拒答、原文片段”的池中抽样，**不是从全部192条未经结构筛选的响应抽样**；参考目标组使用与随机教师完全相同的问题。历史 dev 严格 EM：

| 历史组 | seed 20260920 | seed 20260921 | seed 20260922 | 均值±样本标准差 |
|---|---:|---:|---:|---:|
| 参考筛选教师（等于本次中文accept） | 26.5625% | 31.2500% | 26.5625% | 28.1250% ± 2.7063 |
| 等量随机教师 | 26.5625% | 28.1250% | 25.0000% | 26.5625% ± 1.5625 |
| 同题参考答案SFT | 34.3750% | 34.3750% | 31.2500% | 33.3333% ± 1.8042 |

历史dev均值字符LCS F1分别68.48%、67.10%、71.76%，全部题目有答案，严格拒答基线EM为0，不能据此评价无答案表现。三个seed累计监督token分别1659、1975、2299；固定样本量和步数仍未匹配token预算。筛选组还改变问题覆盖与难度。完整每次指标/训练步骤/逐题预测在[原始训练记录](../reports/quality-study-20260920/supervised-07/run/)，本次未重写。

历史一次同来源holdout的均值EM分别22.57%、17.36%、24.31%，主要同题目标替换差+6.94个百分点；该留出集已经消耗。本次没有重新利用留出标签选择方案。[原始结果与区间边界](../reports/quality-study-20260920/RESULTS.md)。已有[外部DRCD评估](EXTERNAL_DRCD.md)主要对照两组均57.29%，未复现正向差异，继续保留负结果。

当前受限执行环境实测 torch2.8.0、`torch.backends.mps.is_available()==False`，默认0.5B模型缓存不存在；未新增下载、付费调用、CPU长时间训练或声称异机复现。英文新规则仅选出8条且类别分布变化，未完成新的过滤训练对照。已有24题英文训练不能当作8条过滤训练结果。复用已记录中文等价训练避免无意义重复；其他未执行对照不补造指标。

现有三组训练复现（需已有模型缓存、锁定依赖与可用MPS）：

```bash
HF_HUB_OFFLINE=1 python scripts/reproduce_quality_release.py train --output work/qc-historical-retrain-01
```

这是复现已有中文三组，**不运行新的英文8条方案或全192条raw教师组**。新增这两种实验仍需预登记新的训练协议、同题/等量随机子集对照、多个seed、步数/token计数和dev门槛；不能直接将本次review记录自动转成训练目标。本次没有执行这些新增训练，不能以旧入口命令暗示已经支持或完成它们。

## 历史数字核对

原始逐题预测及 `verify_teacher_study.py` 验证：33.78%→55.86%（整体EM）、51.52%→38.38%（有答案EM）是**原提示教师目标与新提示教师目标训练的学生，在74题dev上的三seed均值对比**；无答案EM为19.51%→69.92%。这些数字不是教师自身评测，更不是本次过滤带来的收益。原始教师在24条训练题上的 normalized EM 为37.50%→41.67%。历史方案未通过所有seed的采用门槛；dev始终拒答EM为41/74=55.41%。保留原始口径及负结果，未改写历史日期或Git历史。

## 归属与验证边界

SQuAD/Wikipedia与CMRC2018内容及其在新审计记录中的再现延续[数据许可](../DATA_LICENSE.md)的CC BY-SA4.0；模型不重新分发。新代码沿用MIT。人工提供目标、约束与事实边界；本次实现、测试、命令执行与文档由Codex辅助完成，没有新增人工语义标注。详细交付清单见[变更说明](synthetic-qc-20260922/CHANGELOG.md)。
