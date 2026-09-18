# 数据许可与归属

`data/complexity-v1/*.jsonl` 与 `data/teacher-pilot-v1/` 中的文段、问题、答案，以及报告中再现的这些内容，来自 **SQuAD 2.0**，遵循 **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**，不适用仓库代码的 MIT 许可。

- 创作者/归属：SQuAD 团队，Pranav Rajpurkar、Robin Jia、Percy Liang；基础文段来自 Wikipedia 贡献者。
- 官方项目与许可声明：https://rajpurkar.github.io/SQuAD-explorer/
- 原始文件：https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
- 许可全文：https://creativecommons.org/licenses/by-sa/4.0/legalcode.en
- Wikipedia 文章：https://en.wikipedia.org/wiki/Computational_complexity_theory （数据是 SQuAD 所收录的历史版本，并非当前网页快照）。
- 修改：筛选 `Computational_complexity_theory`；保留原问题 ID、文段和答案；去除重复；增加来源、家族和本地 split 字段；重新分为 train/dev/test。未改写问题或答案。
- 原始文件 SHA-256、处理规则和分组审计见 `data/complexity-v1/manifest.json`。

再分发此数据改编须保留归属、来源、变更说明和相同许可。当前仅本地准备，尚未公开发布。样例文件仅包含训练集内容。

引用：Rajpurkar, Jia, Liang (2018), *Know What You Don't Know: Unanswerable Questions for SQuAD*, https://arxiv.org/abs/1806.03822 。

新增 `data/repair-v1/` 是原SQuAD训练材料的再次选样；`data/distilled-local-v1/` 保留SQuAD输入，target由Apache-2.0 Qwen本地教师生成，可能复制源文段。源数据及其再现继续保留CC BY-SA归属；生成过程与模型版本见manifest。`data/teacher-received-v1/` 为用户提交的GPT回答审计记录，未用于训练；这里不以数据许可说明替代该服务适用条款或声称已取得蒸馏许可。

## teacher-study-v2

`data/teacher-study-v2-gold`与`data/teacher-study-v2-prompted`复用同一24条许可训练样本。前者使用SQuAD gold，后者使用本地Apache-2.0 Qwen教师的新提示回答；来源hash、模型revision、完整提示和采样条件见manifest与reports。文段和问题继续遵守上文CC BY-SA4.0及归属要求。提示中的Luma/Rivo四个示例是AI辅助人工创作的虚构文本，不来源于dev/test或第三方业务数据。

## coverage-v3 / cross-article-v3

`data/coverage-v3-gold242`使用已有Computational_complexity_theory训练集的全部242条gold。`data/cross-article-v3`从同一官方SQuAD2.0 dev文件的Packet_switching、Prime_number各取32条。文段、问题及标注按原数据CC BY-SA4.0与归属要求提供；来源URL、原文件SHA256、文章名、确定性抽样、去重和本地改动在manifest、configs/coverage-v3中记录。本项目重新抽样后的64题不是SQuAD官方隐藏测试。未添加用户业务数据或付费教师回答。

## chinese-v5（CMRC2018）

新来源为Yiming Cui等作者的CMRC2018官方仓库，固定revision c0eb1b6ba219847457e6af3180da722bbeb656af。96题子集按CC BY-SA4.0提供；完整作者、论文、原始URL、修改说明和许可证见data/chinese-v5/ATTRIBUTION.md、SOURCE_LICENSE.txt及manifest.json。代码MIT许可不覆盖数据内容。此为官方公开dev的重新抽样，不是官方隐藏test。未上传或公开发布。
