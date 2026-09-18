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
