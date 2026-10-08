# 新文段留出评估的数据归属

本说明补充新目录的归属，不改变仓库代码许可或任何已有数据的许可。

[留出集](../reports/teacher-gated-confirmation-v1/cohort/)来自 Rajpurkar、Jia、Liang 的 [SQuAD 2.0 官方公开 dev](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)，底层文段归属 Wikipedia 贡献者；数据按 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en) 提供。修改包括：确定性筛选 128 个文段、每文段各一条可回答和不可回答问题、拆分无标签输入与标签、增加文章/文段身份及哈希；没有改写原问题、文段或参考答案。预测中再现的原文片段保留原数据归属。它不是官方隐藏测试。

[历史排除元数据](../configs/confirmation-exclusions-v1.json)含既有公开问题的归一化文本、ID、文段哈希，以及 35 个公开来源文件的身份。其 `licensing.sources` 保留各来源 URL、作者与改动说明：SQuAD 2.0 和 CMRC2018 按 CC BY-SA 4.0，DRCD 保留 CC BY-SA 3.0 原来源归属。该文件的混合数据不能按代码 MIT 许可处理。归一化只用于去重，不改写历史样本或预测。

[冻结源码与输入快照](../reports/teacher-gated-confirmation-v1/frozen-source/)的代码沿用仓库代码许可；其中 TRAIN 数据及历史排除材料继续遵守各自的数据许可，参见既有 [DATA_LICENSE.md](../DATA_LICENSE.md)。Qwen、PyTorch、Transformers、PEFT 的贡献和许可归属见 [THIRD_PARTY.md](../THIRD_PARTY.md)。本轮没有公开模型或适配器权重、完整 teacher logits 文件、凭据、私人业务数据或付费服务输出。
