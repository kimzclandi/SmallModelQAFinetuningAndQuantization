# Logits 缓存的训练前身份校验

2026-10-08，Codex 辅助实现。此修复不训练模型、不改变温度/目标函数，也不修改 v1/v2 冻结记录或质量结论。

## 问题与修复

旧训练入口检查缓存文件 hash 和源 artifact 的 manifest hash，却没有重新读取 artifact 的训练文件并核验其内容；完整缓存也未检查重复/缺失 ID。仅 manifest 保持不变时，源训练文件变化不能被这条入口发现；误混入其他缓存记录也缺少独立的 TRAIN 行核对。历史缓存记录的绝对 artifact 路径还使搬迁后的 checkout 依赖旧目录。

[训练入口](../qa_lab/logits_distillation.py)现在在模型更新前完成：

1. 检查每条缓存的规范路径、文件 hash、非空和 ID 唯一性。
2. 调用既有 `verified_rows`，重新核验 TRAIN 源文件及 artifact 内容 hash、TRAIN 输入与目标策略。完整缓存必须按冻结顺序覆盖全部 artifact 行；smoke 只能覆盖原先声明的前缀，不能冒充完整运行。
3. 核验 teacher/student tokenizer 身份一致，并对实际加载的学生 tokenizer 再做指纹检查。
4. 逐条从已验证的训练行重新生成 token IDs 和 answer-only labels，与缓存精确比较；检查监督位置数、分布形状、float16 类型、有限值，以及实际学生输出层的词表维度。以 float64 重算概率和并要求绝对误差不超过 0.005，允许 float16 log-probability 的舍入；这是格式容差，不是质量门槛，也不对缓存重新归一化。全部通过后才创建优化器。

新 `train --artifact` 参数只允许指定搬迁后的同内容 artifact，仍必须满足历史 manifest hash。`--data-dir` 指向对应冻结 TRAIN 来源。未传 `--artifact` 时继续使用历史路径。缓存、原始配置及报告无需重写。

当前入口还增加[模型文件身份预检](MODEL_IDENTITY.md)。新缓存记录学生/教师 checkpoint 字节身份；旧缓存缺少该回执时，新训练需显式兼容参数，并标记教师字节身份未绑定。它补充本页的缓存内容检查，不追认历史运行。

0.005 容差在新版真实缓存预检前固定：151,936 词表的均匀分布有 `log p≈−11.93`，该量级 FP16 的半 ULP 约为 0.003906，映射到概率的相对舍入误差约为 `exp(0.003906)−1≈0.003914`。0.005 用作保守格式容差；它不是所有分布的严格误差定理，也不能排除容差以内的分布扰动。不依据缓存验证结果放宽门槛。全零 log-probability（3 词时概率和为 3）和与学生输出维度不符的分布均有拒绝回归。

```sh
# 在原有 v2 训练命令中添加这两个参数（重新训练需要新输出目录）：
--artifact data/coverage-v3-gold242 --data-dir data/complexity-v1

# 无模型、无原始缓存张量的公开离线来源核验：
PYTHONPATH=. python scripts/verify_logits_distillation_v2.py
python -m pytest -q tests/test_logits_cache_lineage.py tests/test_logits_distillation.py
```

[回归测试](../tests/test_logits_cache_lineage.py)覆盖搬迁、重复/缺失/开发集 ID、顺序错配、来源内容变化但 manifest 不变、tokenizer/目标来源错配、smoke冒充完整运行、路径逃逸、token/mask 改动和非有限张量。CI 的独立 CPU PyTorch 任务执行张量测试，避免仅因可选依赖缺失而跳过这些检查。

本机验证：190 项测试全部通过；15 组统一离线验收通过；原有 242 条真实缓存的文件 hash、TRAIN 来源、token/answer-mask 及 1,289 个监督位置全部通过预检。[机器可读记录](maintenance/2026-10-08-cache-integrity/verification-final.json)只公开结果和身份 hash，不包含完整缓存张量或个人路径。

## 验证边界

- 公开 v2 离线核验重新检查 242 条 TRAIN 的血缘、缓存 ID 和协议中的 teacher/student revision；公开仓库没有完整缓存张量，因此 CI 不重算全部教师分布。
- 本机匹配原缓存可逐条执行新预检；这仍不证明历史训练曾使用新预检，更不改变保存的训练结果。
- 哈希检查用于发现文件变更和意外混用，不是可信签名。若攻击者同时篡改全部输入和 hash，不能据此证明真实性。
- TRAIN/dev/test 的原始切分由既有数据协议与核验负责；本修复约束缓存回到已验证 TRAIN 行，不能证明模型预训练未见数据，也不能把多次使用的 dev 变成独立确认集。
- 完整预检额外读取一次缓存，内存中一次仅持有一条完整词表分布。未测得训练速度收益，不宣称吞吐优化。

v2 仍为 dev EM 60.36%±2.81%，低于匹配 gold-SFT 64.86%±1.35%；没有新质量或泛化结论。
