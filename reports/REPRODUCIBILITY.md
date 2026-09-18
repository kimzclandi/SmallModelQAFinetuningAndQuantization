# 独立环境验收记录

日期：2026-09-18。验收在同一台 Mac 的独立源码目录与新建 Python 3.12.11 虚拟环境进行；不是异机/跨操作系统验证。

| 检查 | 真实结果 | 范围 |
|---|---|---|
| 全新 venv 按 requirements.lock.txt 安装 | 成功，32 个锁定依赖 | 从本地 uv 下载缓存离线安装；首次网络下载已在主环境验证 |
| 重建数据 | 全部 5 个文件逐字节匹配 | 使用已下载的原始 SQuAD 文件；校验原始 SHA-256 |
| 必要测试 | 11 passed | 含相似家族隔离、ID 覆盖、F1、拒答和硬件权限降级 |
| 原始基线重算 | dev 74 / test 102 的逐项分数与汇总一致 | 仅重算已有预测，不再生成 test 输出 |
| 新环境 CPU 真实生成 | 2/2 dev 完成 | 主环境已下载的同一固定模型缓存；不比较 CPU/GPU 速度 |
| 训练 | 两步 gold-label LoRA，非零梯度，adapter 保存成功 | 不是正式 SFT 对照或教师蒸馏 |
| adapter 重新加载 | 2/2 dev 完成 | 仅加载/执行验证，不做优化收益声明 |
| GPU 解码一致性 | 两条 train 样本 token IDs 与标准 generate 一致 | 小规模实现 smoke，不是全面形式证明 |

发现并修复：CPU 首次在受限沙箱读取 sysctl 硬件名称时失败。改为捕获权限异常、记录 unknown，CPU 推理成功。原始失败日志（路径脱敏）保留于 cpu-smoke-initial-failure.txt；基线旧代码快照保留于 baseline-v1/source/。模型的 torch_dtype 弃用提示不影响本次锁定版本运行；PEFT 保存时的配置查询 warning 后已实际验证 adapter 可加载。

CPU 实际输出、运行配置和结果见 clean-cpu-smoke/；LoRA 记录见 train-smoke.training.json；模型文件 hash 见 model-artifacts.json。无 API key、教师 API 调用、公开上传或线上 GitHub Actions 运行。
