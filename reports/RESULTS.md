# 真实基线结果（2026-09-18）

本报告由已保存的逐条预测计算；只有原始学生 baseline，无蒸馏/量化/数据修复结果。

| 数据集 | n | EM | token F1 | 格式合规 | 有答案 EM | 无答案严格拒答 | 始终拒答 EM |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev | 74 | 25.68% | 30.13% | 95.95% | 57.58% | 0.00% | 55.41% |
| test | 102 | 24.51% | 31.56% | 74.51% | 52.08% | 0.00% | 52.94% |

结论：当前学生会提取部分正确片段，但严格拒答能力未达到本任务要求。总分低于始终拒答对照，不具备可用性结论。拒答 token 的格式变体属于协议失败，不能据此断言模型从未表现拒答意图。

## 性能与资源

| 数据集 | 平均 TTFT | 平均生成耗时 | decode tokens/s | 输出 tokens 合计 | 达到长度上限 |
|---|---:|---:|---:|---:|---:|
| dev | 84.91 ms | 181.43 ms | 35.70 | 329 | 0 |
| test | 59.01 ms | 179.02 ms | 40.60 | 599 | 0 |

硬件：Apple M4 Max，48 GiB 统一内存，40 核 GPU；macOS 27.0。FP32、MPS、batch=1、greedy、KV cache、eager attention。完整实验进程 33.93 秒，模型加载 1.17 秒。

该性能数字来自一次本机温启动串行运行；不含网络、队列或服务并发，不用于量化加速结论。输出长度并不固定。
- rss_peak_sampled_bytes: 3,231,580,160 bytes（3.010 GiB）
- mps_alloc_peak_sampled_bytes: 2,233,534,464 bytes（2.080 GiB）
- mps_driver_peak_sampled_bytes: 5,061,328,896 bytes（4.714 GiB）

模型权重文件：988,097,824 bytes；加载后的 FP32 参数张量：1,976,131,072 bytes。磁盘权重采用模型原始存储精度，加载为 FP32，不是进行了量化。RSS/MPS 是采样最大值，统一内存口径重叠，不能相加。

## 可追溯证据

- `baseline-v1/run.json`：配置、源文件 hash、数据 hash、包版本、硬件和时间。
- `baseline-v1/{dev,test}.predictions.jsonl`：原始文本、token IDs、停止原因、prompt hash、逐条耗时。
- `baseline-v1/{dev,test}.scored.jsonl`：逐条 EM/F1/格式与自动症状标签。
- `baseline-v1/{dev,test}.failures.jsonl`：EM<1 的案例；格式失败但 EM=1 的条目保留在 scored 文件，两个维度不同。
- `baseline-v1/manifest.json`：实验输出 hash，验证脚本重新计算指标并检查完整覆盖。
- `model-artifacts.json`：模型下载文件大小与 SHA-256。

## 训练入口验证

两步 gold-label LoRA 成功，训练参数 540,672。loss 为 1.941955 和 2.562346，来自不同训练样本，不能用来宣称提升/退化。仅验证前向、反向、优化器与 adapter 保存；不评估其 test 质量。详见 `train-smoke.training.json`。
