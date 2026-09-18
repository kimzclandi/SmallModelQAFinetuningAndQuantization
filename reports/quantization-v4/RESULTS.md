# 第四轮：FP16 / Q4 / Q8 同框架量化对照

已运行验证。通过本地dev压缩筛选的候选：['q8']。没有新test推理，没有模型训练，也没有公开发布。

## 实测结果

质量样本为同一74题dev；性能为同一8个train输入、每题强制32tokens，每精度3个独立进程运行的中位数。

| 指标 | FP16 | affine Q4 | affine Q8 |
|---|---:|---:|---:|
| dev EM | 25.68% | 21.62% | 24.32% |
| dev token F1 | 30.13% | 24.44% | 29.68% |
| 有答案 EM | 57.58% | 48.48% | 54.55% |
| 权重文件 / decimal MB | 988.10 | 278.06 | 525.05 |
| decode tokens/s | 266.21 | 390.87 | 313.86 |
| 平均TTFT的跨run中位数 / ms | 26.31 | 21.78 | 21.86 |
| 采样RSS峰值的中位数 / GiB | 2.028 | 0.710 | 1.169 |
| MLX active峰值的中位数 / GiB | 1.279 | 0.626 | 0.839 |

Q8权重减少46.86%，采样RSS峰值中位数减少42.35%，本轮固定工作量解码速度比为1.179×。这些是本机此工作量的测量，不是跨硬件通用加速承诺。

RSS和MLX计数在统一内存中重叠，不相加。RSS每10ms采样，不保证捕捉真实峰值；MLX active不含allocator cache，后者单独保存在每个run与summary中。进程内存包括加载和预热。TTFT从已tokenize/传输后的同步prefill开始，不含模型加载、网络与排队。

## 固定门槛与逐题回归

方案在commit 3525a20提前登记：相对同轮FP16，整体EM最多少1/74、有答案EM最多少1/33、F1最多下降2个百分点；权重<=60%、RSS<=80%、decode>=95%、TTFT<=110%。全部满足才能成为本地dev压缩候选。

| 候选 | EM | 有答案EM | F1 | 权重 | RSS | decode | TTFT | 全部通过 |
|---|---|---|---|---|---|---|---|---|
| q4 | 未通过 | 未通过 | 未通过 | 通过 | 通过 | 通过 | 通过 | 未通过 |
| q8 | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 |

| 与FP16相比 | 修复题数 | 回归题数 |
|---|---:|---:|
| q4 | 3 | 6 |
| q8 | 0 | 1 |

Q8回归样例 `56e1fc57e3433e140042322f`：

Intractable problems lacking polynomial time solutions necessarily negate the practical efficacy of what type of algorithm?

FP16：`exponential-time algorithms`；Q8：`intractable problems`；参考答案：['exponential-time algorithms', 'exponential-time']。

## 每次性能测量（不挑最好的一次）

| 顺序 | 精度 | decode tokens/s | TTFT ms |
|---:|---|---:|---:|
| 1 | fp16 | 265.92 | 26.27 |
| 2 | q4 | 392.48 | 21.76 |
| 3 | q8 | 313.86 | 21.86 |
| 4 | q4 | 390.87 | 21.78 |
| 5 | q8 | 315.46 | 21.78 |
| 6 | fp16 | 266.21 | 26.31 |
| 7 | q8 | 313.85 | 21.87 |
| 8 | fp16 | 266.27 | 26.38 |
| 9 | q4 | 388.61 | 21.83 |

顺序为三轮轮换：FP16/Q4/Q8，Q4/Q8/FP16，Q8/FP16/Q4。每次单独进程、串行运行、2次虚构输入预热；固定32输出tokens，包括遇到EOS后继续，仅用于性能，不计质量。decode排除首token，分子每题31个；质量按EOS停止、最多48tokens。

## 机制与适用边界

- 原始学生、同一FP16导出权重、同MLX实现与版本、同prompt/tokenizer、greedy、batch1；Q4/Q8均为group64 affine整数权重量化，KV Cache保持浮点。Q8不是FP8。
- Q8通过2个训练输入与MLX-LM上游greedy逐token一致性检查；全部质量/性能输入token hash均核对。两题parity不是完整实现正确性的证明。
- 开发集已经被多轮查看，本轮通过的是预设容差内的压缩筛选，未取得独立test验证，不能包装为无损量化。
- 基础模型自身仍缺乏可靠拒答能力；dev始终拒答基线为55.41%，本轮各模型总体EM均更低。压缩候选通过不表示QA业务达标。
- Q4仍保留更小更快但质量差的结果。不同轮次测速受系统状态影响，不把旧速度与本轮混在一起计算增益。
- 模型权重只存本地.cache，不入Git/源码包；转换命令、来源hash、预测、时延、内存、配置与源码快照均可追溯。

复现与面试自测见docs/QUANTIZATION_V4.md。
