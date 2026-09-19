# 第四轮：8-bit能否成为更合适的压缩候选

## 输入、机制、输出

输入是未接受本项目训练的Qwen2.5-0.5B-Instruct之同一MLX FP16导出权重。比较FP16、group64 affine Q4、group64 affine Q8；输出是74题开发集逐条预测、固定工作量时延/内存、3次/精度测量，以及提前固定的质量—资源门槛决定。这里没有训练、没有加入教师标签、没有量化KV Cache。

Affine量化将每组64个权重映射到有限整数档位，另存scale和bias用于计算；8-bit提供更多档位、通常能减小权重近似误差，但具体任务输出仍须实测。模型文件还包含scale/bias与未量化参数，不能只用“参数量×位宽”代表完整文件大小。Q8是整数权重量化，不是浮点FP8。

已核对[MLX官方量化文档](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)的affine模式，并检查本地mlx-lm 0.26.3实现。网页当前版本比本项目新；实际能力以锁定的MLX0.29.3转换和真实运行记录为准，不默默升级依赖。

## 实际结论与边界

Q8在本轮dev上少答对1题，EM从25.68%降到24.32%；权重988.10→525.05MB，固定工作量decode中位数266.21→313.86tokens/s。它通过预登记的全部筛选条件，成为**本地开发集压缩候选**。它不是无损量化，也不表示原学生具备可部署的问答质量。完整数字与唯一回归样例见[报告](../reports/quantization-v4/RESULTS.md)。

本轮门槛是在观察Q8结果前提交的，允许整体EM最多下降1/74；不能在结果出来后扩大容差。没有再次跑旧test或第三轮跨文章holdout，因此不宣称独立测试上保真。未来业务接受什么质量损失，应由预先确定的成本与风险目标决定。

## 为什么要重新同时测三种精度

不同运行时段可能受系统负载、温度和缓存影响。此次用FP16/Q4/Q8、Q4/Q8/FP16、Q8/FP16/Q4三轮顺序，每个精度位于首、中、尾各一次；保留每次结果，以中位数比较。同一8个train输入、batch1、每题强制32token，避免某模型回答更短而显得更快。

这不等于完整负载测试：只测一个小输入集合、一个输出长度和一台机器。首token、decode、模型加载、tokenization是不同阶段；报告中的TTFT排除了加载和编码。不能将本机MLX结果用于宣称CUDA或Ascend加速。

## 复现

按README与完整复现手册建立.venv-mlx并下载学生；首次先导出FP16到新目录（见docs/REPRODUCE_CLOSURE.md）。以下使用已有本地FP16导出，所有输出均换新路径。

```bash
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
.venv-mlx/bin/python -m qa_lab.mlx_experiment convert \
  --model .cache/mlx/student-fp16 --quantize --bits 8 \
  --output work/quantization-v4-reproduce-q8
.venv-mlx/bin/python -m qa_lab.mlx_experiment parity \
  --model work/quantization-v4-reproduce-q8 \
  --protocol configs/quantization-v4.json \
  --output work/quantization-v4-reproduce-parity
.venv-mlx/bin/python -m qa_lab.mlx_experiment quality \
  --model work/quantization-v4-reproduce-q8 --splits dev \
  --protocol configs/quantization-v4.json \
  --output work/quantization-v4-reproduce-quality
.venv-mlx/bin/python -m qa_lab.mlx_experiment benchmark \
  --model work/quantization-v4-reproduce-q8 \
  --protocol configs/quantization-v4.json \
  --output work/quantization-v4-reproduce-bench
```

仅跑以上一次Q8 benchmark不能计算公平加速比。完整顺序见configs/quantization-v4.json与commands.jsonl；对FP16/Q4也用同一入口逐个进程运行。不要并行运行GPU任务。原实验脚本拒绝覆盖已有输出；权重不放入Git。

离线核验不加载模型、不测速、不再次接触test推理：

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_quantization.py
```

工程检查：固定模型与来源hash；确认位宽/分组与算子支持；输入token与输出工作量一致；同步GPU计时；区分TTFT/decode/加载；记录全部重复；质量与资源同时过门槛；保存回归与版本。

后续冻结Q8的中文新来源检查见[第五轮方法](CHINESE_V5.md)，不改写本轮dev-only范围。
