# 开源组件与官方依据

核实日期：2026-09-18。精确安装版本见 `requirements.lock.txt`，不是对“最新版本”的声明。

| 组件 | 用途 | 许可/官方来源 |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.49B 参数学生，已有预训练和指令微调，本项目 baseline 未更新权重 | Apache-2.0；https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct |
| SQuAD 2.0 | 可回答/不可回答的文段问答 | CC BY-SA 4.0；https://rajpurkar.github.io/SQuAD-explorer/ |
| PyTorch | MPS/CPU 推理和训练 | BSD-style；https://github.com/pytorch/pytorch/blob/main/LICENSE |
| Transformers | tokenizer、模型、聊天模板 | Apache-2.0；https://huggingface.co/docs/transformers/model_doc/qwen2 |
| PEFT | LoRA 训练入口 | Apache-2.0；https://huggingface.co/docs/peft/package_reference/lora |
| pytest / psutil | 必要测试 / RSS 采样 | MIT / BSD-3-Clause；各包发布信息 |

Apple MLX/MLX-LM已实际用于同框架FP16、affine Q4与Q8权重量化，官方仓库：https://github.com/ml-explore/mlx-lm 。结果见reports/quantization-v4与reports/chinese-v5；KV保持浮点，不把PyTorch与MLX的跨框架时延差作为量化收益。

下载的模型附带的 LICENSE/模型卡保留在被 Git 忽略的本地缓存中；仓库不包含模型权重。

新增本地教师：[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，Apache-2.0，固定revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。输出为本地生成响应，不使用GPT候选训练。新增Apple MLX/MLX-LM（MIT）用于同框架FP16/Q4/Q8实验，精确版本见 requirements-mlx.lock.txt；保留本地源模型许可证与转换血缘。

### CMRC2018 中文数据

Yiming Cui, Ting Liu, Wanxiang Che, Li Xiao, Zhipeng Chen, Wentao Ma, Shijin Wang, Guoping Hu, EMNLP-IJCNLP2019. [官方数据仓库](https://github.com/ymcui/cmrc2018) / [论文](https://aclanthology.org/D19-1600/)。CC BY-SA4.0；96题子集与修改说明见data/chinese-v5。中文指标由本项目独立实现并明确命名，不是复制的官方评测脚本，不宣称官方榜单分数。
