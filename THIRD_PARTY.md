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

后续量化候选为 Apple MLX/MLX-LM 权重量化，官方支持文档：https://github.com/ml-explore/mlx-lm 。尚未实现、未运行，也未承诺加速。必须在 MLX 内另建浮点基线，再与同框架量化模型比较；不能用本仓 PyTorch 时延直接证明 MLX 量化收益。

下载的模型附带的 LICENSE/模型卡保留在被 Git 忽略的本地缓存中；仓库不包含模型权重。
