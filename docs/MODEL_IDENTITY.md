# Logits 蒸馏的 checkpoint 文件身份预检

2026-10-09，AI 辅助实现。此改动补齐实际文件身份与加载路径的绑定，不训练模型、不更改目标函数，也不追认历史实验曾执行新检查。

## 问题与范围

旧 logits 入口记录 `model_id/revision` 和 tokenizer 指纹，但没有消费已有模型文件清单。同一个 revision 字符串不足以发现本地权重损坏或目录混用；缓存生成时的学生 tokenizer 还单独解析一次模型位置。

新 [身份检查模块](../qa_lab/model_identity.py)显式登记两个 `(model_id, 完整 revision)` 与受保护文件清单及清单 SHA256 的对应关系：

|对象|身份清单|边界|
|---|---|---|
|Qwen2.5-0.5B-Instruct 学生|[学生清单](../reports/model-artifacts.json)|原始 checkpoint；不含 LoRA adapter 或 MLX 转换权重|
|Qwen2.5-1.5B-Instruct 教师|[教师清单](../reports/closure-v1/teacher-model-artifacts.json)|原始 checkpoint；不推断其他模型或 revision|

严格入口仅用于 [logits cache/train](../qa_lab/logits_distillation.py)。其他历史 inference、3B teacher、MLX 转换和数据准备入口没有因此获得“已做文件身份预检”的声明。已有清单、冻结配置与历史结果保持原样。

## 检查与加载顺序

1. 离线定位已存在的 pinned snapshot，不下载或转换模型。未知模型与 revision 组合拒绝执行。
2. 分块读取文件，逐项核对字节数及 SHA256；不把约 3 GB 的教师权重一次读入内存。拒绝缺失、损坏、断链和未登记的加载相关覆盖文件。正常 Hugging Face snapshot 到同模型 `blobs/` 的链接不等于文件损坏。
3. 缓存生成先完成学生和教师两个 snapshot 的全部检查，再加载任一 tokenizer/model。训练检查当前学生；使用缓存时不要求重新加载教师。
4. 加载器直接使用刚刚核验的具体目录，开启离线模式并禁止 remote code；不再按模型 ID 二次解析。新记录保存不含主机绝对路径的身份回执。
5. 模型与 tokenizer 的逻辑名称恢复为公开 model ID，避免后续 PEFT adapter 元数据写入本机 snapshot 绝对路径。这不改变已加载的权重字节。

新缓存增加 `model_identities.student/teacher`。新训练记录增加当前学生身份、`teacher_identity_binding` 和 `legacy_teacher_unbound`，将新检查与旧缓存证据分开。

## 历史缓存兼容

缺少身份回执的旧缓存仍可运行原有离线证据验收，但新的训练入口默认拒绝它。确需以旧缓存进行一项新的、预先冻结的训练时，显式传入 `--allow-legacy-unbound-cache`。新记录必须保留 `legacy_teacher_unbound=true`；该参数不补写旧记录，也不证明旧教师分布来自本轮核验的权重字节。现有缓存的 TRAIN、token/mask、文件 hash 和词表检查仍然执行。

带有新身份字段却不完整或不匹配的缓存不能借此兼容参数绕过检查。

## 不加载模型的实际文件核验

从仓库根运行，使用已有依赖环境和本地 HF hub 目录；输出文件必须是新的：

```sh
python -m qa_lab.model_identity \
  --model-cache-dir /path/to/existing/hub \
  --role both --output work/model-identity-check.json

python -m pytest -q tests/test_model_identity.py
```

纯 CPU 命令只读取和哈希已有文件，不导入 torch/transformers、执行推理或生成教师分布。`cache` 与 `train` 也接受 `--model-cache-dir`；省略时使用环境配置的本地 HF 缓存。没有完整本地文件时直接失败，不自动下载。

## 证据边界

- 这是文件一致性与加载路径检查，不是可信签名、模型质量确认、CUDA/Ascend 经验或速度收益。
- 核验与加载之间要求目录不被并发修改；不宣称防御并发文件替换。原始 checkpoint 之外的 adapter 和转换产物需要各自身份检查。
- CI 用小文件与 mock 检查损坏拒绝、同一目录加载、双模型检查顺序和元数据；不下载权重，也不重建原始教师分布。
- 原 v2 仍是复用 dev 上的 EM 60.36%±2.81%，低于匹配 gold-SFT 的 64.86%±1.35%。本修复没有新增训练、质量或泛化结果。
