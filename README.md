# Domain QA Lab · 小模型领域问答优化实验室

在普通个人硬件上建立可复现的「基线 → 教师响应蒸馏 → 量化 → 失败驱动数据改进」实验。**当前完成阶段一：真实基线。** 有可运行的 gold-label LoRA 训练入口与两步冒烟记录；教师蒸馏、量化对比和改进闭环尚未完成。没有公开发布、调用付费 API 或上传数据。

## 任务与当前证据

给定英文计算复杂性文段和问题，输出最短原文片段；文段没有答案则严格输出 `NO_ANSWER`。输入不包含 gold。采用 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)（Apache-2.0，固定 revision）与 [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) 的 `Computational_complexity_theory` 子集（CC BY-SA 4.0）。模型已接受上游预训练和指令微调，baseline 仅指未接受**本项目**训练。

418 道题、48 个文段/家族；按来源文段和近似问题连通家族划分：train 242、dev 74、test 102。使用公开 SQuAD dev 重新切分，**并非官方隐藏测试，也不能排除预训练污染**。

| 实际运行 | 样本数 | EM | token F1 | 有答案 EM | 无答案严格拒答 | 始终拒答 EM |
|---|---:|---:|---:|---:|---:|---:|
| 原始学生 · dev | 74 | 25.68% | 30.13% | 57.58% | 0.00% | 55.41% |
| 原始学生 · test | 102 | 24.51% | 31.56% | 52.08% | 0.00% | 52.94% |

**保留负结果：原始学生总分低于始终拒答对照。** 严格拒答为零包含格式变体影响，不等于已经逐条确认“幻觉”。这为下一步拒答诊断提供方向，不能作为模型优化成功的证据。

硬件实测：M4 Max、48 GiB 统一内存、40 核 GPU、macOS 27.0，MPS FP32、batch=1。测试集平均 TTFT 59.01 ms，decode 40.60 tokens/s；输出自然停止、长度不固定，不能据此推断量化加速。RSS 采样最大约 3.01 GiB、MPS driver 约 4.71 GiB，两者不可相加。更多结果和测量口径见 [真实实验报告](reports/RESULTS.md)。

## 快速开始

从仓库根目录执行。Python **3.12**；安装 `uv` 或使用已有 Python 的 venv/pip。首次下载免费模型约 1 GB，无需 API key。推荐至少 8 GiB 可用内存与 5 GB 磁盘空间，最低硬件门槛未实测；已有验证硬件为上述 48 GiB Mac。CPU 可以运行小规模验证，不能与 GPU 时延混比。默认不支持 CUDA 路径。

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.lock.txt
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1
.venv/bin/python scripts/download_model.py
.venv/bin/python -m pytest -q
PYTHONPATH=. .venv/bin/python scripts/verify_artifacts.py

# 小规模真实生成：使用 dev，不消耗测试集访问
HF_HUB_OFFLINE=1 .venv/bin/python -m qa_lab.inference \
  --device cpu --splits dev --limit 2 --output work/smoke-01
```

Apple Silicon 将 `--device cpu` 换为 `--device mps`。若 MPS 不可用会明确报错，不会静默切换；受限执行沙箱可能看不到 Metal GPU。离线指标核验无需模型，只用标准库；pytest 用于测试。

完整基线复现（执行前理解测试集使用纪律，后续每次必须新建输出目录）：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m qa_lab.inference \
  --device mps --output work/baseline-reproduction-01

# 从原始公开源重建同一数据，禁止覆盖已冻结版本
.venv/bin/python -m qa_lab.data --output work/rebuilt-data

# 从预测单独重算评分
.venv/bin/python -m qa_lab.metrics \
  --data data/complexity-v1/dev.jsonl \
  --predictions reports/baseline-v1/dev.predictions.jsonl \
  --output work/recomputed-dev

# 真正可执行的训练入口：仅两步 gold-label LoRA 冒烟
HF_HUB_OFFLINE=1 .venv/bin/python -m qa_lab.train \
  --output checkpoints/smoke-02

# 可选：载入保存的 adapter 跑 dev 冒烟，不作为收益评估
HF_HUB_OFFLINE=1 .venv/bin/python -m qa_lab.inference \
  --adapter checkpoints/smoke-02 --splits dev --limit 2 \
  --output work/adapter-smoke-02
```

训练仅接受原始冻结 train 文件，不接收 dev/test。`labels=-100` 屏蔽 prompt，只监督回答和结束标记；参数矩阵形状与公式见面试文档。训练配置可另存为新文件增加步数，但在设计正式对照前不要把 smoke 当作训练结果。推理全部为本地，无 API 网络请求；下载步骤例外。

## 目录与证据链

- `qa_lab/data.py`：可重建数据、精确去重、近似家族合并和切分。
- `qa_lab/inference.py`：离线模型、输入白名单、KV-cache greedy 解码、逐条计时及内存采样。
- `qa_lab/metrics.py`：多参考 EM/F1、拒答/格式/家族切片、严格 ID 覆盖校验。
- `qa_lab/train.py`：真实 gold-label LoRA 训练与保存；不是教师蒸馏。
- `configs/`、`requirements.lock.txt`：固定协议和全量依赖版本。
- `data/complexity-v1/`：小规模许可数据、训练样例、来源 hash 与分组 manifest。
- `reports/baseline-v1/`：176 条预测、逐条评分、汇总、失败与运行/文件 hash。
- `reports/train-smoke.training.json`：两步前向/反向/优化器记录，无收益声明。
- `tests/`：指标、输入边界、近重复隔离和数据完整性测试。
- `.github/workflows/tests.yml`：离线测试工作流定义；尚未在 GitHub 运行。

模型权重、adapter、虚拟环境和缓存均被 Git 忽略。证据链为：模型 revision/文件 hash + 数据 hash + 代码 hash + config → prediction → per-item score → metrics。后续改核心代码时建立新的版本/报告，不覆盖本轮证据。

## 阅读顺序与限制

1. [冻结协议](docs/PROTOCOL.md)：指标、切分、测量、test 使用纪律。
2. [开发集失败分析](reports/DEV_FAILURE_ANALYSIS.md)：观察与根因假设分开。
3. [面试与知识手册](docs/INTERVIEW.md)：领域地图、五个模型、三个争议、30 秒/3 分钟讲解、10 道主动回忆题。
4. [后续阶段状态](docs/ROADMAP.md)：未完成事项及条件。
5. [贡献与本人验证](CONTRIBUTIONS.md)、[数据许可](DATA_LICENSE.md)、[组件归属](THIRD_PARTY.md)。

局限：单篇文章、样本少、同文章相关性；启发式近重复检测不保证语义无泄漏；公开 benchmark 可能进入预训练；英文结果不能外推中文业务；EM/F1 不等同语义蕴含；单次 seed/运行无统计显著性证明；内存是采样最大值。未实现检索服务、teacher API、量化、RLHF 或生产数据飞轮。

## 本地验收

已在独立源码目录和全新 Python 3.12 虚拟环境按锁定依赖验证：11 项测试通过、数据重建逐字节一致、176 条已有预测离线重算一致、CPU 两条 dev 真正生成成功、LoRA adapter 重新加载成功。详见 [复现记录](reports/REPRODUCIBILITY.md)。这是同机新环境验证；异机和 GitHub CI 尚未验证。基线所用代码位于 `reports/baseline-v1/source/`，当前代码另含硬件查询权限容错修复。
