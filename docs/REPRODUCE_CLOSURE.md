# 完整实验复现：先隔离输出，再安装与运行

历史 `reports/` 和版本化 `data/` 是冻结证据，不覆盖、不手改。新实验只写入新建 `work/` 目录或新版本目录。下列命令从仓库根目录执行，已存在的输出会报错；换新名字，不删除旧结果。首次下载学生约1GB、教师约3GB；不调用付费API、不上传数据。

## 运行环境

主实验 Python3.12、PyTorch2.8.0、Transformers4.56.2、PEFT0.17.1；Mac MPS FP32。教师是 Qwen2.5-1.5B-Instruct，revision固定在 `configs/teacher-local-fp32-v1.json`。不能用初始FP16配置替代：本机已观测其final logits非有限。

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.lock.txt
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
.venv/bin/python scripts/download_model.py
.venv/bin/python scripts/download_teacher.py
export HF_HUB_OFFLINE=1

# 所有新数据/报告/adapter都隔离在这个新目录
mkdir -p work/reproduce-closure-01
.venv/bin/python -m qa_lab.closure_data teacher \
  --output work/reproduce-closure-01/teacher
.venv/bin/python -m qa_lab.closure_data distill \
  --teacher-report work/reproduce-closure-01/teacher \
  --output work/reproduce-closure-01/distilled-data
.venv/bin/python -m qa_lab.closure_data repair \
  --output work/reproduce-closure-01/repaired-data
.venv/bin/python -m qa_lab.train_artifact \
  --artifact work/reproduce-closure-01/distilled-data --config configs/distilled-response-v1.json \
  --output work/reproduce-closure-01/distilled-adapter
.venv/bin/python -m qa_lab.train_artifact \
  --artifact work/reproduce-closure-01/repaired-data --config configs/data-repair-v1.json \
  --output work/reproduce-closure-01/repaired-adapter
.venv/bin/python -m qa_lab.inference \
  --adapter work/reproduce-closure-01/distilled-adapter --splits dev \
  --output work/reproduce-closure-01/distilled-dev
.venv/bin/python -m qa_lab.inference \
  --adapter work/reproduce-closure-01/repaired-adapter --splits dev \
  --output work/reproduce-closure-01/repaired-dev
```

普通gold-SFT匹配对照见 `reports/gold-sft-pilot-v1/RESULTS.md`。同24个ID、同48步、同初始化与优化器，教师目标token数可能不同，不声称token预算完全一致。数据修订为原24题再加24道未用的有答案训练题；48步不变，原数据每题曝光次数也随之改变。这只能验证这次数据修订的整体效果，不能单独证明类别比例或覆盖度哪个因素起效。

日常复现只跑dev。若要复现本次最终test评估，先固定模型和所有配置，再将 `--splits dev` 改为 `--splits test` 且使用新输出目录。记录访问；不要根据test结果改候选或重新调整协议。

## Apple Silicon量化

MLX是独立环境，避免改变PyTorch依赖。当前只验证Apple Silicon；不称Linux/CUDA可直接运行此模块。下载后完全离线转换。

```bash
uv venv .venv-mlx --python 3.12
uv pip install --python .venv-mlx/bin/python -r requirements-mlx.lock.txt
.venv-mlx/bin/python -m qa_lab.mlx_experiment convert \
  --model .cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --output work/reproduce-closure-01/mlx-fp16
.venv-mlx/bin/python -m qa_lab.mlx_experiment convert \
  --model work/reproduce-closure-01/mlx-fp16 --quantize \
  --output work/reproduce-closure-01/mlx-q4
.venv-mlx/bin/python -m qa_lab.mlx_experiment quality \
  --model work/reproduce-closure-01/mlx-fp16 --splits dev \
  --output work/reproduce-closure-01/mlx-fp16-dev
.venv-mlx/bin/python -m qa_lab.mlx_experiment quality \
  --model work/reproduce-closure-01/mlx-q4 --splits dev \
  --output work/reproduce-closure-01/mlx-q4-dev
.venv-mlx/bin/python -m qa_lab.mlx_experiment benchmark \
  --model work/reproduce-closure-01/mlx-fp16 \
  --output work/reproduce-closure-01/bench-1-fp16
.venv-mlx/bin/python -m qa_lab.mlx_experiment benchmark \
  --model work/reproduce-closure-01/mlx-q4 \
  --output work/reproduce-closure-01/bench-2-q4
.venv-mlx/bin/python -m qa_lab.mlx_experiment benchmark \
  --model work/reproduce-closure-01/mlx-q4 \
  --output work/reproduce-closure-01/bench-3-q4
.venv-mlx/bin/python -m qa_lab.mlx_experiment benchmark \
  --model work/reproduce-closure-01/mlx-fp16 \
  --output work/reproduce-closure-01/bench-4-fp16
```

测速过程串行，不同时运行其他模型。每个进程加载单个模型、预热2次；固定8个train输入，每个强制32 tokens（包括EOS后继续生成，仅测工作量，不用于质量计分），进程顺序FP16/Q4/Q4/FP16。FP16与Q4都保留浮点KV Cache。质量评估则按EOS停止，上限48tokens。

初次使用上游convert时，它尝试自动查询模型卡并在离线环境失败。当前转换调用同一MLX加载/量化/保存函数，但只复制本地模型卡和LICENSE；不发起Hub请求。4-bit表示被量化的权重分组，scale/bias/未量化层仍有存储开销，不等于整个文件恰好每参数4bit。

## 不运行模型的证据核验

```bash
.venv/bin/python -m pytest -q
PYTHONPATH=. .venv/bin/python scripts/verify_artifacts.py
PYTHONPATH=. .venv/bin/python scripts/verify_closure.py
```

这些检查重算保存的预测、比较输入与源码hash，不会重新访问模型或追加测试集推理。CI定义只能说明工作流已准备，GitHub线上执行需发布后另行核验。

元数据说明：实际首轮蒸馏/修订复用了gold配置以匹配数值超参数，其中purpose文字是控制组历史说明；真实标签来源以training.json的method和artifact_path/hash为准。新增两份复现配置只修正purpose说明，数值参数完全相同，历史run.json未改写。
