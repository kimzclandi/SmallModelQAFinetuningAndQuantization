# 发布前工程验收：v6 历史记录

发布前历史验收快照，以下发布状态按验收时刻理解。后续已公开，首次CI记录见[发布状态](ROADMAP.md)，最新运行见[Actions](https://github.com/kimzclandi/domain-qa-lab/actions/workflows/tests.yml)。

日期：2026-09-18。结论：已有实验的主要数值与结论可从保存证据核验；该次修复复现入口问题，未追加质量实验。以下状态限定于2026-09-18，当前发布与CI状态见README。

## 当前环境与历史保护

接手时仓库为main，HEAD为648b02587d2270d683ca8694798afbf0480ebb08，工作区干净，remote为空。macOS27.0 / arm64 / Apple M4 Max / 48GB统一内存。原仓库有被忽略的.venv、.venv-ci、.venv-mlx、模型缓存与checkpoints；不把它们打入源码包。

读取了README、ROADMAP、CONTRIBUTIONS及全部7份RESULTS.md后再编辑。原有681个reports/data/configs文件的SHA256前后相同；历史v5 ZIP哈希也未变，见[claim-audit.json](acceptance-v6/claim-audit.json)。历史RESULTS/source快照保持不变，新增文档只澄清当前状态。Git协议提交链包含c1c6045、1bacfca、97942f9、3525a20、6c2929f等已有冻结节点；不重写提交历史。

## README结论核对表

数值逐项回溯结构化summary；六套核验进一步重算逐条预测、检查模型/数据/代码记录与配置。模型文件哈希是记录的血缘信息，不能仅由哈希反推所有历史操作；本轮学生CPU缓存文件额外与model-artifacts.json逐文件匹配。

| README当前主张 | 核对依据 | 审计结论与边界 |
|---|---|---|
| 418题，242/74/102，48家族 | data/complexity-v1 manifest与重建 | 五个生成文件逐字节相同；同文章关联/预训练污染仍存在 |
| 原始学生test EM24.51%、F1 31.56% | baseline逐条预测/评分 | 重算一致；始终拒答52.94%，未获得任务可用性 |
| gold/蒸馏/修订表中的dev、test EM/F1 | closure-v1 summary和预测 | 表格数值一致，训练候选未过dev门槛；只做离线重算，未重新推理旧test |
| 教师9/24、数据修订修复1退化5 | teacher-fp32、closure paired记录 | 保留教师错误与修订负结果，非logits蒸馏 |
| 首轮Q4 test EM14.71%、988.10/278.06MB、255.70/370.17tokens/s | closure MLX质量与四次bench | 同MLX、两次/精度的中位数；不与第四轮速度混算 |
| 教师三seed均值/标准差与有答案退化 | teacher-study-v2 summary | 表格一致，均未过门槛；标准差不是置信区间 |
| 新文章36.46%/44.27%，有答案38.54%/47.92% | coverage-v3 aggregate/逐题预测 | 表格一致；策略整体收益，非等token或单因素归因 |
| Q8 525.05MB、313.86tokens/s、dev24.32% | quantization-v4 summary及9次timings | 表格、计算分母、顺序与门槛一致；英文dev少1/74，不代表无损 |
| 中文44/96、LCS F1 72.68%/72.12% | chinese-v5预测、summary与模型冻结 | 正确ID集合相同，两条文本不同；仅有答案题、自定义指标、新来源有限保真 |
| 模型版本与训练配置 | baseline/teacher配置、training.json、source快照 | 学生7ae5576…、教师989aa79…；推理输入不使用gold；LoRA训练与MLX量化独立 |
| 许可、快速开始、CI与验证范围 | 官方来源、锁文件、隔离运行 | 本轮新环境安装/CPU冒烟成立；无全新模型下载验证、无MLX重推理、无异机/线上CI |

完整机器可读数值表见[claim-audit.json](acceptance-v6/claim-audit.json)。这些核验支持“记录可重算”，不把其表述为独立复现所有模型输出。

## 实际修复

1. 四个汇总函数增加read_only模式，六套核验中的对应入口使用它。冻结summary缺失时明确失败，不自动补建。四个故障注入测试分别验证缺失不能被修复掩盖。
2. 新增scripts/acceptance.py，统一本地与CI：测试、六套核验、gold对照、教师审计。拒绝-O/PYTHONOPTIMIZE；只接受新work子目录，并在前后校验reports/data/configs。结果与逐检查日志可复核。
3. 推理默认split改为dev；显式--splits test仍可使用，但不再因省略参数默认访问test。新环境两题实际运行验证此行为。
4. 新增仅5个依赖的requirements-ci.lock.txt；CI改用它及统一入口，权限为contents: read。MLX与PyTorch继续分环境。
5. pyproject构建setuptools与主锁统一为84.0.0，避免两处固定版本互相矛盾。快速开始仍从源码根目录运行，本轮未做独立wheel构建验证。
6. 修正THIRD_PARTY中的“MLX未运行”旧描述；README先离线验收后实际推理，区分锁文件平台/哈希边界；ROADMAP收敛三个缺口。

## 实际执行及证据层级

| 执行 | 结果 | 证据层级 |
|---|---|---|
| 拷贝739个原追踪文件到独立work目录，再同步本轮修复 | 不带旧环境与权重 | 隔离源码副本，同机 |
| 新Python3.12.13轻量环境执行统一入口 | 51测试、6核验、gold对照、教师审计均通过 | 已有证据核验，未调用模型 |
| 新PyTorch环境按锁安装 | 32包，freeze逐行与锁一致，uv pip check通过 | 同机新环境依赖验证 |
| 新MLX环境按锁安装 | 40包，freeze逐行与锁一致，uv pip check通过 | 安装验证；没有MLX重推理或转换 |
| 新PyTorch环境、CPU、默认dev、limit2 | status=complete，2条预测，新work目录 | 重新模型推理；同机新环境冒烟，不是质量估计 |
| 学生已缓存9个文件hash与原model-artifacts匹配 | 全部一致 | 重用只读模型缓存；未测试空缓存首次下载 |
| 从已缓存固定SQuAD源重建 | train/dev/test/manifest/samples逐字节一致 | 数据处理可复现；本轮没有重新下载原数据 |

原始验收记录见[offline/result.json](acceptance-v6/offline/result.json)、[测试日志](acceptance-v6/offline/tests.log)、[新环境与重建检查](acceptance-v6/validation-details.json)、[CPU run](acceptance-v6/cpu-smoke/run.json)。run中的绝对执行路径是本次历史记录，复现应使用README中的相对命令。CPU日志的torch_dtype弃用提示来自现有Transformers接口，不影响此次执行；没有为消除提示改动冻结实验代码。

没有把两题结果加入正式74/96/102题汇总；没有新训练、重跑完整质量集、调参、测速或新增业务实验。同机新环境仍共享操作系统和硬件；它不能证明CUDA、Ascend、Linux训练、异机或逐bit复现。Linux CI定义尚未在GitHub执行。

## 下载、依赖与官方许可

首次安装依赖需要PyPI网络；本轮真实安装成功。现有锁是精确版本列表，**不含发行文件哈希，也不是跨平台通用锁**。新环境安装性已验证，不声称长期可用、可离线重装全部依赖或供应链认证。独立MLX锁只面向macOS arm64；轻量CI不安装torch/MLX。

学生/教师下载脚本固定revision并禁止隐式Hub token，无付费推理调用。数据已含许可子集，离线验收不需要重新下载。qa_lab.data对源文件SHA256和已有输出有检查。完整重训/模型转换属于可选复现，先用新work目录；历史批量实验脚本是冻结实验入口，不是可任意重跑覆盖的快速开始。

本轮再次查验的官方依据：

- [SQuAD官方数据说明](https://rajpurkar.github.io/SQuAD-explorer/)明确数据为CC BY-SA4.0；项目是公开dev重新切分，非官方隐藏test。
- [CMRC固定版本LICENCE](https://raw.githubusercontent.com/ymcui/cmrc2018/c0eb1b6ba219847457e6af3180da722bbeb656af/LICENCE)为CC BY-SA4.0；保留原许可、作者和修改说明。归属文件为data/chinese-v5/ATTRIBUTION.md。
- [固定学生LICENSE](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE)与[固定教师LICENSE](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE)为Apache-2.0。模型不随源码分发。

代码MIT不覆盖数据；公开前仍保留数据归属和模型上游许可。没有引入新数据集或宣称新的框架能力。

## 后续验证条件

中文拒答评测需要可靠的业务定义与标签，目标硬件验证需要相应设备。后续条件见 [ROADMAP](ROADMAP.md)。已查看的英文test、跨文章64题和中文96题不能继续调参后再声称独立验证。公开后的离线CI记录见README；本页保留的是发布前工程验收事实。
