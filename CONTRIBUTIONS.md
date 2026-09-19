# 贡献与 AI 辅助开发

项目使用 Codex 辅助实验设计、代码与文档编写、命令执行、结果整理及发布维护。AI 辅助工作覆盖数据隔离、训练/评测入口、离线证据核验与复现工具；不声称独立研发基础模型、蒸馏算法或训练框架。

Qwen 提供学生与本地教师模型，SQuAD/Wikipedia 和 CMRC2018 提供数据，PyTorch、Transformers、PEFT 与 MLX 提供推理、训练和量化基础设施。归属见 [THIRD_PARTY](THIRD_PARTY.md) 与 [DATA_LICENSE](DATA_LICENSE.md)。

已保存的实验包括 gold-SFT、本地教师响应蒸馏、训练数据覆盖对照、同 MLX 框架精度比较及冻结 Q8 的中文验证。手工采集的 GPT 候选仅用于审计，未进入训练；实际蒸馏使用本地 Qwen 教师。生成记录、预测、失败样例和门槛决定均保留。

验证层级分开记录：单元测试与离线核验检查已保存证据；同机新环境 CPU 冒烟验证少量实际推理；历史训练和量化结果由对应运行记录支持。Linux CI 不加载模型，不代表异机训练复现。实验限制与当前状态见 [README](README.md) 和 [研究范围](docs/ROADMAP.md)。
