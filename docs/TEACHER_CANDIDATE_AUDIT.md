# 手工教师候选的导入与审计

该流程保留24条手工采集回答的原始字节、解析修复与质量审计，仅用于检查候选采集工具。它们未参与训练。实际响应蒸馏使用本地Qwen教师，见[复现指南](REPRODUCE_CLOSURE.md)。

输入为训练集的id/context/question，输出为同ID的原文答案片段或严格`NO_ANSWER`。请求包按固定hash排序选12条有答案和12条无答案；gold用于抽样，不进入请求。数据来源与再分发条件见[数据许可](../DATA_LICENSE.md)。

导入检查schema、ID完整覆盖、来源hash、请求内容和原文片段格式，不验证语义正确性、不按gold筛掉错误、不自动开始训练。原始响应按字节保留；转换为训练数据前还须检查质量及来源服务的适用条款。不能仅凭订阅资格推定训练或再分发许可。

```bash
.venv/bin/python -m qa_lab.teacher_io pack --output work/teacher-pilot-rebuild
.venv/bin/python -m qa_lab.teacher_io import --help
```

导入参数包含请求包、原始JSON、实际模型标识、采集日期与新的输出目录。拒绝覆盖旧记录；未知底层版本或采样参数必须记录为未知。无法固定教师服务时，只能冻结已采集响应以复现后续审计，不能保证重新请求一致。

24条候选中的13处非法JSON转义按独立解析副本处理，原始文本保留。结果和用途边界见[历史审计](../reports/teacher-audit-v1/REVIEW.md)。测试中的合成响应只存在临时目录，不作为教师运行记录。
