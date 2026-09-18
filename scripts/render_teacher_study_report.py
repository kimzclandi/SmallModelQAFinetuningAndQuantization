"""Render a reviewable report from frozen teacher-study-v2 metrics (no inference)."""
import json
from pathlib import Path
from qa_lab.common import read_jsonl,write_json
from scripts.teacher_study import ROOT,DATA,SELECTION,paired,pilot_rows
from qa_lab.metrics import evaluate
s=json.loads((ROOT/'summary.json').read_text())
labels={'gold':'标准答案 SFT','original_teacher':'原提示词教师蒸馏','prompted_teacher':'新提示词教师蒸馏'}
lines=['# 第二轮真实结果：教师提示与三随机种子对照','',
'状态：本地训练和评测已完成；未公开发布。3组×3seed，共9次48步LoRA训练和666条dev预测（重复评估同一74题，不是666道独立问题）。没有新增test推理。',
'', '## 主要结论','',
'新提示提高了教师拒答倾向；必须同时看总EM与有答案EM。三个seed用于描述本轮训练波动，不是统计显著性证明。最终门槛结果：`'+s['decision']+'`，通过的方法：'+str(s['passing_arms'])+'。',
'', '## 三seed均值与样本标准差','',
'下表全部为开发集，单位%；±表示三个seed的样本标准差（不是置信区间或标准误）。所有方法使用相同24道训练题；未训练学生参考值为整体19/74=25.68%、有答案19/33=57.58%。', '',
'| 标签来源 | 整体 EM | 有答案 EM | 无答案 EM |','|---|---:|---:|---:|']
for a,ag in s['aggregate'].items():
 vals=[f"{ag[k]['mean']*100:.2f} ± {ag[k]['sample_std']*100:.2f}" for k in ['overall_em','answerable_em','unanswerable_em']]
 lines.append('| '+labels[a]+' | '+' | '.join(vals)+' |')
lines+=['','## 每次运行与监督token预算','','| seed | 标签来源 | EM | token F1 | 有答案 EM | 无答案 EM | 监督tokens | 通过门槛 |','|---|---|---:|---:|---:|---:|---:|---|']
for r in s['runs'].values():
 m=r['metrics'];vals=[f'{v*100:.2f}' for v in [m['overall']['em'],m['overall']['f1'],m['answerable']['em'],m['unanswerable']['em']]]
 lines.append(f"| {r['seed']} | {labels[r['arm']]} | "+' | '.join(vals)+f" | {r['supervised_tokens']} | {'是' if r['gate_pass'] else '否'} |")
lines+=['','48步和每题两次曝光固定，但标签长度不同，监督token预算不同；不能把性能差异全部解释成教师知识迁移。','','## 教师回答审计（训练题，不是独立泛化测试）','','| 教师提示 | EM | F1 | 有答案正确数 | 无答案正确数 | 格式有效数 |','|---|---:|---:|---:|---:|---:|']
for name,m in s['teacher'].items():
 lines.append(f"| {name} | {m['overall']['em']*100:.2f}% | {m['overall']['f1']*100:.2f}% | {round(m['answerable']['em']*12)}/12 | {round(m['unanswerable']['em']*12)}/12 | {round(m['overall']['format_valid_rate']*24)}/24 |")
lines+=['','所有24条非空、EOS完成回答均保留，未按gold正确性筛选。normalized EM与原文连续片段格式是独立指标：大小写、标点或冠词归一化后可能匹配，但不满足严格输出格式。因此EM正确数可以高于category=correct的数量。','','## 配对修复与回归','','方向是左侧方法→右侧方法；每个单元比较相同seed的74个ID。','','| 对照方向 / seed | 修复数 | 回归数 | EM差值（百分点） |','|---|---:|---:|---:|']
for name,p in s['paired'].items():lines.append(f"| {name} | {len(p['fixes'])} | {len(p['regressions'])} | {p['delta_em']*100:+.2f} |")
train=pilot_rows(DATA,SELECTION)
old=read_jsonl('reports/closure-v1/teacher-fp32/predictions.jsonl');new=read_jsonl(ROOT/'teacher/predictions.jsonl')
_,os=evaluate(train,old);_,ns=evaluate(train,new)
op={r['id']:r for r in old};np={r['id']:r for r in new};om={r['id']:r for r in os};nm={r['id']:r for r in ns}
changes=[{'id':r['id'],'question':r['question'],'gold':r['answers'],'is_impossible':r['is_impossible'],
          'old':op[r['id']]['prediction'],'new':np[r['id']]['prediction'],'old_em':om[r['id']]['em'],'new_em':nm[r['id']]['em']} for r in train if op[r['id']]['prediction']!=np[r['id']]['prediction']]
teacher_delta={'paired':paired(os,ns),'changed_responses':changes}
if (ROOT/'teacher-paired.json').exists():assert json.loads((ROOT/'teacher-paired.json').read_text())==teacher_delta
else:write_json(ROOT/'teacher-paired.json',teacher_delta)
lines+=['','逐条ID见summary.json的paired字段；教师具体变更见teacher-paired.json；学生错误题见各run/dev/dev.failures.jsonl。保留全部错误，不只挑选成功案例。','','## 为什么没有采用','','预登记要求每个seed都满足整体EM>25.68%且有答案EM>=57.58%。不能用总EM上升抵消有答案能力的退化。保留基准只是保留对照，不代表基准已合格。',
'', '## 证据与局限','',
'- 协议在运行前提交：1bacfca；完整提示、3个seed与数值超参数见configs/teacher-study-v2。',
'- 每条输入仅文段和问题；推理不读取gold。teacher只生成train；学生只评估dev。commands.jsonl记录实际命令，source保留执行源码。',
'- 同一seed内三组相同初始化种子、同样本顺序、同48步；adapter文件hash与评测加载hash匹配。固定seed并不保证跨平台逐位复现。',
'- 新提示词含4个人工虚构示例，只根据旧训练题上的教师失误设计；没有把dev/test题写入prompt。提示长短及多个指令一起改变，不能做单句因果归因。',
'- 本轮所有标签只来自24题，且新提示教师仍错得多。dev被多轮使用；3个seed不是3个独立数据集。需要新来源评测、更多样本及受控token预算才能进一步判断泛化。',
'- 无付费API、无新模型下载、无公开上传。本轮不声称取得可采用的模型优化，不刷新旧test分数。',
'', '复现、输入输出和学习题见docs/TEACHER_STUDY_V2.md。']
p=ROOT/'RESULTS.md'
text='\n'.join(lines)+'\n'
if p.exists():assert p.read_text()==text
else:p.write_text(text)
print(p)
