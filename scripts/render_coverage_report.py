"""Render coverage-v3 tables from saved metrics; no model execution."""
import json,statistics
from pathlib import Path
from scripts.coverage_study import R,C,H
s=json.loads((R/'summary.json').read_text());p=json.loads((C/'protocol.json').read_text())
lines=['# 第三轮结果：扩大gold覆盖与跨文章检查','','已完成6次242步LoRA训练、6次74题dev评估、7次64题新文章评估。没有重跑旧102题test；不把重复评估次数当成独立样本数。','',
'开发集决定：`'+s['dev']['decision']+'`；符合三个seed门槛的方法：'+str(s['dev']['passing_arms'])+'。决定在新文章推理前提交和冻结，见freeze.json。',
'', '## 同步数对照','',
'每组3个seed。以下±是样本标准差，单位为百分点，不是置信区间。','',
'| 数据与评测 | 整体EM | 有答案EM | 无答案EM |','|---|---:|---:|---:|']
for split in ['dev','external']:
 for arm in p['arms']:
  a=s['aggregate'][split][arm]
  vals=[f"{a[k]['mean']*100:.2f}% ± {a[k]['sample_std']*100:.2f}" for k in ['overall','answerable','unanswerable']]
  lines.append('| '+split+' / '+arm+' | '+' | '.join(vals)+' |')
b=s['external']['baseline']
lines+=['',f"新文章原始学生：EM {b['overall']['em']*100:.2f}%，F1 {b['overall']['f1']*100:.2f}%，有答案EM {b['answerable']['em']*100:.2f}%，无答案EM {b['unanswerable']['em']*100:.2f}%。始终拒答基线为50.00%。",'',
'原始学生的旧dev为整体25.68%、有答案57.58%。跨文章基线只说明新集合上的实测表现，不能拿它与旧test直接算提升。','',
'## 每个seed的新文章结果','',
'| 模型 | EM | F1 | 有答案EM | 无答案EM | 训练监督tokens |','|---|---:|---:|---:|---:|---:|']
for name,m in s['external'].items():
 tokens='—' if name=='baseline' else str(json.loads((R/name/'training.json').read_text())['total_supervised_tokens'])
 vals=[f'{v*100:.2f}%' for v in [m['overall']['em'],m['overall']['f1'],m['answerable']['em'],m['unanswerable']['em']]]
 lines.append('| '+name+' | '+' | '.join(vals)+' | '+tokens+' |')
lines+=['','## 分文章结果','','每篇32题，各16道有答案和无答案题。表中训练方法是三个seed的平均EM；两篇文章不代表完整业务域。','','| 方法 | Packet_switching | Prime_number |','|---|---:|---:|']
for arm in ['baseline',*p['arms']]:
 names=['baseline'] if arm=='baseline' else [f'{arm}-{seed}' for seed in p['seeds']]
 vals=[f"{statistics.mean(s['by_article'][n][t]['overall']['em'] for n in names)*100:.2f}%" for t in p['holdout_titles']]
 lines.append('| '+arm+' | '+' | '.join(vals)+' |')
lines+=['','## 扩展组相对重复训练组的逐题变化（新文章）','','| seed | 修复 | 回归 | EM差值（百分点） |','|---|---:|---:|---:|']
for seed,d in s['paired'].items():lines.append(f"| {seed} | {len(d['fixes'])} | {len(d['regressions'])} | {100*d['delta_em']:+.2f} |")
lines+=['','具体题ID保存在summary.json的paired字段；每个模型都有逐题预测、评分和错误记录。不能从总分推断哪些类别改善，需读有答案、拒答与按文章切片。',
'', '## 方法和限制','',
'- 24题控制组和242题扩展组均为同242步、同LoRA/学习率/原始学生、3个seed；前者每题10或11次，后者每题一次。监督token、epoch、样本分布和覆盖度不相同。',
'- 扩展组使用全部旧train标准答案，没有新建教师标签；gold是公开标注，不等于已逐题人工复核的绝对正确答案。第一答案选择也可能损失可接受表述的多样性。',
'- 新64题按文章和类别固定hash抽样，去重阈值沿用第一轮；对旧train/dev/test文本做近重复排除，但不做旧test模型推理。不是自然类别占比，不是官方隐藏测试。',
'- 文章未用于本项目训练，但同属公开SQuAD dev，可能出现在模型预训练中；英文数学/网络问答不等于中文企业业务。',
'- 所有候选包括dev门槛失败者都按预登记统一评估；不依据新文章结果挑seed、追加训练或改阈值。',
'- 本轮同机M4 Max本地免费运行；模型、配置、数据hash、源码、adapter加载hash和每次命令均保存。未上传或发布。',
'', '复现与主动回忆见docs/COVERAGE_V3.md；原阶段一/闭环/第二轮证据保持不变。']
out=R/'RESULTS.md';text='\n'.join(lines)+'\n'
if out.exists():assert out.read_text()==text
else:out.write_text(text)
print(out)
