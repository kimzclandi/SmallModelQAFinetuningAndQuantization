"""Render frozen Chinese portability results, with explicit non-official metrics."""
import json
from collections import Counter
from qa_lab.common import read_jsonl,write_json
from scripts.chinese_study import R,D
s=json.loads((R/'summary.json').read_text());audit=json.loads((D/'manifest.json').read_text())['audit']
rows={r['id']:r for r in read_jsonl(D/'external.jsonl')}
pm={v:{x['id']:x for x in read_jsonl(R/v/'predictions.jsonl')} for v in ['fp16','q8']}
changes=[dict(id=i,question=r['question'],answers=r['answers'],fp16=pm['fp16'][i]['prediction'],q8=pm['q8'][i]['prediction']) for i,r in rows.items() if pm['fp16'][i]['prediction']!=pm['q8'][i]['prediction']]
out=R/'changed-answers.json'
if out.exists():assert json.loads(out.read_text())==changes
else:write_json(out,changes)
lines=['# 第五轮：Q8候选的中文新来源验证','','已完成：同一原始学生FP16与已选Q8，在冻结的96题/96篇文章上各推理一次。模型与数据在commit 6c2929f冻结；提示词与指标更早预登记。没有训练、教师采集或旧test推理。',
'', '## 结果','',
'本轮是CMRC2018公开dev的项目级新来源子集，**不是官方隐藏测试或CMRC榜单分数**。严格EM是主指标；另外两项为明确命名的自定义补充指标。', '',
'| 指标 | FP16 | Q8 |','|---|---:|---:|']
for label,key in [('严格EM','strict_em'),('归一化EM（自定义）','normalized_em'),('字符LCS F1（自定义）','char_lcs_f1'),('原文片段/拒答格式有效率','format_valid'),('拒答率','abstain')]:lines.append('| '+label+' | '+' | '.join(f"{s['metrics'][v][key]*100:.2f}%" for v in ['fp16','q8'])+' |')
lines+=['',f"两者严格答对均为44/96题；严格正确集合相同，Q8相对FP16修复{len(s['fixes'])}题、回归{len(s['regressions'])}题。但有{len(changes)}题输出文本不同，不能称为逐token或逐条输出无损。所有变更见changed-answers.json。",'',
'质量保持门槛：严格EM最多下降1/96、字符LCS F1最多下降2个百分点、格式有效率最多下降1/96。实际各项：'+str(s['checks'])+'；全部通过：'+str(s['pass_all'])+'。没有在结果出来后改阈值。',
'', '## 数据处理与失败分母','',
f"官方源文件共{audit['raw_n']}题。按预登记规则，全部参考答案必须非空且offset与原文一致；176道题未通过该校验而在抽样前排除。hash遍历至填满96题时，另跳过2道已选文章的题。没有因模型答错而删样本。审计明细：{dict(Counter(x['reason'] for x in audit['excluded']))}。",'',
'这不表示176题语义标注都错误，而是它们未满足本次要求的全部参考答案位置校验。位置错误、空文本等不同原因如需修复，应另开数据清洗实验，不能事后回填影响本轮样本。',
'', '## 范围与局限','',
'- 全部题目有答案：不能验证信息不足时的拒答、抗幻觉或拒答阈值。始终拒答在本集合上的严格EM为0%。',
'- 通用中文百科阅读理解，不是计算机专门任务或企业知识库；没有业务部署或用户反馈验证。',
'- 这是第四轮已选候选在新来源数据上的一次检查，支持此范围内的质量保持；不是统计等价证明或预训练无污染证明。',
'- 用新的中文提示，不把分数与旧英文SQuAD分数直接比较，更不能把差异当成训练收益。',
'- 严格EM区分全部内容；归一化会移除Unicode标点和空白，可能掩盖例如3.14/314的差异，因此仅作辅助，不替代严格EM与格式检查。',
'- 默认最多48输出token，截断输出仍计入96题分母。',
'- 没有重新测速，不能把第四轮英文工作量的速度优势直接推断为中文端到端加速。',
'', '## 复现验证','',
'新增隔离输出的reproduce入口，已用同一Q8模型运行2题smoke，结果保存在reproduction-smoke；不并入正式96题指标。这是同机入口验证，不是异机复现。',
'', '依赖、输入输出、指标公式和学习题见docs/CHINESE_V5.md。数据按CC BY-SA4.0提供，作者、固定源版本与原许可证见data/chinese-v5/ATTRIBUTION.md和SOURCE_LICENSE.txt。']
out=R/'RESULTS.md';text='\n'.join(lines)+'\n'
if out.exists():assert out.read_text()==text
else:out.write_text(text)
print(out)
