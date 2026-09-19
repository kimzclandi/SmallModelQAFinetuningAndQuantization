"""Generate the Q8 report from frozen actual metrics."""
from scripts.report_output import report_output
OUTPUT = report_output()
import json
from pathlib import Path
from qa_lab.common import read_jsonl,write_json
from scripts.quantization_study import R,C
s=json.loads((R/'summary.json').read_text());p=json.loads(C.read_text())
lines=['# 第四轮：FP16 / Q4 / Q8 同框架量化对照','','已运行验证。通过本地dev压缩筛选的候选：'+str(s['qualifying_local_candidates'])+'。没有新test推理，没有模型训练，发布状态见README。','','## 实测结果','','质量样本为同一74题dev；性能为同一8个train输入、每题强制32tokens，每精度3个独立进程运行的中位数。','',
'| 指标 | FP16 | affine Q4 | affine Q8 |','|---|---:|---:|---:|']
vs=[s['variants'][v] for v in p['variants']]
for label,fn in [('dev EM',lambda x:f"{100*x['quality']['overall']['em']:.2f}%"),('dev token F1',lambda x:f"{100*x['quality']['overall']['f1']:.2f}%"),('有答案 EM',lambda x:f"{100*x['quality']['answerable']['em']:.2f}%"),('权重文件 / decimal MB',lambda x:f"{x['weight_bytes']/1e6:.2f}"),('decode tokens/s',lambda x:f"{x['performance']['median_decode_tps']:.2f}"),('平均TTFT的跨run中位数 / ms',lambda x:f"{x['performance']['median_ttft_ms']:.2f}"),('采样RSS峰值的中位数 / GiB',lambda x:f"{x['performance']['median_rss_bytes']/2**30:.3f}"),('MLX active峰值的中位数 / GiB',lambda x:f"{x['performance']['median_mlx_peak_active_bytes']/2**30:.3f}")]:
 lines.append('| '+label+' | '+' | '.join(fn(x) for x in vs)+' |')
b,q=vs[0],vs[2]
lines+=['',f"Q8权重减少{100*(1-q['weight_bytes']/b['weight_bytes']):.2f}%，采样RSS峰值中位数减少{100*(1-q['performance']['median_rss_bytes']/b['performance']['median_rss_bytes']):.2f}%，本轮固定工作量解码速度比为{q['performance']['median_decode_tps']/b['performance']['median_decode_tps']:.3f}×。这些是本机此工作量的测量，不是跨硬件通用加速承诺。",'',
'RSS和MLX计数在统一内存中重叠，不相加。RSS每10ms采样，不保证捕捉真实峰值；MLX active不含allocator cache，后者单独保存在每个run与summary中。进程内存包括加载和预热。TTFT从已tokenize/传输后的同步prefill开始，不含模型加载、网络与排队。',
'', '## 固定门槛与逐题回归','',
'方案在commit 3525a20提前登记：相对同轮FP16，整体EM最多少1/74、有答案EM最多少1/33、F1最多下降2个百分点；权重<=60%、RSS<=80%、decode>=95%、TTFT<=110%。全部满足才能成为本地dev压缩候选。','',
'| 候选 | EM | 有答案EM | F1 | 权重 | RSS | decode | TTFT | 全部通过 |','|---|---|---|---|---|---|---|---|---|']
for v,g in s['gate'].items():lines.append('| '+v+' | '+' | '.join('通过' if x else '未通过' for x in [*g['checks'].values(),g['pass_all']])+' |')
lines+=['','| 与FP16相比 | 修复题数 | 回归题数 |','|---|---:|---:|']
for v,d in s['paired'].items():lines.append(f"| {v} | {len(d['fixes'])} | {len(d['regressions'])} |")
rows={r['id']:r for r in read_jsonl('data/complexity-v1/dev.jsonl')}
pm={v:{r['id']:r for r in read_jsonl(R/f'{v}-quality/dev.predictions.jsonl')} for v in p['variants']}
details=[{'id':i,'question':rows[i]['question'],'answers':rows[i]['answers'],'fp16':pm['fp16'][i]['prediction'],'q8':pm['q8'][i]['prediction']} for i in s['paired']['q8']['regressions']]
out=OUTPUT/'q8-regressions.json'
write_json(out,details)
for d in details:lines+=['',f"Q8回归样例 `{d['id']}`：",'',d['question'],'',f"FP16：`{d['fp16']}`；Q8：`{d['q8']}`；参考答案：{d['answers']}。"]
lines+=['','## 每次性能测量（不挑最好的一次）','','| 顺序 | 精度 | decode tokens/s | TTFT ms |','|---:|---|---:|---:|']
for n,v in enumerate(p['benchmark_order'],1):
 x=next(x for x in s['variants'][v]['performance']['runs'] if x['run']==n)
 lines.append(f"| {n} | {v} | {x['decode_tps']:.2f} | {x['ttft_ms']:.2f} |")
lines+=['','顺序为三轮轮换：FP16/Q4/Q8，Q4/Q8/FP16，Q8/FP16/Q4。每次单独进程、串行运行、2次虚构输入预热；固定32输出tokens，包括遇到EOS后继续，仅用于性能，不计质量。decode排除首token，分子每题31个；质量按EOS停止、最多48tokens。',
'', '## 机制与适用边界','',
'- 原始学生、同一FP16导出权重、同MLX实现与版本、同prompt/tokenizer、greedy、batch1；Q4/Q8均为group64 affine整数权重量化，KV Cache保持浮点。Q8不是FP8。',
'- Q8通过2个训练输入与MLX-LM上游greedy逐token一致性检查；全部质量/性能输入token hash均核对。两题parity不是完整实现正确性的证明。',
'- 开发集已经被多轮查看，本轮通过的是预设容差内的压缩筛选，未取得独立test验证，不能包装为无损量化。',
'- 基础模型自身仍缺乏可靠拒答能力；dev始终拒答基线为55.41%，本轮各模型总体EM均更低。压缩候选通过不表示QA业务达标。',
'- Q4仍保留更小更快但质量差的结果。不同轮次测速受系统状态影响，不把旧速度与本轮混在一起计算增益。',
'- 模型权重只存本地.cache，不入Git/源码包；转换命令、来源hash、预测、时延、内存、配置与源码快照均可追溯。',
'', '复现与实验方法见docs/QUANTIZATION_V4.md。']
out=OUTPUT/'RESULTS.md';txt='\n'.join(lines)+'\n'
out.write_text(txt)
print(out)
