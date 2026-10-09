"""Stratified blind human-review preparation, separate from automatic QC.

Sampling units are candidate responses, not independent questions. A reviewer
origin is a declaration, not proof that a human performed the annotation.
"""
from collections import Counter, defaultdict
import hashlib
import json
import random

ERRORS = {'none', 'wrong_answer', 'unsupported', 'false_abstention', 'missed_abstention', 'ambiguous_question', 'uncertain'}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def prepare(population, per_stratum=3, seed=20260922):
    if type(per_stratum) is not int or per_stratum < 1 or type(seed) is not int:
        raise ValueError('Positive per-stratum sample size and integer seed required')
    groups=defaultdict(list);seen=set()
    for batch,row in population:
        identity=(batch,row['record_id'])
        if identity in seen:raise ValueError('Duplicate population identity')
        seen.add(identity)
        if row['decision'] not in {'accept','review','reject'} or row['slice'] not in {'answerable','unanswerable','unknown'}:
            raise ValueError('Unknown QC stratum')
        source=row.get('source')
        if not source or any(not isinstance(source.get(k),str) for k in ['context','question']) or not isinstance(row.get('raw_response'),str):
            raise ValueError('Blind review requires raw text and source context/question')
        groups[(batch,row['decision'],row['slice'])].append(row)
    if not seen:raise ValueError('Empty population')
    rng=random.Random(seed);items=[];key=[];strata=[]
    for group,rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:r['record_id']);sample=rng.sample(rows,min(len(rows),per_stratum))
        sid=digest(list(group))[:24]
        strata.append(dict(stratum_id=sid,batch=group[0],decision=group[1],answerability=group[2],population_n=len(rows),sample_n=len(sample)))
        for row in sample:
            bid=digest([seed,group[0],row['record_id']])[:24]
            items.append(dict(blind_id=bid,context=row['source']['context'],question=row['source']['question'],response=row['raw_response']))
            key.append(dict(blind_id=bid,stratum_id=sid,batch=group[0],record_id=row['record_id']))
    rng.shuffle(items)
    packet_sha=digest(items)
    template=dict(schema='blind-annotations-v1',packet_sha256=packet_sha,reviewer_alias='reviewer-a',origin='pending_human',annotations=[dict(blind_id=r['blind_id'],judgment=None,error=None,note='') for r in items])
    return dict(schema='blind-review-v1',seed=seed,per_stratum=per_stratum,population_n=len(seen),items=items,key=key,strata=strata,template=template,packet_sha256=packet_sha)


def analyze(bundle, annotations):
    if bundle.get('schema')!='blind-review-v1' or digest(bundle['items'])!=bundle['packet_sha256']:
        raise ValueError('Packet binding mismatch')
    ids=[r['blind_id'] for r in bundle['items']]
    if len(ids)!=len(set(ids)) or Counter(r['blind_id'] for r in bundle['key'])!=Counter(ids):raise ValueError('Key coverage mismatch')
    groups={s['stratum_id']:s for s in bundle['strata']}
    counts=Counter(r['stratum_id'] for r in bundle['key'])
    if len(groups)!=len(bundle['strata']) or set(counts)!=set(groups) or any(counts[k]!=s['sample_n'] or s['population_n']<s['sample_n'] for k,s in groups.items()):
        raise ValueError('Stratum coverage mismatch')
    if not isinstance(annotations,list) or not 1 <= len(annotations) <= 2:raise ValueError('One or two reviewer templates required')
    reviewers=[];completed=[];aliases=set();pending=False
    for a in annotations:
        if not isinstance(a,dict):raise ValueError('Reviewer template must be an object')
        alias=a.get('reviewer_alias')
        if not isinstance(alias,str) or not alias.strip() or alias in aliases:raise ValueError('Reviewer aliases must be distinct and nonempty')
        aliases.add(alias)
        if a.get('schema')!='blind-annotations-v1' or a.get('packet_sha256')!=bundle['packet_sha256'] or a.get('origin') not in {'pending_human','human_attested','synthetic_fixture'}:
            raise ValueError('Annotation schema/origin/binding mismatch')
        rows=a.get('annotations')
        if not isinstance(rows,list) or any(not isinstance(r,dict) or not isinstance(r.get('blind_id'),str) for r in rows) or Counter(r.get('blind_id') for r in rows)!=Counter(ids):raise ValueError('Annotation ID coverage mismatch')
        filled=0
        for r in rows:
            j,e,n=r.get('judgment'),r.get('error'),r.get('note')
            if not isinstance(n,str):raise ValueError('Note must be text')
            if j is None:
                if e is not None:raise ValueError('Pending row has error label')
                continue
            if j not in {'correct','incorrect','uncertain'} or e not in ERRORS:raise ValueError('Unknown annotation label')
            if (j=='correct' and e!='none') or (j=='incorrect' and e in {'none','uncertain','ambiguous_question'}) or (j=='uncertain' and e not in {'uncertain','ambiguous_question'}) or (j!='correct' and not n.strip()):
                raise ValueError('Inconsistent judgment/error or missing explanation')
            filled+=1
        complete=filled==len(ids) and a['origin']!='pending_human'
        result=dict(reviewer_alias=alias,origin=a['origin'],completed_n=filled,total_n=len(ids),strata=None)
        if complete:
            lookup={r['blind_id']:r for r in rows};stats=[]
            for sid,s in groups.items():
                selected=[lookup[r['blind_id']] for r in bundle['key'] if r['stratum_id']==sid]
                c=Counter(r['judgment'] for r in selected)
                stats.append(dict(**s,correct_n=c['correct'],incorrect_n=c['incorrect'],uncertain_n=c['uncertain'],correct_fraction=c['correct']/len(selected),error_counts=dict(Counter(r['error'] for r in selected))))
            result['strata']=stats;completed.append((alias,lookup))
        else:pending=True
        reviewers.append(result)
    agreement=None;queue=[]
    if len(completed)==2:
        a,b=completed
        for bid in ids:
            if a[1][bid]['judgment']!=b[1][bid]['judgment']:
                queue.append(dict(blind_id=bid,judgments={a[0]:a[1][bid]['judgment'],b[0]:b[1][bid]['judgment']},resolution=None))
        agreement=dict(n=len(ids),disagreements=len(queue),exact_judgment_agreement=(len(ids)-len(queue))/len(ids))
    origins={r['origin'] for r in reviewers}
    status='pending' if pending else ('human_declared_complete' if origins=={'human_attested'} else 'synthetic_test_only' if origins=={'synthetic_fixture'} else 'mixed_origins_not_human_estimate')
    return dict(schema='blind-review-analysis-v1',status=status,reviewers=reviewers,agreement=agreement,adjudication_queue=queue,
        limitations=['Human origin is self-declared, not independently verified.','Fractions describe each sampled stratum; do not pool unequal sampling rates.','Uncertain judgments remain in denominators; no adjudicated ground truth is inferred.','Candidate responses may share questions/families; no independent-sample confidence interval.','No training or causal model-quality claim.'])
