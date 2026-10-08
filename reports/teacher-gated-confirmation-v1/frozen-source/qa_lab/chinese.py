"""Explicit non-official Chinese metrics and output-independent sample isolation."""
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
import unicodedata
from .common import digest


def normalize_zh(text):
    return ''.join(c for c in unicodedata.normalize('NFC',text).casefold() if not c.isspace() and not unicodedata.category(c).startswith('P'))


def lcs_length(a,b):
    previous=[0]*(len(b)+1)
    for x in a:
        current=[0]
        for j,y in enumerate(b):current.append(previous[j]+1 if x==y else max(previous[j+1],current[-1]))
        previous=current
    return previous[-1]


def evaluate_zh(rows,predictions):
    expected=[r['id'] for r in rows];ids=[r['id'] for r in predictions]
    if len(expected)!=len(set(expected)) or len(ids)!=len(set(ids)) or set(expected)!=set(ids):raise ValueError('Duplicate, missing or extra IDs')
    by={r['id']:r for r in predictions};scored=[]
    for row in rows:
        if row['is_impossible'] or not row['answers']:raise ValueError('This metric version only supports answerable rows')
        raw=by[row['id']]['prediction'].strip();normal=normalize_zh(raw)
        valid=bool(raw) and (raw=='NO_ANSWER' or raw in row['context'])
        if not normal or raw=='NO_ANSWER':strict=em=f1=0.0
        else:
            strict=float(any(raw==a.strip() for a in row['answers']))
            refs=[normalize_zh(a) for a in row['answers']]
            em=float(normal in refs)
            f1=max(2*lcs_length(normal,a)/(len(normal)+len(a)) for a in refs)
        category='correct' if strict else ('over_abstention' if raw=='NO_ANSWER' else ('format_or_nonextractive' if not valid else 'wrong_or_partial_span'))
        scored.append(dict(id=row['id'],strict_em=strict,normalized_em=em,char_lcs_f1=f1,format_valid=valid,abstain=raw=='NO_ANSWER',category=category,root_cause=None if strict else 'unverified'))
    if not rows:raise ValueError('Empty evaluation')
    m={key:sum(r[key] for r in scored)/len(scored) for key in ['strict_em','normalized_em','char_lcs_f1','format_valid','abstain']}
    m.update(n=len(rows),categories=dict(Counter(r['category'] for r in scored)),metric_version='zh-strict-em-and-char-lcs-v1-NOT-official-CMRC',unanswerable_n=0)
    return m,scored


@lru_cache(maxsize=10000)
def grams(text):
    s=normalize_zh(text)
    return frozenset(s[i:i+5] for i in range(max(0,len(s)-4))) or frozenset([s])


def near(a,b):
    x,y=grams(a['context']),grams(b['context'])
    if len(x&y)/len(x|y)>=.7:return True
    q,t=normalize_zh(a['question']),normalize_zh(b['question'])
    matcher=SequenceMatcher(None,q,t,autojunk=False)
    return matcher.quick_ratio()>=.9 and matcher.ratio()>=.9


def select(raw,old,token_count,n=96):
    pool=[];excluded=[];seen_raw=set()
    for article in raw['data']:
        for paragraph in article['paragraphs']:
            for q in paragraph['qas']:
                if q['id'] in seen_raw:raise ValueError('Duplicate raw ID')
                seen_raw.add(q['id'])
                valid=bool(q['answers']) and all(a['text'].strip() and normalize_zh(a['text']) and paragraph['context'][a['answer_start']:a['answer_start']+len(a['text'])]==a['text'] for a in q['answers'])
                if not valid:excluded.append({'id':q['id'],'reason':'invalid_reference'});continue
                pool.append(dict(id=q['id'],source_title=article['title'],context=paragraph['context'],question=q['question'],answers=list(dict.fromkeys(a['text'] for a in q['answers'])),is_impossible=False,split='external',family_id=digest(normalize_zh(article['title']))[:16]))
    out=[];titles=set();old_ids={r['id'] for r in old}
    for r in sorted(pool,key=lambda r:digest('cmrc-v5:'+r['id'])):
        title=normalize_zh(r['source_title'])
        if title in titles:excluded.append({'id':r['id'],'reason':'article_already_selected'});continue
        count=token_count(r)
        if count>2048:excluded.append({'id':r['id'],'reason':'input_over_2048','input_tokens':count});continue
        if r['id'] in old_ids or any(near(r,o) for o in old+out):excluded.append({'id':r['id'],'reason':'near_duplicate'});continue
        r['input_tokens']=count;out.append(r);titles.add(title)
        if len(out)==n:break
    if len(out)!=n:raise ValueError('Insufficient isolated eligible questions')
    return out,{'raw_n':len(seen_raw),'selected_n':len(out),'excluded':excluded,'selected_articles':len(titles),'note':'Excluded includes all invalid references and only candidates traversed before fixed sample filled; remaining candidates not selected.'}
