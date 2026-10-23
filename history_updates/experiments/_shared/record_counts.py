"""Exact descriptive counts from the retained training and token schedules."""
import gzip
import json

def records(path):
    with gzip.open(path,'rt',encoding='utf-8') as source:
        for line in source:
            if line.strip():yield json.loads(line)

def training_counts(path):
    pairs=set();samples=set();questions=set();roots=set();count=0
    for row in records(path):
        if row['sample_id'] in samples:raise ValueError('Duplicate training presentation identity')
        count+=1;samples.add(row['sample_id']);pairs.add(row['pair_id'])
        questions.add(row['question_id']);roots.add(row['source_root'])
    return {'draws_per_fit':count,'unique_candidate_pairs':len(pairs),
            'unique_presented_questions':len(questions),'source_cases':len(roots)}

def token_matching(path):
    groups={};ranges={arm:[] for arm in ('ISO','HIST','CURRENT_EXTRA')}
    for row in records(path):
        key=(row['case_id'],row['history'],row['query_id'],row['candidate_sha256'])
        group=groups.setdefault(key,{})
        if row['arm'] in group:raise ValueError('Duplicate candidate-context arm')
        if row['arm'] not in ranges or type(row['n_prompt']) is not int or row['n_prompt']<=0:
            raise ValueError('Invalid token schedule record')
        group[row['arm']]=row['n_prompt'];ranges[row['arm']].append(row['n_prompt'])
    if not groups or any(set(row)!=set(ranges) for row in groups.values()):
        raise ValueError('Incomplete token-matching triplet')
    delta=[row['HIST']-row['CURRENT_EXTRA'] for row in groups.values()]
    return {'paired_candidate_contexts':len(groups),
            'HIST_and_CURRENT_EXTRA_equal_token_lengths':sum(v==0 for v in delta),
            'mean_HIST_minus_CURRENT_EXTRA_tokens':sum(delta)/len(delta),
            'min_HIST_minus_CURRENT_EXTRA_tokens':min(delta),'max_HIST_minus_CURRENT_EXTRA_tokens':max(delta),
            **{arm+'_token_range':[min(values),max(values)] for arm,values in ranges.items()}}
