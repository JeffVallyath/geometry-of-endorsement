"""Deterministic full logical schedule. No candidate output enters selection."""
def make_schedule(cases,arms):
    rows=[];notes=[]
    for case in cases:
        for phase,histories in [('post',['H0','H1','S0','S1','C','C+']),('pre',['H0','H1','S0','S1'])]:
            for h in histories:
                notes.append({'case_id':case['case_id'],'history':h,'phase':phase})
                for q in case['queries']:
                    if phase=='pre' and not q['query_id'].startswith('J'):continue
                    for a in arms:
                        rows.append({'case_id':case['case_id'],'history':h,'phase':phase,'query_id':q['query_id'],'arm':a})
    return rows,notes
