"""New common source-only answer codec; old scores stay under old rules.
No substring search, fuzzy matching, gold-based repair, or semantic retry.
"""
import unicodedata

def norm(s):return ' '.join(unicodedata.normalize('NFC',s).split())

def decode(text,aliases,known_ids,*,truncated=False,invalid_control_tokens=False):
    if truncated or invalid_control_tokens:return {'entity':None,'status':'INVALID_OUTPUT'}
    lines=text.strip().splitlines()
    if not lines:return {'entity':None,'status':'INVALID_OUTPUT'}
    if len(lines)>1:return {'entity':None,'status':'MULTILINE_OUTPUT'}
    line=norm(lines[0])
    if line=='UNSUPPORTED':return {'entity':None,'status':'REFUSAL'}
    forms={line}
    if line.startswith('Answer:'):forms.add(line[len('Answer:'):].strip())
    for x in list(forms):
        forms.add(x[:-1].rstrip() if x.endswith('.') else x)
    for x in list(forms):
        for l,r in [('"','"'),("'","'"),('“','”'),('‘','’')]:
            if len(x)>=2 and x.startswith(l) and x.endswith(r):
                y=x[1:-1].strip();forms.add(y);forms.add(y[:-1].rstrip() if y.endswith('.') else y)
    ids={aliases[norm(x)] for x in forms if norm(x) in aliases}
    ids|={x for x in forms if x in known_ids}
    if len(ids)!=1:return {'entity':None,'status':'AMBIGUOUS_OUTPUT' if ids else 'UNRECOGNIZED_OUTPUT'}
    return {'entity':next(iter(ids)),'status':'VALID'}
