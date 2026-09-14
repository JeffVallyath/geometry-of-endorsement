"""Independent integer/reference-label checker; does not call task evaluate/apply_semantics."""
from pathlib import Path
import json, hashlib
from . import protocol as P

def verify(root):
    root=Path(root);seen=set();counts={};checks=0
    for name in ['fit','cal','final','workflow']:
        rows=[json.loads(x) for x in (root/f'{name}_scenes.jsonl').read_text().splitlines()]
        counts[name]=len(rows)
        for d in rows:
            assert d['scene_id'] not in seen;seen.add(d['scene_id'])
            for k in ['actors','projects','layout']:d[k]=tuple(d[k])
            d['values']=tuple(tuple(x) for x in d['values'])
            s=P.Scene(**d)
            if name!='final':continue
            for key,prog in P.programs(s).items():
                updated=[list(x) for x in s.values]
                for command in prog:updated[command.actor][command.project]=command.value
                for draw in [0,1]:
                    for q in P.program_queries(s,key,draw):
                        spec=q['spec'];k=spec['kind'];a,b,p=spec['a'],spec['b'],spec['p']
                        def solve(m):
                            x,y=m[a][p],m[b][p]
                            return {'direct':x,'opposes':1-x,'same':int(x==y),'both':x*y,'either':int(x+y>0)}[k]
                        before,after=solve(s.values),solve(updated)
                        assert before==q['source_gold'] and after==q['gold'] and (before!=after)==q['changed']
                        checks+=1
    return {'source_scenes':counts,'program_query_labels_checked':checks,'split_ids_unique':True,'model_forwards':0}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('data');ap.add_argument('--out');args=ap.parse_args();res=verify(args.data)
    if args.out:Path(args.out).write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2))
