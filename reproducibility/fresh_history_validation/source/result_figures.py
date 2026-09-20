"""Two local SVG figures from completed, verified CPU reconstruction only."""
import argparse
from html import escape
import json
import math
from pathlib import Path

def read(path):return json.loads(path.read_text())

def text(x,y,value,size=16,color='#17202a'):
    return f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" fill="{color}">{escape(str(value))}</text>'

def line(x1,y1,x2,y2,color='#d5d8dc',width=1,dash=''):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>'

def canvas(title):
    return ['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="570" viewBox="0 0 920 570" role="img">',
        f'<title>{escape(title)}</title>','<rect width="920" height="570" fill="white"/>',text(38,42,title,24)]

def write_new(path,parts):
    with path.open('x',encoding='utf8') as handle:handle.write('\n'.join(parts+['</svg>']))

def main(source,out):
    proof=read(source/'CPU_REPLAY.json')
    if proof['status']!='PASS' or proof['final_cases']!=64 or proof['learned_seeds']!=[0,1]:
        raise ValueError('Completed two-seed CPU replay required before plotting')
    primary=read(source/'PRIMARY.json');utility=read(source/'UTILITY.json')
    out.mkdir(parents=True,exist_ok=True)
    p=canvas('Fresh-case confirmation: full matched-history criterion')
    p.append(text(38,73,'64 cases; fixed denominator of 1,152 coded targets per seed',15))
    values=[r['matched']/r['denominator'] for r in primary['per_seed']]+[primary['rate']]
    ceiling=max(0.01,max(values+[primary['interval95'][1]])*1.25)
    y=lambda value:430-300*value/ceiling
    for i in range(5):
        value=ceiling*i/4;p.extend([line(115,y(value),840,y(value)),text(42,y(value)+5,f'{100*value:.2f}%',14)])
    labels=['Stored seed 0','Stored seed 1','Case-level mean']
    for x,value,label in zip((195,425,655),values,labels):
        p.append(f'<rect x="{x}" y="{y(value)}" width="110" height="{430-y(value)}" fill="#225ea8"/>')
        p.extend([text(x,y(value)-12,f'{100*value:.2f}%',17),text(x-5,464,label,16)])
    lo,hi=primary['interval95'];p.extend([line(710,y(lo),710,y(hi),'#17202a',2),line(700,y(lo),720,y(lo),'#17202a',2),line(700,y(hi),720,y(hi),'#17202a',2)])
    p.append(text(38,515,'Error bar: 95% paired case-bootstrap interval; seeds are repeated measurements.',15))
    p.append(text(38,544,f"Contributing cases: {primary['contributing_cases']}/64. Ineligible targets remain in the denominator.",15))
    write_new(out/'PRIMARY_CONFIRMATION.svg',p)
    u=canvas('Held-out errors at equal retention: literal No/Yes')
    u.append(text(38,73,'A: six checks at one history. B: three checks across two histories.',15))
    plot_values=[r['policies'][p]['equal_retention'][str(f)]['task_error']
                 for r in utility['per_seed_format'] if r['draw']==0
                 for p in ('A','B') for f in (.25,.5,.75)]
    ceiling=min(1.,max(.05,math.ceil(max((v for v in plot_values if v is not None),default=0.)*125)/100))
    y=lambda value:410-280*value/ceiling
    xs={'.25':240,'.5':490,'.75':740}
    for i in range(5):
        value=ceiling*i/4;u.extend([line(115,y(value),840,y(value)),text(55,y(value)+5,f'{100*value:.1f}%',14)])
    for fraction,x in xs.items():u.append(text(x-28,440,f'{100*float(fraction):.0f}% retained',14))
    colors={'A':'#225ea8','B':'#238443'}
    for seed in (0,1):
        row=next(r for r in utility['per_seed_format'] if r['seed']==seed and r['draw']==0)
        for procedure in ('A','B'):
            points=[]
            for f in (.25,.5,.75):
                value=row['policies'][procedure]['equal_retention'][str(f)]['task_error']
                if value is not None:
                    x=xs[str(f).replace('0.','.')];points.append((x,y(value)))
                    u.append(f'<circle cx="{x}" cy="{y(value)}" r="5" fill="{colors[procedure]}"/>')
            for a,b in zip(points,points[1:]):u.append(line(*a,*b,colors[procedure],3,'7 5' if seed else ''))
            xx=120+seed*400;yy=103+(0 if procedure=='A' else 20)
            u.extend([line(xx,yy-5,xx+30,yy-5,colors[procedure],3,'7 5' if seed else ''),text(xx+40,yy,f'{procedure}, seed {seed}; common cases: {row["common_cases"]}',14)])
    endpoint=next(r for r in utility['combined'] if r['draw']==0)
    value=endpoint['B_minus_A_50pct'];interval=endpoint['interval95']
    summary='Named 50% contrast (B minus A): '+('unavailable' if value is None else f'{100*value:+.2f} percentage points')
    if interval is not None:summary+=f'; 95% interval [{100*interval[0]:+.2f}, {100*interval[1]:+.2f}]'
    u.append(text(38,491,summary,15))
    u.append(text(38,521,'Lower error is better. Equal query counts do not imply equal GPU cost.',15))
    if not endpoint['quantitative_headline_available']:u.append(text(38,548,'Descriptive only: fewer than 20 common cases in at least one seed.',15,'#8c2d04'))
    elif interval is not None and interval[0]<=0<=interval[1]:
        u.append(text(38,548,'The 95% interval includes zero; a utility advantage is not established.',15,'#555555'))
    write_new(out/'VALIDATION_UTILITY.svg',u)
    print(json.dumps({'figures':['PRIMARY_CONFIRMATION.svg','VALIDATION_UTILITY.svg'],'source':str(source)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.source,args.output)
