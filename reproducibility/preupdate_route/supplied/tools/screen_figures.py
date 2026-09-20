"""Descriptive no-go figures. Reuses the prior SVG + resvg rendering approach."""
import argparse
import hashlib
from html import escape
import importlib.metadata
import json
from pathlib import Path


def load(path): return json.loads(path.read_text(encoding='utf8'))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def text(x,y,value,size=18,color='#172b3a',anchor='start'):
    return f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(str(value))}</text>'
def line(x1,y1,x2,y2,color='#dbe2e8',width=1,dash=''):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>'
def rect(x,y,w,h,color):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}"/>'
def canvas(title):
    return ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="650" viewBox="0 0 1100 650" role="img">',f'<title>{escape(title)}</title>',rect(0,0,1100,650,'white'),text(40,48,title,27)]
def write(path,parts):
    with path.open('x',encoding='utf8') as f:f.write('\n'.join(parts+['</svg>']))


def create(source,out):
    screen=load(source/'replay/SCREEN.json');proof=load(source/'replay/CPU_REPLAY.json');totals=load(source/'SCREEN_TOTALS.json')
    if screen['status']!='NO_ACTIONABLE_DIRECT_PREUPDATE_ROUTE' or proof['final_cases']!=0:
        raise ValueError('Only the complete prespecified no-go branch is supported')
    out.mkdir(parents=True,exist_ok=True)
    colors={'N':'#496b8a','K':'#b44a38','SCTRL':'#397d72'}
    p=canvas('Historical screen: corrections did not replicate across seeds')
    p += [text(40,82,'16 exposed cases. A correction requires both selected histories and preserved controls.',18),
          text(40,109,'14 previous positive cases and 2 fixed controls. This is not a prevalence sample.',17)]
    for x,arm,label in [(520,'K','Message removal'),(790,'SCTRL','Matched control')]:
        p += [rect(x,134,18,18,colors[arm]),text(x+28,150,label,18)]
    p.append(text(75,165,'Corrected distinct cases',17))
    y=lambda n:455-n*68
    for n in range(5):p += [line(115,y(n),1025,y(n)),text(93,y(n)+6,n,17,'#526779','end')]
    values={}
    for x,seed,label in [(260,0,'Stored seed 0'),(550,1,'Stored seed 1'),(840,None,'Either seed')]:
        for dx,arm in [(-73,'K'),(7,'SCTRL')]:
            n=len(screen['corrected_cases'][arm]) if seed is None else len({r['root_id'] for r in screen['details'] if r['seed']==seed and r['arm']==arm and r['corrected']})
            values[f'{label}/{arm}']=n
            p += [rect(x+dx,y(n),65,455-y(n),colors[arm]),text(x+dx+32.5,y(n)-10,n,22,colors[arm],'middle')]
        p.append(text(x,488,label,20,anchor='middle'))
    p += [text(40,542,'K corrected 3 cases, all in seed 1. The matched control corrected 2.',21),
          text(40,577,'Required: at least 3 K cases, at least 1 per seed, and 2 more than control.',18),
          text(40,613,'Result: the cross-seed and comparison gates failed. No final cases were run.',18,'#8e3628')]
    write(out/'SCREEN_CORRECTIONS.svg',p)
    p=canvas('Historical screen: message removal increased answer errors')
    p += [text(40,83,'Both stored seeds, both selected histories, all scheduled questions in each family.',18),
          text(40,110,'Descriptive rates on the exposed safety panel. No population confidence intervals.',17)]
    for left,family,ceiling,title,new_only in [(100,'joint',15,'Joint-answer errors',False),(655,'direct',6,'New direct-answer errors',True)]:
        p.append(text(left,162,title,22))
        y=lambda n:445-n/ceiling*235
        for frac in (0,.25,.5,.75,1):
            val=ceiling*frac
            p += [line(left,y(val),left+345,y(val)),text(left-12,y(val)+5,f'{val:g}%',15,'#526779','end')]
        if new_only:
            p += [line(left,y(2),left+345,y(2),'#8e3628',2,'6 4'),text(left+347,y(2)+5,'2% cap',15,'#8e3628')]
        for x,arm,label in [(left+20,'N','Original'),(left+133,'K','Removal'),(left+246,'SCTRL','Control')]:
            row=next(r for r in totals if r['family']==family and r['arm']==arm)
            count=row['new_errors'] if new_only else row['errors'];value=100*count/row['scheduled']
            p += [rect(x,y(value),70,445-y(value),colors[arm]),text(x+35,y(value)-29,f'{value:.2f}%',19,colors[arm],'middle'),
                  text(x+35,y(value)-9,f"{count}/{row['scheduled']:,}",15,'#526779','middle'),text(x+35,478,label,17,anchor='middle')]
    p += [text(40,533,'Joint errors: 75 to 129 of 1,152 responses, an increase of 4.69 percentage points.',20),
          text(40,568,'Message removal introduced 67 new joint errors and corrected 13 previous joint errors.',18),
          text(40,603,'New direct errors: 38 of 768 (4.95%), above the fixed 2% limit. The branch is closed.',18,'#8e3628')]
    write(out/'SCREEN_DAMAGE.svg',p)
    receipt=dict(scope='Historical resource-admission screen only',final_cases=0,screen_sha256=digest(source/'replay/SCREEN.json'),
        totals_sha256=digest(source/'SCREEN_TOTALS.json'),correction_values=values,
        requested_final_example='Not applicable: FINAL was not admitted; no repeated baseline-positive final case exists',
        requested_population_figure='Not applicable: fixed historical screen shown explicitly instead',
        figures={p.name:digest(p) for p in out.glob('*.svg')})
    with (out/'FIGURE_DATA.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
    return receipt


def render(out):
    import resvg_py
    entries=[]
    for name in ('SCREEN_CORRECTIONS','SCREEN_DAMAGE'):
        source=out/(name+'.svg');dest=out/(name+'.png')
        with dest.open('xb') as f:f.write(resvg_py.svg_to_bytes(svg_string=source.read_text(encoding='utf8')))
        entries.append(dict(name=name,svg_sha256=digest(source),png_sha256=digest(dest)))
    receipt=dict(renderer='resvg-py',version=importlib.metadata.version('resvg-py'),figures=entries,scientific_runtime_unchanged=True)
    with (out/'RENDER_RECEIPT.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--render',action='store_true');a=p.parse_args()
    print(json.dumps(render(a.out) if a.render else create(a.source,a.out)))
