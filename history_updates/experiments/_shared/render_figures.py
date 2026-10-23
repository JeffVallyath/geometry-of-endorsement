"""Bind retained scientific renderers to recomputed paper-table data.

These are available source renderings, not editable replicas of the final PDF.
The frozen plotting functions and their numerical values are left unchanged.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import types


def render(legacy,tables,destination):
    tables,destination=Path(tables),Path(destination)
    destination.mkdir(exist_ok=False)
    package=types.ModuleType('_paper_source_figures')
    package.__path__=[str(Path(legacy)/'src/repro')]
    sys.modules[package.__name__]=package
    style=importlib.import_module(package.__name__+'.main_figure_style')
    original=importlib.import_module(package.__name__+'.main_figure_renderers')
    cleaning=importlib.import_module(package.__name__+'.main_figures')
    import matplotlib.pyplot as plt
    style.setup()
    claims=json.loads((tables/'claims.json').read_bytes())
    inputs={};jobs=[]
    if 'synthetic_primary' in claims:
        inputs[original.PLOT+'v10_source_history.json']={'rows':[
            {'actor':{'Gemma-2-9B':'gemma','Qwen3-8B':'qwen'}[r['Model']],
             'update_type':'Learned' if r['Source_condition'] in {'INV_PAIR_NLL','FREE_PAIR_CONSISTENCY'} else 'Textual',
             'mean':r['Rate_percent']/100,
             'corrected_interval':[r['Lower_percent']/100,r['Upper_percent']/100]}
            for r in claims['synthetic_primary']]}
        jobs.append(('figure_02_source','history','table_03.json'))
    if 'cache_crossover' in claims:
        inputs[original.PLOT+'cache_crossover.json']=claims['cache_crossover']
        jobs.append(('figure_03_source','crossover','claims.json#/cache_crossover'))
    if 'cache_refresh' in claims:
        inputs[original.QBR+'replay/SUMMARY.json']={'family':claims['cache_refresh']['result']['primary']['uncertainty_family']}
        jobs.append(('figure_08_source','repair','claims.json#/cache_refresh/result/primary/uncertainty_family'))
    original.load=lambda name:inputs[name]
    outputs={}
    for name,function,source in jobs:
        figure,values=getattr(original,function)()
        figure.savefig(destination/(name+'.svg'),dpi=144,metadata={'Date':None,'Creator':None})
        cleaning.clean_svg_metadata(destination/(name+'.svg'))
        figure.savefig(destination/(name+'.png'),dpi=144,metadata={'Software':None})
        plt.close(figure)
        outputs[name]={'data':source,'renderer':'main_figure_renderers.'+function,'values':values}
    result={'status':'PASS','scope':'Available scientific source renderings; approved final layouts remain in the supplied PDF','renderings':outputs}
    (destination/'plot_data.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy',type=Path,required=True)
    parser.add_argument('--tables',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=render(args.legacy,args.tables,args.output)
    print(json.dumps({'status':result['status'],'renderings':list(result['renderings'])}))
