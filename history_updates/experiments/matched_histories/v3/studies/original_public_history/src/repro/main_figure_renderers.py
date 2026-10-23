"""Compact publication figures drawn from retained, hash-verified measurements."""
import numpy as np
from matplotlib.patches import Rectangle
from .main_figure_data import load, PLOT, FRESH, QBR
from . import main_figure_style as S

def design():
    fig=S.figure(1.05,'',width=5.5); ax=fig.add_axes([.01,.04,.98,.92]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    S.box(ax,.02,.56,.14,.32,'History H₁',S.HISTORY['A'],size=7.8);S.box(ax,.02,.12,.14,.32,'History H₂',S.HISTORY['B'],size=7.8)
    xs=[.215,.405,.625];ws=[.15,.18,.19];labs=['Same update\ncommands','Same intended\ncurrent record','Same downstream\nquestion']
    for x,w,l in zip(xs,ws,labs): S.box(ax,x,.30,w,.40,l,S.INK,size=7.8)
    S.arrow(ax,(.168,.72),(.207,.56));S.arrow(ax,(.168,.28),(.207,.44))
    S.arrow(ax,(.371,.50),(.397,.50));S.arrow(ax,(.591,.50),(.617,.50))
    S.box(ax,.86,.56,.12,.32,'Answer\u2009A₁',S.HISTORY['A'],size=7.8);S.box(ax,.86,.12,.12,.32,'Answer\u2009A₂',S.HISTORY['B'],size=7.8)
    S.arrow(ax,(.823,.56),(.852,.72));S.arrow(ax,(.823,.44),(.852,.28))
    return fig,{'schematic':'two starting histories -> same commands -> same intended current record -> same question -> answers compared across histories'}

def history():
    d=load(PLOT+'v10_source_history.json');fig=S.figure(2.25,'',width=5.5)
    for col,actor in enumerate(['gemma','qwen']):
        ax=fig.add_axes([.39+col*.30,.18,.27,.73]);rr=[r for r in d['rows'] if r['actor']==actor]
        for i,r in enumerate(rr):
            learned=r['update_type']=='Learned';S.ci(ax,100*r['mean'],*[100*v for v in r['corrected_interval']],3-i,S.METHOD['learned' if learned else 'textual'],'o' if learned else 's')
        ax.set(ylim=(-.5,3.5),xlim=(0,10),xticks=[0,5,10],yticks=[3,2,1,0],xlabel='Witness rate (%)')
        ax.set_yticklabels(['Constrained learned','Consistency-trained','Text correction','Latest-value'] if col==0 else [])
        S.panel(ax,chr(65+col),actor.title());S.grid(ax)
    return fig,d

def construct():
    d=load(PLOT+'figure_data.json');keys=['difference_in_means','logistic','sbert_interaction','deterministic_random','situation_only_activation','consideration_only_activation','separate_encoding_additive']
    labels=['Relation direction','Logistic probe','Text-only MiniLM','Random scores','Situation only','Consideration only','Separate additive'];fig=S.figure(3.25,'',width=5.5)
    for col,name in enumerate(['llama','gemma']):
        m=d['models'][name];ax=fig.add_axes([.38+col*.31,.12,.28,.82])
        for i,k in enumerate(keys):
            v=m['evaluations'][k];color=S.SCORER.get(k,S.INK);marker={'difference_in_means':'o','logistic':'s','sbert_interaction':'D','deterministic_random':'x'}.get(k,'+')
            if v['ci_low'] is not None:S.ci(ax,v['I_b'],v['ci_low'],v['ci_high'],6-i,color,marker)
            else:ax.plot(v['I_b'],6-i,marker=marker,mfc='white',mec=color,ls='none')
        ax.set(xlim=(-.65,3),xticks=[0,1,2,3],ylim=(-.5,6.5),yticks=np.arange(7)[::-1],xlabel='Checkerboard interaction');ax.set_yticklabels(labels if col==0 else [])
        S.panel(ax,chr(65+col),f'{name.title()} · layer {m["selected_layer"]}');S.grid(ax)
    return fig,{'models':{k:{'evaluations':v['evaluations'],'selected_layer':v['selected_layer'],'selection_criterion':v['selection_criterion']} for k,v in d['models'].items()}}

def consequences():
    d=load(PLOT+'consequence_comparisons.json')['counterfactual'];fig=S.figure(2.10,'',width=5.5);ax=fig.add_axes([.42,.17,.55,.72])
    for i,r in enumerate(d['arms']):S.ci(ax,r['recovery'],r['lower'],r['upper'],2-i,[S.INTERVENTION['targeted'],S.INTERVENTION['comparison'],S.INTERVENTION['full']][i],['o','s','D'][i])
    ax.axvline(1,color=S.MUTED,ls='--',lw=.8);ax.set(yticks=[2,1,0],yticklabels=['Relation steering','Whole-state interpolation','Full replacement'],xlim=(-1.2,1.12),ylim=(-.5,2.5),xticks=[-1,0,1],xlabel='Recovery');S.grid(ax)
    return fig,d

def crossover():
    d=load(PLOT+'cache_crossover.json');states=d['example']['states'];fig=S.figure(2.55,'',width=5.5);ax=fig.add_axes([.04,.16,.40,.72]);ax.set(xlim=(0,1),ylim=(-.5,3.5));ax.axis('off')
    ax.text(.27,3.45,'Early layers',ha='center',fontsize=8);ax.text(.73,3.45,'Later layers',ha='center',fontsize=8)
    for i,r in enumerate(states):
        for j,donor in enumerate(r['state']):
            x=.06+j*.47;y=3-i;ax.add_patch(Rectangle((x,y-.24),.40,.48,facecolor='white',edgecolor=S.HISTORY[donor],lw=1.1,hatch='///' if donor=='B' else None));ax.text(x+.20,y,'H1' if donor=='A' else 'H2',ha='center',va='center',fontsize=8.5,bbox=dict(facecolor='white',edgecolor='none',pad=2.6))
    S.panel(ax,'A','Activation sources');ax=fig.add_axes([.54,.16,.40,.72])
    for i,r in enumerate(states):
        value=r['joint']['p_yes'];donor=r['state'][1];ax.barh(3-i,value,height=.48,color=S.HISTORY[donor],hatch='///' if donor=='B' else None);ax.text(value+.025,3-i,f'{value:.2f}',ha='left',va='center',fontsize=8)
    ax.set(xlim=(0,1.12),xticks=[0,.5,1],ylim=(-.5,3.5),yticks=[],xlabel='P(Yes)');S.panel(ax,'B','Downstream answer');S.grid(ax)
    return fig,d

def fresh():
    p=load(FRESH+'expected/PRIMARY.json');u=load(FRESH+'expected/UTILITY.json');fig=S.figure(3.1,'',width=5.5);ax=fig.add_axes([.13,.70,.34,.18]);yy=[2,1,0]
    for y,r,m in zip(yy[:2],p['per_seed'],['v','^']):ax.plot(100*r['matched']/r['denominator'],y,m,color=S.MUTED)
    S.ci(ax,100*p['rate'],*[100*v for v in p['interval95']],0,S.HISTORY['A']);ax.set(yticks=yy,yticklabels=['Seed 0','Seed 1','Mean'],xlim=(0,1.7),ylim=(-.6,2.6),xticks=[0,.5,1,1.5],xlabel='Witness rate (%)');S.panel(ax,'A','Fresh confirmation');S.grid(ax)
    for col,seed in enumerate([0,1]):
        ax=fig.add_axes([.15+col*.45,.11,.34,.22]);row=next(r for r in u['per_seed_format'] if r['seed']==seed and r['draw']==0)
        for method,color,marker,ls in [('A',S.HISTORY['A'],'o','-'),('B',S.HISTORY['B'],'s','--')]:
            vals=[100*row['policies'][method]['equal_retention'][str(f)]['task_error'] for f in [.25,.5,.75]];ax.plot([25,50,75],vals,color=color,marker=marker,ls=ls,label={'A':'One history','B':'Two histories'}[method])
        ax.set(xlim=(20,80),xticks=[25,50,75],ylim=(0,10),yticks=[0,5,10],xlabel='Cases retained (%)');ax.set_ylabel('Error (%)' if col==0 else '');S.panel(ax,chr(66+col),f'Utility · seed {seed}');S.grid(ax,'y')
        if col==0:ax.legend(loc='upper left')
    return fig,{'primary':p,'utility':u}

def repair():
    d=load(QBR+'replay/SUMMARY.json');fig=S.figure(2.9,'',width=5.5);panels=[('efficacy','Repair gain vs. later layers',(0,.65)),('net_joint_error','Net joint-error change',(-1.15,.1)),('direct_harm','New direct errors',(0,.65)),('joint_harm','New joint errors',(0,.65))]
    for i,(key,title,lim) in enumerate(panels):
        ax=fig.add_axes([.13+(i%2)*.48,.66-(i//2)*.42,.34,.18]);v=d['family'][key];S.ci(ax,100*v['estimate'],*[100*x for x in v['interval9875']],0,S.INTERVENTION['targeted']);ax.set(xlim=lim,ylim=(-.5,.5),yticks=[],xlabel='Percentage points');S.panel(ax,chr(65+i),title);S.grid(ax)
    return fig,d
