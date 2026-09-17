"""One supporting comparison figure, including mixed results and absolute rates."""
from .update_methods import reconstructed, BASELINE, RECIPES


def plot(module):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    result, _ = reconstructed()
    fig, (ax, absolute) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={'width_ratios': [1.6, 1]})
    labels = []
    for i, row in enumerate(result['contrasts']):
        y = 3-i
        color = '#2166ac' if row['actor'] == 'gemma' else '#b2182b'
        effect, low, high = [100*row[k] for k in ('effect', 'lower', 'upper')]
        ax.errorbar(effect, y, xerr=[[effect-low], [high-effect]], fmt='o', color=color, capsize=3)
        for seed, marker, dy in [('0', 'v', -.12), ('1', '^', .12)]:
            ax.scatter(100*row['per_seed'][seed]['effect'], y+dy, marker=marker, s=17, color='#777777')
        method = 'Constrained' if row['recipe'] == RECIPES[0] else 'Consistency-trained'
        labels.append(row['actor'].title()+' / '+method)
    ax.set_yticks([3, 2, 1, 0], labels)
    ax.axvline(0, color='#555555', lw=.8)
    ax.set_xlim(-10, 55)
    ax.set_ylim(-.5, 3.5)
    ax.set_xlabel('Change in complete-case success (percentage points)')
    ax.set_title('A  Change vs. factual-update baseline', loc='left', fontsize=10)
    ax.grid(axis='y', visible=False)
    rates = {(r['actor'], r['condition']): r['rate'] for r in result['absolute_rates']}
    for i, (actor, color) in enumerate([('gemma', '#2166ac'), ('qwen', '#b2182b')]):
        values = [rates[actor, BASELINE]] + [sum(rates[actor, recipe+f'_s{s}'] for s in (0, 1))/2 for recipe in RECIPES]
        absolute.bar([x+(i-.5)*.32 for x in range(3)], [100*v for v in values], width=.3, color=color, label=actor.title())
    absolute.set_xticks(range(3), ['Baseline', 'Constrained', 'Consistency-\ntrained'], rotation=15)
    absolute.set_ylim(0, 100)
    absolute.set_ylabel('Complete-case success (%)')
    absolute.set_title('B  Absolute reliability', loc='left', fontsize=10)
    absolute.legend(frameon=False)
    legends = [Line2D([], [], marker='o', color='#333333', linestyle='none', label='Seed mean, 98.75% interval'),
               Line2D([], [], marker='v', color='#777777', linestyle='none', label='Seed 0'),
               Line2D([], [], marker='^', color='#777777', linestyle='none', label='Seed 1')]
    fig.legend(handles=legends, loc='lower center', bbox_to_anchor=(.5, .04), ncol=3, frameon=False, fontsize=8)
    fig.text(.5, .012, '64 benchmark cases per model; success requires all 34 questions correct from all four histories. Models are not pooled.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .15, 1, 1))
    module.save(fig, 'fig10_update_methods')
