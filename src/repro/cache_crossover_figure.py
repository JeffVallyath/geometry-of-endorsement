"""One selected causal example, plotted from independently replayed saved scores."""
import json

from .cache_crossover import BASE, replay
from .common import sha256

STATE_LABELS = {
    'AA': 'Original history A',
    'BB': 'Original history B',
    'AB': 'Earlier-layer activations from A\n+ later-layer activations from B',
    'BA': 'Earlier-layer activations from B\n+ later-layer activations from A',
}


def figure_data():
    result = replay()
    return dict(example=result['example'], counts=result['counts']['witness'],
                layers=result['layers'],
                sources={name: sha256(BASE / name) for name in
                         ('raw.jsonl.gz', 'case_plan.json', 'terminal_worlds.json')})


def plot(module):
    import matplotlib.pyplot as plt

    data = figure_data()
    example = data['example']
    fig = plt.figure(figsize=(10.2, 5.6))
    fig.text(.025, .94, 'Swapping later activations changes the joint answer', fontsize=15, weight='bold')
    fig.text(.025, .885, 'Gemma: the current records say Dion and Orla both support Bridge.', fontsize=10)
    ax = fig.add_axes([.32, .25, .37, .48])
    facts = fig.add_axes([.75, .25, .235, .48])
    ax.set_title('Do both support Bridge?', loc='left', fontsize=11, pad=26)
    facts.set_title('Direct facts stay correct', fontsize=11, pad=26)
    facts.text(.24, 1.04, 'Dion', transform=facts.transAxes, ha='center', fontsize=10)
    facts.text(.76, 1.04, 'Orla', transform=facts.transAxes, ha='center', fontsize=10)
    for i, row in enumerate(example['states']):
        y = 3 - i
        color = '#276882' if row['state'][-1] == 'A' else '#806296'
        value = row['joint']['p_yes']
        ax.barh(y, value, height=.48, color=color)
        ax.text(value - .025 if value > .5 else value + .025, y, f'{value:.6f}',
                ha='right' if value > .5 else 'left', va='center',
                color='white' if value > .5 else '#333333', fontsize=9)
        ax.text(-.08, y, STATE_LABELS[row['state']], transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=10)
        for x, direct, fact in zip((.24, .76), row['direct'], example['direct_facts']):
            probability = direct['p_yes'] if fact['correct_semantic_answer'] else 1 - direct['p_yes']
            facts.text(x, y, f'{probability:.6f}', ha='center', va='center', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_yticks([])
    ax.set_ylim(-.55, 3.55)
    ax.set_xlabel('Normalized probability of Yes', fontsize=10)
    ax.grid(axis='y', visible=False)
    ax.set_axisbelow(True)
    ax.axhline(1.5, color='#b9c4ce', linewidth=.7)
    facts.set_xlim(0, 1)
    facts.set_ylim(ax.get_ylim())
    facts.axhline(1.5, color='#b9c4ce', linewidth=.7)
    facts.axis('off')
    facts.text(.5, -.13, 'Probability of each correct\ndirect answer', transform=facts.transAxes,
               ha='center', va='top', fontsize=9)
    fig.text(.025, .105, 'Earlier-layer activations: first 16 layers. Later-layer activations: remaining 26 layers.', fontsize=9)
    fig.text(.025, .055, 'Selected example, main answer format. Weights stay fixed; activations are combined before the question.', fontsize=9)
    fig.text(.025, .015, 'This example illustrates the intervention, not how often the effect occurs.', fontsize=9, color='#52606d')
    module.save(fig, 'fig08_cache_crossover')
    (module.OUT / 'cache_crossover.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf8', newline='\n')
