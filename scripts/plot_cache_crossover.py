"""Selected extension figure, adapted from the supplied reporting overlay.

Uses ordinary-language labels and the repo's existing style/save helpers.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from repro.cache_crossover_extension import ARTIFACTS, replay
from repro.common import new_output, sha256

STATES = ('AA', 'AB', 'BA', 'BB')
LABELS = ('Original history 1',
          'Earlier-layer activations from history 1\n+ later-layer activations from history 2',
          'Earlier-layer activations from history 2\n+ later-layer activations from history 1',
          'Original history 2')


def figure_data():
    checked = replay()
    with (ARTIFACTS / 'extension_joint.csv').open(newline='', encoding='utf8') as stream:
        rows = list(csv.DictReader(stream))
    return dict(rows=rows, direct_checks=checked['strict_direct_checks'],
                minimum_direct_probability=checked['minimum_direct_probability'],
                maximum_direct_change=checked['maximum_direct_change'],
                sources={n: sha256(ARTIFACTS / n) for n in ('extension_joint.csv', 'extension_direct.csv')})


def plot(module):
    import matplotlib.pyplot as plt

    data = figure_data()
    fig = plt.figure(figsize=(12.0, 6.0))
    fig.text(.025, .94, 'Changing the later stored activations changes the downstream answer', fontsize=15, weight='bold')
    fig.text(.025, .885, 'Gemma: one retrospectively selected history pair, tested with two answer formats.', fontsize=11)
    fig.text(.025, .835, 'Current facts: Ada and Gita both oppose Theater. Question: does at least one support it? Correct answer: No.', fontsize=10)
    axes = [fig.add_axes([.32, .31, .29, .41]), fig.add_axes([.67, .31, .29, .41])]
    for ax, code, title in zip(axes, ('literal_No_Yes', 'coded_A_B'), ('Literal Yes/No answers', 'Coded answers (A/B)')):
        rows = [r for r in data['rows'] if r['code'] == code]
        if [r['state'] for r in rows] != list(STATES):
            raise ValueError('Figure state inventory changed')
        values = [float(r['p_yes']) for r in rows]
        ax.barh([3, 2, 1, 0], values, height=.5,
                color=['#276882', '#806296', '#276882', '#806296'])
        for y, value in zip([3, 2, 1, 0], values):
            ax.text(value-.025 if value > .5 else value+.025, y, f'{value:.6f}',
                    ha='right' if value > .5 else 'left', va='center', fontsize=9,
                    color='white' if value > .5 else '#333333')
        ax.set_title(title, loc='left', pad=15, fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(-.55, 3.55)
        ax.set_yticks([])
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xlabel('Normalized probability of Yes', fontsize=10)
        ax.grid(axis='y', visible=False)
        ax.set_axisbelow(True)
    for y, label in zip([3, 2, 1, 0], LABELS):
        axes[0].text(-.07, y, label, transform=axes[0].get_yaxis_transform(), ha='right', va='center', fontsize=10)
    fig.text(.025, .195, f'Direct factual answers remain correct in all {data["direct_checks"]} strict checks.', fontsize=11, weight='bold')
    fig.text(.025, .145, f'Minimum correct-answer probability: {data["minimum_direct_probability"]:.6f}.  '
             f'Largest change from the corresponding original state: {data["maximum_direct_change"]:.8f}.', fontsize=10)
    fig.text(.025, .09, 'Both histories: two changed facts and 517-token prefixes. Split: first 16 layers / remaining 26 layers.', fontsize=9)
    fig.text(.025, .045, 'Activations are combined before the question; weights stay fixed. Colors identify the history supplying the later layers.', fontsize=9)
    fig.text(.025, .005, 'The formats test the same example, not independent cases or a prevalence estimate.', fontsize=9, color='#52606d')
    module.save(fig, 'fig09_cache_crossover_extension')
    (module.OUT / 'cache_crossover_extension.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh figure directory')
    args = parser.parse_args()
    import make_figures as style
    destination = new_output(args.output)
    if any(destination.iterdir()):
        raise FileExistsError('Choose a fresh figure output directory')
    style.style()
    style.OUT = destination
    plot(style)
