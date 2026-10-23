"""Physical-size layout. All sizes are final PDF points; never bbox-tight resize."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = '#20272c'
MUTED = '#515b63'
GRID = '#d9dee2'
HISTORY = {'A': '#246b92', 'B': '#bb7420'}
SCORER = {'difference_in_means': '#246b92', 'logistic': '#875681',
          'native_answer_margin': '#3e484f', 'sbert_interaction': '#74685b',
          'held_out_template': '#246b92', 'deterministic_random': '#74685b'}
METHOD = {'learned': '#246b92', 'textual': '#74685b'}
INTERVENTION = {'targeted': '#246b92', 'comparison': '#875681',
                'baseline': '#66727a', 'full': '#377566', 'removal': '#bb7420'}

def setup():
    font_manager.findfont('DejaVu Sans',fallback_to_default=False)
    matplotlib.rcParams.update({
        'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
        'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK,
        'xtick.color': INK, 'ytick.color': INK, 'axes.linewidth': .6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 1.3, 'lines.markersize': 4,
        'grid.color': GRID, 'grid.linewidth': .5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'geometry-figure-system-v1',
        'savefig.bbox': None, 'savefig.facecolor': 'white',
        'figure.facecolor': 'white', 'figure.dpi': 144,
    })

def figure(height, title='', subtitle='', width=5.5):
    return plt.figure(figsize=(width, height))

def panel(ax, letter, title):
    ax.set_title(f'{letter}  {title}', loc='left', pad=7, fontweight='bold')

def foot(fig, text):
    # Captions and methods carry explanatory prose in the manuscript.
    return None

def grid(ax, axis='x', zero=True):
    ax.set_axisbelow(True)
    ax.grid(axis=axis)
    if zero:
        (ax.axvline if axis == 'x' else ax.axhline)(0, lw=.65, color=INK)

def ci(ax, value, low, high, y, color=INK, marker='o', open_marker=False):
    ax.errorbar(value, y, xerr=[[value-low], [high-value]], fmt=marker,
                color=color, markerfacecolor='white' if open_marker else color,
                markersize=4, capsize=2.5, elinewidth=1.1, markeredgewidth=.9)

def box(ax, x, y, w, h, text, color=INK, face='white', size=8.5):
    patch = FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=.016',
                           facecolor=face,edgecolor=color,linewidth=.8)
    ax.add_patch(patch)
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=INK,linespacing=1.4)

def arrow(ax, a, b, color=INK, style='-'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,
                                linewidth=.9,color=color,linestyle=style))
