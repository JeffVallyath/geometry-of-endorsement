"""Bind manuscript plotting functions to anonymous, retained data only."""
from .common import ROOT, load_json

PLOT = 'artifacts/figures/'
FRESH = 'reproducibility/fresh_history_validation/'
QBR = 'reproducibility/question_blind_repair/supplied/primary/'
DIRECT = {
    PLOT + 'figure_data.json', PLOT + 'cache_crossover.json',
    FRESH + 'expected/PRIMARY.json', FRESH + 'expected/UTILITY.json',
    QBR + 'replay/SUMMARY.json',
}


def load(relative):
    if relative in DIRECT:
        return load_json(ROOT / relative)
    selections = {PLOT + 'v10_source_history.json': 'matched_history',
                  PLOT + 'consequence_comparisons.json': 'consequences'}
    if relative not in selections:
        raise ValueError('Input is outside the six retained figures')
    return load_json(ROOT / 'figures/source/plot_data.json')[selections[relative]]
