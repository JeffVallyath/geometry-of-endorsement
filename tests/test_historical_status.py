import json

from repro.common import ROOT
from geometry_of_truth.project.results import development_intervals


def test_historical_null_is_labeled_without_replacing_original_values():
    status = json.loads((ROOT / 'artifacts/project/status.json').read_text(encoding='utf8'))
    result = status['moral_relation_development']
    context = result['permutation_context']
    assert context['status'] == 'historical_200_draw_summary'
    assert context['draws'] == 200
    assert (ROOT / context['later_refit_null']).is_file()
    for probe in ('difference_in_means', 'logistic'):
        assert result[probe]['original_200_draw_permutation_p'] == 1 / 201
    assert any('Historical' in column and '200' in column for column in development_intervals(status).columns)
