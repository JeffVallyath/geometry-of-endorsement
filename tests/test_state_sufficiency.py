"""Primary-study regression checks retained from the supplied scientific suite."""
import copy
import pytest
from repro import state_sufficiency as ss

@pytest.fixture(scope='module')
def summaries():
    return ss.replay()


def test_eight_primary_cells_and_all_condition_recurrence(summaries):
    assert len(summaries['v10']) == len(summaries['strong']) == 8
    assert all(r['lower'] > 0 for r in summaries['v10'])
    assert [r['roots'] for r in summaries['v10']] == [50,41,30,25,24,25,38,40]
    assert [r['count'] for r in summaries['strong']] == [49,24,20,22,32,7,36,33]


@pytest.mark.parametrize('mutation', ['missing_seed','duplicate','denominator','root_identity','witness_subset','bootstrap','example'])
def test_compact_reconstruction_rejects_corruption(monkeypatch, mutation):
    original=ss.load_json
    def altered(path):
        data=copy.deepcopy(original(path))
        if path.name=='root_counts.json':
            if mutation=='missing_seed': data=[r for r in data if r['condition']!='INV_PAIR_NLL_s1']
            elif mutation=='duplicate': data[-1]=copy.deepcopy(data[0])
            elif mutation=='denominator': data[0]['questions']=data[0]['AC']
            elif mutation=='root_identity': data[0]['root_id']='unobserved-root'
            elif mutation=='witness_subset': data[0]['W']=19
        if path.name=='specification_summary.json' and mutation=='bootstrap': data['bootstrap_seed']+=1
        if path.name=='selected_repeat_checked_example.json' and mutation=='example': data['half_probability_range']=0.1
        return data
    monkeypatch.setattr(ss,'load_json',altered)
    with pytest.raises(ValueError): ss.reconstruct()


def test_export_refuses_existing_output(tmp_path):
    ss.export(tmp_path)
    with pytest.raises(FileExistsError): ss.export(tmp_path)


