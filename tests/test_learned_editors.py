import pytest
from repro import learned_editors


def test_original_editor_cpu_checks(tmp_path):
    result = learned_editors.check(tmp_path / 'editor-checks')
    assert result['tests_run'] == result['passed'] == 18
    assert not result['skipped'] and not result['pretrained_models_loaded'] and not result['cuda_initialized']


def test_editor_checks_refuse_an_existing_output(tmp_path):
    with pytest.raises(ValueError, match='fresh output'):
        learned_editors.check(tmp_path)
