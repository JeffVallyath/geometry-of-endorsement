"""Main-paper captions stay brief without delegating essential caveats to the appendix."""
import importlib.util
import re

import pytest

from repro.common import ROOT
from repro.evidence import check, sections


def main_captions():
    text = (ROOT / 'figures/CAPTIONS.md').read_text(encoding='utf8')
    return [(heading, body) for heading, body in sections(text)
            if re.match(r'^Figure \d+\b', heading)]


def test_main_captions_are_short_and_link_to_matching_appendix():
    appendix = (ROOT / 'figures/CAPTION_DETAILS.md').read_text(encoding='utf8')
    appendix_headings = {heading for heading, _ in sections(appendix)}
    captions = main_captions()
    assert len(captions) == 10
    for heading, body in captions:
        prose = re.sub(r'<a\s+id="[^"]+"\s*>\s*</a>', '', body)
        prose = re.sub(r'!\[[^\n]*\]\([^\n]*\)', '', prose)
        prose = re.sub(r'\[Methods and evidence\]\([^)]*\)\.', '', prose).strip()
        assert len(prose.split()) <= 160, (heading, len(prose.split()))
        assert len(prose.split('\n\n')) == 2, heading
        assert heading in appendix_headings
        slug = re.sub(r'[^\w\-\s]', '', heading.lower()).replace(' ', '-')
        assert f'(CAPTION_DETAILS.md#{slug})' in body
        assert not re.search(r'INV_PAIR_NLL|FREE_PAIR_CONSISTENCY|2609141002|'
                             r'python -m repro|\(k\+1\)/\(B\+1\)|regularization', body)


@pytest.mark.parametrize('unit,old,new', [
    ('figure1', 'selected on separate data', 'selected on evaluation data'),
    ('figure2', 'normal intervals', 'bootstrap intervals'),
    ('figure3', 'p-value resolution differs', 'p-value resolution is identical'),
    ('figure3', 'a different analysis', 'the same analysis'),
    ('figure4', 'open marker has no retained interval', 'open marker has a retained interval'),
    ('figure5', 'not\nrandom activation directions', 'random activation directions'),
    ('figure6', 'not perfect hard-answer accuracy', 'perfect hard-answer accuracy'),
    ('figure7', 'questions failing the witness criteria remain in the denominator',
     'questions failing the witness criteria are excluded from the denominator'),
    ('figure7', 'Intervals are\nmultiplicity-adjusted', 'Intervals are unadjusted'),
    ('figure7', 'All eight multiplicity-adjusted intervals lie above', 'Some adjusted intervals cross'),
    ('figure7', 'not case prevalence', 'case prevalence'),
    ('figure7', 'correct direct facts in every history', 'correct direct facts in one history'),
    ('figure8', 'not a prevalence estimate', 'a prevalence estimate'),
    ('figure9', 'not independent cases or prevalence', 'independent cases and prevalence'),
    ('figure10', 'touches zero', 'excludes zero'),
    ('figure10', 'used fallback', 'did not use fallback'),
])
def test_short_caption_caveats_cannot_be_masked_by_valid_appendix(unit, old, new):
    text = (ROOT / 'figures/CAPTIONS.md').read_text(encoding='utf8')
    assert old in text
    mutated = text.replace(old, new, 1)
    report = check(document='captions', overrides={'captions': mutated})
    assert any(row['unit'] == unit and row['rule'] == 'material-qualification-missing'
               for row in report.errors), report.errors


def test_figure_package_includes_appendix_without_regenerating_figures():
    spec = importlib.util.spec_from_file_location('figure_packager', ROOT / 'scripts/package_figures.py')
    packager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packager)
    assert 'figures/CAPTIONS.md' in packager.MEMBERS
    assert 'figures/CAPTION_DETAILS.md' in packager.MEMBERS
