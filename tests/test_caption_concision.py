"""Captions stay brief and preserve the interpretation of each scientific comparison."""
import importlib.util
import re

import pytest

from repro.common import ROOT
from repro.evidence import check, sections


def figure_captions():
    text = (ROOT / 'figures/CAPTIONS.md').read_text(encoding='utf8')
    return [(heading, body) for heading, body in sections(text)
            if re.match(r'^(?:Supplementary )?Figure S?\d+\b', heading)]


def test_captions_are_short_and_link_to_matching_appendix():
    appendix = (ROOT / 'figures/CAPTION_DETAILS.md').read_text(encoding='utf8')
    assert '\u2014' not in (ROOT / 'figures/CAPTIONS.md').read_text(encoding='utf8')
    appendix_headings = {heading for heading, _ in sections(appendix)}
    captions = figure_captions()
    assert len(captions) == 13
    for heading, body in captions:
        prose = re.sub(r'<a\s+id="[^"]+"\s*>\s*</a>', '', body)
        prose = re.sub(r'!\[[^\n]*\]\([^\n]*\)', '', prose)
        prose = re.sub(r'^\[[^\n]*$', '', prose, flags=re.M).strip()
        assert len(prose.split()) <= 160, (heading, len(prose.split()))
        assert len(prose.split('\n\n')) == 2, heading
        assert heading in appendix_headings
        slug = re.sub(r'[^\w\-\s]', '', heading.lower()).replace(' ', '-')
        assert f'(CAPTION_DETAILS.md#{slug})' in body
        assert not re.search(r'INV_PAIR_NLL|FREE_PAIR_CONSISTENCY|2609141002|'
                             r'python -m repro|\(k\+1\)/\(B\+1\)|regularization', body)


@pytest.mark.parametrize('unit,old,new', [
    ('figure1', 'layers selected on\nseparate data', 'layers selected on evaluation data'),
    ('figure2', 'normal intervals', 'bootstrap intervals'),
    ('figure3', 'coarser p-values', 'more precise p-values'),
    ('figure4', 'point estimate without an interval', 'point estimate with an interval'),
    ('figure5', 'arbitrary per-item scores', 'random activation directions'),
    ('figure6', "did not satisfy the study's stricter qualification requirements",
     "satisfied the study's stricter qualification requirements"),
    ('figure7', 'Questions failing the checks stay in the denominator',
     'Questions failing the checks are removed from the denominator'),
    ('figure7', 'multiplicity-adjusted', 'unadjusted'),
    ('figure7', 'intervals lie above zero', 'intervals cross zero'),
    ('figure7', 'Rates count qualifying cross-history disagreements over all scheduled questions',
     'Rates count individual answer errors'),
    ('figure7', 'every history reports the required facts', 'one history reports the required facts'),
    ('figure7', 'differing valid answers', 'differing answers'),
    ('figure8', 'selected Gemma example', 'representative Gemma example'),
    ('figure9', 'same selected example', 'independent examples'),
    ('figure10', 'touch or cross zero', 'exclude zero'),
    ('figure10', 'often used its fallback', 'never used its fallback'),
    ('supplementary_s1', 'averages success after two and three updates', 'requires success on both update lengths'),
    ('supplementary_s2', 'leave the direction of the effect unresolved', 'establish equivalent accuracy'),
    ('supplementary_s3', 'Right-hand counts use only cases passing every direct-fact check',
     'Right-hand counts use all cases'),
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
