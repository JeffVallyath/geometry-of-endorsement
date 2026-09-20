"""Bounded checks on current reader-facing links, excluding frozen source docs."""
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import pytest

from repro.common import ROOT

GUIDES = [
    'reproducibility/text_scope/README.md',
    'reproducibility/completed_followups/README.md',
    'reproducibility/qwen32_full_forward/README.md',
    'reproducibility/qwen32_full_forward/WORKED_EXAMPLE.md',
    'docs/PRIVACY_BOUNDARY.md', 'docs/TEAM_READINESS_CHECKS.md',
    'docs/history/READABILITY_INTEGRATION_20260919.md',
    'README.md', 'PAPER_GUIDE.md', 'PROJECT_STRATEGY.md', 'REPRODUCIBILITY.md',
    'DATA_NOTICE.md', 'docs/RESULTS_AND_CLAIMS.md', 'figures/CAPTIONS.md',
    'figures/CAPTION_DETAILS.md',
    'docs/TERMINOLOGY.md',
    'reproducibility/state_sufficiency/README.md',
    'reproducibility/cache_crossover/README.md',
    'reproducibility/in_context_updates/README.md',
    'reproducibility/update_method_comparison/README.md',
    'reproducibility/fresh_inference/README.md',
    'reproducibility/fresh_history_validation/README.md',
    'docs/FRESH_HISTORY_REPRODUCTION.md',
    'reproducibility/preupdate_route/README.md',
    'reproducibility/question_blind_repair/README.md',
    'reproducibility/question_blind_repair/WORKED_EXAMPLE.md',
    'reproducibility/question_blind_repair/INTEGRATION_REPORT.md',
    'reproducibility/audits/preupdate_evidence_audit_20260919/README.md',
    'reproducibility/fresh_history_validation/PORTABILITY.md',
    'reproducibility/fresh_history_validation/examples/CASE_0017.md',
    'reproducibility/state_sufficiency/independent_verification/README.md',
    'reproducibility/memit_qualification/README.md',
    'docs/CACHE_CROSSOVER_APPENDIX.md',
]


def anchors(path):
    found, counts = set(), {}
    found.update(re.findall(r'<a\s+id="([^"]+)"\s*>', path.read_text(encoding='utf8')))
    for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', path.read_text(encoding='utf8'), re.M):
        slug = re.sub(r'[^\w\-\s]', '', heading.lower()).replace(' ', '-')
        count = counts.get(slug, 0)
        found.add(slug + (f'-{count}' if count else ''))
        counts[slug] = count + 1
    return found


@pytest.mark.parametrize('name', GUIDES)
def test_current_local_links_and_anchors(name):
    path = ROOT / name
    text = re.sub(r'```.*?```', '', path.read_text(encoding='utf8'), flags=re.S)
    for target in re.findall(r'\]\(([^)]+)\)', text):
        link = urlsplit(target.strip('<>'))
        if link.scheme:
            continue
        destination = (path.parent / unquote(link.path)).resolve() if link.path else path
        assert destination.is_relative_to(ROOT.resolve()) and destination.exists(), (name, target)
        if link.fragment and destination.suffix == '.md':
            assert unquote(link.fragment) in anchors(destination), (name, target)


@pytest.mark.parametrize('extension', ('png', 'pdf'))
def test_main_figure_filenames_sort_in_paper_order(extension):
    names = sorted(p.name for p in (ROOT / 'figures').glob(f'fig*.{extension}'))
    assert [int(re.match(r'fig(\d+)_', name).group(1)) for name in names] == list(range(1, 11))


def test_main_captions_follow_figure_order():
    text = (ROOT / 'figures/CAPTIONS.md').read_text(encoding='utf8')
    numbers = re.findall(r'^## Figure (\d+)[.: -]', text, re.M)
    assert list(map(int, numbers)) == list(range(1, 11))
