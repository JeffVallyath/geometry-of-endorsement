"""Reader-facing terminology guards, separate from frozen scientific records."""
import re
from pathlib import Path

from repro.common import ROOT
from repro.evidence import number_tokens


def test_main_headings_use_scientific_questions_not_version_codes():
    for name in ('README.md','PAPER_GUIDE.md','PROJECT_STRATEGY.md','docs/RESULTS_AND_CLAIMS.md','figures/CAPTIONS.md'):
        headings=re.findall(r'^#{1,3}\s+(.+)$',(ROOT/name).read_text(encoding='utf8'),re.M)
        assert all(not re.search(r'\bV\d+\b|ICMH|LP1|INV_PAIR|FREE_PAIR',h) for h in headings),name


def test_navigation_anchor_numbers_are_not_claims_but_visible_numbers_are():
    assert number_tokens('<a id="figure-7-old-title"></a> We measured 18 questions.')==['18']
    assert number_tokens('probability < 0.25 or > 0.95')==['0.25','0.95']


def test_reader_glossary_does_not_replace_local_definitions():
    text=(ROOT/'docs/RESULTS_AND_CLAIMS.md').read_text(encoding='utf8')
    assert '**benchmark case**' in text and '**individual-fact question**' in text
    assert '**current-facts-from-the-start reference**' in text
    assert '**already-correct update reference**' in text
    assert 'source-history sufficiency' in (ROOT/'README.md').read_text(encoding='utf8').lower()


def test_archived_figure_is_separate_from_reader_friendly_rendition():
    text=(ROOT/'figures/CAPTIONS.md').read_text(encoding='utf8')
    assert 'figS3_source_history_readable.png' in text
    assert 'figS3_v6_source_history.png' in text and 'figS3_v6_source_history.pdf' in text
    assert 'remain byte-for-byte' in text


def test_public_data_display_labels_explain_the_units():
    text=(ROOT/'reproducibility/in_context_updates/public_data_table.md').read_text(encoding='utf8')
    assert 'Qualifying history-dependent questions' in text
    assert 'Questions passing factual and reference checks' in text
    assert 'Benchmark cases with a qualifying question' in text
