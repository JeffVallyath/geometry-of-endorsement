"""Regressions for the recovered composition wrapper, without rerendering figures."""
import xml.etree.ElementTree as ET

import pytest

from repro import main_figures
from repro.main_figure_data import load


def test_seven_compositions_have_bound_public_inputs():
    manifest = main_figures.verify_inputs()
    assert set(manifest['figures']) == set(main_figures.NAMES)
    assert len(manifest['files']) == 6
    assert all(row['width_pt'] == 396 for row in manifest['figures'].values())


def test_unretained_input_cannot_be_loaded():
    with pytest.raises(ValueError, match='outside the six retained'):
        load('unretained.json')


def test_render_wrapper_cannot_overwrite_delivered_figures():
    with pytest.raises(ValueError, match='new figure output'):
        main_figures.run(main_figures.ROOT / 'figures/main')


def test_svg_metadata_cleanup_preserves_scientific_drawing(tmp_path):
    path = tmp_path / 'figure.svg'
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
                    '<metadata><description>renderer metadata</description></metadata>'
                    '<g id="data"><path d="M 1 2 L 3 4"/><text x="2">0.9136</text></g></svg>',
                    encoding='utf-8')
    before = ET.parse(path).getroot()
    drawing = next(n for n in before if n.tag.endswith('}g'))
    expected = [(node.tag, dict(node.attrib), node.text, node.tail) for node in drawing.iter()]
    main_figures.clean_svg_metadata(path)
    after = ET.parse(path).getroot()
    assert not any(n.tag.endswith('}metadata') for n in after.iter())
    assert after.attrib == before.attrib
    actual = next(n for n in after if n.tag.endswith('}g'))
    assert [(node.tag, dict(node.attrib), node.text, node.tail) for node in actual.iter()] == expected
