"""Render five main-text and two appendix figures from retained measurements.

The composition functions use hash-verified public inputs and write only to a new
reviewer output folder. Display text and layout leave the measured values fixed.
It performs no model execution, fitting, bootstrap or statistical reconstruction.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .common import ROOT, load_json, new_output, safe_path, sha256

NAMES = ('matched_history_design', 'matched_history', 'construct_validity', 'consequence_qualification',
         'selected_crossover', 'fresh_confirmation_utility', 'repair_benefit_damage')


def clean_svg_metadata(path):
    """Remove renderer metadata without changing any drawing node or attribute."""
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    tree = ET.parse(path)
    for parent in tree.iter():
        for child in list(parent):
            if child.tag.rsplit('}', 1)[-1] == 'metadata':
                parent.remove(child)
    tree.write(path, encoding='utf-8', xml_declaration=True)


def verify_inputs():
    manifest = load_json(ROOT / 'figures/source/INPUTS.json')
    if manifest['schema'] != 'ANONYMOUS_MAIN_FIGURE_INPUTS_V1' or set(manifest['figures']) != set(NAMES):
        raise ValueError('Unknown figure input manifest')
    seen = set()
    for row in manifest['files']:
        name = row['path']
        if name.casefold() in seen:
            raise ValueError('Duplicate figure input')
        seen.add(name.casefold())
        path = safe_path(name, ROOT)
        if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError('Figure input changed: ' + name)
    return manifest


def run(output):
    inputs = verify_inputs()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new figure output directory')
    destination = new_output(output)
    from . import main_figure_style as style
    from . import main_figure_renderers as renderers
    import matplotlib
    import matplotlib.pyplot as plt

    style.setup()
    values = {}
    for name in NAMES:
        specification = inputs['figures'][name]
        figure, values[name] = getattr(renderers, specification['function'])()
        figure.set_size_inches(specification['width_pt'] / 72, specification['height_inches'])
        figure.savefig(destination / (name + '.pdf'), dpi=144,
                       metadata={'Creator': None, 'Producer': None, 'CreationDate': None, 'ModDate': None})
        figure.savefig(destination / (name + '.png'), dpi=144, metadata={'Software': None})
        figure.savefig(destination / (name + '.svg'), dpi=144, metadata={'Date': None, 'Creator': None})
        clean_svg_metadata(destination / (name + '.svg'))
        plt.close(figure)
    (destination / 'plot_data.json').write_text(json.dumps(values, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    result = {'status': 'PASS', 'figures': list(NAMES), 'verified_data_files': len(inputs['files']),
              'matplotlib': matplotlib.__version__, 'scope': 'Manuscript figure compositions from retained measurements',
              'pretrained_execution': 'not performed', 'new_statistical_analysis': 'not performed'}
    (destination / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))


if __name__ == '__main__':
    main()
