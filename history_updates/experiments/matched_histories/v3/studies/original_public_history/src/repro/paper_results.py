"""Run the retained paper analyses offline, or run their CPU regression tests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import ROOT, new_output
from .release_manifest import verify


def jobs(destination):
    """Only retained paper studies; each scientific analyzer remains unchanged."""
    independent = 'reproducibility/state_sufficiency/independent_verification/reconstruct.py'
    module_jobs = [
        ('relation', 'relation_readout'), ('consequences', 'counterfactual'),
        ('primary', 'state_sufficiency'),
        ('controls', 'cpu_robustness'), ('fresh', 'fresh_history'),
        ('qwen32', 'qwen32_full_forward'), ('textual-breadth', 'text_scope'),
        ('crossover', 'cache_crossover'), ('crossover-extension', 'cache_crossover_extension'),
        ('refresh', 'question_blind_repair'), ('update-methods', 'update_methods'),
        ('ripple', 'in_context_updates'), ('mechanism', 'completed_followups'),
        ('editors', 'learned_editors'),
    ]
    result = [(name, ['-m', 'repro.' + module, '--output', str(destination / name)])
              for name, module in module_jobs]
    result[3:3] = [
        ('independent-analyze', [independent, 'analyze', '--output', str(destination / 'independent')]),
        ('independent-compare', [independent, 'compare', '--output', str(destination / 'independent')]),
    ]
    return result


def offline_environment(destination):
    # Preserve only process-launch essentials, never inherit model caches or credentials.
    allowed = {'systemroot', 'windir', 'path', 'comspec', 'pathext', 'lang', 'lc_all'}
    env = {key: value for key, value in os.environ.items() if key.lower() in allowed}
    runtime = destination / '.runtime'
    runtime.mkdir()
    temporary = runtime / 'tmp'
    temporary.mkdir()
    env.update(PYTHONPATH=os.pathsep.join([str(ROOT / 'scripts/offline_site'), str(ROOT / 'src')]),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg', MPLCONFIGDIR=str(runtime / 'matplotlib'),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary))
    return env


def portable_log(value, destination):
    for path, label in ((destination, '<output>'), (ROOT, '<review-root>'), (Path(sys.prefix), '<python-env>')):
        for spelling in (str(path), path.as_posix()):
            for encoded in (json.dumps(spelling)[1:-1], json.dumps(spelling, ensure_ascii=False)[1:-1]):
                value = value.replace(encoded, label)
            value = value.replace(spelling, label)
    return value


def remove_pytest_current_links(destination):
    """Remove only pytest's sibling-directory aliases after its process exits."""
    destination = Path(destination).resolve()
    runtime = destination / '.runtime'
    base = runtime / 'pytest'
    for path in (runtime, base):
        if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
            raise ValueError('Unexpected linked pytest runtime directory')
    if not base.exists():
        return 0
    base = base.resolve(strict=True)
    if base.parent != runtime or not base.is_dir():
        raise ValueError('Pytest runtime escapes its output directory')
    aliases = []
    for path in base.iterdir():
        if not path.is_symlink():
            continue
        prefix = path.name.removesuffix('current')
        raw_target = path.parent / path.readlink()
        if (prefix == path.name or not prefix or raw_target.is_symlink()
                or getattr(raw_target, 'is_junction', lambda: False)()):
            raise ValueError('Unexpected pytest runtime alias')
        try:
            target = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError('Invalid pytest runtime alias target') from error
        suffix = target.name[len(prefix):] if target.name.startswith(prefix) else ''
        if (target.parent != base or not target.is_dir() or not suffix
                or not suffix.isascii() or not suffix.isdigit()):
            raise ValueError('Unexpected pytest runtime alias target')
        aliases.append(path)
    # Validate the whole set before removing any links. Target directories and
    # their contents are preserved; release_manifest retains its no-link guard.
    for path in aliases:
        path.unlink()
    return len(aliases)


def run(output, *, test=False):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new output directory; existing outputs are preserved')
    integrity = verify()
    destination = new_output(output)
    environment = offline_environment(destination)
    commands = ([('tests', ['-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp',
                            str(destination / '.runtime/pytest')])]
                if test else jobs(destination))
    receipts = []
    for label, arguments in commands:
        print('Running ' + label, flush=True)
        try:
            result = subprocess.run([sys.executable, '-B', *arguments], cwd=ROOT, env=environment,
                                    capture_output=True, text=True, encoding='utf-8', timeout=1200)
            text = result.stdout + result.stderr
            code = result.returncode
        except subprocess.TimeoutExpired:
            text, code = 'CPU review step exceeded its 1200-second bound.\n', 124
        (destination / (label + '.log')).write_text(portable_log(text, destination), encoding='utf-8', newline='\n')
        removed_links = remove_pytest_current_links(destination) if test and code == 0 else 0
        receipts.append({'step': label, 'status': 'PASS' if code == 0 else 'FAIL', 'exit_code': code,
                         'log': label + '.log'})
        report = {'status': 'PASS' if len(receipts) == len(commands) and not code else 'INCOMPLETE',
                  'mode': 'CPU tests' if test else 'saved-output paper analyses',
                  'input_integrity': integrity, 'steps': receipts,
                  'pretrained_inference': 'not performed', 'network': 'blocked in child Python processes'}
        if test:
            report['pytest_current_links_removed'] = removed_links
        (destination / 'CHECKS.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        if code:
            raise RuntimeError('CPU review failed at ' + label + '; see its output log')
        print(label + ': PASS', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test', action='store_true', help='Run the retained CPU tests instead of the analyses')
    args = parser.parse_args()
    result = run(args.output, test=args.test)
    print(json.dumps({'status': result['status'], 'steps': len(result['steps'])}, indent=2))


if __name__ == '__main__':
    main()
