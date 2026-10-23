"""Run the unchanged submitted runner against an exact external snapshot.

Git and collaborator documents belong to this review overlay. The submitted
checker still rejects extra files inside the separate software/data extraction.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'tools'))
from review_support import RELEASE, REPO, digest, release_api


def command(snapshot, arguments):
    snapshot = Path(snapshot).resolve()
    api = release_api()
    expected = digest(RELEASE / 'metadata/files.sha256')
    if digest(snapshot / 'metadata/files.sha256') != expected:
        raise ValueError('Snapshot manifest differs from the retained submitted manifest')
    # This performs the original complete, closed-inventory verification.
    api.verify(snapshot)
    if snapshot.is_relative_to(REPO) or REPO.is_relative_to(snapshot):
        raise ValueError('Use a separate submitted snapshot, outside this Git overlay')
    result = list(arguments)
    for flag in ('--output', '--historical-python', '--gmello-python'):
        if flag in result:
            index = result.index(flag) + 1
            if index >= len(result):
                raise ValueError('Missing value for ' + flag)
            value = Path(result[index]).resolve()
            if flag == '--output' and (value.is_relative_to(REPO) or REPO.is_relative_to(value)):
                raise ValueError('Output must be outside the Git overlay')
            result[index] = str(value)
    return [sys.executable, '-B', '-X', 'utf8', str(snapshot / 'reproduce.py'), *result]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Replay saved observations from a separate, exact software/data extraction.\n'
                    'Help needs no snapshot or data. Actual replay verifies every submitted input.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Replay options forwarded to the unchanged submitted runner:
  --all                 Run all five sections.
  --section SECTION     Run one section (names and scope below).
  --verify              Verify the exact submitted inventory without replaying.
  --test                Run the submitted protection tests.
  --output PATH         Write to a new directory outside both input trees.
  --historical-python PATH, --gmello-python PATH
                        Select the two additional pinned CPU interpreters.
  --hash-seed INTEGER    Set the replay process hash seed (default: 0).
  --diagnostics         Keep successful stage logs.

Sections:
  matched_histories     Synthetic and public history-dependence measurements.
  cache_interventions  Recorded cache interventions and their checks.
  external_state       Current-state repair and interpreter results.
  interpreter_checks   History exposure and trained QA/PROGRAM comparisons.
  gmello_comparison    Corrected graph-edit baseline and contrasts.

Small example (no separate archives or model weights required):
  python tools/review_examples.py --all --output "../review examples"

Full replay example:
  python reproduce.py --snapshot "../submitted/history_updates" --all --output "../full replay"

Start with README.md. Archive handoff and environment setup:
  history_updates/docs/reproduction.md
''')
    parser.add_argument('--snapshot', type=Path, required=True, metavar='PATH',
                        help='Exact combined software/data extraction; required except for help')
    args, forwarded = parser.parse_known_args(argv)
    cmd = command(args.snapshot, forwarded)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    return subprocess.run(cmd, cwd=args.snapshot.resolve(), env=env, timeout=7200,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).returncode


if __name__ == '__main__':
    raise SystemExit(main())
