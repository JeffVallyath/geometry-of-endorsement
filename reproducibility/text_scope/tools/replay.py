"""Path-only adapter: execute the original saved-score analyzer unchanged."""
from pathlib import Path
import runpy
import sys

source, output = map(Path, sys.argv[1:])
sys.path.insert(0, str(source / 'code'))
sys.argv = ['analyze', '--runtime', str(source / 'runtime'), '--output', str(output)]
runpy.run_module('text_scope.analyze', run_name='__main__')
