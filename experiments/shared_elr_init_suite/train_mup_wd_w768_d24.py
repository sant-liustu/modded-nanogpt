"""Double-depth variant of the w768 muP, free-norm WD=0.1 experiment."""
from pathlib import Path
import runpy
import sys

# Keep the experimental treatment fixed; other trainer options remain available.
for option in ('--arm', '--model-width', '--model-depth'):
    if any(arg == option or arg.startswith(option + '=') for arg in sys.argv[1:]):
        raise SystemExit(f'{option} is fixed by this depth-24 entry point')
sys.argv[1:1] = [
    '--arm', 'mup_wd', '--model-width', '768', '--model-depth', '24',
    '--expected-world-size', '2', '--device-batch-size', '64',
    '--run-name', 'mup_wd_w768_d24',
    '--log-root', 'logs/mup_wd_w768_d24',
]
runpy.run_path(str(Path(__file__).with_name('train.py')), run_name='__main__')
