"""w768/depth12 fixed-initial-RMS entry point, peak matrix ELR 0.48."""
from pathlib import Path
import runpy
import sys

# Prevent CLI/JSON overrides from silently changing the named treatment.
for option in ('--arm', '--model-width', '--model-depth', '--peak-elr', '--schedule-json'):
    if any(arg == option or arg.startswith(option + '=') for arg in sys.argv[1:]):
        raise SystemExit(f'{option} is fixed by this ELR entry point')
sys.argv[1:1] = [
    '--arm', 'fanin_fixed', '--model-width', '768', '--model-depth', '12',
    '--peak-elr', '0.48', '--schedule', 'cosine',
    '--expected-world-size', '2', '--device-batch-size', '64',
    '--run-name', 'fanin_fixed_w768_d12_elr0p48',
    '--log-root', 'logs/fanin_fixed_w768_d12_elr0p48',
]
runpy.run_path(str(Path(__file__).with_name('train.py')), run_name='__main__')
