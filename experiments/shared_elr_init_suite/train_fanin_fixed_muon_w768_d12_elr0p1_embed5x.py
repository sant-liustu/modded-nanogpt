"""w768/depth12 fixed-initial-RMS entry point, embedding ELR 5x, peak matrix ELR 0.1."""
from pathlib import Path
import runpy
import sys

# Prevent CLI/JSON overrides from silently changing the named treatment.
for option in ('--arm', '--model-width', '--model-depth', '--embedding-elr-multiplier', '5', '--peak-elr', '--schedule-json', '--embedding-elr-multiplier'):
    if any(arg == option or arg.startswith(option + '=') for arg in sys.argv[1:]):
        raise SystemExit(f'{option} is fixed by this ELR entry point')
sys.argv[1:1] = [
    '--arm', 'fanin_fixed', '--model-width', '768', '--model-depth', '12',
    '--embedding-elr-multiplier', '5', '--peak-elr', '0.1', '--schedule', 'cosine',
    '--expected-world-size', '2', '--device-batch-size', '64',
    '--run-name', 'fanin_fixed_muon_w768_d12_elr0p1',
    '--log-root', 'logs/fanin_fixed_muon_w768_d12_elr0p1',
]
runpy.run_path(str(Path(__file__).with_name('train_muon.py')), run_name='__main__')
