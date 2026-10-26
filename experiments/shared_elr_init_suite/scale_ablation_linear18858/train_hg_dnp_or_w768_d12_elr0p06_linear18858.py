"""AdamW hg_dnp_or; historical w768/d12 ELR0.06 linear18858 protocol."""
from pathlib import Path
import runpy
import sys
for option in ('--scale-variant', '--schedule-json'):
    if any(arg == option or arg.startswith(option + '=') for arg in sys.argv[1:]):
        raise SystemExit(f'{option} is fixed by this entry point')
sys.argv[1:1] = ['--scale-variant', 'hg_dnp_or', '--run-name', 'train_hg_dnp_or_w768_d12_elr0p06_linear18858',
                  '--log-root', 'logs/matched_linear18858/hg_dp']
runpy.run_path(str(Path(__file__).with_name('train_adamw_matched.py')), run_name='__main__')
