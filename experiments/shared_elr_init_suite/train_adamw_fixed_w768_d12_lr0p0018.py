"""Two-GPU current GPT/FineWeb, old lr0p0018 AdamW baseline LR and fixed initial matrix RMS."""
from pathlib import Path
import runpy
import sys

for option in ('--arm','--lr-mode','--model-width','--model-depth','--schedule-json',
               '--peak-elr','--embed-learning-rate','--gamma-learning-rate'):
    if any(arg==option or arg.startswith(option+'=') for arg in sys.argv[1:]):
        raise SystemExit(f'{option} is fixed or inapplicable in this absolute-LR entry')
sys.argv[1:1]=[
    '--arm','fanin_fixed','--lr-mode','absolute','--model-width','768','--model-depth','12',
    '--learning-rate','0.0009','--embedding-lr-multiplier','2',
    '--embed-learning-rate','0.0018','--gamma-learning-rate','0.0009','--min-lr-ratio','0.1','--weight-decay','0.1','--grad-clip','0',
    '--schedule','cosine','--expected-world-size','2','--device-batch-size','64',
    '--run-name','adamw_fixed_w768_d12_lr0p0018','--log-root','logs/adamw_fixed_w768_d12_lr0p0018',
]
runpy.run_path(str(Path(__file__).with_name('train.py')),run_name='__main__')
