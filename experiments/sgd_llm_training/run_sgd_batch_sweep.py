"""Run the fixed-LR batch sweep sequentially on one GPU; stop on failure."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    repo = here.parent.parent
    for batch in (8, 4, 2, 16):
        script = here / f'train_sgd_rmsnorm_gamma_w384_B{batch:04d}_devB{batch:03d}_blocklr0p1.py'
        print(f'Running batch={batch}: {script.name}', flush=True)
        subprocess.run([sys.executable, str(script)], cwd=repo, check=True)