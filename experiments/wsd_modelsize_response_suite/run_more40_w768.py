"""Another20 w768/d12 pairs: same protocol,new shard seed(default1)."""
from pathlib import Path
import subprocess
import sys

if __name__=='__main__':
    subprocess.run([sys.executable,str(Path(__file__).with_name('run_suite.py')),
        '--models','w768_d12','--phase','repeats','--repeats','20','--shard-seed','1',
        *sys.argv[1:]],check=True)
