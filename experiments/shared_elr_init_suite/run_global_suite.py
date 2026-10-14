"""Launch four global-block ELR runs in disjoint GPU waves; inspect with --dry-run first."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import os
import time

p = argparse.ArgumentParser()
p.add_argument('--train-glob', required=True)
p.add_argument('--val-glob', required=True)
p.add_argument('--output', type=Path, default=Path('outputs/global_block_elr_suite'))
p.add_argument('--dry-run', action='store_true')
p.add_argument('--gpu-ids', default='0,1,2,3,4,5,6,7', help='unique visible GPU indices')
p.add_argument('--arm', choices=['mup_wd'], action='append')
p.add_argument('--width', type=int, choices=[384,768], action='append')
p.add_argument('--schedule', choices=['cosine','wsd'], action='append')
a = p.parse_args()
try:
    gpu_ids = [int(v) for v in a.gpu_ids.split(',')]
except ValueError:
    p.error('GPU ids must be comma-separated integers')
if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or min(gpu_ids)<0:
    p.error('GPU ids must be unique nonnegative integers')
jobs = []
for arm in a.arm or ['mup_wd']:
    for width in a.width or [384,768]:
        for schedule in a.schedule or ['cosine','wsd']:
            name = f'global_block_{arm}_w{width}_{schedule}'
            ranks = 1 if width == 384 else 2
            cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={ranks}',
                   str(Path(__file__).with_name('train_global_block_elr.py')), '--arm', arm, '--model-width', str(width),
                   '--expected-world-size', str(ranks), '--device-batch-size', str(128//ranks),
                   '--schedule', schedule, '--input-bin', a.train_glob, '--input-val-bin', a.val_glob,
                   '--log-root', str(a.output/'logs'), '--run-name', name,
                   '--checkpoint-dir', str(a.output/'checkpoints'/name)]
            jobs.append(dict(name=name, command=cmd, ranks=ranks, end_step=20400,
                             checkpoint_steps=[14600,20400] if schedule == 'wsd' else [20400]))
waves=[]
wave=[]
used=0
for job in jobs:
    if job['ranks']>len(gpu_ids):
        p.error('not enough selected GPUs for a job')
    if used+job['ranks']>len(gpu_ids):
        waves.append(wave)
        wave=[]
        used=0
    job['wave']=len(waves)+1
    job['gpu_ids']=gpu_ids[used:used+job['ranks']]
    wave.append(job)
    used+=job['ranks']
if wave: waves.append(wave)
if a.dry_run:
    print(json.dumps(jobs, indent=2))
    sys.exit(0)
import torch
import glob
if not glob.glob(a.train_glob) or not glob.glob(a.val_glob):
    p.error('training and validation shards must exist')
if not torch.cuda.is_available() or max(gpu_ids)>=torch.cuda.device_count():
    p.error('selected GPU ids must exist in the current CUDA-visible device inventory')
# Respect any outer CUDA_VISIBLE_DEVICES mapping.
outer_mapping=os.environ.get('CUDA_VISIBLE_DEVICES')
physical_ids=outer_mapping.split(',') if outer_mapping else [str(i) for i in range(torch.cuda.device_count())]
for job in jobs:
    if Path(job['command'][job['command'].index('--checkpoint-dir')+1]).exists():
        p.error(f"checkpoint already exists for {job['name']}; choose a fresh output path")
a.output.mkdir(parents=True, exist_ok=True)
(a.output/'manifest.json').write_text(json.dumps(jobs, indent=2), encoding='utf-8')
results=[]
for wave in waves:
    running=[]
    try:
        for job in wave:
            print(f"Starting {job['name']} on GPUs {job['gpu_ids']}",flush=True)
            output=(a.output/(job['name']+'.console.log')).open('w',encoding='utf-8')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(physical_ids[g] for g in job['gpu_ids']))
            try:
                proc=subprocess.Popen(job['command'],stdout=output,stderr=subprocess.STDOUT,env=env)
            except OSError as error:
                output.write(str(error)+'\n'); output.close()
                results.append(dict(name=job['name'],exit_code=-1,status='launch_failed',error=str(error)))
                continue
            running.append((job,proc,output))
        while any(proc.poll() is None for _,proc,_ in running):
            time.sleep(.5)
        for job,proc,_ in running:
            results.append(dict(name=job['name'],exit_code=proc.returncode,
                                status='success' if proc.returncode==0 else 'failed'))
            print(f"{job['name']}: exit {proc.returncode}",flush=True)
        (a.output/'run_summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    finally:
        for _,proc,output in running:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
            output.close()
failed=[r['name'] for r in results if r['exit_code']!=0]
print(json.dumps(dict(completed=len(results),failed=failed),indent=2))
sys.exit(1 if failed else 0)
