"""Run three continuation arms sequentially after a shared1000-update stable extension."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',type=Path,required=True)
    p.add_argument('--input-bin',required=True)
    p.add_argument('--input-val-bin',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nproc-per-node',type=int,default=2)
    p.add_argument('--device-batch-size',type=int,default=64)
    p.add_argument('--eval-every',type=int,choices=[1],default=1)
    p.add_argument('--probe-val-tokens',type=int,default=1048576)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    checkpoint=a.resume.resolve();output=a.output.resolve()
    manifest=json.loads((checkpoint/'complete.json').read_text())
    if manifest.get('version')!=1 or manifest.get('step') not in (15000,16000) or manifest.get('world_size')!=a.nproc_per_node:
        p.error('requires complete step15000 or common step16000 checkpoint with matching rank count')
    files=[f'rank{r:05d}.pt' for r in range(a.nproc_per_node)]
    if manifest.get('files')!=files or not all((checkpoint/f).is_file() for f in files):
        p.error('incomplete checkpoint rank files')
    if a.nproc_per_node<=0 or a.device_batch_size<=0 or 128%(a.nproc_per_node*a.device_batch_size):
        p.error('rank count times microbatch must divide global128')
    if a.eval_every<=0 or a.probe_val_tokens<=0 or a.probe_val_tokens%(a.nproc_per_node*a.device_batch_size*1024):
        p.error('invalid validation interval/token budget')
    jobs=[]
    shared_checkpoint = output/'checkpoints'/'stable'/'step016000' if manifest['step']==15000 else checkpoint
    for arm in (('stable','high','low','pulse') if manifest['step']==15000 else ('high','low','pulse')):
        input_checkpoint = checkpoint if arm=='stable' else shared_checkpoint
        command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={a.nproc_per_node}',str(ROOT/'train_w768_d12_continue.py'),'--experiment',arm,'--resume',str(input_checkpoint),'--checkpoint-dir',str(output/'checkpoints'/arm),'--input-bin',a.input_bin,'--input-val-bin',a.input_val_bin,'--log-root',str(output/'logs'),'--run-name',arm,'--expected-world-size',str(a.nproc_per_node),'--device-batch-size',str(a.device_batch_size),'--eval-every',str(a.eval_every),'--probe-val-tokens',str(a.probe_val_tokens)]
        jobs.append(dict(experiment=arm,command=command))
    if a.dry_run:print(json.dumps(jobs,indent=2));return
    if output.exists():p.error('choose a fresh output root')
    (output/'console').mkdir(parents=True)
    (output/'manifest.json').write_text(json.dumps(jobs,indent=2),encoding='utf-8')
    for job in jobs:
        print('START '+job['experiment'],flush=True)
        with (output/'console'/(job['experiment']+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run(job['command'],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('All continuation arms completed',flush=True)
if __name__=='__main__':main()
