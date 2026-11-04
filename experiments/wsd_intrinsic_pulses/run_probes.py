"""Run six intrinsic-width probes sequentially for each supplied stable checkpoint."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',nargs='+',type=Path,required=True,help='complete checkpoint directories at5000/10000/15000/16000; any subset')
    p.add_argument('--input-bin',required=True)
    p.add_argument('--input-val-bin',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nproc-per-node',type=int,default=2)
    p.add_argument('--device-batch-size',type=int,default=64)
    p.add_argument('--probe-val-tokens',type=int,default=1048576)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.nproc_per_node<=0 or a.device_batch_size<=0 or 128%(a.nproc_per_node*a.device_batch_size):p.error('world size times microbatch must divide128')
    if a.probe_val_tokens<=0 or a.probe_val_tokens%(a.nproc_per_node*a.device_batch_size*1024):p.error('validation tokens must divide global validation microbatch tokens')
    output=a.output.resolve(); jobs=[];seen=set()
    for checkpoint in a.checkpoints:
        checkpoint=checkpoint.resolve()
        manifest=json.loads((checkpoint/'complete.json').read_text())
        step=manifest.get('step')
        files=[f'rank{r:05d}.pt' for r in range(a.nproc_per_node)]
        if manifest.get('version')!=1 or step not in (5000,10000,15000,16000) or manifest.get('world_size')!=a.nproc_per_node or manifest.get('files')!=files or not all((checkpoint/f).is_file() for f in files):p.error('requires complete supported-step checkpoint and matching rank count: '+str(checkpoint))
        if step in seen:p.error('duplicate fork step')
        seen.add(step)
        for arm in ('baseline','elr005','elr004','elr003','elr002','elr001'):
            name=f'step{step:06d}_{arm}'
            cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={a.nproc_per_node}',str(ROOT/'train_intrinsic_pulse.py'),'--fork-step',str(step),'--experiment',arm,'--resume',str(checkpoint),'--checkpoint-dir',str(output/'checkpoints'/name),'--input-bin',a.input_bin,'--input-val-bin',a.input_val_bin,'--log-root',str(output/'logs'),'--run-name',name,'--expected-world-size',str(a.nproc_per_node),'--device-batch-size',str(a.device_batch_size),'--probe-val-tokens',str(a.probe_val_tokens)]
            jobs.append(dict(name=name,fork_step=step,experiment=arm,input_checkpoint=str(checkpoint),command=cmd))
    if a.dry_run:print(json.dumps(jobs,indent=2));return
    if output.exists():p.error('choose a fresh output root')
    (output/'console').mkdir(parents=True)
    (output/'manifest.json').write_text(json.dumps(jobs,indent=2),encoding='utf-8')
    for job in jobs:
        print('START '+job['name'],flush=True)
        with (output/'console'/(job['name']+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(job['command'],stdout=log,stderr=subprocess.STDOUT)
        (output/(job['name']+'_result.json')).write_text(json.dumps(dict(exit_code=result.returncode)),encoding='utf-8')
        if result.returncode:raise SystemExit(result.returncode)
    print('All requested probes completed',flush=True)
if __name__=='__main__':main()
