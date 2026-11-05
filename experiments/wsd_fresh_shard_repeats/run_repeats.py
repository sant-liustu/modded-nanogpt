"""Run20 paired continuations on20 distinct provably untouched shards."""
import argparse,glob,json,subprocess,sys
from pathlib import Path
from fresh_shards import audit_history,check_fresh_shard
ROOT=Path(__file__).resolve().parent

def make_plan(state,a):
    audit=audit_history(state);B=state['train_loader']['B'];world=state['train_loader']['world']
    if a.nproc_per_node!=world or a.device_batch_size!=B:raise ValueError('rank/microbatch must match the original checkpoint')
    if a.probe_val_tokens<=0 or a.probe_val_tokens%(B*world*state['train_loader']['T']):raise ValueError('invalid validation budget')
    if a.repeats<2:raise ValueError('at least2 pairs required')
    required=(620*audit['gradient_accumulation']+1)*audit['global_microbatch_tokens']+1
    eligible=[i for i in audit['unvisited_indices'] if audit['token_counts'][i]>=required]
    indices=a.fresh_shard_indices if a.fresh_shard_indices is not None else eligible[:a.repeats]
    if len(indices)!=a.repeats or len(set(indices))!=len(indices):raise ValueError(f'need{a.repeats} distinct large enough untouched shards; only{len(eligible)} eligible; no automatic reuse')
    for i in indices:check_fresh_shard(state,i,620)
    out=a.output.resolve();checkpoint=a.resume.resolve();jobs=[]
    for rep,index in enumerate(indices):
        for arm in ('baseline','elr001'):
            name=f'rep{rep:03d}_shard{index:05d}_{arm}';log_root=out/'logs'/name
            command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',str(ROOT/'train_fresh_repeat.py'),'--experiment',arm,'--replicate-id',str(rep),'--fresh-shard-index',str(index),'--resume',str(checkpoint),'--checkpoint-dir',str(out/'checkpoints'/name),'--input-bin',a.input_bin,'--input-val-bin',a.input_val_bin,'--log-root',str(log_root),'--run-name',name,'--expected-world-size',str(world),'--device-batch-size',str(B),'--probe-val-tokens',str(a.probe_val_tokens)]
            jobs.append(dict(name=name,replicate_id=rep,fresh_shard_index=index,experiment=arm,log_root=str(log_root),input_checkpoint=str(checkpoint),command=command))
    return dict(version=1,checkpoint=str(checkpoint),fork_step=16000,replicates=list(range(a.repeats)),selected_shard_indices=indices,validation_tokens=a.probe_val_tokens,history_audit=audit,jobs=jobs)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',type=Path,required=True);p.add_argument('--input-bin',required=True);p.add_argument('--input-val-bin',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--fresh-shard-indices',type=int,nargs='+')
    p.add_argument('--nproc-per-node',type=int,default=2);p.add_argument('--device-batch-size',type=int,default=64)
    p.add_argument('--probe-val-tokens',type=int,default=1048576);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    checkpoint=a.resume.resolve();manifest=json.loads((checkpoint/'complete.json').read_text());files=[f'rank{r:05d}.pt' for r in range(a.nproc_per_node)]
    if manifest.get('version')!=1 or manifest.get('step')!=16000 or manifest.get('world_size')!=a.nproc_per_node or manifest.get('files')!=files or not all((checkpoint/f).is_file() for f in files):p.error('complete shared-stable16000 checkpoint required')
    import torch
    # mmap reads metadata without eagerly allocating full model/Adam tensor storage.
    state=torch.load(checkpoint/'rank00000.pt',map_location='cpu',weights_only=False,mmap=True)
    if [str(Path(f).resolve()) for f in sorted(glob.glob(a.input_bin))]!=state['train_loader']['files']:p.error('training glob must match the original checkpoint; choose fresh shard by index')
    try:plan=make_plan(state,a)
    except ValueError as exc:p.error(str(exc))
    del state
    if a.dry_run:print(json.dumps(plan,indent=2));return
    out=a.output.resolve()
    if out.exists():p.error('choose a fresh output directory')
    (out/'console').mkdir(parents=True);(out/'manifest.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    for job in plan['jobs']:
        print('START '+job['name'],flush=True)
        with (out/'console'/(job['name']+'.log')).open('w',encoding='utf-8') as log:result=subprocess.run(job['command'],stdout=log,stderr=subprocess.STDOUT)
        (out/(job['name']+'_result.json')).write_text(json.dumps(dict(exit_code=result.returncode)),encoding='utf-8')
        if result.returncode:raise SystemExit(result.returncode)
    subprocess.run([sys.executable,str(ROOT/'aggregate_pairs.py'),'--run-root',str(out)],check=True)
if __name__=='__main__':main()
