"""Plan140 paired mixed-two-shard continuations,without changing the validation prefix."""
import argparse,hashlib,json,os,random,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def make_plan(info,checkpoint,output,pairs=140,seed_start=2000,fixed_shards=None,val_tokens=1048576,save_checkpoints=False):
    if pairs<2:raise ValueError('need at least2 pairs')
    if val_tokens<=0 or val_tokens%(info['ranks']*info['device_batch_size']*info['context']):raise ValueError('invalid validation budget')
    remaining=list(range(info['train_shard']+1,len(info['train_files'])))
    if len(remaining)<2:raise ValueError('need at least2 shards after the checkpoint current shard')
    width=info['ranks']*info['device_batch_size']*info['context'];accum=info['batch_size']//(info['ranks']*info['device_batch_size'])
    if 16000*accum+1>=sum((((n-1024)//2)-1)//width for n in info['train_sizes']):raise ValueError('full corpus pass reached;no unused suffix')
    if fixed_shards is not None and (len(set(fixed_shards))!=2 or any(i not in remaining for i in fixed_shards)):raise ValueError('fixed pool must contain two different unused-suffix shards')
    out=Path(output).resolve();jobs=[];seeds=list(range(seed_start,seed_start+pairs))
    for seed in seeds:
        shards=list(fixed_shards) if fixed_shards is not None else random.Random(seed^0x51A2).sample(remaining,2)
        for arm in ('baseline','elr001'):
            name=f'seed{seed}_{arm}';log_root=out/'logs'/name
            cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f"--nproc_per_node={info['ranks']}",str(ROOT/'train_w768_mix2.py'),'--experiment',arm,'--mix-seed',str(seed),'--mix-shards',*[str(i) for i in shards],'--resume',str(checkpoint),'--checkpoint-dir',str(out/'checkpoints'/name),'--input-bin',info['input_bin'],'--input-val-bin',info['input_val_bin'],'--expected-world-size',str(info['ranks']),'--device-batch-size',str(info['device_batch_size']),'--compile-model',str(info['compile_model']),'--probe-val-tokens',str(val_tokens),'--log-root',str(log_root),'--run-name',name]
            if save_checkpoints:cmd+=['--save-checkpoints']
            jobs.append(dict(name=name,replicate_id=seed,experiment=arm,mix_shards=shards,log_root=str(log_root),command=cmd))
    return dict(version=1,model='w768_d12',protocol='mix2_context_shuffle',checkpoint=str(checkpoint),fork_step=16000,replicates=seeds,validation_tokens=val_tokens,jobs=jobs)

def load_info(checkpoint):
    import torch
    m=json.loads((checkpoint/'complete.json').read_text());ranks=m['world_size'];files=[f'rank{i:05d}.pt' for i in range(ranks)]
    if m.get('version')!=1 or m.get('step')!=16000 or m.get('files')!=files or not all((checkpoint/f).is_file() for f in files):raise ValueError('complete shared16000 checkpoint required')
    s=torch.load(checkpoint/'rank00000.pt',map_location='cpu',weights_only=False,mmap=True);cfg=s['config']
    if (cfg['model_width'],cfg['model_depth'])!=(768,12):raise ValueError('requires w768/d12 checkpoint')
    prov=json.loads((ROOT/'provenance.json').read_text());allowed={prov['parent_runtime_sha256'],'92b8b365e80a7d738bd292014f9adbb67bc515358175b50060ad81956faac6c1','bf1ba81bad94b4dad48b10d441730286d10f7f1bf456f02f5db5ee8c32930e37'}
    if s['source_sha256'] not in allowed or s.get('probe_experiment')!='stable' or s['schedule_sha256']!='9e3a4b48e9c2de544e2c1c7f35775c80f910b39717eed85ac45645434961c387':raise ValueError('known shared-stable16000 checkpoint required')
    return dict(ranks=ranks,device_batch_size=cfg['device_batch_size'],context=cfg['sequence_length'],batch_size=cfg['batch_size'],compile_model=cfg['compile_model'],input_bin=cfg['input_bin'],input_val_bin=cfg['input_val_bin'],cuda_visible_count=len(s['rng']['cuda']),train_shard=s['train_loader']['shard'],train_files=s['train_loader']['files'],train_sizes=s['train_loader']['sizes'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=140);p.add_argument('--seed-start',type=int,default=2000);p.add_argument('--pool-shards',type=int,nargs=2,help='optional fixed pool for all seeds;default randomly choose two per pair')
    p.add_argument('--save-checkpoints',action='store_true')
    p.add_argument('--probe-val-tokens',type=int,default=1048576);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    cp=a.resume.resolve();info=load_info(cp);plan=make_plan(info,cp,a.output,a.pairs,a.seed_start,a.pool_shards,a.probe_val_tokens,a.save_checkpoints)
    if a.dry_run:print(json.dumps(plan,indent=2));return
    out=a.output.resolve()
    if out.exists():p.error('choose a fresh output root')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        import torch
        inventory=[str(i) for i in range(torch.cuda.device_count())]
    else:inventory=visible.split(',')
    count=info['cuda_visible_count']
    if len(inventory)<count:p.error('not enough visible GPUs to restore checkpoint RNG environment')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=','.join(inventory[:count])
    (out/'console').mkdir(parents=True);(out/'manifest.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    for job in plan['jobs']:
        print('START '+job['name'],flush=True)
        with (out/'console'/(job['name']+'.log')).open('w',encoding='utf-8') as log:result=subprocess.run(job['command'],env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(result.returncode)
    subprocess.run([sys.executable,str(ROOT/'aggregate_mix_pairs.py'),'--run-root',str(out)],check=True)
if __name__=='__main__':main()
