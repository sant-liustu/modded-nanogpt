"""Plan400 pairs per smaller model,without changing the validation prefix."""
import argparse,hashlib,json,os,random,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODELS=json.loads((ROOT/'models.json').read_text(encoding='utf-8'))

def make_plan(info,checkpoint,output,pairs=400,seed_start=4000,fixed_shards=None,val_tokens=1048576,save_checkpoints=False,model=None):
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
            cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f"--nproc_per_node={info['ranks']}",str(ROOT/MODELS[model]['script']),'--experiment',arm,'--mix-seed',str(seed),'--mix-shards',*[str(i) for i in shards],'--resume',str(checkpoint),'--checkpoint-dir',str(out/'checkpoints'/name),'--input-bin',info['input_bin'],'--input-val-bin',info['input_val_bin'],'--expected-world-size',str(info['ranks']),'--device-batch-size',str(info['device_batch_size']),'--compile-model',str(info['compile_model']),'--probe-val-tokens',str(val_tokens),'--log-root',str(log_root),'--run-name',name]
            if save_checkpoints:cmd+=['--save-checkpoints']
            jobs.append(dict(name=name,replicate_id=seed,experiment=arm,mix_shards=shards,log_root=str(log_root),command=cmd))
    return dict(version=1,model=model,protocol='mix2_context_shuffle',checkpoint=str(checkpoint),fork_step=16000,replicates=seeds,validation_tokens=val_tokens,jobs=jobs)

def load_info(checkpoint,model):
    import torch
    m=json.loads((checkpoint/'complete.json').read_text(encoding='utf-8'));ranks=m['world_size'];files=[f'rank{i:05d}.pt' for i in range(ranks)]
    if m.get('version')!=1 or m.get('step')!=16000 or m.get('files')!=files or not all((checkpoint/f).is_file() for f in files):raise ValueError('complete shared16000 checkpoint required')
    s=torch.load(checkpoint/'rank00000.pt',map_location='cpu',weights_only=False,mmap=True);cfg=s['config']
    spec=MODELS[model]
    if (cfg['model_width'],cfg['model_depth'])!=(spec['width'],spec['depth']):raise ValueError('wrong model-size checkpoint')
    allowed={spec['parent_runtime_sha256']}
    if s['source_sha256'] not in allowed or s.get('probe_experiment')!='stable' or s['schedule_sha256']!='9e3a4b48e9c2de544e2c1c7f35775c80f910b39717eed85ac45645434961c387':raise ValueError('known shared-stable16000 checkpoint required')
    return dict(ranks=ranks,device_batch_size=cfg['device_batch_size'],context=cfg['sequence_length'],batch_size=cfg['batch_size'],compile_model=cfg['compile_model'],input_bin=cfg['input_bin'],input_val_bin=cfg['input_val_bin'],cuda_visible_count=len(s['rng']['cuda']),train_shard=s['train_loader']['shard'],train_files=s['train_loader']['files'],train_sizes=s['train_loader']['sizes'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--model',choices=list(MODELS));g.add_argument('--config',type=Path,help='JSON:models maps model key to shared16000 checkpoint path')
    p.add_argument('--resume',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=400);p.add_argument('--seed-start',type=int,default=4000);p.add_argument('--pool-shards',type=int,nargs=2)
    p.add_argument('--save-checkpoints',action='store_true');p.add_argument('--probe-val-tokens',type=int,default=1048576);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.model:
        if a.resume is None:p.error('--model requires --resume')
        requests={a.model:a.resume}
    else:
        if a.resume is not None:p.error('--resume is only for single-model mode')
        requests=json.loads(a.config.read_text(encoding='utf-8'))['models']
    if not requests:p.error('config must contain at least one model')
    plans={};infos={};out=a.output.resolve()
    for model,path in requests.items():
        if model not in MODELS:p.error('unknown model '+model)
        cp=Path(path).resolve();info=load_info(cp,model);infos[model]=info
        plans[model]=make_plan(info,cp,out/model,a.pairs,a.seed_start,a.pool_shards,a.probe_val_tokens,a.save_checkpoints,model)
    if a.dry_run:print(json.dumps(dict(total_runs=sum(len(v['jobs']) for v in plans.values()),models=plans),indent=2));return
    if out.exists():p.error('choose a fresh output root')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        import torch
        inventory=[str(i) for i in range(torch.cuda.device_count())]
    else:inventory=visible.split(',')
    for model in plans:
        if len(inventory)<infos[model]['cuda_visible_count']:p.error(model+':not enough visible GPUs for checkpoint RNG environment')
    out.mkdir(parents=True)
    (out/'manifest.json').write_text(json.dumps(dict(models=list(plans),pairs_per_model=a.pairs,total_runs=sum(len(v['jobs']) for v in plans.values())),indent=2),encoding='utf-8')
    for model,plan in plans.items():
        d=out/model;(d/'console').mkdir(parents=True);(d/'manifest.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=','.join(inventory[:infos[model]['cuda_visible_count']])
        for job in plan['jobs']:
            print('START '+model+'/'+job['name'],flush=True)
            with (d/'console'/(job['name']+'.log')).open('w',encoding='utf-8') as log:result=subprocess.run(job['command'],env=env,stdout=log,stderr=subprocess.STDOUT)
            (d/(job['name']+'_result.json')).write_text(json.dumps(dict(exit_code=result.returncode)),encoding='utf-8')
            if result.returncode:raise SystemExit(result.returncode)
        subprocess.run([sys.executable,str(ROOT/'aggregate_mix_pairs.py'),'--run-root',str(d)],check=True)
    (out/'complete.json').write_text(json.dumps(dict(status='complete')),encoding='utf-8')
if __name__=='__main__':main()
