"""Plan/run the five-model time,amplitude and fresh-shard response experiments."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODELS=json.loads((ROOT/'models.json').read_text())

def make_plan(config,output,phases,selected,load_info,repeats=20,shard_seed=0,val_tokens=1048576):
    output=Path(output).resolve();jobs=[];repeat_manifests={};summary={}
    if repeats<2:raise ValueError('at least2 paired repeats required')
    if val_tokens<=0:raise ValueError('validation budget must be positive')
    for key in selected:
        entry=config['models'][key];paths={int(k):Path(v).resolve() for k,v in entry['checkpoints'].items() if v}
        wanted={5000,10000,15000,16000} if 'time' in phases else {16000}
        if 16000 not in paths:wanted.add(15000)
        infos={step:load_info(path,key,step) for step,path in paths.items() if step in wanted}
        need16=bool(set(phases)&{'prepare','time','amplitude','repeats'})
        prepare=need16 and 16000 not in paths
        if prepare:
            if 15000 not in paths:raise ValueError(key+': need15000 to prepare missing16000')
            paths[16000]=output/key/'checkpoints'/'prepare16000'/'step016000'
            infos[16000]=infos[15000]
        model_jobs=[];dedup={}
        def job(step,arm,purpose,rep=-1):
            mode='fresh' if rep>=0 else 'historical';identity=(step,arm,rep)
            if identity in dedup:
                dedup[identity]['purposes'].append(purpose);return dedup[identity]
            info=infos[step];ranks=info['ranks'];micro=info['device_batch_size']
            if val_tokens%(ranks*micro*1024):raise ValueError(key+': validation tokens must divide global microbatch tokens')
            name=('prepare16000' if arm=='stable' else (f'rep{rep:03d}_{arm}' if rep>=0 else f'fork{step:06d}_{arm}'))
            log_root=output/key/('repeats' if rep>=0 else 'probes')/'logs'/name
            command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={ranks}',str(ROOT/MODELS[key]['script']),'--fork-step',str(step),'--experiment',arm,'--resume',str(paths[step]),'--checkpoint-dir',str(output/key/'checkpoints'/name),'--input-bin',entry.get('input_bin',info['input_bin']),'--input-val-bin',entry.get('input_val_bin',info['input_val_bin']),'--expected-world-size',str(ranks),'--device-batch-size',str(micro),'--compile-model',str(info['compile_model']),'--probe-val-tokens',str(val_tokens),'--log-root',str(log_root),'--run-name',name]
            if rep>=0:command+=['--fresh-replicate',str(rep),'--fresh-repeats',str(repeats),'--shard-seed',str(shard_seed)]
            record=dict(model=key,name=name,experiment=arm,fork_step=step,suite_mode=mode,replicate_id=rep,fresh_shard_index=None,log_root=str(log_root),input_checkpoint=str(paths[step]),purposes=[purpose],ranks=ranks,cuda_visible_count=info.get('cuda_visible_count',ranks),command=command)
            dedup[identity]=record;model_jobs.append(record);return record
        if prepare:job(15000,'stable','prepare')
        if 'time' in phases:
            for step in (5000,10000,15000,16000):
                if step not in paths:raise ValueError(f'{key}:missing checkpoint{step}')
                for arm in ('baseline','elr001','elr003'):job(step,arm,'time')
        if 'amplitude' in phases:
            for arm in ('baseline','elr005','elr004','elr003','elr002','elr001'):job(16000,arm,'amplitude')
        if 'repeats' in phases:
            for rep in range(repeats):
                for arm in ('baseline','elr001'):job(16000,arm,'repeats',rep)
            fresh=[j for j in model_jobs if j['suite_mode']=='fresh']
            repeat_manifests[key]=dict(version=1,checkpoint=str(paths[16000]),fork_step=16000,replicates=list(range(repeats)),validation_tokens=val_tokens,shard_seed=shard_seed,jobs=fresh)
        summary[key]=dict(prepare=int(prepare),historical=sum(j['suite_mode']=='historical' and j['experiment']!='stable' for j in model_jobs),fresh=sum(j['suite_mode']=='fresh' for j in model_jobs),total=len(model_jobs))
        jobs.extend(model_jobs)
    return dict(version=1,phases=phases,models=summary,jobs=jobs,repeat_manifests=repeat_manifests)

def checkpoint_info(path,key,step):
    import torch
    m=json.loads((path/'complete.json').read_text());ranks=m['world_size']
    if m.get('version')!=1 or m.get('step')!=step or m.get('files')!=[f'rank{r:05d}.pt' for r in range(ranks)]:raise ValueError('bad checkpoint manifest: '+str(path))
    if not all((path/f).is_file() for f in m['files']):raise ValueError('missing rank files: '+str(path))
    state=torch.load(path/'rank00000.pt',map_location='cpu',weights_only=False,mmap=True)
    spec=MODELS[key]
    if (state['config']['model_width'],state['config']['model_depth'])!=(spec['width'],spec['depth']) or state['step']!=step:raise ValueError('wrong model or step: '+str(path))
    own=hashlib.sha256((ROOT/spec['script']).read_text(encoding='utf-8').encode()).hexdigest()
    if step==16000:
        allowed={own}
        if key=='w768_d12':allowed|={'92b8b365e80a7d738bd292014f9adbb67bc515358175b50060ad81956faac6c1','bf1ba81bad94b4dad48b10d441730286d10f7f1bf456f02f5db5ee8c32930e37'}
        if state['source_sha256'] not in allowed or state.get('probe_experiment')!='stable' or state['schedule_sha256']!='9e3a4b48e9c2de544e2c1c7f35775c80f910b39717eed85ac45645434961c387':raise ValueError('16000 must be a constant-ELR shared stable checkpoint')
    elif state['source_sha256']!=spec['original_source_sha256'] or state['schedule_sha256']!='38e74728f90116c44727d8da1985cde02668d1d7747bc30a699589640e7f18be':
        raise ValueError('expected this model original stable checkpoint')
    cfg=state['config'];return dict(ranks=ranks,device_batch_size=cfg['device_batch_size'],compile_model=cfg['compile_model'],input_bin=cfg['input_bin'],input_val_bin=cfg['input_val_bin'],cuda_visible_count=len(state['rng']['cuda']))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['all','prepare','time','amplitude','repeats'],default='all');p.add_argument('--models',nargs='+',choices=list(MODELS))
    p.add_argument('--repeats',type=int,default=20);p.add_argument('--shard-seed',type=int,default=0);p.add_argument('--probe-val-tokens',type=int,default=1048576);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();config=json.loads(a.config.read_text());selected=a.models or list(config['models']);phases=['time','amplitude','repeats'] if a.phase=='all' else [a.phase]
    plan=make_plan(config,a.output,phases,selected,checkpoint_info,a.repeats,a.shard_seed,a.probe_val_tokens)
    if a.dry_run:print(json.dumps(plan,indent=2));return
    output=a.output.resolve()
    if output.exists():p.error('choose a fresh output root')
    (output/'console').mkdir(parents=True);(output/'manifest.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    import torch
    visible=os.environ.get('CUDA_VISIBLE_DEVICES');inventory=visible.split(',') if visible is not None else [str(i) for i in range(torch.cuda.device_count())]
    for key,manifest in plan['repeat_manifests'].items():
        d=output/key/'repeats';d.mkdir(parents=True);(d/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for job in plan['jobs']:
        count=job['cuda_visible_count']
        if len(inventory)<count:raise RuntimeError(f"{job['model']}:checkpoint needs{count} visible GPUs;available{len(inventory)}")
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=','.join(inventory[:count])
        print('START '+job['model']+'/'+job['name'],flush=True)
        with (output/'console'/(job['model']+'_'+job['name']+'.log')).open('w',encoding='utf-8') as log:result=subprocess.run(job['command'],env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(result.returncode)
        if job['suite_mode']=='fresh':
            metas=list(Path(job['log_root']).rglob('fresh_shard_metadata.json'))
            if len(metas)!=1:raise RuntimeError('expected exactly one fresh data metadata file')
            job['fresh_shard_index']=json.loads(metas[0].read_text())['fresh_shard']['index']
            manifest=plan['repeat_manifests'][job['model']]
            (output/job['model']/'repeats'/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for key in plan['repeat_manifests']:subprocess.run([sys.executable,str(ROOT/'aggregate_pairs.py'),'--run-root',str(output/key/'repeats')],check=True)
    subprocess.run([sys.executable,str(ROOT/'collect_recovery.py'),'--run-root',str(output)],check=True)
    (output/'complete.json').write_text(json.dumps(dict(status='complete',jobs=len(plan['jobs']))),encoding='utf-8')
if __name__=='__main__':main()
