"""Serial 4-GPU baseline -> checkpoint/reference -> matching pipeline.

Run from any directory. --smoke uses one GPU, a tiny real model and aot_eager
torch.compile (Windows-compatible). Production uses inductor by default.
"""
import argparse, hashlib, json, math, os, signal, subprocess, sys, tempfile, time, uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent

def atomic_json(path, value):
    tmp=Path(str(path)+'.tmp'); tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,path)

def verify_stage(path, cfg):
    done=json.loads((path/'complete.json').read_text(encoding='utf-8'))
    assert done['completed_step']==cfg['num_iterations'] and done['config']==cfg
    rows=[json.loads(s) for s in (path/'dense_mean_elr.jsonl').read_text(encoding='utf-8').splitlines()]
    first=1 if cfg['role']=='baseline' else cfg['warmup_iters']+1
    assert [r['update_step'] for r in rows]==list(range(first,cfg['num_iterations']+1))
    expected=6*cfg['n_layer']+1
    for r in rows:
        names=[]
        for n in r['names']:
            while n.startswith(('_orig_mod.','module.')):n=n.split('.',1)[1]
            names.append(n)
        assert len(names)==expected and len(set(names))==expected
        assert len(r['rms'])==expected and all(math.isfinite(v) and v>0 for v in r['rms'])
        assert math.isfinite(r['actual_mean_elr']) and r['actual_mean_elr']>0
        assert r['pre_update_step']==r['update_step']-1
        assert abs(r['embed_lr']/r['block_lr']-2)<1e-12
        if cfg['role']=='matching':
            assert r['relative_error']<1e-10 and r['weight_decay']==cfg['post_weight_decay']
        else:
            assert r['weight_decay']==cfg['weight_decay']
    log=(path/'train.log').read_text(encoding='utf-8')
    assert f"step:{cfg['num_iterations']}/{cfg['num_iterations']} val_loss:" in log
    if cfg['role']=='baseline':
        ready=json.loads((path/'warmup_checkpoint/ready.json').read_text(encoding='utf-8'))
        assert ready['step']==cfg['warmup_iters']
        for rank in range(ready['world_size']): assert (path/f'warmup_checkpoint/rank{rank}.pt').is_file()
        assert (path/'warmup_checkpoint/shared.pt').is_file()
    return rows

def launch(cmd, path, cfg, env, events):
    start=time.time()
    events.append(dict(stage=path.name,event='start',time=start,command=cmd));atomic_json(path.parent/'events.json',events)
    options=dict(env=env,cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
    if os.name!='nt':options['start_new_session']=True
    proc=subprocess.Popen(cmd,**options)
    try:
        with (path/'console.log').open('w',encoding='utf-8') as log:
            for line in proc.stdout:
                log.write(line);log.flush();print(f'[{path.name}] {line}',end='',flush=True)
        rc=proc.wait()
        if rc:raise subprocess.CalledProcessError(rc,cmd)
    except BaseException:
        if proc.poll() is None:
            if os.name=='nt':subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
            else:os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name!='nt':os.killpg(proc.pid,signal.SIGKILL)
                else:proc.kill()
                proc.wait()
        raise
    verify_stage(path,cfg)
    events.append(dict(stage=path.name,event='verified_complete',time=time.time()))
    atomic_json(path.parent/'events.json',events)

def acquire_gpu_locks(ids):
    # Per-device OS locks prevent two pipeline instances sharing any selected GPU.
    # The kernel releases the locks on exit/crash; no stale PID-based lock removal.
    handles=[]
    for identity in sorted(ids):
        key=hashlib.sha256(identity.encode()).hexdigest()[:20]
        handle=open(Path(tempfile.gettempdir())/f'wte_mean_elr_gpu_{key}.lock','a+b')
        if os.name=='nt':
            import msvcrt
            handle.seek(0);handle.write(b'0');handle.flush();handle.seek(0)
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        handles.append(handle)
    return handles

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    if hasattr(sys.stderr,'reconfigure'):sys.stderr.reconfigure(encoding='utf-8',errors='replace')
    def interrupted(signum, frame):raise KeyboardInterrupt(f'Pipeline interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New directory; existing directories are refused')
    parser.add_argument('--train-data',default='data/fineweb10B/fineweb_train_*.bin')
    parser.add_argument('--val-data',default='data/fineweb10B/fineweb_val_*.bin')
    parser.add_argument('--gpus',default='0,1,2,3',help='Physical GPU indexes/UUIDs; preserve supplied CUDA_VISIBLE_DEVICES if set')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--compile-backend',default=None)
    args=parser.parse_args()
    # Do not initialize CUDA in the orchestrator; torchrun children own the devices.
    selected=os.environ.get('CUDA_VISIBLE_DEVICES',args.gpus).split(',')
    if args.smoke:selected=selected[:1]
    if len(selected)!=(1 if args.smoke else 4):raise ValueError('Production requires exactly four visible GPUs')
    handles=acquire_gpu_locks(selected)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=','.join(selected);env['PYTHONUNBUFFERED']='1'
    env['PYTHONIOENCODING']='utf-8'
    for k in ['RANK','LOCAL_RANK','WORLD_SIZE','GROUP_RANK','ROLE_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','TORCHELASTIC_RUN_ID']:
        env.pop(k,None)
    common=dict(input_bin=str(Path(args.train_data).absolute()),input_val_bin=str(Path(args.val_data).absolute()),
        batch_size=128,device_batch_size=32,sequence_length=1024,num_iterations=20400,
        embed_learning_rate=.0036,warmup_iters=1000,warmdown_iters=5800,weight_decay=0.,
        val_loss_every=500,val_tokens=10485760,save_every=0,compile_model=1,tensor_norm_every=4,
        adamw_update_norm_every=4,activation_probe_every=0,spectral_norm_estimate_enabled=1,
        activation_probe_eps=1e-12,seed=0,vocab_size=50304,n_layer=12,n_head=6,n_embd=768,
        compile_backend=args.compile_backend or ('aot_eager' if args.smoke else 'inductor'),
        role='baseline',resume_dir='',reference_file='',post_weight_decay=.1)
    if args.smoke:
        import numpy as np
        data=out/'tiny_data';data.mkdir()
        rng=np.random.default_rng(731)
        for name in ['train_0.bin','train_1.bin','val_0.bin']:
            tokens=rng.integers(0,128,size=97,dtype=np.uint16)
            header=np.zeros(256,dtype=np.int32);header[:3]=[20240520,1,len(tokens)]
            with (data/name).open('wb') as f:header.tofile(f);tokens.tofile(f)
        common.update(input_bin=str(data/'train_*.bin'),input_val_bin=str(data/'val_*.bin'),batch_size=4,device_batch_size=2,
            sequence_length=8,num_iterations=8,warmup_iters=4,warmdown_iters=2,val_loss_every=4,val_tokens=32,
            tensor_norm_every=1,adamw_update_norm_every=1,spectral_norm_estimate_enabled=0,vocab_size=128,n_layer=2,n_head=2,n_embd=32)
    events=[]
    try:
        configs={}
        for stage,wd,role,post in [('baseline_wd0',0.,'baseline',.1),('baseline_wd0p1',.1,'baseline',0.),
                                  ('matching_wd0_to_wd0p1',0.,'matching',.1),('matching_wd0p1_to_wd0',.1,'matching',0.)]:
            path=out/stage;path.mkdir()
            cfg=dict(common,weight_decay=wd,role=role,post_weight_decay=post,output_dir=str(path))
            if role=='matching':
                base=out/('baseline_wd0' if wd==0 else 'baseline_wd0p1')
                verify_stage(base,configs[base.name])
                cfg.update(resume_dir=str(base/'warmup_checkpoint'),reference_file=str(base/'reference.jsonl'))
            config=path/'config.json';atomic_json(config,cfg);configs[stage]=cfg
            if args.smoke:cmd=[sys.executable,str(HERE/'train.py'),'--config',str(config)]
            else:
                # Port 0 is allocated atomically by the rendezvous store (no probe/rebind race).
                cmd=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--nproc-per-node=4',
                     '--rdzv-backend=c10d','--rdzv-endpoint=localhost:0','--rdzv-id='+uuid.uuid4().hex,
                     '--max-restarts=0',str(HERE/'train.py'),'--config',str(config)]
            launch(cmd,path,cfg,env,events)
            if role=='baseline':
                rows=verify_stage(path,cfg)
                # Exact per-update RMS-ELR from this run; no interpolation or old targets.
                tmp=path/'reference.jsonl.tmp'
                with tmp.open('w',encoding='utf-8') as f:
                    for r in rows:
                        if args.smoke:
                            # Exercise compile/DDP prefix compatibility through the real loader.
                            r=dict(r,names=['module._orig_mod.'+n for n in r['names']])
                        f.write(json.dumps(r)+'\n')
                os.replace(tmp,path/'reference.jsonl')
        atomic_json(out/'pipeline_complete.json',dict(stages=list(configs),completed=True,world_size=len(selected),smoke=args.smoke))
        print('PIPELINE COMPLETE: '+str(out),flush=True)
    except BaseException as exc:
        atomic_json(out/'pipeline_failed.json',dict(error=repr(exc),completed=False))
        raise
    finally:
        for handle in handles:handle.close()

if __name__=='__main__':main()
