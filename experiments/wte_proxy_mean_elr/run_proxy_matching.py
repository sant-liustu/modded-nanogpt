import argparse, hashlib, json, math, os, signal, subprocess, sys, tempfile, time, uuid
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent

def atomic_json(path, value):
    tmp=Path(str(path)+'.tmp'); tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,path)

def verify_stage(path, cfg):
    done=json.loads((path/'complete.json').read_text(encoding='utf-8'))
    assert done['config']==cfg and done['start_step']==0 and done['completed_step']==cfg['num_iterations']
    rows=[json.loads(line) for line in (path/'dense_mean_elr.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['update_step'] for r in rows]==list(range(1,cfg['num_iterations']+1))
    for r in rows:
        assert math.isclose(r['embed_lr'],2*r['block_lr'],rel_tol=1e-12)
        if r['update_step']<=cfg['warmup_iters']:
            assert r['relative_error'] is None and r['weight_decay']==cfg['weight_decay']
        else:
            assert r['relative_error']<1e-10 and r['weight_decay']==cfg['post_weight_decay']
        proxy=sum(r['block_lr']/v for v in r['rms'])/len(r['rms'])
        assert math.isclose(proxy,r['proxy_mean_elr'],rel_tol=1e-12)
    assert not list(path.glob('restored_rank*.json'))
    return rows

EVENT_LOCK = threading.Lock()

def event(path, events, **record):
    with EVENT_LOCK:
        events.append(dict(time=time.time(), **record))
        atomic_json(path / 'events.json', events)

def stop_process(proc):
    if proc.poll() is not None: return
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        try: proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if os.name != 'nt': os.killpg(proc.pid, signal.SIGKILL)
            else: proc.kill()
            proc.wait()
    except ProcessLookupError:
        pass

def launch(cmd, path, cfg, env, events, control):
    options = dict(env=env, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                   text=True, encoding='utf-8', errors='replace', bufsize=1)
    if os.name != 'nt': options['start_new_session'] = True
    # Launch and registration are atomic with respect to group cancellation.
    with control['lock']:
        if control['cancelled']: raise RuntimeError('Peer job failed; launch cancelled')
        proc = subprocess.Popen(cmd, **options)
        control['processes'].append(proc)
    event(path.parent, events, stage=path.name, event='start', command=cmd,
          visible_gpus=env['CUDA_VISIBLE_DEVICES'])
    try:
        with (path/'console.log').open('w', encoding='utf-8') as log:
            for line in proc.stdout:
                log.write(line); log.flush()
                print(f'[{path.name}] {line}', end='', flush=True)
        rc = proc.wait()
        if rc: raise subprocess.CalledProcessError(rc, cmd)
        verify_stage(path, cfg)
        event(path.parent, events, stage=path.name, event='verified_complete')
    except BaseException:
        stop_process(proc)
        raise

def run_pair(jobs):
    control = dict(lock=threading.Lock(), cancelled=False, processes=[])
    pool = ThreadPoolExecutor(max_workers=2)
    futures = [pool.submit(launch, *job, control) for job in jobs]
    try:
        for future in as_completed(futures): future.result()
    except BaseException:
        with control['lock']:
            control['cancelled'] = True
            processes = list(control['processes'])
        for proc in processes: stop_process(proc)
        for future in futures: future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

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
    from extract_proxy_reference import extract
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    def interrupted(signum,frame):raise KeyboardInterrupt(signum)
    signal.signal(signal.SIGTERM,interrupted)
    p=argparse.ArgumentParser(description='Extract new proxy references from existing baselines; run two fresh-start matching jobs concurrently.')
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--smoke',action='store_true',help='Use existing tiny smoke baselines, one local GPU and aot_eager')
    p.add_argument('--train-data');p.add_argument('--val-data')
    a=p.parse_args();root=a.baseline_root.resolve();out=a.output.resolve()
    selected=os.environ.get('CUDA_VISIBLE_DEVICES',a.gpus).split(',')
    if a.smoke:selected=selected[:1]
    assert len(selected)==(1 if a.smoke else 4) and len(set(selected))==len(selected)
    locks=acquire_gpu_locks(selected);events=[];created=False
    try:
        out.mkdir(parents=True,exist_ok=False);created=True
        tags=['wd0','wd0p1']
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(extract,root/f'baseline_{tag}',out/f'proxy_reference_{tag}.jsonl') for tag in tags]
            configs=[f.result() for f in futures]
        event(out,events,stage='references',event='both_proxy_references_ready')
        jobs=[]
        for i,(tag,cfg) in enumerate(zip(tags,configs)):
            cfg=dict(cfg);stage=out/f'proxy_matching_{tag}';stage.mkdir()
            assert cfg['weight_decay']==(0. if tag=='wd0' else .1)
            if not a.smoke:assert cfg['batch_size']==128 and cfg['device_batch_size']==64
            else:
                assert cfg['num_iterations']<=32 and cfg['n_embd']<=64
                cfg['compile_backend']='aot_eager'
            cfg.update(role='matching',resume_dir='',reference_file=str(out/f'proxy_reference_{tag}.jsonl'),
                       post_weight_decay=.1 if tag=='wd0' else 0.,output_dir=str(stage))
            if a.train_data:cfg['input_bin']=str(Path(a.train_data).absolute())
            if a.val_data:cfg['input_val_bin']=str(Path(a.val_data).absolute())
            config=stage/'config.json';atomic_json(config,cfg)
            env=os.environ.copy()
            for k in ['RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','TORCHELASTIC_RUN_ID']:env.pop(k,None)
            env.update(PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',CUDA_VISIBLE_DEVICES=','.join(selected if a.smoke else selected[i*2:i*2+2]))
            worker=str(HERE/f'train_proxy_{tag}_fresh.py')
            cmd=[sys.executable,worker,'--config',str(config)] if a.smoke else [sys.executable,'-m','torch.distributed.run',
                 '--nnodes=1','--nproc-per-node=2','--rdzv-backend=c10d','--rdzv-endpoint=localhost:0',
                 '--rdzv-id='+uuid.uuid4().hex,'--max-restarts=0',worker,'--config',str(config)]
            jobs.append((cmd,stage,cfg,env,events))
        run_pair(jobs)
        atomic_json(out/'pipeline_complete.json',dict(completed=True,metric='proxy_mean_elr',fresh_start=True,stages=[j[1].name for j in jobs]))
        print('PROXY MATCHING COMPLETE:',out)
    except BaseException as e:
        if created:atomic_json(out/'pipeline_failed.json',dict(error=repr(e)))
        raise
    finally:
        for lock in locks:lock.close()
if __name__=='__main__':main()
