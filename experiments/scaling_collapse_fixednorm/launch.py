"""Eight-GPU large model first, then the remaining model stages."""
import argparse
import glob
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def make_plan(gpus, output, train_glob, val_glob):
    def job(name, indices):
        ids=[gpus[i] for i in indices]
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(ids)}',
             str(ROOT/f'train_fixednorm_{name}.py'),'--expected-world-size',str(len(ids)),
             '--device-batch-size',str(128//len(ids)), '--input-bin',train_glob,
             '--input-val-bin',val_glob,'--checkpoint-dir',str(output/'checkpoints'/name),
             '--log-root',str(output/'logs'),'--run-name',name]
        return dict(name=name,gpu_ids=ids,command=cmd)
    return dict(large=[[job('w1536',list(range(8)))]],remaining=[
        [job('w768_d24',[4,5,6,7])],
        [job('w768',[4,5]),job('w768_d6',[6]),job('w384',[7])]])

def stop_process(proc):
    if proc.poll() is not None:
        return
    if os.name=='posix':
        try: os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError: pass
    else:
        proc.terminate()
    try: proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name=='posix':
            try: os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError: pass
        else: proc.kill()
        proc.wait()

def run_lane(name, stages, output):
    results=[]
    for stage_index, jobs in enumerate(stages,1):
        processes=[]
        try:
            for job in jobs:
                env=os.environ.copy()
                env['CUDA_VISIBLE_DEVICES']=','.join(job['gpu_ids'])
                log=(output/'console'/f"{job['name']}.log").open('w',encoding='utf-8')
                try:
                    proc=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=(os.name=='posix'))
                except BaseException:
                    log.close(); raise
                processes.append((job,proc,log))
                print(f"START {name} stage{stage_index}: {job['name']} GPUs={env['CUDA_VISIBLE_DEVICES']}",flush=True)
            while True:
                codes=[proc.poll() for _,proc,_ in processes]
                if any(code is not None and code!=0 for code in codes):
                    raise RuntimeError(f'{name} stage{stage_index} failed; next stage will not start')
                if all(code is not None for code in codes): break
                time.sleep(1)
            results.extend(dict(name=job['name'],exit_code=proc.returncode) for job,proc,_ in processes)
        except BaseException:
            for _,proc,_ in processes: stop_process(proc)
            raise
        finally:
            for _,_,log in processes: log.close()
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lane',choices=['all','large','remaining'],default='all')
    p.add_argument('--gpu-ids',default='0,1,2,3,4,5,6,7',help='eight indices into the current CUDA-visible GPU inventory')
    p.add_argument('--train-glob',required=True)
    p.add_argument('--val-glob',required=True)
    p.add_argument('--output',type=Path,required=True,help='fresh run root; separate roots if invoking lanes separately')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    try: ids=[int(x) for x in a.gpu_ids.split(',')]
    except ValueError: p.error('GPU indices must be integers')
    if len(ids)!=8 or len(set(ids))!=8 or min(ids)<0: p.error('exactly eight distinct nonnegative GPU indices required')
    outer=os.environ.get('CUDA_VISIBLE_DEVICES')
    visible=outer.split(',') if outer is not None else None
    if visible is not None and max(ids)>=len(visible): p.error('GPU indices exceed CUDA_VISIBLE_DEVICES')
    gpus=[visible[i] if visible is not None else str(i) for i in ids]
    output=a.output.resolve()
    plan=make_plan(gpus,output,a.train_glob,a.val_glob)
    selected=plan if a.lane=='all' else {a.lane:plan[a.lane]}
    if a.dry_run:
        print(json.dumps(selected,indent=2)); return
    import torch
    used={int(ids[i]) for i in (range(8) if a.lane=='all' else (range(8) if a.lane=='large' else range(4,8)))}
    if not torch.cuda.is_available() or max(used)>=torch.cuda.device_count(): p.error('selected GPUs are unavailable')
    if not glob.glob(a.train_glob) or not glob.glob(a.val_glob): p.error('training/validation files not found')
    if output.exists(): p.error('output root exists; choose a fresh path')
    (output/'console').mkdir(parents=True)
    (output/'manifest.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    results={}
    # The large model occupies all eight GPUs; never overlap the lanes.
    for name,stages in selected.items():
        try:
            results[name]=dict(status='PASS',jobs=run_lane(name,stages,output))
        except Exception as exc:
            results[name]=dict(status='FAIL',error=str(exc))
            break
    (output/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    if any(row['status']=='FAIL' for row in results.values()): raise SystemExit(1)

if __name__=='__main__': main()
