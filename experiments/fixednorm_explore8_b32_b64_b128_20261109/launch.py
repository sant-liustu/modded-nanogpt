"""Run eight single-GPU trials in waves, one process per selected GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',required=True)
    parser.add_argument('--val',required=True)
    parser.add_argument('--output-root',type=Path,default=ROOT/'runs_explore8')
    parser.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    parser.add_argument('--loss-chunk-tokens',type=int,default=1024)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    gpus=[x.strip() for x in args.gpus.split(',')]
    if any(not x for x in gpus) or len(set(gpus))!=len(gpus):
        parser.error('Provide distinct nonempty GPU IDs')
    if args.loss_chunk_tokens<=0:
        parser.error('Use a positive head-loss chunk size')
    jobs=json.loads((ROOT/'manifest.json').read_text())['jobs']
    plans=[]
    for index,job in enumerate(jobs):
        script=ROOT/job['script']
        if hashlib.sha256(script.read_bytes()).hexdigest()!=job['sha256']:
            raise ValueError(f'Script changed: {script}')
        gpu=gpus[index%len(gpus)]
        cmd=[sys.executable,str(script),'--train',args.train,'--val',args.val,
             '--output',str(args.output_root.resolve()/job['id']),
             '--loss-chunk-tokens',str(args.loss_chunk_tokens)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
        plans.append((job,gpu,cmd,env))
        print(f'CUDA_VISIBLE_DEVICES={gpu} {shlex.join(cmd)}',flush=True)
    if args.dry_run:
        return
    for _,_,cmd,env in plans:
        subprocess.run(cmd+['--preflight'],cwd=ROOT,env=env,check=True)
    logdir=args.output_root.resolve()/'console'
    logdir.mkdir(parents=True,exist_ok=True)
    results=[]
    for start in range(0,len(plans),len(gpus)):
        children=[]
        try:
            for job,gpu,cmd,env in plans[start:start+len(gpus)]:
                with (logdir/(job['id']+'.log')).open('a',encoding='utf-8') as log:
                    process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                children.append((job['id'],gpu,process))
            for run_id,gpu,process in children:
                code=process.wait()
                results.append(dict(id=run_id,gpu=gpu,returncode=code))
                print(f'{run_id}: GPU={gpu}, exit={code}',flush=True)
            (args.output_root.resolve()/'launcher_results.json').write_text(
                json.dumps(results,indent=2),encoding='utf-8')
            if any(code['returncode'] for code in results):
                raise SystemExit(1)
        except BaseException:
            for _,_,process in children:
                if process.poll() is None:
                    process.terminate()
            for _,_,process in children:
                process.wait()
            raise


if __name__=='__main__':
    main()
