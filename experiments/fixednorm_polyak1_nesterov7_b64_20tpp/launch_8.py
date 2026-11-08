"""Launch eight independent single-GPU jobs; this launcher uses only stdlib."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, help='Quoted training shard glob')
    parser.add_argument('--val', required=True, help='Quoted validation shard glob')
    parser.add_argument('--output-root', type=Path, default=ROOT/'runs')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    gpus = [x.strip() for x in args.gpus.split(',')]
    if len(gpus) != 8 or any(not x for x in gpus) or len(set(gpus)) != 8:
        parser.error('Provide exactly eight distinct GPU IDs')
    manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    plans = []
    for job, gpu in zip(manifest['jobs'], gpus):
        script = ROOT/job['script']
        if hashlib.sha256(script.read_bytes()).hexdigest() != job['sha256']:
            raise ValueError(f'Training script changed: {script}')
        cmd = [sys.executable, str(script), '--train', args.train, '--val', args.val,
               '--output', str(args.output_root.resolve()/job['id'])]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        plans.append((job, gpu, cmd, env))
        print(f'CUDA_VISIBLE_DEVICES={gpu} {shlex.join(cmd)}', flush=True)
    if args.dry_run:
        return
    # Check all data/config contracts before any training begins.
    for _, _, cmd, env in plans:
        subprocess.run(cmd+['--preflight'], env=env, cwd=ROOT, check=True)
    logdir = args.output_root.resolve()/'console'
    logdir.mkdir(parents=True, exist_ok=True)
    processes = []
    try:
        for job, gpu, cmd, env in plans:
            with (logdir/(job['id']+'.log')).open('a', encoding='utf-8') as log:
                process = subprocess.Popen(cmd, env=env, cwd=ROOT,
                                           stdout=log, stderr=subprocess.STDOUT)
            processes.append((job['id'], gpu, process))
        results = []
        for run_id, gpu, process in processes:
            code = process.wait()
            results.append(dict(id=run_id, gpu=gpu, returncode=code))
            print(f'{run_id}: GPU={gpu}, exit={code}', flush=True)
        (args.output_root.resolve()/'launcher_results.json').write_text(
            json.dumps(results, indent=2), encoding='utf-8')
        if any(row['returncode'] for row in results):
            raise SystemExit(1)
    except BaseException:
        # Only stop children owned by this launcher, not other training processes.
        for _, _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, _, process in processes:
            process.wait()
        raise


if __name__ == '__main__':
    main()
