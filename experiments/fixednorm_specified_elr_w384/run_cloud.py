"""Launch w384/w768/w1536 on disjoint 1/2/4 GPU groups, then audit their data order."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import uuid


def build_jobs(here, output, gpu_ids, args):
    jobs = []
    for width, devices, batch in [(384, gpu_ids[:1], 128), (768, gpu_ids[1:3], 64), (1536, gpu_ids[3:7], 32)]:
        command = [sys.executable]
        if len(devices) > 1:
            command += ['-m', 'torch.distributed.run', '--standalone',
                        '--rdzv_backend=c10d', '--rdzv_endpoint=localhost:0',
                        f'--nproc_per_node={len(devices)}']
        command += [str(here / 'train.py'), f'--model_width={width}', '--batch_size=128',
                    f'--device_batch_size={batch}', '--sequence_length=1024',
                    '--num_iterations=10200', f'--train_shards={args.shards}',
                    f'--data_seed={args.data_seed}', '--seed=0', '--shuffle_each_epoch=1',
                    '--elr=0.03', '--elr_warmup_iters=500', '--elr_warmdown_iters=2900',
                    '--gamma_lr=0.0018', '--gamma_warmup_iters=500',
                    '--val_loss_every=100', '--val_tokens=10485760',
                    f'--compile_model={args.compile_model}', f'--save_every={args.save_every}',
                    '--spectral_norm_estimate_enabled=0', '--tensor_norm_every=4',
                    '--adamw_update_norm_every=4', '--activation_probe_every=0',
                    f'--input_bin={args.input_bin}', f'--input_val_bin={args.input_val_bin}',
                    f'--log_root={output / str(width)}', f'--run_name=w{width}']
        jobs.append(dict(width=width, gpu_ids=devices, command=command, status='planned'))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6', help='Seven distinct indices within the GPUs visible to this launcher')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--data-seed', type=int, default=20261012)
    parser.add_argument('--input-bin', default='data/fineweb10B/fineweb_train_*.bin')
    parser.add_argument('--input-val-bin', default='data/fineweb10B/fineweb_val_*.bin')
    parser.add_argument('--compile-model', type=int, choices=[0, 1], default=1)
    parser.add_argument('--save-every', type=int, default=0, help='0 saves only the final checkpoint')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    ids = args.gpus.split(',')
    if len(ids) != 7 or len(set(ids)) != 7 or any(not v.isdigit() for v in ids):
        parser.error('--gpus requires seven distinct nonnegative integer indices')
    if args.shards < 0 or args.data_seed < 0 or args.save_every < 0:
        parser.error('shards, data seed and save interval must be nonnegative')
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    output = repo / 'logs/fixednorm_specified_elr_suite' / str(uuid.uuid4())
    jobs = build_jobs(here, output, ids, args)
    if args.dry_run:
        print(json.dumps(dict(cwd=str(repo), output=str(output), jobs=jobs), indent=2))
        return
    # Preflight before spawning any training process.
    import torch
    from data_order import EpochSampleDataLoader
    from verify_run_order import compare
    if not torch.cuda.is_available() or max(map(int, ids)) >= torch.cuda.device_count():
        parser.error('selected GPU indices are unavailable; this suite requires seven visible GPUs')
    train = EpochSampleDataLoader(str(repo / args.input_bin), 1, 1024, 0, 1,
                                  max_shards=args.shards, data_seed=args.data_seed, device='cpu')
    data_order = train.order_metadata()
    train.close()
    import glob
    validation = sorted(glob.glob(str(repo / args.input_val_bin)))
    if not validation:
        parser.error('validation files not found')
    for path in validation:
        import numpy as np
        with open(path, 'rb') as f:
            header = np.frombuffer(f.read(1024), dtype='<i4')
        if len(header) != 256 or tuple(header[:2]) != (20240520, 1) or int(header[2]) < 128 * 1024 + 1:
            parser.error(f'validation shard has invalid header or is too short: {path}')
        if Path(path).stat().st_size != 1024 + int(header[2]) * 2:
            parser.error(f'validation shard byte count does not match header: {path}')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    visible_tokens = visible.split(',') if visible else None
    output.mkdir(parents=True)
    manifest = output / 'suite_manifest.json'
    def save():
        manifest.write_text(json.dumps(dict(data_order=data_order, jobs=jobs), indent=2) + '\n', encoding='utf-8')
    save()
    print(f'Suite output: {output}', flush=True)
    processes, streams = [], []
    try:
        for job in jobs:
            env = os.environ.copy()
            # Avoid inheriting an outer distributed launch. Run this entry with plain python.
            for name in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
                env.pop(name, None)
            masks = [visible_tokens[int(i)] if visible_tokens else i for i in job['gpu_ids']]
            env['CUDA_VISIBLE_DEVICES'] = ','.join(masks)
            env['PYTHONUNBUFFERED'] = '1'
            stream = (output / f"w{job['width']}.console.log").open('w', encoding='utf-8')
            streams.append(stream)
            process = subprocess.Popen(job['command'], cwd=repo, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=(os.name != 'nt'))
            processes.append(process)
            job.update(status='running', pid=process.pid, cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'])
            save()
        while True:
            for job, process in zip(jobs, processes):
                code = process.poll()
                if code is not None and job['status'] == 'running':
                    job.update(exit_code=code, status='completed' if code == 0 else 'failed')
                    save()
                    if code:
                        raise RuntimeError(f"w{job['width']} failed (exit {code}); see its console log")
            if all(job['status'] == 'completed' for job in jobs):
                break
            time.sleep(1)
        run_dirs = []
        for job in jobs:
            directories = list((output / str(job['width'])).glob('*/data_order_audit.jsonl'))
            if len(directories) != 1:
                raise RuntimeError(f"missing/ambiguous audit output for w{job['width']}")
            run_dirs.append(directories[0].parent)
        try:
            checks = [compare(run_dirs[0], run_dirs[1]), compare(run_dirs[0], run_dirs[2])]
        except ValueError as exc:
            (output / 'data_order_verification.json').write_text(json.dumps(dict(status='FAIL', error=str(exc)), indent=2), encoding='utf-8')
            raise
        (output / 'data_order_verification.json').write_text(json.dumps(dict(status='PASS', checks=checks), indent=2), encoding='utf-8')
        print('PASS: all three runs completed and actual consumed data order matches.', flush=True)
    finally:
        for job, process in zip(jobs, processes):
            if process.poll() is None:
                if os.name == 'nt':
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name == 'nt':
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                job.update(status='interrupted', exit_code=process.returncode)
        for stream in streams:
            stream.close()
        save()


if __name__ == '__main__':
    main()
