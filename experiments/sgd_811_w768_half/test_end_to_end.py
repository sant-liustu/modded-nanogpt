"""Small-model subprocess tests of the actual runner, including interruption."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-source', required=True)
    parser.add_argument('--val-source', required=True)
    cli = parser.parse_args()
    root = Path(cli.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    data = root/'data'
    data.mkdir()
    # Real prepared FineWeb tokens, tiny fixtures; intentionally split a context.
    def read(name, count):
        h = np.fromfile(name, dtype=np.int32, count=256)
        assert tuple(h[:2]) == (20240520, 1) and h[2] >= count
        return np.fromfile(name, dtype=np.uint16, offset=1024, count=count)
    def write(name, values):
        h = np.zeros(256, dtype=np.int32)
        h[:3] = 20240520, 1, len(values)
        with name.open('wb') as f:
            f.write(h.tobytes())
            f.write(values.tobytes())
    tokens = read(cli.train_source, 6401)
    write(data/'train_0.bin', tokens[:3001])
    write(data/'train_1.bin', tokens[3001:])
    write(data/'val.bin', read(cli.val_source, 529))
    base = dict(train=str(data/'train_*.bin'), val=str(data/'val.bin'), updates=100,
                lr=.01, batch=4, microbatch=None, validation_batch_size=32,
                context=16, width=16, heads=2, layers=1, seed=0,
                validation_contexts=33, eval_every=1, device='cpu', preflight=False)
    child = '''import json,sys,torch
from argparse import Namespace
from pathlib import Path
import train
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
args=json.loads(sys.argv[1]); stop=json.loads(sys.argv[2]); real=train.save
def save(path,*values):
    real(path,*values)
    if stop and Path(path).stem==stop[0] and values[1]==stop[1]:
        raise SystemExit(17)
train.save=save
train.run(Namespace(**args))
'''
    events = []
    def run(name, arm='all', stop=None, device='cpu', fail=False):
        args = dict(base, output=str(root/name), arm=arm, device=device)
        t = time.perf_counter()
        result = subprocess.run([sys.executable, '-c', child, json.dumps(args), json.dumps(stop)],
                                cwd=Path(__file__).parent, capture_output=True, text=True,
                                timeout=240, env=dict(os.environ, PYTHONIOENCODING='utf-8',
                                                      CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        (root/f'{len(events):02d}_{name}_{arm}.log').write_text(result.stdout+result.stderr, encoding='utf-8')
        expected = 17 if stop else 0
        if fail:
            assert result.returncode != 0 and ('contract' in result.stderr or
                                               'different experiment' in result.stderr), result.stderr
        else:
            assert result.returncode == expected, result.stderr
        events.append(dict(run=name,arm=arm,stop=stop,code=result.returncode,
                           seconds=time.perf_counter()-t))
        print(events[-1], flush=True)

    run('continuous')
    run('resumed', 'prefix', ['prefix', 40])
    # Simulate a log flushed after the saved checkpoint; it must be removed.
    with (root/'resumed/prefix.jsonl').open('a') as f:
        f.write(json.dumps(dict(macro=41, injected_stale_record=True))+'\n')
    with (root/'resumed/prefix_steps.jsonl').open('a') as f:
        f.write(json.dumps(dict(updates=41, injected_stale_record=True))+'\n')
        f.write('{"updates":42')  # interrupted write of the last line
    run('resumed', 'prefix')
    run('resumed', 'lr', ['lr', 79])
    run('resumed', 'lr')
    run('resumed', 'batch', ['batch', 81])
    run('resumed', 'batch')
    stats = {}
    for arm in ('prefix','lr','batch'):
        a=torch.load(root/'continuous'/f'{arm}.pt',map_location='cpu',weights_only=False)
        b=torch.load(root/'resumed'/f'{arm}.pt',map_location='cpu',weights_only=False)
        for key in ('macro','position','updates'):
            assert a[key]==b[key]
        assert torch.equal(a['rng'], b['rng'])
        for key in a['model']:
            assert torch.equal(a['model'][key],b['model'][key]), (arm,key)
        def rows(folder):
            values=[json.loads(s) for s in (root/folder/f'{arm}.jsonl').read_text().splitlines()]
            assert not any('injected_stale_record' in x for x in values)
            for x in values:
                x.pop('seconds',None)
            return {x['macro']:x for x in values}
        assert rows('continuous') == rows('resumed'), arm
        def steps(folder):
            return [json.loads(s) for s in (root/folder/f'{arm}_steps.jsonl').read_text().splitlines()]
        assert steps('continuous')==steps('resumed'), arm
        step_rows=steps('resumed')
        expected_first=1 if arm=='prefix' else 79
        assert [r['updates'] for r in step_rows]==list(range(expected_first,a['updates']+1))
        assert all(np.isfinite(r['train_loss']) for r in step_rows)
        stats[arm] = {k:a[k] for k in ('macro','position','updates')}
    assert stats['lr']==dict(macro=82,position=400,updates=100)
    assert stats['batch']==dict(macro=82,position=400,updates=82)
    lr_rows=[json.loads(s) for s in (root/'continuous/lr.jsonl').read_text().splitlines()]
    bs_rows=[json.loads(s) for s in (root/'continuous/batch.jsonl').read_text().splitlines()]
    for a,b in zip(lr_rows,bs_rows):
        for k in ('macro','T','contexts','tokens','ratio'):
            assert a[k]==b[k], (k,a,b)
    assert {r['batch'] for r in bs_rows}=={4,16,40}
    assert lr_rows[0]['validation_loss']==bs_rows[0]['validation_loss']
    run('resumed')  # completed run can be safely reopened
    old=base['lr']; base['lr']=.02
    run('resumed', fail=True)  # changed scientific settings must not silently resume
    base['lr']=old
    if torch.cuda.is_available():
        run('gpu', device='cuda')
        for arm in ('prefix','lr','batch'):
            state=torch.load(root/'gpu'/f'{arm}.pt',map_location='cpu',weights_only=False)
            assert all(torch.isfinite(v).all() for v in state['model'].values())
            for line in (root/'gpu'/f'{arm}.jsonl').read_text().splitlines():
                assert np.isfinite(json.loads(line)['validation_loss'])
    report=dict(status='passed', model=dict(width=16,depth=1,heads=2,context=16),
                actual_updates=100, validation_contexts=33, validation_batch=32,
                microbatch=None, final_states=stats, events=events,
                checks=['real-token shard crossing','full batches 4/16/40',
                        'validation batch 32 plus remainder 1','shared checkpoint fork',
                        'prefix and both tails interrupted after checkpoint',
                        'CPU resumed weights and logs exactly equal to continuous',
                        'data cursor and ratio-path pairing','stale log rollback',
                        'every-update logs equal after resume; partial final line recovered',
                        'completed-run reopening','mismatched config rejection',
                        'CUDA full-chain finite checks when available'])
    (root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS: full-chain report',root/'report.json',flush=True)


if __name__=='__main__':
    main()
