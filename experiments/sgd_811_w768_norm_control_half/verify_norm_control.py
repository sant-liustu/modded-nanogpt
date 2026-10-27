import math
"""Standalone CPU checks: projection, paired training and exact resume."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
import torch
import train
from model import GPT, GPTConfig


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = GPT(GPTConfig(n_layer=1,n_head=2,n_embd=16,vocab_size=256))
    targets = train.initialize_norm_targets(model)
    assert model.transformer.wte.weight is model.lm_head.weight
    assert len(targets) == len({id(p) for p in model.parameters() if p.ndim == 2})
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(1.7)
    train.project_matrix_norms(model)
    assert train.norm_control_error(model)<2e-6
    for name,p in model.named_parameters():
        if p.ndim == 2:
            torch.testing.assert_close(p,before[name],rtol=2e-6,atol=2e-7)
        else:
            assert name not in targets
            torch.testing.assert_close(p,before[name]*1.7)
    g,prefix,_=train.schedule(301728)
    assert prefix==241386 and len(g)==251946 and g.sum()==301728
    assert set(g)=={1,4,10}
    events=[]
    with tempfile.TemporaryDirectory(prefix='smoke_verify_',dir=Path(__file__).parent) as temp:
        root=Path(temp)
        def write(name, values):
            header=np.zeros(256,dtype=np.int32)
            header[:3]=20240520,1,len(values)
            with name.open('wb') as f:
                f.write(header.tobytes()); f.write(values.astype(np.uint16).tobytes())
        rng=np.random.default_rng(17)
        data=rng.integers(0,256,size=6401,dtype=np.uint16)
        write(root/'train0.bin',data[:3001])
        write(root/'train1.bin',data[3001:])
        write(root/'val.bin',rng.integers(0,256,size=529,dtype=np.uint16))
        # Verify independent 200k checkpoint includes targets and cannot overwrite.
        contract={'norm_control_policy':train.NORM_CONTROL_POLICY}
        assert train.save_prefix_milestone(root,model,'prefix',199999,799996,199999,contract) is None
        milestone=train.save_prefix_milestone(root,model,'prefix',200000,800000,200000,contract)
        stamp=milestone.stat().st_mtime_ns
        train.save_prefix_milestone(root,model,'prefix',200000,800000,200000,contract)
        assert milestone.stat().st_mtime_ns==stamp
        state=torch.load(milestone,weights_only=False)
        assert state['norm_targets']==targets
        train.restore_norm_targets(model,state['norm_targets'])
        try:
            train.restore_norm_targets(model,None)
            raise AssertionError('missing targets accepted')
        except ValueError:
            pass
        child='''import json,sys,torch
from argparse import Namespace
from pathlib import Path
import train
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
args=json.loads(sys.argv[1]); stop=json.loads(sys.argv[2])
original_update=train.update
def update(*a,**kw):
    result=original_update(*a,**kw)
    assert train.norm_control_error(a[0])<2e-6
    return result
train.update=update
original_save=train.save
def save(path,*a):
    original_save(path,*a)
    if stop and Path(path).stem==stop[0] and a[1]==stop[1]:
        raise SystemExit(17)
train.save=save
train.run(Namespace(**args))
'''
        common=dict(train=str(root/'train*.bin'),val=str(root/'val.bin'),updates=100,
                    lr=.01,batch=4,microbatch=None,validation_batch_size=32,
                    context=16,width=16,layers=1,heads=2,seed=0,
                    validation_contexts=33,eval_every=1,device='cpu',preflight=False)
        def run(folder,arm='all',stop=None):
            start=time.perf_counter()
            result=subprocess.run([sys.executable,'-c',child,
                                   json.dumps(dict(common,output=str(root/folder),arm=arm)),json.dumps(stop)],
                                  cwd=Path(__file__).parent,capture_output=True,text=True,timeout=180)
            assert result.returncode==(17 if stop else 0), result.stderr
            events.append(dict(folder=folder,arm=arm,interrupt=stop,seconds=time.perf_counter()-start))
            print('PASS',events[-1],flush=True)
        run('continuous')
        run('resumed','prefix',['prefix',40])
        run('resumed','prefix')
        run('resumed','lr',['lr',79])
        run('resumed','lr')
        run('resumed','batch',['batch',81])
        run('resumed','batch')
        max_error=0.
        final={}
        for arm in ('prefix','lr','batch'):
            a=torch.load(root/'continuous'/f'{arm}.pt',weights_only=False)
            b=torch.load(root/'resumed'/f'{arm}.pt',weights_only=False)
            assert a['norm_targets']==b['norm_targets']
            for key in ('macro','position','updates'):
                assert a[key]==b[key]
            for name,value in a['model'].items():
                assert torch.equal(value,b['model'][name]),(arm,name)
            rows=[]
            for folder in ('continuous','resumed'):
                entries=[json.loads(line) for line in (root/folder/f'{arm}.jsonl').read_text().splitlines()]
                for row in entries:
                    row.pop('seconds')
                    assert row['norm_control_max_relative_error']<2e-6
                    max_error=max(max_error,row['norm_control_max_relative_error'])
                rows.append({row['macro']:row for row in entries})
            assert rows[0]==rows[1],arm
            train_rows = [[json.loads(line) for line in
                           (root/folder/f'{arm}_train.jsonl').read_text().splitlines()]
                          for folder in ('continuous','resumed')]
            assert train_rows[0] == train_rows[1], (arm, 'per-update training logs')
            first_update = 1 if arm == 'prefix' else final['prefix']['updates']+1
            assert [row['updates'] for row in train_rows[0]] == list(range(first_update,a['updates']+1))
            assert all(math.isfinite(row['train_loss']) for row in train_rows[0])
            final[arm]=dict(updates=a['updates'],contexts=a['position'],macro=a['macro'])
        lr=json.loads((root/'continuous/lr.jsonl').read_text().splitlines()[0])
        bs=json.loads((root/'continuous/batch.jsonl').read_text().splitlines()[0])
        assert lr['validation_loss']==bs['validation_loss']
        lrrows=[json.loads(line) for line in (root/'continuous/lr.jsonl').read_text().splitlines()]
        bsrows=[json.loads(line) for line in (root/'continuous/batch.jsonl').read_text().splitlines()]
        for l,b in zip(lrrows,bsrows):
            for key in ('T','contexts','tokens','ratio'):
                assert l[key]==b[key]
        assert {row['batch'] for row in bsrows}=={4,16,40}
        assert final['lr']==dict(updates=100,contexts=400,macro=82)
        assert final['batch']==dict(updates=82,contexts=400,macro=82)
        run('resumed')
    report=dict(status='passed',device='cpu',test_model=dict(width=16,layers=1,heads=2,context=16),
                fixture='synthetic token shards',canonical_updates=100,validation_batch=32,
                max_recorded_relative_rms_error=max_error,events=events,final_states=final,
                checks=['matrix projection after every update','tied matrix counted once',
                        'gamma excluded from projection','200k milestone targets and no overwrite',
                        'missing targets rejected','full global batches 4/16/40',
                        'CPU continuous and interrupted/resumed weights bitwise equal',
                        'resumed JSONL identical except timing','paired clocks/data/ratios',
                        'every update loss retained and resume logs identical','same controlled prefix fork','completed run can be reopened'],
                limitations='No formal width-768 run or GPU/DDP validation; no claim of loss collapse.')
    Path(__file__).with_name('verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
