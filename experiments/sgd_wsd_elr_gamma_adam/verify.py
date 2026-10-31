"""Small real-token runs, resume equivalence and schedule checks. No long run."""
import contextlib
import io
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
import train


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    defaults=train.parser().parse_args(['--arm','fixed','--output','unused'])
    assert (defaults.validation_contexts, defaults.eval_every, defaults.dense_window)==(256,500,0)
    assert defaults.gamma_lr == .0003
    warm,decay=train.boundaries(defaults)
    assert abs(train.gamma_lr(defaults,0)-.0003/warm)<1e-18
    for step in (warm-1, warm, decay, defaults.updates-1):
        assert train.gamma_lr(defaults,step)==.0003
    report = {}
    with tempfile.TemporaryDirectory(prefix='smoke_', dir=Path(__file__).parent, ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        for name, count in [('train',769),('val',41)]:
            source = Path('F:/datasets/fineweb10B-gpt2/raw') / ('fineweb_train_000001.bin' if name=='train' else 'fineweb_val_000000.bin')
            tokens=np.fromfile(source,dtype=np.uint16,offset=1024,count=count)
            header=np.zeros(256,dtype=np.int32);header[:3]=[20240520,1,len(tokens)]
            with (root/f'{name}.bin').open('wb') as f:
                f.write(header.tobytes());f.write(tokens.tobytes())
        def args(arm,path,device='cpu'):
            return train.parser().parse_args(['--arm',arm,'--output',str(path),
                '--train',str(root/'train.bin'),'--val',str(root/'val.bin'),
                '--updates','24','--width','16','--layers','1','--heads','2',
                '--context','8','--batch','4','--validation-contexts','5',
                '--validation-batch-size','3','--eval-every','4',
                '--checkpoint-every','4','--dense-every','2','--dense-window','2',
                '--device',device])
        def run(a):
            with contextlib.redirect_stdout(io.StringIO()):train.run(a)
        def rows(path):
            return [{k:v for k,v in json.loads(line).items() if k!='seconds'}
                    for line in path.read_text().splitlines()]
        for arm,power in train.ARMS.items():
            full=root/(arm+'_full'); resumed=root/(arm+'_resume')
            a=args(arm,full)
            for i in range(25):
                s=train.scale_at(arm,i,24)
                assert .5 <= s <= 1.5
                eta=train.base_lr(a,min(i,23))
                assert abs(eta*s**power/s**power-eta)<1e-12
            run(a)
            original_save=train.save
            def interrupted(path,*values):
                original_save(path,*values)
                if values[2]==8:raise InterruptedError('test checkpoint interrupt')
            train.save=interrupted
            try:
                run(args(arm,resumed))
                raise AssertionError('interruption did not happen')
            except InterruptedError:pass
            finally:train.save=original_save
            for filename in ('steps.jsonl','metrics.jsonl'):
                with (resumed/filename).open('a') as f:
                    f.write('{"updates":9,"stale":true}\n{"updates":')
            run(args(arm,resumed))
            x=torch.load(full/'latest.pt',weights_only=False)
            y=torch.load(resumed/'latest.pt',weights_only=False)
            assert all(torch.equal(v,y['model'][k]) for k,v in x['model'].items())
            assert torch.equal(x['rng'],y['rng'])
            ax,ay=x['gamma_optimizer'],y['gamma_optimizer']
            assert ax['param_groups']==ay['param_groups']
            assert len(ax['state'])==5  # four block gamma vectors plus final gamma
            for key, state in ax['state'].items():
                assert set(state)=={'step','exp_avg','exp_avg_sq'}
                for field,value in state.items():
                    assert torch.equal(value,ay['state'][key][field])
                assert state['step'].item()==24
            assert all(r['gamma_lr']==a.gamma_lr for r in rows(full/'steps.jsonl')
                       if r['updates']>=train.boundaries(a)[0])
            assert x['position']==y['position']==96
            assert x['norm_targets']==y['norm_targets']
            for filename in ('steps.jsonl','metrics.jsonl'):
                assert rows(full/filename)==rows(resumed/filename),filename
            steps=rows(full/'steps.jsonl')
            assert [r['updates'] for r in steps]==list(range(1,25))
            maxerr=max(v.get('relative_error',0) for r in rows(full/'metrics.jsonl') for v in r['tensors'].values())
            assert maxerr<3e-6
            stable=torch.load(resumed/'stable_end.pt',weights_only=False)
            assert stable['updates']==train.boundaries(a)[1]==17
            assert stable['position']==17*a.batch
            assert all(torch.equal(v,torch.load(full/'stable_end.pt',weights_only=False)['model'][k]) for k,v in stable['model'].items())
            stable_digest=hashlib.sha256((resumed/'stable_end.pt').read_bytes()).hexdigest()
            before=(resumed/'steps.jsonl').read_bytes()
            run(args(arm,resumed))
            assert (resumed/'steps.jsonl').read_bytes()==before
            assert hashlib.sha256((resumed/'stable_end.pt').read_bytes()).hexdigest()==stable_digest
            bad=args(arm,resumed);bad.lr=.02
            try:run(bad);raise AssertionError('contract accepted changed LR')
            except ValueError:pass
            report[arm]=dict(cpu_resume_bitwise_equal=True,adam_moments_equal=True,logs_equal=True,stable_checkpoint_preserved=True,
                            max_norm_relative_error=maxerr,updates=24)
        if torch.cuda.is_available():
            torch.use_deterministic_algorithms(False)
            for arm in train.ARMS:
                out=root/(arm+'_cuda');run(args(arm,out,'cuda'))
                assert all(np.isfinite(r['train_loss']) for r in rows(out/'steps.jsonl'))
                report[arm]['cuda_24_updates_finite']=True
    Path(__file__).with_name('verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
