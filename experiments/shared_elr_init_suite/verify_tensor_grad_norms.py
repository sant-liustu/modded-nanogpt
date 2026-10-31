"""Gradient attribution: known norms and equivalence with norm observation disabled."""
from pathlib import Path
import json
import math
import subprocess
import sys
import time
import torch
import train_llama025b_fineweb as run

torch.set_num_threads(2)
root=Path(__file__).parent
out=root/'local_checks'/('tensor_grad_'+time.strftime('%Y%m%d_%H%M%S'))
out.mkdir(parents=True)
model=run.Llama(run.LlamaConfig(17,16,64,1,2))
for p in model.parameters(): p.grad=torch.zeros_like(p)
model.model.embed_tokens.weight.grad.flatten()[:2]=torch.tensor([3.,4.])
model.lm_head.weight.grad.flatten()[0]=12.
saved=[p.grad.clone() for p in model.parameters()]
rng=torch.get_rng_state().clone()
rows=run.tensor_gradient_norms(model)
assert rows['model.embed_tokens.weight']['grad_norm']==5.
assert rows['lm_head.weight']['grad_norm']==12.
assert abs(rows['lm_head.weight']['squared_norm_fraction']-144/169)<1e-12
assert abs(sum(v['squared_norm_fraction'] for v in rows.values())-1)<1e-12
assert torch.equal(rng,torch.get_rng_state())
for p,g in zip(model.parameters(),saved): assert torch.equal(p.grad,g)
total=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
assert total.item()==13.
coefficient=(1/(total+1e-6)).clamp(max=1).item()
for name,p in model.named_parameters():
    assert abs(p.grad.norm().item()-rows[name]['grad_norm']*coefficient)<1e-7
for p in model.parameters(): p.grad.zero_()
assert all(v['grad_norm']==v['grad_rms']==v['squared_norm_fraction']==0 for v in run.tensor_gradient_norms(model).values())

# Portable baseline: disable only per-tensor observation; no unpublished Git SHA needed.
baseline=out/'baseline.py'
source=(root/'train_llama025b_fineweb.py').read_text(encoding='utf8')
observed='per_tensor_grads = tensor_gradient_norms(raw) if rank == 0 else None'
assert source.count(observed)==1
baseline.write_text(source.replace(observed,'per_tensor_grads = {} if rank == 0 else None'),encoding='utf8')
common=['--input-bin','F:/datasets/fineweb10B-gpt2/raw/fineweb_train_000001.bin',
        '--hidden-size','32','--intermediate-size','128','--num-layers','1','--num-heads','4',
        '--device','cuda' if torch.cuda.is_available() else 'cpu','--expected-world-size','1',
        '--batch-size','2','--micro-batch-size','1','--sequence-length','8','--stop-step','4',
        '--warmup-steps','1','--val-every','0','--fisher-every','0']
states=[]
for name,script in [('baseline',baseline),('logged',root/'train_llama025b_fineweb.py')]:
    with open(out/(name+'.log'),'w') as f:
        subprocess.run([sys.executable,str(script),*common,'--output',str(out/name)],stdout=f,stderr=subprocess.STDOUT,check=True)
    states.append(torch.load(out/name/'checkpoints'/'step000004'/'rank0.pt',weights_only=False))
def equal(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b
for key in ('model','optimizer','data_state','torch_rng','cuda_rng','python_rng'):
    equal(states[0][key],states[1][key])
logs=[json.loads(s) for s in (out/'logged'/'tensor_grad_norm_history.jsonl').read_text().splitlines()]
summary=[json.loads(s) for s in (out/'logged'/'train_history.jsonl').read_text().splitlines()]
assert [v['step'] for v in logs]==[1,2,3,4]
names=set(states[1]['model'])
for row,train in zip(logs,summary):
    assert set(row['parameters'])==names
    assert row['global_grad_norm']==train['grad_norm']
    assert row['clip_coef']==train['clip_coef'] and row['clipped']==train['clipped']
    values=row['parameters'].values()
    recovered=math.sqrt(sum(v['grad_norm']**2 for v in values))
    assert math.isclose(recovered,row['global_grad_norm'],rel_tol=1e-6)
    assert abs(sum(v['squared_norm_fraction'] for v in values)-1)<1e-12
    assert all(math.isfinite(v['grad_rms']) for v in values)
report=dict(known_gradients=True,zero_gradients=True,non_mutating=True,all_steps_all_tensors=True,
            global_norm_reconstruction=True,training_matches_observer_disabled_exactly=True,steps=4)
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(output=str(out),**report),indent=2))
