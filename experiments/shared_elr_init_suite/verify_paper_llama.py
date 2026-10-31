"""Small correctness checks; Transformers is optional and used only as a reference."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
import train_llama025b_fineweb as run

torch.set_num_threads(2)
root = Path(__file__).parent
out = root/'local_checks'/('paper_llama_'+time.strftime('%Y%m%d_%H%M%S'))
out.mkdir(parents=True)
results = []


def equal(a, b):
    if isinstance(a, torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a, dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a, (tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b


def math_checks(device, bf16):
    from transformers import LlamaConfig, LlamaForCausalLM
    c = run.LlamaConfig(vocab_size=41,hidden_size=32,intermediate_size=128,
                        num_hidden_layers=2,num_attention_heads=4)
    torch.manual_seed(19)
    model = run.Llama(c).to(device)
    config=LlamaConfig(**run.asdict(c),attention_dropout=0.0,tie_word_embeddings=False,use_cache=False)
    config._attn_implementation='sdpa'
    reference=LlamaForCausalLM(config).to(device)
    reference.load_state_dict(model.state_dict(),strict=True)
    assert model.lm_head.weight is not model.model.embed_tokens.weight
    x=torch.randint(41,(3,5),device=device)
    y=torch.randint(41,(3,5),device=device)
    with torch.autocast(device,dtype=torch.bfloat16,enabled=bf16):
        z,loss=model(x,y)
        zr=reference(x).logits
        lr=run.F.cross_entropy(zr.float().reshape(-1,41),y.reshape(-1))
    tolerance=0.02 if bf16 else 2e-5
    torch.testing.assert_close(z.float(),zr.float(),atol=tolerance,rtol=tolerance)
    g=torch.autograd.grad(loss,tuple(model.parameters()))
    gr=torch.autograd.grad(lr,tuple(reference.parameters()))
    for a,b in zip(g,gr): torch.testing.assert_close(a,b,atol=tolerance,rtol=tolerance)
    # Probe must leave existing gradients, modes and global RNG unchanged.
    for v in model.parameters(): v.grad=torch.randn_like(v)
    old_grads=[v.grad.clone() for v in model.parameters()]
    old=copy.deepcopy(model.state_dict())
    rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    y[1,2:]=-1
    record=run.probe(model,x,y,2,53,run.parameter_rms(model),2,False,3,use_bf16=bf16)
    assert set(record['groups'])=={'Emb','Head','Norm','QK','VO','FFN'}
    equal(old,model.state_dict())
    assert torch.equal(rng,torch.get_rng_state()) and model.training
    if cuda_rng is not None: assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
    for v,saved in zip(model.parameters(),old_grads): assert torch.equal(v.grad,saved)
    # Reference whole differentiable objective with exactly the same sampled labels.
    worst=0.0
    for draw,seed in enumerate((53,54)):
        generator=torch.Generator(device=device).manual_seed(seed)
        objective=0
        with torch.autocast(device,dtype=torch.bfloat16,enabled=bf16):
            for b in range(0,len(x),2):
                hidden=model(x[b:b+2],return_hidden=True).reshape(-1,32)
                mask=y[b:b+2].reshape(-1)!=-1
                for t in range(0,len(hidden),3):
                    valid=mask[t:t+3]
                    if not valid.any(): continue
                    logits=model.lm_head(hidden[t:t+3])
                    labels=torch.multinomial(logits.detach().float().softmax(-1),1,generator=generator).squeeze(-1)
                    labels[~valid]=-1
                    objective+=run.F.cross_entropy(logits.float(),labels,ignore_index=-1,reduction='sum')/record['valid_tokens']
            gradients=torch.autograd.grad(objective,tuple(model.parameters()))
        for (name,v),grad in zip(model.named_parameters(),gradients):
            expected=record['valid_tokens']*grad.double().square().sum().item()/v.numel()
            actual=record['parameters'][name]['mean_fisher_draws'][draw]
            error=abs(actual-expected)/max(expected,1e-15)
            worst=max(worst,error)
            assert error < (0.02 if bf16 else 2e-5),(name,error)
    results.append(dict(test='HF_forward_backward_and_Fisher_reference',device=device,bf16=bf16,max_fisher_relative_error=worst))


def sequential_reader_checks():
    import numpy as np
    directory=out/'reader'
    directory.mkdir()
    arrays=[np.arange(29,dtype=np.uint16),np.arange(10,39,dtype=np.uint16)]
    for i,values in enumerate(arrays):
        header=np.zeros(256,dtype=np.int32); header[:3]=[20240520,1,len(values)]
        with open(directory/f'{i}.bin','wb') as f:
            f.write(header.tobytes()); f.write(values.tobytes())
    readers=[run.TokenData(str(directory/'*.bin'),4,41,r,2) for r in range(2)]
    # 16 input tokens per global microbatch; the short remainder is dropped together.
    for shard in range(2):
        for rank,reader in enumerate(readers):
            x,y=reader.batch(2)
            expected=torch.from_numpy(arrays[shard].astype(np.int64))
            begin=rank*8
            assert torch.equal(x.reshape(-1),expected[begin:begin+8])
            assert torch.equal(y.reshape(-1),expected[begin+1:begin+9])
            assert reader.shard==(shard+1)%2 and reader.position==0
    assert all(reader.epoch==1 for reader in readers)
    for rank,reader in enumerate(readers):
        state=reader.state_dict()
        restored=run.TokenData(str(directory/'*.bin'),4,41,rank,2)
        restored.load_state_dict(state)
        equal(reader.batch(2),restored.batch(2))
    results.append(dict(test='sequential_reader',rank_slices_disjoint=True,
                        shard_order=True,tail_drop=True,wrap_and_cursor_resume=True))


sequential_reader_checks()
math_checks('cpu',False)
if torch.cuda.is_available():
    math_checks('cuda',False)
    math_checks('cuda',True)

# Schedule must retain 50k horizon despite 20k cutoff; official first update LR=0.
assert run.learning_rate(0)==0 and run.learning_rate(1000)==8e-4
assert abs(run.learning_rate(50000)-4e-5)<1e-15
assert run.learning_rate(19999)>4e-5
for it in range(50001):
    expected=8e-4*it/1000 if it<=1000 else 4e-5+0.5*(1+run.math.cos(run.math.pi*(it-1000)/49000))*7.6e-4
    assert abs(run.learning_rate(it)-expected)<1e-15

common=[sys.executable,str(root/'train_llama025b_fineweb.py'),
        '--input-bin','F:/datasets/fineweb10B-gpt2/raw/fineweb_train_000001.bin',
        '--hidden-size','32','--intermediate-size','128','--num-layers','1','--num-heads','4',
        '--expected-world-size','1','--batch-size','2','--micro-batch-size','1','--sequence-length','8',
        '--device','cuda' if torch.cuda.is_available() else 'cpu','--warmup-steps','1',
        '--val-every','0','--fisher-sequences','4','--fisher-micro-batch-size','2',
        '--fisher-token-chunk-size','4','--save-every','2']
def launch(name,stop,every,resume=None):
    command=common+['--output',str(out/name),'--stop-step',str(stop),'--fisher-every',str(every)]
    if resume: command+=['--resume',str(resume)]
    with open(out/(name+'.log'),'w',encoding='utf8') as f:
        subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    return torch.load(out/name/'checkpoints'/f'step{stop:06d}'/'rank0.pt',weights_only=False)

off=launch('off',4,0)
on=launch('on',4,2)
for key in ('model','optimizer','data_state','torch_rng','cuda_rng'): equal(off[key],on[key])
resumed=launch('resume',4,2,out/'on'/'checkpoints'/'step000002')
for key in ('model','optimizer','data_state','torch_rng','cuda_rng'): equal(on[key],resumed[key])
rows=[json.loads(v) for v in (out/'on'/'fisher_history.jsonl').read_text().splitlines()]
assert [v['step'] for v in rows]==[0,2,4]
for row in rows:
    snapshot=torch.load(out/'on'/row['model_checkpoint'],weights_only=False)
    assert snapshot['step']==row['step']
    assert snapshot['initial_rms']==on['initial_rms']
    for key in row['parameters']:
        assert row['parameters'][key]['initial_rms']==on['initial_rms'][key]
train=[json.loads(v) for v in (out/'on'/'train_history.jsonl').read_text().splitlines()]
assert any(v['grad_norm']>1 for v in train),'Clipping should be exercised by this test'
for group in on['optimizer']['param_groups']:
    assert group['weight_decay']==0.1 and group['betas']==(0.9,0.95) and group['eps']==1e-8
assert off['step']==on['step']==4
results.append(dict(test='real_FineWeb_small_end_to_end',steps=4,measurement_state_equivalence=True,
                    exact_resume=True,clipping_exercised=True,snapshots_and_RMS=True))
(out/'report.json').write_text(json.dumps(results,indent=2))
print(json.dumps(dict(output=str(out),results=results),indent=2))
