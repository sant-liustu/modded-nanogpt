"""Actual tied-head GPT: chain-rule equivalence and CUDA memory comparison."""
from pathlib import Path
import ast, json, math, types
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from fisher_diagnostics import probe, parameter_rms, _accumulate_hidden_head

torch.set_num_threads(2)
root=Path(__file__).parent
names={'Rotary','apply_rotary_emb','RMSNorm','CausalSelfAttention','MLP','Block','GPTConfig','GPT'}
source=ast.parse((root/'train.py').read_text())
module=ast.Module(body=[n for n in source.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names],type_ignores=[])
ns=dict(torch=torch,nn=torch.nn,F=torch.nn.functional,math=math,dataclass=dataclass,
        args=SimpleNamespace(arm='fanin_fixed'),__name__='__main__')
exec(compile(module,'train.py','exec'),ns)

def correctness(device,use_bf16):
    torch.manual_seed(27)
    m=ns['GPT'](ns['GPTConfig'](vocab_size=32,n_layer=1,n_head=1,n_embd=128)).to(device)
    x=torch.randint(32,(3,5),device=device)
    y=torch.zeros_like(x); y[1,2:]=-1
    for p in m.parameters(): p.grad=torch.randn_like(p)
    saved=[p.grad.clone() for p in m.parameters()]
    rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    head_rows=[]
    hook=m.lm_head.register_forward_hook(lambda module,inputs,output: head_rows.append(inputs[0].numel()//128))
    out=probe(m,x,y,2,53,parameter_rms(m),micro_batch_size=2,token_chunk_size=3,use_bf16=use_bf16)
    hook.remove()
    assert max(head_rows)<=3 and out['vocabulary_head']=='chunked_hidden_chain_rule'
    reference=[]
    for seed in (53,54):
        gen=torch.Generator(device=device).manual_seed(seed)
        objective=0
        with torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda' and use_bf16):
            for b in range(0,len(x),2):
                hidden=m(x[b:b+2],return_hidden=True).reshape(-1,128)
                mask=y[b:b+2].reshape(-1)!=-1
                for t in range(0,len(hidden),3):
                    valid=mask[t:t+3]
                    if not valid.any(): continue
                    z=m.lm_head(hidden[t:t+3])/m.width_multiplier
                    labels=torch.multinomial(z.detach().float().softmax(-1),1,generator=gen).squeeze(-1)
                    labels[~valid]=-1
                    objective=objective+torch.nn.functional.cross_entropy(z.float(),labels,ignore_index=-1,reduction='sum')/out['valid_tokens']
            gs=torch.autograd.grad(objective,list(m.parameters()))
        buffers=[torch.zeros_like(p) for p in m.parameters()]
        gen=torch.Generator(device=device).manual_seed(seed)
        with torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda' and use_bf16):
            for b in range(0,len(x),2):
                _accumulate_hidden_head(m,x[b:b+2],y[b:b+2],list(m.named_parameters()),buffers,
                                        out['valid_tokens'],gen,3)
        for (name,p),actual,expected in zip(m.named_parameters(),buffers,gs):
            relative=torch.linalg.vector_norm((actual-expected).float())/torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
            assert relative.item() < (0.005 if use_bf16 else 1e-4), (name,relative.item())
        reference.append({n:out['valid_tokens']*g.double().square().sum().item()/p.numel()
                          for (n,p),g in zip(m.named_parameters(),gs)})
    max_relative=0
    for n,v in out['parameters'].items():
        for draw in range(2):
            expected=reference[draw][n]
            err=abs(v['mean_fisher_draws'][draw]-expected)/max(expected,1e-14)
            max_relative=max(max_relative,err)
            assert err < (0.005 if use_bf16 else 1e-4), (n,err)
    assert m.transformer.wte.weight is m.lm_head.weight
    assert torch.equal(rng,torch.get_rng_state()) and m.training
    if cuda_rng is not None: assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
    for p,g in zip(m.parameters(),saved): assert torch.equal(p.grad,g)
    print(json.dumps(dict(check='tied_GPT_full_objective_vs_streamed',device=device,bf16=use_bf16,max_relative_trace_error=max_relative)))

if __name__=='__main__':
    correctness('cpu',False)
    if torch.cuda.is_available():
        correctness('cuda',False)
        correctness('cuda',True)
