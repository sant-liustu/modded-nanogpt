"""Golden plain AdamW update (optional projection) and all-step old-baseline LR audit."""
from pathlib import Path
import ast, copy, math, os
from dataclasses import dataclass
from types import SimpleNamespace
import torch

torch.set_num_threads(2)
root=Path(__file__).parent
source=(root/'train.py').read_text()
names={'Rotary','apply_rotary_emb','RMSNorm','CausalSelfAttention','MLP','Block','GPTConfig','GPT',
       'is_rmsnorm_gamma_name','build_fixed_norm_state','apply_fixed_norm_control',
       'apply_tensor_rms_elr','actual_learning_rate','get_lr'}
module=ast.Module(body=[n for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names],type_ignores=[])
a=SimpleNamespace(arm=os.environ.get('ADAMW_TEST_ARM','fanin_wd'),lr_mode='absolute',learning_rate=.0009,embedding_lr_multiplier=2.,
                  min_lr_ratio=.1,grad_clip=0.,warmup_iters=1000,num_iterations=20400,warmdown_iters=5800,
                  schedule='cosine',weight_decay=.1,embed_learning_rate=.0036,gamma_learning_rate=.0018)
ns=dict(torch=torch,nn=torch.nn,F=torch.nn.functional,math=math,dataclass=dataclass,args=a,__name__='__main__')
exec(compile(module,'train.py','exec'),ns)
# Exact zero-based cosine policy used by the verified old lr0p0018 source.
for update in range(1,20401):
    it=update-1
    ratio=(it+1)/1000 if it<1000 else .1+.9*.5*(1+math.cos(math.pi*min(1.,max(0.,(it-1000)/19400))))
    assert abs(ns['actual_learning_rate'](update)-.0009*ratio)<1e-18

torch.manual_seed(31)
device='cuda' if torch.cuda.is_available() else 'cpu'
m=ns['GPT'](ns['GPTConfig'](vocab_size=32,n_layer=1,n_head=1,n_embd=128)).to(device)
ref=copy.deepcopy(m)
ns.update(raw_model=m,width_multiplier=m.width_multiplier)
# Execute the actual runtime optimizer construction.
exec(source[source.index('rmsnorm_gamma_parameters = ['):source.index('# learning rate decay scheduler')],ns)
ns['tensor_name_by_id']={id(p):n for n,p in m.named_parameters()}
entries=ns['build_fixed_norm_state'](m)
refs={n:p.detach().float().square().mean().sqrt() for n,p in ref.named_parameters() if p.ndim==2}
ref_optimizer=torch.optim.AdamW([dict(params=[p],lr=.0018 if n=='transformer.wte.weight' else .0009,
                                     weight_decay=0. if 'norm' in n else .1)
                               for n,p in ref.named_parameters()],betas=(.9,.95),eps=1e-8,fused=True)
for step in (1,2,3):
    x=torch.randint(32,(2,4),device=device);y=torch.randint(32,(2,4),device=device)
    for model in (m,ref):
        _,loss=model(x,y)
        loss.backward()
    ns['apply_tensor_rms_elr'](ns['optimizers'],step)
    ratio=step/1000
    for group,(name,p) in zip(ref_optimizer.param_groups,ref.named_parameters()):
        group['lr']=(.0018 if name=='transformer.wte.weight' else .0009)*ratio
    for opt in ns['optimizers']:opt.step();opt.zero_grad(set_to_none=True)
    ref_optimizer.step();ref_optimizer.zero_grad(set_to_none=True)
    ns['apply_fixed_norm_control'](entries,step)
    with torch.no_grad():
        for name,p in ref.named_parameters():
            if a.arm=='fanin_fixed' and name in refs:p.mul_(refs[name]/p.float().square().mean().sqrt())
    for (name,p),(_,q) in zip(m.named_parameters(),ref.named_parameters()):
        assert torch.equal(p,q),name
if a.arm!='fanin_fixed':
    assert any(p.float().square().mean().sqrt().item()!=refs[n].item() for n,p in m.named_parameters() if n in refs)
print('PASS:', a.arm, 'all 20400 LR steps match old lr0p0018 baseline; three runtime updates exactly match plain PyTorch reference; free-norm mode has no projection; differing parameter RMS does not rescale LR')
