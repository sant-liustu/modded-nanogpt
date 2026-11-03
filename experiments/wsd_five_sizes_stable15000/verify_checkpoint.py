"""Exercise the actual checkpoint functions on a tiny CUDA model and synthetic loaders."""
import ast, hashlib, json, math, os, random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
root=Path(__file__).resolve().parent
source=root/'train_fixednorm_w768.py'
code=source.read_text(encoding='utf-8')
tree=ast.parse(code)
names={'Rotary','apply_rotary_emb','RMSNorm','CausalSelfAttention','MLP','Block','GPTConfig','GPT','build_fixed_norm_state','apply_fixed_norm_control','project_fixed_rms_eager','checkpoint_tree','loader_state','restore_loader','capture_training_state','restore_training_state'}
ns=dict(torch=torch,nn=nn,F=F,math=math,dataclass=dataclass,Path=Path,os=os,random=random,np=np,hashlib=hashlib,code=code)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names],type_ignores=[]),str(source),'exec'),ns)
assert torch.cuda.is_available(), 'CUDA required for this synthetic check'
torch.manual_seed(17); np.random.seed(17); random.seed(17)
args=SimpleNamespace(arm='fanin_fixed',gamma_learning_rate=.0018,log_root='synthetic',run_name='tiny',continue_wsd=0)
ns['args']=args
m=ns['GPT'](ns['GPTConfig'](vocab_size=128,n_embd=128,n_head=1,n_layer=1)).cuda()
fixed=ns['build_fixed_norm_state'](m)
opts=[torch.optim.AdamW(m.lm_head.parameters(),lr=.001),torch.optim.AdamW([p for n,p in m.named_parameters() if n!='transformer.wte.weight'],lr=.001)]
scheds=[torch.optim.lr_scheduler.LambdaLR(o,lambda i: min((i+1)/924,1)) for o in opts]
def loader():
    return SimpleNamespace(files=[source],current_shard=0,current_position=7,B=2,T=8,process_rank=0,num_processes=1,tokens=torch.arange(100))
train,val=loader(),loader()
ns.update(args=args,raw_model=m,optimizers=opts,schedulers=scheds,fixed_norm_state=fixed,cli=SimpleNamespace(branch='fanin_fixed',fork_step=0),schedule_sha256='synthetic',train_loader=train,val_loader=val,device='cuda',project_fixed_rms=ns['project_fixed_rms_eager'],_load_data_shard=lambda f: torch.arange(100))
x=torch.randint(0,128,(2,8),device='cuda')
def update():
    for o in opts:o.zero_grad(set_to_none=True)
    _,loss=m(x,x);loss.backward()
    for o in opts:o.step();o.zero_grad(set_to_none=True)
    for s in scheds:s.step()
    ns['apply_fixed_norm_control'](fixed,1)
    return loss.detach().clone()
update()
state=ns['capture_training_state'](1,x,x)
# Disk serialization is included.
path=root/'checkpoint_synthetic.pt'
torch.save(state,path)
state=torch.load(path,weights_only=False)
expected_loss=update(); expected=ns['checkpoint_tree'](m.state_dict())
expected_opts=ns['checkpoint_tree']([o.state_dict() for o in opts])
expected_rng=(random.random(),float(np.random.rand()),torch.rand(1),torch.rand(1,device='cuda'))
train.current_position=99;val.current_position=99
ns['cli'].fork_step=1
step,batch=ns['restore_training_state'](state)
assert step==1 and train.current_position==val.current_position==7
assert torch.equal(batch[0],x)
assert torch.equal(expected_loss,update())
def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(v,w) for v,w in zip(a,b))
    return a==b
assert equal(expected,ns['checkpoint_tree'](m.state_dict()))
assert equal(expected_opts,ns['checkpoint_tree']([o.state_dict() for o in opts]))
actual_rng=(random.random(),float(np.random.rand()),torch.rand(1),torch.rand(1,device='cuda'))
assert equal(expected_rng,actual_rng)
path.unlink()
(root/'checkpoint_verification.json').write_text(json.dumps(dict(status='PASS',scope='tiny CUDA eager synthetic, actual capture/restore functions, disk roundtrip, next update/model/optimizer/RNG bitwise equal; synthetic loader positions restored',torch=str(torch.__version__)),indent=2))
print('PASS: tiny CUDA checkpoint roundtrip and next update bitwise equality')
