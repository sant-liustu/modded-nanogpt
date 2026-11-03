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
source=root/'train_w768_d12_continue.py'
code=source.read_text(encoding='utf-8')
tree=ast.parse(code)
names={'Rotary','apply_rotary_emb','RMSNorm','CausalSelfAttention','MLP','Block','GPTConfig','GPT','build_fixed_norm_state','apply_fixed_norm_control','project_fixed_rms_eager','checkpoint_tree','loader_state','restore_loader','capture_training_state','restore_training_state','shared_elr','get_lr','apply_tensor_rms_elr','is_rmsnorm_gamma_name'}
ns=dict(torch=torch,nn=nn,F=F,math=math,dataclass=dataclass,Path=Path,os=os,random=random,np=np,hashlib=hashlib,code=code)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names],type_ignores=[]),str(source),'exec'),ns)
assert torch.cuda.is_available(), 'CUDA required for this synthetic check'
torch.manual_seed(17); np.random.seed(17); random.seed(17)
args=SimpleNamespace(arm='fanin_fixed',gamma_learning_rate=.0018,log_root='synthetic',run_name='tiny',continue_wsd=0,num_iterations=18858,peak_elr=.06,warmup_iters=924)
ns['args']=args
m=ns['GPT'](ns['GPTConfig'](vocab_size=128,n_embd=128,n_head=1,n_layer=1)).cuda()
fixed=ns['build_fixed_norm_state'](m)
opts=[torch.optim.AdamW(m.lm_head.parameters(),lr=.001,betas=(.9,.95),weight_decay=0,fused=True),torch.optim.AdamW([dict(params=[p]) for n,p in m.named_parameters() if n!='transformer.wte.weight'],lr=.001,betas=(.9,.95),weight_decay=0,fused=True)]
scheds=[torch.optim.lr_scheduler.LambdaLR(o,lambda i: min((i+1)/924,1)) for o in opts]
def loader():
    return SimpleNamespace(files=[source],current_shard=0,current_position=7,B=2,T=8,process_rank=0,num_processes=1,tokens=torch.arange(100))
train,val=loader(),loader()
ns.update(args=args,raw_model=m,optimizers=opts,schedulers=scheds,fixed_norm_state=fixed,cli=SimpleNamespace(branch='fanin_fixed',fork_step=15000,experiment='pulse'),schedule_sha256='synthetic',train_loader=train,val_loader=val,device='cuda',project_fixed_rms=ns['project_fixed_rms_eager'],_load_data_shard=lambda f: torch.arange(100))
ns.update(rmsnorm_gamma_param_ids={id(p) for n,p in m.named_parameters() if ns['is_rmsnorm_gamma_name'](n)},fixed_rms_by_id={},tensor_name_by_id={id(p):n for n,p in m.named_parameters()},tensor_elr_schedule={u:{n:ns['shared_elr'](u) for n,p in m.named_parameters() if p.ndim>=2} for u in (15000,15001,16000,16001,16200,16201,18200)})
for step,expected in [(16200,.006),(16201,.06)]:
    ns['apply_tensor_rms_elr'](opts,step)
    for group in [g for opt in opts for g in opt.param_groups]:
        p=group['params'][0]
        if id(p) in ns['rmsnorm_gamma_param_ids']:assert group['lr']==.0018
        else:assert abs(group['lr']/ns['fixed_rms_by_id'][id(p)]-expected)<1e-12
x=torch.randint(0,128,(2,8),device='cuda')
def update(step=16001):
    ns['apply_tensor_rms_elr'](opts,step)
    for o in opts:o.zero_grad(set_to_none=True)
    _,loss=m(x,x);loss.backward()
    for o in opts:o.step();o.zero_grad(set_to_none=True)
    for s in scheds:s.step()
    ns['apply_fixed_norm_control'](fixed,step)
    return loss.detach().clone()
update(15000)
state=ns['capture_training_state'](15000,x,x)
original_state=ns['checkpoint_tree'](state)
provenance=json.loads((root/'provenance.json').read_text())
original_state['source_sha256']=provenance['source_sha256']
original_state['schedule_sha256']=provenance['source_schedule_sha256']
original_state.pop('probe_experiment')
# Original checkpoint is accepted only by the shared stable extension.
ns['cli'].experiment='stable'
ns['restore_training_state'](original_state)
for field,value in [('source_sha256','bad'),('schedule_sha256','bad'),('step',14999)]:
    invalid=ns['checkpoint_tree'](original_state); invalid[field]=value
    try: ns['restore_training_state'](invalid)
    except RuntimeError: pass
    else: raise AssertionError('accepted invalid original checkpoint '+field)
# Common step16000 checkpoint migrates into every branch without changing state.
common=ns['capture_training_state'](16000,x,x)
common['schedule_sha256']=provenance['common_stable_schedule_sha256']
ns['cli'].fork_step=16000
for arm in ('high','low','pulse'):
    ns['cli'].experiment=arm
    step,_=ns['restore_training_state'](common)
    assert step==16000
invalid=ns['checkpoint_tree'](common);invalid['schedule_sha256']='bad'
try: ns['restore_training_state'](invalid)
except RuntimeError: pass
else: raise AssertionError('accepted invalid shared prefix')
state=ns['capture_training_state'](16000,x,x)
# Disk serialization is included.
path=root/'checkpoint_synthetic.pt'
torch.save(state,path)
state=torch.load(path,weights_only=False)
expected_loss=update(); expected=ns['checkpoint_tree'](m.state_dict())
expected_opts=ns['checkpoint_tree']([o.state_dict() for o in opts])
expected_rng=(random.random(),float(np.random.rand()),torch.rand(1),torch.rand(1,device='cuda'))
train.current_position=99;val.current_position=99
ns['cli'].fork_step=16000
step,batch=ns['restore_training_state'](state)
assert step==16000 and train.current_position==val.current_position==7
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
(root/'checkpoint_verification.json').write_text(json.dumps(dict(status='PASS',scope='tiny CUDA eager synthetic, pinned original migration and rejection guards; shared stable16000 migration to all branches; real ELR application at16200/16201; disk roundtrip; next branched update/model/optimizer/RNG bitwise equal; synthetic loader positions restored',torch=str(torch.__version__)),indent=2))

ns['cli'].experiment='high'
try: ns['restore_training_state'](state)
except RuntimeError: pass
else: raise AssertionError('accepted cross-arm checkpoint')
print('PASS: pinned original migration, rejection guards, own checkpoint roundtrip and next update bitwise equality')
