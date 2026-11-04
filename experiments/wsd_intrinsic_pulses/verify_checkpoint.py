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
source=root/'train_intrinsic_pulse.py'
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
ns.update(args=args,raw_model=m,optimizers=opts,schedulers=scheds,fixed_norm_state=fixed,cli=SimpleNamespace(branch='fanin_fixed',fork_step=5000,stage_start=5000,experiment='elr005',pulse_elr=.05,pulse_updates=24,probe_val_tokens=1048576),schedule_sha256='synthetic',train_loader=train,val_loader=val,device='cuda',project_fixed_rms=ns['project_fixed_rms_eager'],_load_data_shard=lambda f: torch.arange(100))
ns.update(rmsnorm_gamma_param_ids={id(p) for n,p in m.named_parameters() if ns['is_rmsnorm_gamma_name'](n)},fixed_rms_by_id={},tensor_name_by_id={id(p):n for n,p in m.named_parameters()})
x=torch.randint(0,128,(2,8),device='cuda')
def update(step):
    ns['apply_tensor_rms_elr'](opts,step)
    for o in opts:o.zero_grad(set_to_none=True)
    _,loss=m(x,x);loss.backward()
    for o in opts:o.step();o.zero_grad(set_to_none=True)
    for sched in scheds:sched.step()
    ns['apply_fixed_norm_control'](fixed,step)
    return loss.detach().clone()
def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(v,w) for v,w in zip(a,b))
    return a==b
provenance=json.loads((root/'provenance.json').read_text())
report=[]
for start in (5000,10000,15000,16000):
    for arm,width,eta in [('baseline',0,.06),('elr005',24,.05),('elr004',30,.04),('elr003',40,.03),('elr002',60,.02),('elr001',120,.01)]:
        cli=ns['cli'];cli.stage_start=start;cli.fork_step=start;cli.experiment=arm;cli.pulse_elr=eta;cli.pulse_updates=width
        ns['schedule_sha256']=f'synthetic-{start}-{arm}'
        ns['tensor_elr_schedule']={u:{n:ns['shared_elr'](u) for n,p in m.named_parameters() if p.ndim>=2} for u in (start+1,start+2,start+width,start+width+1)}
        # Migrate actual capture payload with pinned historical source/schedule identity.
        history=ns['capture_training_state'](start,x,x)
        history['source_sha256']=provenance['original_source_sha256' if start!=16000 else 'shared16000_source_sha256']
        history['schedule_sha256']=provenance['original_schedule_sha256' if start!=16000 else 'shared16000_schedule_sha256']
        if start==16000:history['probe_experiment']='stable'
        else:history.pop('probe_experiment')
        history.pop('probe_fork_step');history.pop('probe_val_tokens')
        before=ns['checkpoint_tree'](m.state_dict());step,batch=ns['restore_training_state'](history)
        assert step==start and equal(before,ns['checkpoint_tree'](m.state_dict())) and torch.equal(batch[0],x)
        for key,bad in [('source_sha256','bad'),('schedule_sha256','bad'),('step',start-1)]:
            invalid=ns['checkpoint_tree'](history);invalid[key]=bad
            try:ns['restore_training_state'](invalid)
            except RuntimeError:pass
            else:raise AssertionError('accepted bad historical '+key)
        # Verify actual matrix LR / fixed target RMS, and gamma LR at both sides of pulse boundary.
        for u in (start+1,start+width+1):
            ns['apply_tensor_rms_elr'](opts,u)
            expected=eta if width and u<=start+width else .06
            for group in [g for opt in opts for g in opt.param_groups]:
                p=group['params'][0]
                if id(p) in ns['rmsnorm_gamma_param_ids']:assert group['lr']==.0018
                else:assert abs(group['lr']/ns['fixed_rms_by_id'][id(p)]-expected)<1e-12
        update(start+1)
        state=ns['capture_training_state'](start+1,x,x)
        path=root/'checkpoint_synthetic.pt';torch.save(state,path);state=torch.load(path,weights_only=False);path.unlink()
        expected_loss=update(start+2)
        expected_model=ns['checkpoint_tree'](m.state_dict());expected_opts=ns['checkpoint_tree']([o.state_dict() for o in opts]);expected_sched=ns['checkpoint_tree']([o.state_dict() for o in scheds])
        expected_rng=(random.random(),float(np.random.rand()),torch.rand(1),torch.rand(1,device='cuda'))
        train.current_position=99;val.current_position=99;cli.fork_step=start+1
        ns['restore_training_state'](state)
        assert train.current_position==val.current_position==7
        assert torch.equal(expected_loss,update(start+2))
        assert equal(expected_model,ns['checkpoint_tree'](m.state_dict())) and equal(expected_opts,ns['checkpoint_tree']([o.state_dict() for o in opts])) and equal(expected_sched,ns['checkpoint_tree']([o.state_dict() for o in scheds]))
        actual_rng=(random.random(),float(np.random.rand()),torch.rand(1),torch.rand(1,device='cuda'))
        assert equal(expected_rng,actual_rng)
        for key,bad in [('probe_experiment','wrong'),('probe_fork_step',start+1),('probe_val_tokens',1),('schedule_sha256','bad')]:
            invalid=ns['checkpoint_tree'](state);invalid[key]=bad
            try:ns['restore_training_state'](invalid)
            except RuntimeError:pass
            else:raise AssertionError('accepted bad continuation '+key)
        report.append(dict(fork_step=start,arm=arm,migration='PASS',boundary_lr='PASS',disk_resume_next_step='bitwise PASS'))
(root/'checkpoint_verification.json').write_text(json.dumps(dict(status='PASS',scope='tiny CUDA eager synthetic:24 historical migrations, exact matrix ELR boundaries/gamma LR, complete disk checkpoint next update/model/optimizer/scheduler/RNG bitwise equality; negative provenance/fork/arm/validation checks; mocked loaders restore positions',torch=str(torch.__version__),cases=report),indent=2))
print('PASS:24 tiny CUDA migrations, LR boundaries and bitwise next-update resume')
