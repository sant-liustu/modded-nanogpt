"""Tiny CUDA tests of actual shuffled loader, paired streams and complete resume."""
import ast,glob,hashlib,json,math,os,random,subprocess,tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import sys
from torch import nn
import torch.nn.functional as F
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
from fresh_shards import check_fresh_shard,audit_history
source=root/'train_fresh_repeat.py';code=source.read_text(encoding='utf-8');tree=ast.parse(code)
names={'Rotary','apply_rotary_emb','RMSNorm','CausalSelfAttention','MLP','Block','GPTConfig','GPT','DistributedDataLoader','_peek_data_shard','_load_data_shard','build_fixed_norm_state','apply_fixed_norm_control','project_fixed_rms_eager','checkpoint_tree','loader_state','restore_loader','capture_training_state','restore_training_state','shared_elr','get_lr','apply_tensor_rms_elr','is_rmsnorm_gamma_name'}
ns=dict(torch=torch,nn=nn,F=F,math=math,dataclass=dataclass,Path=Path,os=os,glob=glob,random=random,np=np,hashlib=hashlib,json=json,code=code,__file__=str(source),check_fresh_shard=check_fresh_shard)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names],type_ignores=[]),str(source),'exec'),ns)
assert torch.cuda.is_available()
torch.manual_seed(17);np.random.seed(17);random.seed(17)
args=SimpleNamespace(arm='fanin_fixed',gamma_learning_rate=.0018,log_root='tiny',run_name='tiny',continue_wsd=0,num_iterations=18858,peak_elr=.06,warmup_iters=924,batch_size=2,sequence_length=8);ns['args']=args
m=ns['GPT'](ns['GPTConfig'](vocab_size=128,n_embd=128,n_head=1,n_layer=1)).cuda();fixed=ns['build_fixed_norm_state'](m)
opts=[torch.optim.AdamW(m.lm_head.parameters(),lr=.001,betas=(.9,.95),weight_decay=0,fused=True),torch.optim.AdamW([dict(params=[p]) for n,p in m.named_parameters() if n!='transformer.wte.weight'],lr=.001,betas=(.9,.95),weight_decay=0,fused=True)]
scheds=[torch.optim.lr_scheduler.LambdaLR(o,lambda i:min((i+1)/924,1)) for o in opts]
def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(v,w) for v,w in zip(a,b))
    return a==b

def update(step):
    m.train();ns['apply_tensor_rms_elr'](opts,step)
    for o in opts:o.zero_grad(set_to_none=True)
    _,loss=m(ns['x'],ns['y']);loss.backward()
    for o in opts:o.step();o.zero_grad(set_to_none=True)
    for sched in scheds:sched.step()
    ns['apply_fixed_norm_control'](fixed,step)
    ns['x'],ns['y']=ns['train_loader'].next_batch()
    return loss.detach().clone()
with tempfile.TemporaryDirectory(dir=root) as tmp:
    tmp=Path(tmp);rng=np.random.default_rng(99)
    for i in range(22):
        name=f'train{i:03d}' if i<21 else 'val000'
        count=270001 if i==0 else 10001
        tokens=rng.integers(0,128,count,dtype=np.uint16);header=np.zeros(256,dtype=np.int32);header[:3]=[20240520,1,len(tokens)]
        with (tmp/(name+'.bin')).open('wb') as f:f.write(header.tobytes());f.write(tokens.tobytes())
    loader=ns['DistributedDataLoader'];pattern=str(tmp/'train*.bin')
    for index in range(1,21):
        a=loader(pattern,1,8,0,2);b=loader(pattern,1,8,1,2);one=loader(pattern,2,8,0,1)
        for obj in (a,b,one):obj.start_fresh_shard(index)
        for _ in range(3):
            xa,ya=a.next_batch();xb,yb=b.next_batch();xo,yo=one.next_batch()
            assert torch.equal(torch.cat((xa,xb)),xo) and torch.equal(torch.cat((ya,yb)),yo)
        # No wrapping/spilling even if a caller requests beyond this shard.
        a.current_position=len(a.tokens)
        try:a.next_batch()
        except RuntimeError:pass
        else:raise AssertionError('fresh shard wrapped')
    train=loader(pattern,2,8,0,1);val=loader(str(tmp/'val*.bin'),2,8,0,1)
    # Replay actual source-loader fetches, not just a fabricated saved cursor.
    for _ in range(16001):oldx,oldy=train.next_batch()
    ns.update(raw_model=m,optimizers=opts,schedulers=scheds,fixed_norm_state=fixed,train_loader=train,val_loader=val,device='cuda',project_fixed_rms=ns['project_fixed_rms_eager'],cli=SimpleNamespace(branch='fanin_fixed',fork_step=16000,stage_start=16000,experiment='baseline',pulse_elr=.06,pulse_updates=0,probe_val_tokens=1048576,replicate_id=0,fresh_shard_index=1,stop_step=16520,fresh_info={},fresh_audit={}),schedule_sha256='synthetic',rmsnorm_gamma_param_ids={id(p) for n,p in m.named_parameters() if ns['is_rmsnorm_gamma_name'](n)},fixed_rms_by_id={},tensor_name_by_id={id(p):n for n,p in m.named_parameters()})
    history=ns['capture_training_state'](16000,oldx,oldy)
    history['source_sha256']='bf1ba81bad94b4dad48b10d441730286d10f7f1bf456f02f5db5ee8c32930e37';history['schedule_sha256']='9e3a4b48e9c2de544e2c1c7f35775c80f910b39717eed85ac45645434961c387';history['probe_experiment']='stable'
    for k in ('probe_fork_step','probe_val_tokens','replicate_id','fresh_shard','fresh_audit','fresh_helper_sha256'):history.pop(k)
    historical_if=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="payload['source_sha256'] != hashlib.sha256(code.encode()).hexdigest()")
    reset_code=compile(ast.Module(body=[historical_if],type_ignores=[]),'<actual-initial-data-reset>','exec')
    checkpoint=tmp/'step016000';checkpoint.mkdir()
    torch.save(history,checkpoint/'rank00000.pt')
    (checkpoint/'complete.json').write_text(json.dumps(dict(version=1,step=16000,world_size=1,files=['rank00000.pt'])))
    command=[sys.executable,str(root/'run_repeats.py'),'--resume',str(checkpoint),'--input-bin',pattern,'--input-val-bin',str(tmp/'val*.bin'),'--output',str(tmp/'out'),'--nproc-per-node','1','--device-batch-size','2','--probe-val-tokens','32','--dry-run']
    plan=json.loads(subprocess.check_output(command,text=True))
    assert len(plan['jobs'])==40 and plan['history_audit']['excluded_indices']==[0] and plan['selected_shard_indices']==list(range(1,21))

    traces={};initial=[];report=[]
    for seed in range(1000,1020):
        for arm,width,eta in [('baseline',0,.06),('elr001',120,.01)]:
            cli=ns['cli'];cli.fork_step=16000;cli.experiment=arm;cli.replicate_id=seed-1000;cli.fresh_shard_index=seed-999;cli.pulse_elr=eta;cli.pulse_updates=width;cli.stop_step=16000+(520 if arm=='baseline' else 620)
            ns['schedule_sha256']='synthetic-'+arm
            ns['tensor_elr_schedule']={u:{n:ns['shared_elr'](u) for n,p in m.named_parameters() if p.ndim>=2} for u in (16001,16002,16120,16121)}
            step,(ns['x'],ns['y'])=ns['restore_training_state'](history)
            before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),[s.clone() for s in torch.cuda.get_rng_state_all()])
            ns['payload']=history;exec(reset_code,ns)
            assert not torch.equal(ns['x'],oldx)
            assert before[0]==random.getstate() and np.array_equal(before[1][1],np.random.get_state()[1]) and torch.equal(before[2],torch.get_rng_state()) and equal(before[3],torch.cuda.get_rng_state_all())
            signature=hashlib.sha256(ns['x'].cpu().numpy().tobytes()).hexdigest();traces[(seed,arm)]=(signature,train.fresh_shard)
            m.eval();val.reset();vx,vy=val.next_batch()
            with torch.no_grad():_,vl=m(vx,vy)
            initial.append(float(vl))
            # Real LR setter at both sides of the pulse switch.
            for u in (16120,16121):
                ns['apply_tensor_rms_elr'](opts,u)
                for opt in opts:
                    for group in opt.param_groups:
                        p=group['params'][0]
                        if id(p) in ns['rmsnorm_gamma_param_ids']:assert group['lr']==.0018
                        else:assert math.isclose(group['lr']/ns['fixed_rms_by_id'][id(p)],ns['shared_elr'](u),rel_tol=1e-12)
            update(16001);state=ns['capture_training_state'](16001,ns['x'],ns['y'])
            path=tmp/'state.pt';torch.save(state,path);state=torch.load(path,weights_only=False)
            expected_loss=update(16002);expected_model=ns['checkpoint_tree'](m.state_dict());expected_opts=ns['checkpoint_tree']([o.state_dict() for o in opts]);expected_schedulers=ns['checkpoint_tree']([o.state_dict() for o in scheds]);expected_next=ns['checkpoint_tree']((ns['x'],ns['y']));expected_position=ns['loader_state'](train)
            cli.fork_step=16001;step,(ns['x'],ns['y'])=ns['restore_training_state'](state)
            ns['payload']=state;position=ns['loader_state'](train);batch=ns['checkpoint_tree']((ns['x'],ns['y']));exec(reset_code,ns)
            assert position==ns['loader_state'](train) and equal(batch,ns['checkpoint_tree']((ns['x'],ns['y'])))
            assert torch.equal(expected_loss,update(16002)) and equal(expected_model,ns['checkpoint_tree'](m.state_dict())) and equal(expected_opts,ns['checkpoint_tree']([o.state_dict() for o in opts])) and equal(expected_schedulers,ns['checkpoint_tree']([o.state_dict() for o in scheds]))
            assert equal(expected_next,ns['checkpoint_tree']((ns['x'],ns['y']))) and expected_position==ns['loader_state'](train)
            wrong=ns['checkpoint_tree'](state);wrong['replicate_id']=seed+1
            try:ns['restore_training_state'](wrong)
            except RuntimeError:pass
            else:raise AssertionError('wrong seed accepted')
            report.append(dict(seed=seed,arm=arm,status='bitwise resume PASS'))
    assert max(initial)==min(initial)
    assert len({traces[(seed,'baseline')][0] for seed in range(1000,1020)})==20
    for seed in range(1000,1020):assert traces[(seed,'baseline')]==traces[(seed,'elr001')]
    assert audit_history(history)['excluded_indices']==[0]
    used=ns['checkpoint_tree'](history)
    try:check_fresh_shard(used,0,620)
    except ValueError:pass
    else:raise AssertionError('used historical shard accepted')
(root/'checkpoint_verification.json').write_text(json.dumps(dict(status='PASS',torch=str(torch.__version__),scope='40 tinyCUDA runs; actual historical16001-fetch replay and mmap checkpoint launcher dry-run;real sequential-shard loader/rank partition;20 distinct paired fresh shards;unchanged initial validation/globalRNG;actual initial reset and no-reset resume;LR switch and gamma;disk resume next loss/params/optimizers/schedulers/batch/cursor bitwise;wrongreplicate/used-shard rejection',cases=report),indent=2),encoding='utf-8')
print('PASS:40 tinyCUDA paired migrations,actual untouched shards/ranks and bitwise resume')
