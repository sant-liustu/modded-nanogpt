"""Actual mixed-context loader,rank partition and280 complete tinyCUDA resumes."""
import ast,glob,hashlib,json,math,os,random,subprocess,sys,tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root));source=root/'train_w768_mix2.py';code=source.read_text(encoding='utf-8');tree=ast.parse(code)
names={'Rotary','apply_rotary_emb','RMSNorm','CausalSelfAttention','MLP','Block','GPTConfig','GPT','DistributedDataLoader','_peek_data_shard','_load_data_shard','make_context_order','build_fixed_norm_state','apply_fixed_norm_control','project_fixed_rms_eager','checkpoint_tree','loader_state','restore_loader','capture_training_state','restore_training_state','shared_elr','get_lr','apply_tensor_rms_elr','is_rmsnorm_gamma_name'}
ns=dict(torch=torch,nn=nn,F=F,math=math,dataclass=dataclass,Path=Path,os=os,glob=glob,random=random,np=np,hashlib=hashlib,json=json,code=code)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names],type_ignores=[]),str(source),'exec'),ns)
assert torch.cuda.is_available();torch.manual_seed(17);np.random.seed(17);random.seed(17)
args=SimpleNamespace(arm='fanin_fixed',gamma_learning_rate=.0018,log_root='tiny',run_name='tiny',continue_wsd=0,num_iterations=18858,peak_elr=.06,warmup_iters=924,batch_size=2,sequence_length=8,device_batch_size=2,expected_world_size=1,compile_model=0,model_width=768,model_depth=12,input_bin='train*.bin',input_val_bin='val*.bin');ns['args']=args
m=ns['GPT'](ns['GPTConfig'](vocab_size=128,n_embd=128,n_head=1,n_layer=1)).cuda();fixed=ns['build_fixed_norm_state'](m)
opts=[torch.optim.AdamW(m.lm_head.parameters(),lr=.001,betas=(.9,.95),weight_decay=0,fused=True),torch.optim.AdamW([dict(params=[p]) for n,p in m.named_parameters() if n!='transformer.wte.weight'],lr=.001,betas=(.9,.95),weight_decay=0,fused=True)];scheds=[torch.optim.lr_scheduler.LambdaLR(o,lambda i:min((i+1)/924,1)) for o in opts]
def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def update(step):
    m.train();ns['apply_tensor_rms_elr'](opts,step)
    for opt in opts:opt.zero_grad(set_to_none=True)
    _,loss=m(ns['x'],ns['y']);loss.backward()
    for opt in opts:opt.step();opt.zero_grad(set_to_none=True)
    for sched in scheds:sched.step()
    ns['apply_fixed_norm_control'](fixed,step);ns['x'],ns['y']=ns['train_loader'].next_batch();return loss.detach().clone()
reports=[]
with tempfile.TemporaryDirectory(dir=root) as tmp:
    tmp=Path(tmp);rng=np.random.default_rng(91)
    for i in range(6):
        name=f'train{i:03d}' if i<5 else 'val000';count=270001 if i==0 else 10001
        tokens=rng.integers(0,128,count,dtype=np.uint16);header=np.zeros(256,dtype=np.int32);header[:3]=[20240520,1,count]
        with (tmp/(name+'.bin')).open('wb') as f:f.write(header.tobytes());f.write(tokens.tobytes())
    loader=ns['DistributedDataLoader'];pattern=str(tmp/'train*.bin');stream_hashes=[]
    # Actual labels preserve every selected contiguous context,including final context/shard edges.
    for seed in range(2000,2140):
        a=loader(pattern,1,8,0,2);b=loader(pattern,1,8,1,2);one=loader(pattern,2,8,0,1)
        for obj in (a,b,one):obj.start_mix([1,2],seed)
        before=random.getstate()
        for _ in range(3):
            cursor=one.mix_cursor;ids=one.mix_order[cursor:cursor+2]
            xa,ya=a.next_batch();xb,yb=b.next_batch();xo,yo=one.next_batch()
            assert torch.equal(torch.cat((xa,xb)),xo) and torch.equal(torch.cat((ya,yb)),yo)
            for row,idx in enumerate(ids):
                shard=0 if idx<one.mix_counts[0] else 1;local=int(idx)-(0 if shard==0 else one.mix_counts[0]);start=local*8
                assert np.array_equal(xo[row].cpu().numpy(),one.mix_arrays[shard][start:start+8]) and np.array_equal(yo[row].cpu().numpy(),one.mix_arrays[shard][start+1:start+9])
        assert random.getstate()==before
        stream_hashes.append(one.mix_digest)
        if seed==2000:
            # Visit the last valid context in each shard explicitly;targets never cross arrays.
            for idx in (one.mix_counts[0]-1,sum(one.mix_counts)-1):
                one.mix_order[:2]=[idx,idx];one.mix_cursor=0;x,y=one.next_batch()
                shard=0 if idx<one.mix_counts[0] else 1;local=idx-(0 if shard==0 else one.mix_counts[0]);start=local*8
                assert np.array_equal(y[0].cpu().numpy(),one.mix_arrays[shard][start+1:start+9])
            one.mix_cursor=len(one.mix_order)
            try:one.next_batch()
            except RuntimeError:pass
            else:raise AssertionError('mixed pool wrapped')
    assert len(set(stream_hashes))==140
    train=loader(pattern,2,8,0,1);val=loader(str(tmp/'val*.bin'),2,8,0,1)
    train.current_position=16000*16;oldx,oldy=train.next_batch()
    cli=SimpleNamespace(branch='fanin_fixed',fork_step=16000,stage_start=16000,experiment='baseline',suite_mode='mix2',mix_seed=2000,mix_shards=[1,2],pulse_elr=.06,pulse_updates=0,stop_step=16320,probe_val_tokens=32)
    ns.update(raw_model=m,optimizers=opts,schedulers=scheds,fixed_norm_state=fixed,train_loader=train,val_loader=val,device='cuda',project_fixed_rms=ns['project_fixed_rms_eager'],cli=cli,schedule_sha256='synthetic',rmsnorm_gamma_param_ids={id(p) for n,p in m.named_parameters() if ns['is_rmsnorm_gamma_name'](n)},fixed_rms_by_id={},tensor_name_by_id={id(p):n for n,p in m.named_parameters()},master_process=False)
    history=ns['capture_training_state'](16000,oldx,oldy);history['probe_experiment']='stable';history['schedule_sha256']='9e3a4b48e9c2de544e2c1c7f35775c80f910b39717eed85ac45645434961c387'
    allowed=[json.loads((root/'provenance.json').read_text())['parent_runtime_sha256'],'bf1ba81bad94b4dad48b10d441730286d10f7f1bf456f02f5db5ee8c32930e37','92b8b365e80a7d738bd292014f9adbb67bc515358175b50060ad81956faac6c1']
    history['source_sha256']=allowed[0]
    for k in ('probe_fork_step','probe_val_tokens','suite_mode','mix_seed','mix_shards'):history.pop(k)
    initial_if=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="payload['source_sha256'] != hashlib.sha256(code.encode()).hexdigest()")
    init_code=compile(ast.Module(body=[initial_if],type_ignores=[]),'<actual-pool-init>','exec')
    traces={};initial_val=[]
    for seed in range(2000,2140):
        shards=random.Random(seed^0x51A2).sample([1,2,3,4],2)
        for arm,width,eta in [('baseline',0,.06),('elr001',120,.01)]:
            cli.fork_step=16000;cli.experiment=arm;cli.mix_seed=seed;cli.mix_shards=shards;cli.pulse_updates=width;cli.pulse_elr=eta;cli.stop_step=16320 if arm=='baseline' else 16420
            history['source_sha256']=allowed[(seed-2000)%3];_,(ns['x'],ns['y'])=ns['restore_training_state'](history)
            before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),torch.cuda.get_rng_state_all())
            ns['payload']=history;exec(init_code,ns)
            assert before[0]==random.getstate() and np.array_equal(before[1][1],np.random.get_state()[1]) and torch.equal(before[2],torch.get_rng_state()) and same(before[3],torch.cuda.get_rng_state_all())
            assert not torch.equal(ns['x'],oldx)
            traces[(seed,arm)]=(tuple(shards),train.mix_digest,hashlib.sha256(ns['x'].cpu().numpy().tobytes()).hexdigest())
            m.eval();val.reset();vx,vy=val.next_batch()
            with torch.no_grad():_,vl=m(vx,vy)
            initial_val.append(float(vl))
            ns['schedule_sha256']='synthetic-'+arm
            ns['tensor_elr_schedule']={u:{n:ns['shared_elr'](u) for n,p in m.named_parameters() if p.ndim>=2} for u in (16001,16002,16120,16121)}
            for u in (16120,16121):
                ns['apply_tensor_rms_elr'](opts,u)
                for opt in opts:
                    for group in opt.param_groups:
                        p=group['params'][0]
                        if id(p) in ns['rmsnorm_gamma_param_ids']:assert group['lr']==.0018
                        else:assert math.isclose(group['lr']/ns['fixed_rms_by_id'][id(p)],ns['shared_elr'](u),rel_tol=1e-12)
            update(16001);state=ns['capture_training_state'](16001,ns['x'],ns['y']);path=tmp/'state.pt';torch.save(state,path);state=torch.load(path,weights_only=False)
            expected_loss=update(16002);expected=ns['checkpoint_tree']((m.state_dict(),[o.state_dict() for o in opts],[o.state_dict() for o in scheds],ns['x'],ns['y'],ns['loader_state'](train)))
            cli.fork_step=16001;_,(ns['x'],ns['y'])=ns['restore_training_state'](state);ns['payload']=state;position=train.mix_cursor;exec(init_code,ns);assert train.mix_cursor==position
            assert torch.equal(expected_loss,update(16002))
            actual=ns['checkpoint_tree']((m.state_dict(),[o.state_dict() for o in opts],[o.state_dict() for o in scheds],ns['x'],ns['y'],ns['loader_state'](train)));assert same(expected,actual)
            invalid=ns['checkpoint_tree'](state);invalid['mix_seed']=seed+1
            try:ns['restore_training_state'](invalid)
            except RuntimeError:pass
            else:raise AssertionError('wrong data seed accepted')
            reports.append(dict(seed=seed,arm=arm,status='bitwise resume PASS'))
    assert max(initial_val)==min(initial_val)
    for seed in range(2000,2140):assert traces[(seed,'baseline')]==traces[(seed,'elr001')]
    assert len({traces[(seed,'baseline')] for seed in range(2000,2140)})==140
    # Actual mmap metadata loading and full280-job dry-run on a synthetic complete checkpoint.
    history['source_sha256']=allowed[0];cp=tmp/'step016000';cp.mkdir();torch.save(history,cp/'rank00000.pt');(cp/'complete.json').write_text(json.dumps(dict(version=1,step=16000,world_size=1,files=['rank00000.pt'])))
    command=[sys.executable,str(root/'run_mix2.py'),'--resume',str(cp),'--output',str(tmp/'plan'),'--probe-val-tokens','32','--dry-run']
    plan=json.loads(subprocess.check_output(command,text=True));assert len(plan['jobs'])==280
(root/'checkpoint_verification.json').write_text(json.dumps(dict(status='PASS',python=sys.version,torch=str(torch.__version__),scope='140 actual context permutations;rank partition,contiguous labels/final-shard edges,pool exhaustion;280 tinyCUDA migrations and disk resumes,next loss/params/Adam/scheduler/batch/order/cursor bitwise;fixed initial val/global RNG;all3 shared16k sources;actual mmap280-job dry-run;formal DDP/compile not run',cases=reports),indent=2),encoding='utf-8')
print('PASS:140 context streams,280 tinyCUDA migrations/resumes and280-job dry-run')

