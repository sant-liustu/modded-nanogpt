"""Five model identities:meta parameter counts and tinyCUDA complete-state migration/resume."""
import ast,glob,hashlib,json,math,os,random,subprocess,sys,tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root));models=json.loads((root/'models.json').read_text())
assert torch.cuda.is_available()
names={'Rotary','apply_rotary_emb','RMSNorm','CausalSelfAttention','MLP','Block','GPTConfig','GPT','DistributedDataLoader','_peek_data_shard','_load_data_shard','build_fixed_norm_state','apply_fixed_norm_control','project_fixed_rms_eager','checkpoint_tree','loader_state','restore_loader','capture_training_state','restore_training_state','shared_elr','select_fresh_shard','get_lr','apply_tensor_rms_elr','is_rmsnorm_gamma_name'}
def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b
reports=[]
with tempfile.TemporaryDirectory(dir=root) as tmp:
    tmp=Path(tmp);rng=np.random.default_rng(97)
    for i in range(22):
        name=f'train{i:03d}' if i<21 else 'val000';count=270001 if i==0 else 10001
        tokens=rng.integers(0,128,count,dtype=np.uint16);header=np.zeros(256,dtype=np.int32);header[:3]=[20240520,1,count]
        with (tmp/(name+'.bin')).open('wb') as f:f.write(header.tobytes());f.write(tokens.tobytes())
    for key,spec in models.items():
        source=root/spec['script'];code=source.read_text(encoding='utf-8');tree=ast.parse(code)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Hyperparameters');cfg={n.target.id:ast.literal_eval(n.value) for n in cls.body if isinstance(n,ast.AnnAssign)}
        ns=dict(torch=torch,nn=nn,F=F,math=math,dataclass=dataclass,Path=Path,os=os,glob=glob,random=random,np=np,hashlib=hashlib,json=json,code=code)
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names],type_ignores=[]),str(source),'exec'),ns)
        ns['args']=SimpleNamespace(**cfg)
        with torch.device('meta'):
            full=ns['GPT'](ns['GPTConfig'](vocab_size=50304,n_embd=spec['width'],n_head=spec['width']//128,n_layer=spec['depth']))
        count=sum(p.numel() for p in full.parameters());expected=50304*spec['width']+spec['depth']*(12*spec['width']**2+2*spec['width']+256)+spec['width'];assert count==expected
        del full
        tinycfg=dict(cfg,batch_size=2,sequence_length=8,device_batch_size=2,expected_world_size=1,compile_model=0)
        ns['args']=SimpleNamespace(**tinycfg)
        torch.manual_seed(17);np.random.seed(17);random.seed(17)
        m=ns['GPT'](ns['GPTConfig'](vocab_size=128,n_embd=128,n_head=1,n_layer=1)).cuda();fixed=ns['build_fixed_norm_state'](m)
        opts=[torch.optim.AdamW(m.lm_head.parameters(),lr=.001,betas=(.9,.95),weight_decay=0,fused=True),torch.optim.AdamW([dict(params=[p]) for n,p in m.named_parameters() if n!='transformer.wte.weight'],lr=.001,betas=(.9,.95),weight_decay=0,fused=True)]
        scheds=[torch.optim.lr_scheduler.LambdaLR(o,lambda i:min((i+1)/924,1)) for o in opts]
        train=ns['DistributedDataLoader'](str(tmp/'train*.bin'),2,8,0,1);val=ns['DistributedDataLoader'](str(tmp/'val*.bin'),2,8,0,1)
        cli=SimpleNamespace(branch='fanin_fixed',fork_step=15000,stage_start=15000,experiment='stable',suite_mode='historical',fresh_replicate=-1,fresh_repeats=20,shard_seed=0,pulse_elr=.06,pulse_updates=0,stop_step=16000,probe_val_tokens=32,fresh_info=None,excluded_shards=[])
        ns.update(raw_model=m,optimizers=opts,schedulers=scheds,fixed_norm_state=fixed,train_loader=train,val_loader=val,device='cuda',project_fixed_rms=ns['project_fixed_rms_eager'],cli=cli,schedule_sha256='synthetic',rmsnorm_gamma_param_ids={id(p) for n,p in m.named_parameters() if ns['is_rmsnorm_gamma_name'](n)},fixed_rms_by_id={},tensor_name_by_id={id(p):n for n,p in m.named_parameters()},master_process=False)
        # Capture synthetic historical states at actual sequential-loader cursors.
        histories={}
        for step in (5000,10000,15000):
            train.current_shard=0;train.current_position=step*16;train.fresh_shard=None
            x,y=train.next_batch();state=ns['capture_training_state'](step,x,y)
            state['source_sha256']=spec['original_source_sha256'];state['schedule_sha256']='38e74728f90116c44727d8da1985cde02668d1d7747bc30a699589640e7f18be'
            for field in ('probe_experiment','probe_fork_step','probe_val_tokens','suite_mode','fresh_replicate','shard_seed','fresh_repeats','fresh_shard','excluded_shard_indices'):state.pop(field)
            histories[step]=state
        # The real restore accepts original15k into preparation, then captured stable16k into arms.
        ns['x'],ns['y']=ns['restore_training_state'](histories[15000])[1]
        train.current_position=16000*16;train.fresh_shard=None;ns['x'],ns['y']=train.next_batch()
        ns['schedule_sha256']='9e3a4b48e9c2de544e2c1c7f35775c80f910b39717eed85ac45645434961c387'
        shared=ns['capture_training_state'](16000,ns['x'],ns['y']);histories[16000]=shared
        resume_if=next(n for n in tree.body if isinstance(n,ast.If) and any(isinstance(b,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='initial_fork' for v in b.targets) for b in n.body))
        start=next(i for i,n in enumerate(resume_if.body) if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='initial_fork' for v in n.targets))
        end=next(i for i,n in enumerate(resume_if.body) if isinstance(n,ast.Delete))
        fork_code=compile(ast.Module(body=resume_if.body[start:end],type_ignores=[]),'<actual-mode-data-setup>','exec')
        def update(step):
            m.train();ns['apply_tensor_rms_elr'](opts,step)
            for opt in opts:opt.zero_grad(set_to_none=True)
            _,loss=m(ns['x'],ns['y']);loss.backward()
            for opt in opts:opt.step();opt.zero_grad(set_to_none=True)
            for sched in scheds:sched.step()
            ns['apply_fixed_norm_control'](fixed,step);ns['x'],ns['y']=train.next_batch()
            return loss.detach().clone()
        cases=[(15000,'stable',0,-1)]
        cases += [(step,arm,width,-1) for step in (5000,10000,15000) for arm,width in [('baseline',0),('elr001',120),('elr003',40)]]
        cases += [(16000,arm,width,-1) for arm,width in [('baseline',0),('elr005',24),('elr004',30),('elr003',40),('elr002',60),('elr001',120)]]
        cases += [(16000,arm,width,rep) for rep in range(20) for arm,width in [('baseline',0),('elr001',120)]]
        traces={}
        for step,arm,width,rep in cases:
            cli.fork_step=step;cli.stage_start=step;cli.experiment=arm;cli.suite_mode='fresh' if rep>=0 else 'historical';cli.fresh_replicate=rep;cli.pulse_updates=width;cli.pulse_elr=(1.2/width if width else .06);cli.stop_step=step+(1000 if arm=='stable' else (width+500 if width else 520));cli.fresh_info=None;cli.excluded_shards=[]
            ns['schedule_sha256']='synthetic-'+str(step)+'-'+arm
            initial=histories[step];_,(ns['x'],ns['y'])=ns['restore_training_state'](initial)
            before=ns['checkpoint_tree']((ns['x'],ns['y']));ns['payload']=initial;exec(fork_code,ns)
            if rep<0:assert same(before,ns['checkpoint_tree']((ns['x'],ns['y'])))
            else:traces[(rep,arm)]=(cli.fresh_info['index'],hashlib.sha256(ns['x'].cpu().numpy().tobytes()).hexdigest())
            ns['tensor_elr_schedule']={u:{n:ns['shared_elr'](u) for n,p in m.named_parameters() if p.ndim>=2} for u in (step+1,step+2,step+width,step+width+1)}
            for u in (step+1,step+width+1):
                ns['apply_tensor_rms_elr'](opts,u)
                for opt in opts:
                    for group in opt.param_groups:
                        p=group['params'][0]
                        if id(p) in ns['rmsnorm_gamma_param_ids']:assert group['lr']==.0018
                        else:assert math.isclose(group['lr']/ns['fixed_rms_by_id'][id(p)],ns['shared_elr'](u),rel_tol=1e-12)
            update(step+1);state=ns['capture_training_state'](step+1,ns['x'],ns['y']);path=tmp/'state.pt';torch.save(state,path);state=torch.load(path,weights_only=False)
            loss=update(step+2);expected=ns['checkpoint_tree']((m.state_dict(),[o.state_dict() for o in opts],[o.state_dict() for o in scheds],ns['x'],ns['y'],ns['loader_state'](train)))
            cli.fork_step=step+1;_,(ns['x'],ns['y'])=ns['restore_training_state'](state);ns['payload']=state;exec(fork_code,ns)
            assert torch.equal(loss,update(step+2))
            actual=ns['checkpoint_tree']((m.state_dict(),[o.state_dict() for o in opts],[o.state_dict() for o in scheds],ns['x'],ns['y'],ns['loader_state'](train)))
            assert same(expected,actual)
            bad=ns['checkpoint_tree'](state);bad['suite_mode']='wrong'
            try:ns['restore_training_state'](bad)
            except RuntimeError:pass
            else:raise AssertionError('wrong mode accepted')
        for rep in range(20):assert traces[(rep,'baseline')]==traces[(rep,'elr001')]
        assert len({traces[(rep,'baseline')][0] for rep in range(20)})==20
        # Wrong model original source must fail even on the shape-matched tiny fixture.
        cli.stage_start=5000;cli.fork_step=5000;cli.experiment='baseline';cli.suite_mode='historical';cli.fresh_replicate=-1
        wrong=ns['checkpoint_tree'](histories[5000]);wrong['source_sha256']=next(v['original_source_sha256'] for k,v in models.items() if k!=key)
        try:ns['restore_training_state'](wrong)
        except RuntimeError:pass
        else:raise AssertionError('wrong model source accepted')
        if key=='w768_d12':
            cli.stage_start=16000;cli.fork_step=16000
            for oldsha in ('bf1ba81bad94b4dad48b10d441730286d10f7f1bf456f02f5db5ee8c32930e37','92b8b365e80a7d738bd292014f9adbb67bc515358175b50060ad81956faac6c1'):
                legacy=ns['checkpoint_tree'](shared);legacy['source_sha256']=oldsha
                ns['restore_training_state'](legacy)
        from run_suite import checkpoint_info
        cp=tmp/(key+'_shared');cp.mkdir();torch.save(shared,cp/'rank00000.pt')
        (cp/'complete.json').write_text(json.dumps(dict(version=1,step=16000,world_size=1,files=['rank00000.pt'])))
        metadata=checkpoint_info(cp,key,16000);assert metadata['ranks']==1 and metadata['device_batch_size']==2
        invalid=ns['checkpoint_tree'](shared);invalid['source_sha256']=spec['original_source_sha256'];torch.save(invalid,cp/'rank00000.pt')
        try:checkpoint_info(cp,key,16000)
        except ValueError:pass
        else:raise AssertionError('accepted original WSD16000 after decay')
        budget=json.loads(subprocess.check_output([sys.executable,str(source),'--fork-step','16000','--experiment','elr001','--describe-only'],text=True))
        assert budget['parameter_count']==count and budget['continuation_updates']==620
        reports.append(dict(model=key,full_meta_parameters=count,tiny_migration_resume_cases=len(cases),status='PASS'))
(root/'checkpoint_verification.json').write_text(json.dumps(dict(status='PASS',python=sys.version,torch=str(torch.__version__),models=reports,scope='full meta parameter counts;280 tinyCUDA model-binding migrations/resumes;real data loader and actual initial/reset blocks;paired20 distinct shards/model;actual LR/gamma;disk resume next loss/params/Adam/schedulers/batch/cursor bitwise;wrongmode/model rejection;legacyw768 shared16k aliases;formal sizes/DDP/compile not run'),indent=2),encoding='utf-8')
print('PASS:five full meta models and280 tinyCUDA migration/resume cases')

