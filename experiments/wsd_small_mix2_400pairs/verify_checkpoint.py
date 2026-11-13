"""Actual context/rank checks plus model-bound tiny resumes and full-shape CUDA smoke."""
import ast,gc,glob,hashlib,json,math,os,random,subprocess,sys,tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root));models=json.loads((root/'models.json').read_text())
from run_small_mix2 import load_info
assert torch.cuda.is_available();torch.set_num_threads(2)
names={'Rotary','apply_rotary_emb','RMSNorm','CausalSelfAttention','MLP','Block','GPTConfig','GPT','DistributedDataLoader','_peek_data_shard','_load_data_shard','make_context_order','build_fixed_norm_state','apply_fixed_norm_control','project_fixed_rms_eager','checkpoint_tree','loader_state','restore_loader','capture_training_state','restore_training_state','shared_elr','get_lr','apply_tensor_rms_elr','is_rmsnorm_gamma_name'}
def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def optimizers_for(model):
    return [torch.optim.AdamW(model.lm_head.parameters(),lr=.001,betas=(.9,.95),weight_decay=0,fused=True),torch.optim.AdamW([dict(params=[p]) for n,p in model.named_parameters() if n!='transformer.wte.weight'],lr=.001,betas=(.9,.95),weight_decay=0,fused=True)]

def configure(ns,model,fixed,opts):
    ns.update(raw_model=model,optimizers=opts,fixed_norm_state=fixed,device='cuda',project_fixed_rms=ns['project_fixed_rms_eager'],rmsnorm_gamma_param_ids={id(p) for n,p in model.named_parameters() if ns['is_rmsnorm_gamma_name'](n)},fixed_rms_by_id={},tensor_name_by_id={id(p):n for n,p in model.named_parameters()})
reports={};cp_paths={}
with tempfile.TemporaryDirectory(dir=root) as tmp:
    tmp=Path(tmp);rng=np.random.default_rng(91)
    for i in range(6):
        name=f'train{i:03d}' if i<5 else 'val000';count=270001 if i==0 else 10001
        tokens=rng.integers(0,128,count,dtype=np.uint16);header=np.zeros(256,dtype=np.int32);header[:3]=[20240520,1,count]
        with (tmp/(name+'.bin')).open('wb') as f:f.write(header.tobytes());f.write(tokens.tobytes())
    for model_key,spec in models.items():
        print('CHECK '+model_key,flush=True)
        code=(root/spec['script']).read_text(encoding='utf-8');tree=ast.parse(code)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Hyperparameters');cfg={n.target.id:ast.literal_eval(n.value) for n in cls.body if isinstance(n,ast.AnnAssign)}
        ns=dict(torch=torch,nn=nn,F=F,math=math,dataclass=dataclass,Path=Path,os=os,glob=glob,random=random,np=np,hashlib=hashlib,json=json,code=code)
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names],type_ignores=[]),'<real-functions>','exec'),ns)
        ns['args']=SimpleNamespace(**cfg)
        with torch.device('meta'):meta=ns['GPT'](ns['GPTConfig'](vocab_size=50304,n_embd=spec['width'],n_head=spec['width']//128,n_layer=spec['depth']))
        assert sum(p.numel() for p in meta.parameters())==spec['parameter_count'];del meta
        loader=ns['DistributedDataLoader'];pattern=str(tmp/'train*.bin');hashes=[]
        for seed in range(4000,4400):
            a=loader(pattern,1,8,0,2);b=loader(pattern,1,8,1,2);one=loader(pattern,2,8,0,1)
            for obj in (a,b,one):obj.start_mix([1,2],seed)
            ids=one.mix_order[:2];xa,ya=a.next_batch();xb,yb=b.next_batch();x,y=one.next_batch()
            assert torch.equal(torch.cat((xa,xb)),x) and torch.equal(torch.cat((ya,yb)),y)
            for row,index in enumerate(ids):
                sh=0 if index<one.mix_counts[0] else 1;local=int(index)-(0 if sh==0 else one.mix_counts[0]);start=local*8
                assert np.array_equal(x[row].cpu().numpy(),one.mix_arrays[sh][start:start+8]) and np.array_equal(y[row].cpu().numpy(),one.mix_arrays[sh][start+1:start+9])
            hashes.append(one.mix_digest)
        assert len(set(hashes))==400
        torch.manual_seed(17);np.random.seed(17);random.seed(17)
        tinycfg=dict(cfg,batch_size=2,sequence_length=8,device_batch_size=2,expected_world_size=1,compile_model=0)
        ns['args']=SimpleNamespace(**tinycfg)
        m=ns['GPT'](ns['GPTConfig'](vocab_size=128,n_embd=128,n_head=1,n_layer=1)).cuda();fixed=ns['build_fixed_norm_state'](m);opts=optimizers_for(m);scheds=[torch.optim.lr_scheduler.LambdaLR(o,lambda i:min((i+1)/924,1)) for o in opts]
        configure(ns,m,fixed,opts)
        train=loader(pattern,2,8,0,1);val=loader(str(tmp/'val*.bin'),2,8,0,1);train.current_position=16000*16;oldx,oldy=train.next_batch()
        cli=SimpleNamespace(branch='fanin_fixed',fork_step=16000,stage_start=16000,experiment='baseline',suite_mode='mix2',mix_seed=4000,mix_shards=[1,2],pulse_elr=.06,pulse_updates=0,stop_step=16270,probe_val_tokens=32)
        ns.update(cli=cli,train_loader=train,val_loader=val,schedulers=scheds,schedule_sha256='synthetic',master_process=False)
        history=ns['capture_training_state'](16000,oldx,oldy);history['source_sha256']=spec['parent_runtime_sha256'];history['probe_experiment']='stable';history['schedule_sha256']='9e3a4b48e9c2de544e2c1c7f35775c80f910b39717eed85ac45645434961c387'
        for k in ('probe_fork_step','probe_val_tokens','suite_mode','mix_seed','mix_shards'):history.pop(k)
        initial_if=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="payload['source_sha256'] != hashlib.sha256(code.encode()).hexdigest()")
        init_code=compile(ast.Module(body=[initial_if],type_ignores=[]),'<actual-init>','exec')
        def update(step):
            m.train();ns['apply_tensor_rms_elr'](opts,step)
            for o in opts:o.zero_grad(set_to_none=True)
            _,loss=m(ns['x'],ns['y']);loss.backward()
            for o in opts:o.step();o.zero_grad(set_to_none=True)
            for sched in scheds:sched.step()
            ns['apply_fixed_norm_control'](fixed,step);ns['x'],ns['y']=train.next_batch();return loss.detach().clone()
        traces={};initial_val=[];resume_cases=0
        for seed in (4000,4001,4199,4399):
            shards=random.Random(seed^0x51A2).sample([1,2,3,4],2)
            for arm,width,eta in [('baseline',0,.06),('elr001',120,.01)]:
                cli.fork_step=16000;cli.experiment=arm;cli.mix_seed=seed;cli.mix_shards=shards;cli.pulse_updates=width;cli.pulse_elr=eta;cli.stop_step=16270 if arm=='baseline' else 16370
                _,(ns['x'],ns['y'])=ns['restore_training_state'](history)
                before=(random.getstate(),torch.get_rng_state().clone(),torch.cuda.get_rng_state_all());ns['payload']=history;exec(init_code,ns)
                assert before[0]==random.getstate() and torch.equal(before[1],torch.get_rng_state()) and same(before[2],torch.cuda.get_rng_state_all())
                traces[(seed,arm)]=(tuple(shards),train.mix_digest,hashlib.sha256(ns['x'].cpu().numpy().tobytes()).hexdigest())
                m.eval();val.reset();vx,vy=val.next_batch()
                with torch.no_grad():_,vl=m(vx,vy)
                initial_val.append(float(vl));ns['schedule_sha256']='synthetic-'+arm
                ns['tensor_elr_schedule']={u:{n:ns['shared_elr'](u) for n,p in m.named_parameters() if p.ndim>=2} for u in (16001,16002,16120,16121)}
                for u in (16120,16121):
                    ns['apply_tensor_rms_elr'](opts,u)
                    for opt in opts:
                        for group in opt.param_groups:
                            p=group['params'][0]
                            if id(p) in ns['rmsnorm_gamma_param_ids']:assert group['lr']==.0018
                            else:assert math.isclose(group['lr']/ns['fixed_rms_by_id'][id(p)],ns['shared_elr'](u),rel_tol=1e-12)
                update(16001);state=ns['capture_training_state'](16001,ns['x'],ns['y']);path=tmp/'state.pt';torch.save(state,path);state=torch.load(path,weights_only=False)
                loss=update(16002);expected=ns['checkpoint_tree']((m.state_dict(),[o.state_dict() for o in opts],[o.state_dict() for o in scheds],ns['x'],ns['y'],ns['loader_state'](train)))
                cli.fork_step=16001;_,(ns['x'],ns['y'])=ns['restore_training_state'](state);position=train.mix_cursor;ns['payload']=state;exec(init_code,ns);assert train.mix_cursor==position
                assert torch.equal(loss,update(16002));actual=ns['checkpoint_tree']((m.state_dict(),[o.state_dict() for o in opts],[o.state_dict() for o in scheds],ns['x'],ns['y'],ns['loader_state'](train)));assert same(expected,actual)
                wrong=ns['checkpoint_tree'](state);wrong['mix_seed']=seed+1
                try:ns['restore_training_state'](wrong)
                except RuntimeError:pass
                else:raise AssertionError('wrong seed accepted')
                resume_cases+=1
        assert max(initial_val)==min(initial_val)
        for seed in (4000,4001,4199,4399):assert traces[(seed,'baseline')]==traces[(seed,'elr001')]
        cli.fork_step=16000
        wrong=ns['checkpoint_tree'](history);wrong['source_sha256']=next(v['parent_runtime_sha256'] for k,v in models.items() if k!=model_key)
        try:ns['restore_training_state'](wrong)
        except RuntimeError:pass
        else:raise AssertionError('wrong model source accepted')
        cp=tmp/model_key;cp.mkdir();torch.save(history,cp/'rank00000.pt');(cp/'complete.json').write_text(json.dumps(dict(version=1,step=16000,world_size=1,files=['rank00000.pt'])))
        cp_paths[model_key]=str(cp);info=load_info(cp,model_key);assert info['ranks']==1
        other=next(k for k in models if k!=model_key)
        try:load_info(cp,other)
        except ValueError:pass
        else:raise AssertionError('wrong-model planning accepted')
        # Real full architecture,short eager BF16 forward/backward+Adam+projection.
        ns['args']=SimpleNamespace(**cfg);fm=ns['GPT'](ns['GPTConfig'](vocab_size=50304,n_embd=spec['width'],n_head=spec['width']//128,n_layer=spec['depth'])).cuda()
        ff=ns['build_fixed_norm_state'](fm);fo=optimizers_for(fm);configure(ns,fm,ff,fo)
        ns['args'].compile_model=0;ns['args'].tensor_norm_every=0
        ns['tensor_elr_schedule']={16001:{n:.01 for n,p in fm.named_parameters() if p.ndim>=2}}
        ns['apply_tensor_rms_elr'](fo,16001)
        for opt in fo:opt.zero_grad(set_to_none=True)
        inputs=torch.randint(0,50304,(1,16),device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):_,floss=fm(inputs,inputs)
        assert torch.isfinite(floss);floss.backward()
        for opt in fo:opt.step();opt.zero_grad(set_to_none=True)
        ns['apply_fixed_norm_control'](ff,16001)
        for entry in ff:
            p=entry['param'];measured=p.float().square().mean().sqrt();assert torch.isclose(measured,entry['target_rms'],rtol=2e-5,atol=1e-8)
        assert sum(p.numel() for p in fm.parameters())==spec['parameter_count']
        reports[model_key]=dict(parameters=spec['parameter_count'],actual_context_permutations=400,representative_disk_resume_cases=resume_cases,full_architecture_short_cuda_step='PASS',full_step_loss=float(floss.detach()))
        del fm,fo,ff;gc.collect();torch.cuda.empty_cache()
    config=tmp/'models.json';config.write_text(json.dumps(dict(models=cp_paths)))
    command=[sys.executable,str(root/'run_small_mix2.py'),'--config',str(config),'--output',str(tmp/'both'),'--probe-val-tokens','32','--dry-run']
    plan=json.loads(subprocess.check_output(command,text=True));assert plan['total_runs']==1600 and all(len(v['jobs'])==800 for v in plan['models'].values())
(root/'checkpoint_verification.json').write_text(json.dumps(dict(status='PASS',python=sys.version,torch=str(torch.__version__),models=reports,scope='actual400 context streams/model,rank+label checks;representative source migrations and exact disk resume;real full parameter shapes withB1/T16 eager BF16 optimizer/projection;actual mmap1600-job plan;formalB128/T1024,DDP/compile and real user checkpoint untested'),indent=2),encoding='utf-8')
print('PASS:both real model architectures,400 streams each,16 tiny resumes and1600-job plan')
