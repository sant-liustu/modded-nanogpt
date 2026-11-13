"""Two-process CPU Gloo numerical comparison against the unchanged serial implementation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import ast,copy,importlib.util,json,math,socket,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
ROOT=Path(__file__).resolve().parent

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

class Tape:
    def batch(self,position,size,device,valid_tokens=None):
        valid=size*4 if valid_tokens is None else valid_tokens
        v=torch.arange(position*4,position*4+valid+1,device=device)%32
        x=torch.zeros(size*4,dtype=torch.long,device=device);y=torch.full_like(x,-1)
        x[:valid]=v[:-1];y[:valid]=v[1:]
        return x.view(size,4),y.view(size,4)

def worker(rank,port,tmp):
    torch.set_num_threads(1)
    print('initializing rank',rank,flush=True)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}?use_libuv=0',rank=rank,world_size=2)
    report=[]
    try:
        for i,j in enumerate(json.loads((ROOT/'manifest.json').read_text())['jobs']):
            m=module(ROOT/j['script'],'ddp_'+str(i));s=module(ROOT/'tests'/j['script'],'serial_'+str(i))
            a=m.parser().parse_args([]);m.resolve_training_plan(a)
            assert (a.batch,a.microbatch,a.updates,a.warmup_updates)==(j['batch'],64,j['updates'],j['warmup_updates'])
            ast.parse((ROOT/j['script']).read_text(),feature_version=(3,10))
            # All unchanged model/optimizer code must match original source.
            defs=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(t).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
            d,e=defs((ROOT/j['script']).read_text()),defs((ROOT/'tests'/j['script']).read_text())
            assert all(d[k]==e[k] for k in ['GPT','GPTConfig','MomentumOptimizer','build_optimizer','Tape','resolve_training_plan','project','base_lr'])
            torch.manual_seed(4)
            cfg=m.GPTConfig(vocab_size=32,n_layer=1,n_head=2,n_embd=16,scale_base_model=16)
            raw=m.GPT(cfg);raw.loss_chunk_tokens=3
            serial=s.GPT(s.GPTConfig(vocab_size=32,n_layer=1,n_head=2,n_embd=16,scale_base_model=16));serial.loss_chunk_tokens=3;serial.load_state_dict(raw.state_dict())
            ddp=m.DDP(raw,broadcast_buffers=False)
            # micro2 x two ranks x 2/4 slots, matching production accumulation counts.
            batch=8 if j['batch']==256 else 16
            a=SimpleNamespace(lr=j['actual_peak_lr'],momentum=j['momentum'],arm='fixed',context=4,batch=batch,microbatch=2,target_train_tokens=2*batch*4+5,device='cpu')
            o=m.build_optimizer(raw,a);so=s.build_optimizer(serial,a);targets=m.targets_for(raw);st=s.targets_for(serial)
            maxerr=0.
            for step in range(3):
                lr=a.lr*(.5 if step==0 else 1.)
                dl,dg=m.update(ddp,o,Tape(),step*batch,a,lr,1.,targets,1.)
                sl,sg=s.update(serial,so,Tape(),step*batch,a,lr,1.,st,1.)
                assert abs(dl-sl)<2e-5,(j['id'],dl,sl)
                assert abs(dg-sg)<2e-3,(dg,sg)
                for p,q in zip(raw.parameters(),serial.parameters()):
                    torch.testing.assert_close(p,q,rtol=8e-5,atol=8e-6)
                    maxerr=max(maxerr,float((p-q).abs().max()))
                flat=torch.cat([p.detach().flatten() for p in raw.parameters()]);ref=flat.clone();dist.broadcast(ref,0);assert torch.equal(flat,ref)
                if step==1:
                    path=Path(tmp)/('trial'+str(i)+'.pt')
                    contract={'tokens_per_update':batch*4,'target_train_tokens':a.target_train_tokens,'device':'cpu','world_size':2}
                    m.save(path,ddp,targets,2,contract,lr*2,o)
                    state=torch.load(path,weights_only=False)
                    assert state['position']==2*batch*4
                    assert not any(k.startswith('module.') for k in state['model'])
                    raw.load_state_dict(state['model']);o.load_state_dict(state['optimizer'])
            report.append(dict(id=j['id'],cpu_gloo_two_ranks=True,serial_match=True,max_parameter_abs_error=maxerr,partial_final_step_5_tokens=True,zero_token_rank=True,checkpoint_restore=True,ranks_bitwise_equal=True,accumulation_slots=j['accumulation_steps']))
            del ddp
        if rank==0:(ROOT/'ddp_verification.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,jobs=report,cloud_linux_nccl_full_model_tested=False),indent=2))
    finally:dist.destroy_process_group()

if __name__=='__main__':
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    with tempfile.TemporaryDirectory() as tmp:mp.spawn(worker,args=(port,tmp),nprocs=2,join=True)
    print('All eight trials passed two-process Gloo vs serial, tail masking and restore checks.')
