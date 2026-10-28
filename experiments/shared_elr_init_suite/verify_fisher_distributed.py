"""Two-process CPU/Gloo SUM-before-square correctness (no CUDA required)."""
from pathlib import Path
import socket
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from fisher_diagnostics import probe, parameter_rms

class Tiny(torch.nn.Module):
    supports_fisher_hidden = True
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Module()
        self.block.mlp = torch.nn.Linear(3,3)
        self.lm_head = torch.nn.Linear(3,5,bias=False)
        self.width_multiplier = 1.0
    def forward(self,x,y=None,return_logits=True,return_hidden=False):
        h=self.block.mlp(x)
        return h if return_hidden else (self.lm_head(h),None)

def worker(rank, init_url):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=init_url,rank=rank,world_size=2,timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(67)
        m=Tiny()
        x=torch.randn(8,4,3)
        y=torch.zeros(8,4,dtype=torch.long)
        y[1,:]=-1
        y[6,1:]=-1
        for p in m.parameters(): p.grad=torch.randn_like(p)
        saved=[p.grad.clone() for p in m.parameters()]
        rng=torch.get_rng_state().clone()
        part=slice(rank*4,(rank+1)*4)
        out=probe(m,x[part],y[part],2,103,parameter_rms(m),micro_batch_size=2,distributed=True)
        assert out['vocabulary_head']=='chunked_hidden_chain_rule'
        assert out['world_size']==2 and out['sequences']==8 and out['local_sequences']==4
        assert out['accumulation_steps']==4 and out['local_accumulation_steps']==2
        assert out['draw_seeds_by_rank']==[[103,104],[105,106]]
        assert out['valid_tokens']==int((y!=-1).sum())
        assert torch.equal(rng,torch.get_rng_state()) and m.training
        for p,g in zip(m.parameters(),saved): assert torch.equal(p.grad,g)
        expected=[]
        # Explicit serial objective over ALL ranks with the identical sampled labels.
        for draw in range(2):
            loss=0
            for r in range(2):
                gen=torch.Generator().manual_seed(103+2*draw+r)
                for offset in range(r*4,(r+1)*4,2):
                    z=m(x[offset:offset+2],y[offset:offset+2])[0].reshape(-1,5)
                    labels=torch.multinomial(z.detach().softmax(-1),1,generator=gen).squeeze(-1)
                    labels[y[offset:offset+2].reshape(-1)==-1]=-1
                    loss=loss+torch.nn.functional.cross_entropy(z,labels,ignore_index=-1,reduction='sum')/out['valid_tokens']
            grads=torch.autograd.grad(loss,list(m.parameters()))
            expected.append({n:out['valid_tokens']*g.double().square().sum().item()/p.numel()
                             for (n,p),g in zip(m.named_parameters(),grads)})
        for name,v in out['parameters'].items():
            for draw in range(2):
                assert abs(v['mean_fisher_draws'][draw]-expected[draw][name])<2e-7
            assert abs(v['mc_standard_error']-abs(expected[0][name]-expected[1][name])/2)<2e-7
        shared=[None,None]
        dist.all_gather_object(shared,out['groups'])
        assert shared[0]==shared[1]
        if rank==0: print('PASS: two-rank Gloo global-token scaling, SUM before square, independent draws, serial objective agreement, equal rank results and state isolation',flush=True)
    finally:
        dist.destroy_process_group()

if __name__=='__main__':
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        port=listener.getsockname()[1]
    mp.spawn(worker,args=(f'tcp://127.0.0.1:{port}?use_libuv=0',),nprocs=2,join=True)
