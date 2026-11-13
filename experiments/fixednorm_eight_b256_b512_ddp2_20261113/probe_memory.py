"""Full-shape two-rank synthetic probe; no data/checkpoint writes."""
import argparse,importlib.util,json,os,sys,time
from pathlib import Path
import torch
import torch.distributed as dist
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--script',default='train_nesterov_b256_elr3p84_beta0p98.py');cli=p.parse_args()
    if int(os.environ.get('WORLD_SIZE','1'))!=2:raise ValueError('Use torchrun --standalone --nproc_per_node=2 probe_memory.py')
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
    try:
        spec=importlib.util.spec_from_file_location('probe_trial',ROOT/cli.script);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
        a=m.parser().parse_args([]);a.lr=a.effective_lr*(1-a.momentum);m.resolve_training_plan(a)
        torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
        raw=m.GPT(m.GPTConfig(vocab_size=a.vocab_size,n_layer=a.layers,n_head=a.heads,n_embd=a.width)).cuda();raw.loss_chunk_tokens=a.loss_chunk_tokens
        targets=m.targets_for(raw);opt=m.build_optimizer(raw,a);model=m.DDP(raw,device_ids=[rank],broadcast_buffers=False)
        class Tape:
            def batch(self,position,size,device):
                t=torch.arange(position*a.context,(position+size)*a.context+1,device=device)%a.vocab_size
                return t[:-1].view(size,a.context),t[1:].view(size,a.context)
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        for step in range(2):loss,grad=m.update(model,opt,Tape(),step*a.batch,a,m.base_lr(a,step),1.,targets,1.)
        torch.cuda.synchronize()
        row=dict(rank=rank,script=cli.script,global_batch=a.batch,per_rank_microbatch=a.microbatch,accumulation_per_rank=a.batch//(2*a.microbatch),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,seconds=time.perf_counter()-start,synthetic=True)
        rows=[None]*2;dist.all_gather_object(rows,row)
        if dist.get_rank()==0:print(json.dumps(rows,indent=2))
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
