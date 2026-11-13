"""Synthetic full-shape memory probe, no dataset/checkpoint/output mutation."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script',default='train_nesterov_b256_elr3p84_beta0p98.py')
    parser.add_argument('--loss-chunk-tokens',type=int,default=1024)
    parser.add_argument('--steps',type=int,default=2)
    cli=parser.parse_args()
    if cli.steps<2 or cli.loss_chunk_tokens<=0:
        parser.error('At least two steps and a positive chunk size are required')
    spec=importlib.util.spec_from_file_location('memory_probe_training',ROOT/cli.script)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    args=module.parser().parse_args(['--loss-chunk-tokens',str(cli.loss_chunk_tokens)])
    args.lr=args.effective_lr*(1-args.momentum)
    module.resolve_training_plan(args)
    if not torch.cuda.is_available():
        raise RuntimeError('This full-shape probe requires a CUDA GPU')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    model=module.GPT(module.GPTConfig(vocab_size=args.vocab_size,n_layer=args.layers,
        n_head=args.heads,n_embd=args.width)).cuda()
    model.loss_chunk_tokens=args.loss_chunk_tokens
    targets=module.targets_for(model)
    optimizer=module.build_optimizer(model,args)

    class SyntheticTape:
        def batch(self,position,size,device):
            tokens=torch.arange(position*args.context,(position+size)*args.context+1,
                                device=device)%args.vocab_size
            return tokens[:-1].view(size,args.context),tokens[1:].view(size,args.context)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    for index in range(cli.steps):
        loss,grad=module.update(model,optimizer,SyntheticTape(),index*args.batch,args,
                               module.base_lr(args,index),1.,targets,1.)
    torch.cuda.synchronize()
    print(json.dumps(dict(script=cli.script,steps=cli.steps,batch=args.batch,
        microbatch=args.microbatch,width=args.width,layers=args.layers,context=args.context,
        loss_chunk_tokens=args.loss_chunk_tokens,last_loss=loss,preclip_gradient_norm=grad,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
        gpu_total_GiB=torch.cuda.get_device_properties(0).total_memory/2**30,
        seconds=time.perf_counter()-started,synthetic_tokens=True),indent=2))


if __name__=='__main__':
    main()
