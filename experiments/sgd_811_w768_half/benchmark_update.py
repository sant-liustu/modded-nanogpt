"""Read-only-to-other-runs diagnostic/update benchmark on an explicit device."""
import argparse
from functools import partial
import json
import math
import statistics
import time

import torch
from model import GPT, GPTConfig
from train import Tape, amp, update


def reference_update(model, optimizer, tape, position, batch, lr, micro, device,
                     fast_loss=False, fast_norm=False):
    model.train()
    optimizer.param_groups[0]['lr']=lr
    optimizer.zero_grad(set_to_none=True)
    total=0.
    for offset in range(0,batch,micro):
        count=min(micro,batch-offset)
        x,y=tape.batch(position+offset,count,device)
        with amp(device):
            _,loss=model(x,y,return_logits=False)
        if fast_loss:
            loss_value=float(loss.detach())
            if not math.isfinite(loss_value):
                raise FloatingPointError('loss')
        elif not torch.isfinite(loss):
            raise FloatingPointError('loss')
        (loss*(count/batch)).backward()
        total+=(loss_value if fast_loss else float(loss.detach()))*count/batch
    if fast_norm:
        grads=[p.grad.detach().float() for p in model.parameters() if p.grad is not None]
        norm=float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads,2))))
    else:
        norm=float(sum(p.grad.detach().float().square().sum()
                       for p in model.parameters() if p.grad is not None).sqrt())
    if not math.isfinite(norm):
        raise FloatingPointError('gradient')
    optimizer.step()
    return total,norm


def check_update(tape):
    torch.manual_seed(3)
    a=GPT(GPTConfig(n_layer=1,n_head=2,n_embd=16))
    b=GPT(GPTConfig(n_layer=1,n_head=2,n_embd=16))
    b.load_state_dict(a.state_dict())
    oa=torch.optim.SGD(a.parameters(),lr=.01)
    ob=torch.optim.SGD(b.parameters(),lr=.01)
    norm_errors=[]
    for i in range(3):
        la,na=reference_update(a,oa,tape,i*4,4,.01,4,'cpu')
        lb,nb=update(b,ob,tape,i*4,4,.01,4,'cpu')
        assert la==lb
        norm_errors.append(abs(na-nb)/max(abs(na),1e-30))
        for pa,pb in zip(a.parameters(),b.parameters()):
            assert torch.equal(pa,pb)
    assert max(norm_errors)<1e-5
    # Nonfinite backward gradients must stop before SGD modifies weights.
    for bad in (float('nan'),float('inf')):
        snapshot={k:v.clone() for k,v in b.state_dict().items()}
        parameter=next(b.parameters())
        hook=parameter.register_hook(lambda grad: torch.full_like(grad,bad))
        try:
            update(b,ob,tape,0,4,.01,4,'cpu')
            raise AssertionError('nonfinite gradient was not rejected')
        except FloatingPointError:
            pass
        finally:
            hook.remove()
        assert all(torch.equal(v,snapshot[k]) for k,v in b.state_dict().items())
    # A nonfinite forward must not reach backward or SGD.
    hook=b.register_forward_hook(lambda m,args,out: (out[0],out[1]*float('nan')))
    snapshot={k:v.clone() for k,v in b.state_dict().items()}
    try:
        update(b,ob,tape,0,4,.01,4,'cpu')
        raise AssertionError('nonfinite loss was not rejected')
    except FloatingPointError:
        pass
    finally:
        hook.remove()
    assert all(torch.equal(v,snapshot[k]) for k,v in b.state_dict().items())
    return dict(cpu_parameter_updates_bitwise_equal=True,
                max_norm_relative_error=max(norm_errors),nonfinite_checks_passed=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--context',type=int,default=1024)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--microbatch',type=int,default=None)
    args=p.parse_args()
    torch.set_num_threads(1)
    correctness=check_update(Tape(args.train,16))
    tape=Tape(args.train,args.context)
    torch.manual_seed(0)
    model=GPT(GPTConfig(n_layer=12,n_head=6,n_embd=768)).cuda()
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    variants=dict(original=reference_update,
                  loss_only=partial(reference_update,fast_loss=True),
                  norm_only=partial(reference_update,fast_norm=True),
                  optimized=update)
    results={name:[] for name in variants}
    micro=args.microbatch or args.batch
    for round_index in range(args.rounds):
        names=list(variants)
        if round_index%2:
            names.reverse()
        for name in names:
            model.load_state_dict(initial)
            fn=variants[name]
            for _ in range(3):
                fn(model,optimizer,tape,0,args.batch,.01,micro,'cuda')
            torch.cuda.synchronize()
            start=time.perf_counter()
            for i in range(args.steps):
                fn(model,optimizer,tape,(i%8)*args.batch,args.batch,.01,micro,'cuda')
            torch.cuda.synchronize()
            ms=1000*(time.perf_counter()-start)/args.steps
            results[name].append(ms)
            print(json.dumps(dict(round=round_index,variant=name,ms=ms)),flush=True)
    grads=[p.grad.detach().float() for p in model.parameters() if p.grad is not None]
    original=float(sum(g.square().sum() for g in grads).sqrt())
    optimized=float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads,2))))
    gpu_error=abs(original-optimized)/max(abs(original),1e-30)
    assert gpu_error<1e-5
    medians={k:statistics.median(v) for k,v in results.items()}
    report=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,
                model='width768_depth12_heads6',context=args.context,batch=args.batch,
                microbatch=micro,steps_per_trial=args.steps,rounds=args.rounds,
                timing_ms=results,median_ms=medians,
                step_time_reduction=1-medians['optimized']/medians['original'],
                correctness=correctness,same_gpu_gradients_norm_relative_error=gpu_error,
                scope='local GPU; excludes validation and checkpoint writes; not server speed proof')
    with open(args.output,'w') as f:
        json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
