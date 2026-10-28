"""Three-arm projected-SGD WSD experiment, single GPU, exact tape resume."""
import argparse
import contextlib
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from model import GPT, GPTConfig


class Tape:
    def __init__(self, pattern, context):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(pattern)
        self.arrays = []
        for name in self.files:
            header = np.fromfile(name, dtype=np.int32, count=256)
            if len(header) != 256 or tuple(header[:2]) != (20240520, 1):
                raise ValueError(f'invalid FineWeb shard: {name}')
            count = int(header[2])
            if Path(name).stat().st_size != 1024 + count * 2:
                raise ValueError(f'truncated shard: {name}')
            self.arrays.append(np.memmap(name, mode='r', dtype=np.uint16,
                                        offset=1024, shape=(count,)))
        self.ends = np.cumsum([len(a) for a in self.arrays])
        self.context = context
        self.tokens = int(self.ends[-1])

    def batch(self, position, size, device):
        start, end = position*self.context, (position+size)*self.context+1
        if end > self.tokens:
            raise ValueError('training tape exhausted; implicit data repetition is disabled')
        chunks = []
        while start < end:
            index = int(np.searchsorted(self.ends, start, side='right'))
            offset = int(self.ends[index-1]) if index else 0
            stop = min(end, int(self.ends[index]))
            chunks.append(np.asarray(self.arrays[index][start-offset:stop-offset]))
            start = stop
        values = torch.from_numpy(np.concatenate(chunks).astype(np.int64)).to(device)
        return values[:-1].view(size, self.context), values[1:].view(size, self.context)


def amp(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, tape, count, micro, device):
    model.eval()
    loss = 0.
    for pos in range(0, count, micro):
        size = min(micro, count-pos)
        x, y = tape.batch(pos, size, device)
        with amp(device):
            _, value = model(x, y, return_logits=False)
        loss += float(value) * size/count
    if not math.isfinite(loss):
        raise FloatingPointError('nonfinite validation loss')
    return loss


@contextlib.contextmanager
def step_trace(path, restored_updates):
    # Stream recovery: discard rows ahead of the checkpoint, without loading
    # hundreds of thousands of records into memory. A partial final line can
    # result from interruption; malformed interior records are an error.
    if path.exists():
        tmp = path.with_suffix('.recovering')
        with path.open() as src, tmp.open('w') as dst:
            for line in src:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if src.read().strip():
                        raise ValueError(f'malformed interior training log: {path}')
                    break
                if row['updates'] <= restored_updates:
                    dst.write(line.rstrip('\n')+'\n')
        os.replace(tmp, path)
    with path.open('a', buffering=65536) as stream:
        yield stream



ARMS = {'fixed': 0, 'sine_p1': 1, 'sine_p2': 2}


def boundaries(args):
    return (max(1, round(args.updates * args.warmup_fraction)),
            args.updates - max(1, round(args.updates * args.decay_fraction)))


def base_lr(args, index):
    warm, decay = boundaries(args)
    if index < warm:
        return args.lr * (index + 1) / warm
    if index >= decay:
        return args.lr * (args.updates - index) / (args.updates - decay)
    return args.lr


def scale_at(arm, completed, total):
    return 1. if arm == 'fixed' else 1. + .5 * math.sin(2 * math.pi * completed / total)


@torch.no_grad()
def targets_for(model):
    targets = {n: float(p.float().square().mean().sqrt())
               for n,p in model.named_parameters() if p.requires_grad and p.ndim == 2}
    if not targets or any(not math.isfinite(v) or v <= 0 for v in targets.values()):
        raise ValueError('invalid initial RMS targets')
    return targets


@torch.no_grad()
def project(model, targets, scale):
    params = dict(model.named_parameters())
    norms = torch.stack([params[n].float().square().mean().sqrt() for n in targets])
    if not bool((torch.isfinite(norms) & (norms > 0)).all()):
        raise FloatingPointError('invalid norm after update')
    for (name, target), current in zip(targets.items(), norms):
        params[name].mul_(target * scale / current)


@torch.no_grad()
def diagnostics(model, targets, scale, lr_matrix, lr_gamma):
    result = {}
    for name, p in model.named_parameters():
        rms = float(p.float().square().mean().sqrt())
        norm = rms * math.sqrt(p.numel())
        lr = lr_matrix if p.ndim == 2 else lr_gamma
        result[name] = dict(norm=norm, rms=rms, lr=lr,
                           lr_over_norm=lr / norm, lr_over_norm2=lr / norm**2,
                           lr_over_rms=lr / rms, lr_over_rms2=lr / rms**2)
        if name in targets:
            result[name]['target_rms'] = targets[name]*scale
            result[name]['relative_error'] = abs(rms/(targets[name]*scale)-1)
    return result


def update(model, optimizer, tape, position, args, lr, scale, targets, next_scale):
    model.train()
    optimizer.param_groups[0]['lr'] = lr * scale**ARMS[args.arm]
    optimizer.param_groups[1]['lr'] = lr
    optimizer.zero_grad(set_to_none=True)
    micro = args.microbatch or args.batch
    total = 0.
    for offset in range(0, args.batch, micro):
        count = min(micro, args.batch-offset)
        x,y = tape.batch(position+offset, count, args.device)
        with amp(args.device):
            _,loss = model(x,y,return_logits=False)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError('nonfinite training loss')
        (loss * (count/args.batch)).backward()
        total += value*count/args.batch
    grads = [p.grad.detach().float() for p in model.parameters() if p.grad is not None]
    grad = float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads,2))))
    if not math.isfinite(grad):
        raise FloatingPointError('nonfinite gradient')
    optimizer.step()
    project(model, targets, next_scale)
    return total,grad


def save(path, model, targets, updates, contract, cumulative_lr):
    tmp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), norm_targets=targets, updates=updates,
                    position=updates*contract['batch'], contract=contract,
                    cumulative_lr=cumulative_lr, rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if contract['device']=='cuda' else []),tmp)
    os.replace(tmp,path)


def run(args):
    if min(args.updates,args.batch,args.context,args.width,args.layers,args.heads,
           args.validation_contexts,args.validation_batch_size,args.eval_every,
           args.checkpoint_every,args.dense_every) <= 0 or args.lr <= 0:
        raise ValueError('positive sizes and LR required')
    if args.microbatch is not None and args.microbatch <= 0:
        raise ValueError('microbatch must be positive')
    if args.width % args.heads or (args.width//args.heads)%2:
        raise ValueError('width must support an even RoPE head dimension')
    if not (0 < args.warmup_fraction < 1 and 0 < args.decay_fraction < 1
            and args.warmup_fraction+args.decay_fraction < 1):
        raise ValueError('invalid WSD fractions')
    warm,decay = boundaries(args)
    if warm >= decay:
        raise ValueError('too few updates for a WSD plateau')
    train,val = Tape(args.train,args.context),Tape(args.val,args.context)
    if args.updates*args.batch*args.context+1 > train.tokens:
        raise ValueError('insufficient training tokens; no implicit repetition')
    if args.validation_contexts*args.context+1 > val.tokens:
        raise ValueError('insufficient validation tokens')
    contract = {k:v for k,v in vars(args).items() if k not in ('output','preflight')}
    for key,tape in [('train',train),('val',val)]:
        contract[key]=[[str(Path(f).resolve()),Path(f).stat().st_size,Path(f).stat().st_mtime_ns] for f in tape.files]
    contract['model_sha256']=hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest()
    contract['policy']='wsd_sine_2d_gamma_base_v1'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    manifest=out/'config.json'
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError('output belongs to a different experiment')
    manifest.write_text(json.dumps(contract,indent=2),encoding='utf-8')
    if args.preflight:
        return
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    if args.device=='cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32=True
    model=GPT(GPTConfig(n_layer=args.layers,n_head=args.heads,n_embd=args.width)).to(args.device)
    targets=targets_for(model)
    optimizer=torch.optim.SGD([
        {'params':[p for p in model.parameters() if p.ndim==2]},
        {'params':[p for p in model.parameters() if p.ndim!=2]}],lr=args.lr,momentum=0,weight_decay=0)
    completed=0;cumulative_lr=0.
    checkpoint=out/'latest.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location=args.device,weights_only=False)
        if state['contract'] != contract:
            raise ValueError('checkpoint contract mismatch')
        model.load_state_dict(state['model'])
        if set(state['norm_targets']) != set(targets) or any(not math.isfinite(v) or v<=0 for v in state['norm_targets'].values()):
            raise ValueError('invalid restored targets')
        targets=state['norm_targets'];completed=state['updates']
        if state['position'] != completed*args.batch or not 0<=completed<=args.updates:
            raise ValueError('invalid checkpoint cursor')
        check=diagnostics(model,targets,scale_at(args.arm,completed,args.updates),args.lr,args.lr)
        if max(r.get('relative_error',0) for r in check.values())>3e-6:
            raise ValueError('checkpoint norm does not match schedule')
        cumulative_lr=state['cumulative_lr']
        torch.set_rng_state(state['rng'].cpu())
        if args.device=='cuda':
            torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        del state
    started=time.monotonic()
    with step_trace(out/'steps.jsonl',completed) as steps, step_trace(out/'metrics.jsonl',completed) as metrics:
        def measure():
            # Diagnostics describe current checkpoint weights and NEXT update LR.
            index=min(completed,args.updates-1)
            scale=scale_at(args.arm,completed,args.updates)
            lr=base_lr(args,index)
            row=dict(updates=completed,tokens=completed*args.batch*args.context,
                     validation_loss=evaluate(model,val,args.validation_contexts,args.validation_batch_size,args.device),
                     parameter_norm=float(torch.stack([p.detach().float().square().sum() for p in model.parameters()]).sum().sqrt()),
                     norm_scale=scale,seconds=time.monotonic()-started,
                     diagnostic_lr_semantics='next_update; final uses last base LR',
                     tensors=diagnostics(model,targets,scale,lr*scale**ARMS[args.arm],lr))
            metrics.write(json.dumps(row)+'\n');metrics.flush()
            print(json.dumps({k:v for k,v in row.items() if k!='tensors'}),flush=True)
        if completed==0:
            # A pre-first-checkpoint restart must not duplicate the initial metric.
            metrics.seek(0,2)
            if (out/'metrics.jsonl').stat().st_size==0:
                measure()
        for index in range(completed,args.updates):
            scale=scale_at(args.arm,index,args.updates)
            next_scale=scale_at(args.arm,index+1,args.updates)
            lr=base_lr(args,index);matrix_lr=lr*scale**ARMS[args.arm]
            loss,grad=update(model,optimizer,train,index*args.batch,args,lr,scale,targets,next_scale)
            completed=index+1;cumulative_lr+=matrix_lr
            steps.write(json.dumps(dict(updates=completed,tokens=completed*args.batch*args.context,
                train_loss=loss,gradient_norm=grad,base_lr=lr,matrix_lr=matrix_lr,gamma_lr=lr,
                norm_scale_used=scale,norm_scale_after=next_scale,
                matrix_lr_over_scale=matrix_lr/scale,matrix_lr_over_scale2=matrix_lr/scale**2,
                cumulative_matrix_lr=cumulative_lr,batch=args.batch))+'\n')
            if completed%100==0:steps.flush()
            special=completed in (warm,decay,args.updates)
            dense=any(abs(completed-b)<=args.dense_window for b in (warm,decay))
            if completed%args.eval_every==0 or special or (dense and completed%args.dense_every==0):
                measure()
            if completed%args.checkpoint_every==0 or special or completed==200000:
                steps.flush();metrics.flush()
                save(checkpoint,model,targets,completed,contract,cumulative_lr)
                if completed==200000:
                    save(out/'step200000.pt',model,targets,completed,contract,cumulative_lr)


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--train',default='F:/datasets/fineweb10B-gpt2/raw/fineweb_train_*.bin')
    p.add_argument('--val',default='F:/datasets/fineweb10B-gpt2/raw/fineweb_val_*.bin')
    p.add_argument('--output',required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    for flag,default in [('updates',301728),('batch',4),('context',1024),('width',768),('layers',12),('heads',6),('seed',0),('validation-contexts',512),('validation-batch-size',32),('eval-every',200),('checkpoint-every',1000),('dense-window',200),('dense-every',20)]:
        p.add_argument('--'+flag,type=int,default=default)
    p.add_argument('--microbatch',type=int,default=None)
    p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--warmup-fraction',type=float,default=4000/81600)
    p.add_argument('--decay-fraction',type=float,default=23200/81600)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--preflight',action='store_true')
    return p


if __name__=='__main__':
    run(parser().parse_args())
