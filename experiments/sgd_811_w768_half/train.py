"""Single-GPU, exactly paired integer-grid 8-1-1 SGD experiment."""
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


def schedule(n, eta=0.01):
    if n < 15:
        raise ValueError('canonical updates must be at least 15')
    # Nearest complete groups for the two 10% tails (half-up rounding).
    # Keep total tokens exact by assigning the remaining updates to the prefix.
    middle = 4 * max(1, (n + 20) // 40)
    last = 10 * max(1, (n + 50) // 100)
    prefix = n - middle - last
    g = np.concatenate((np.ones(prefix, dtype=np.int64),
                        np.full(middle // 4, 4, dtype=np.int64),
                        np.full(last // 10, 10, dtype=np.int64)))
    assert prefix > 0 and int(g.sum()) == n
    return g, prefix, 1.0


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


def update(model, optimizer, tape, position, batch, lr, micro, device):
    model.train()
    optimizer.param_groups[0]['lr'] = lr
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    for offset in range(0, batch, micro):
        count = min(micro, batch-offset)
        x, y = tape.batch(position+offset, count, device)
        with amp(device):
            _, loss = model(x, y, return_logits=False)
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise FloatingPointError('nonfinite training loss; stopping without clipping')
        (loss * (count/batch)).backward()
        total += loss_value * count/batch
    grads = [p.grad.detach().float() for p in model.parameters() if p.grad is not None]
    grad_norm = float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads, 2))))
    if not math.isfinite(grad_norm):
        raise FloatingPointError('nonfinite gradient; stopping without clipping')
    optimizer.step()
    return total, grad_norm


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


def save(path, model, macro, position, updates, contract):
    tmp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), macro=macro, position=position,
                    updates=updates, contract=contract, rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []), tmp)
    os.replace(tmp, path)


def save_prefix_milestone(out, model, arm, macro, position, updates, contract):
    if arm != 'prefix' or updates != 200_000:
        return None
    path = out/'prefix_step200000.pt'
    if not path.exists():
        save(path, model, macro, position, updates, contract)
    return path


def run(args):
    g, prefix, scale = schedule(args.updates, args.lr)
    train = Tape(args.train, args.context)
    val = Tape(args.val, args.context)
    needed = args.updates*args.batch*args.context+1
    if needed > train.tokens:
        raise ValueError(f'need {needed:,} training tokens, have {train.tokens:,}; provide more shards or reduce --updates')
    if args.validation_contexts*args.context+1 > val.tokens:
        raise ValueError('insufficient validation tokens')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    contract = dict(updates=args.updates, lr=args.lr, batch=args.batch, context=args.context,
                    seed=args.seed, width=args.width, layers=args.layers, heads=args.heads,
                    train=[[str(Path(f).resolve()), Path(f).stat().st_size, Path(f).stat().st_mtime_ns] for f in train.files],
                    val=[[str(Path(f).resolve()), Path(f).stat().st_size, Path(f).stat().st_mtime_ns] for f in val.files],
                    validation_contexts=args.validation_contexts,
                    model_sha256=hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest(),
                    schedule_sha256=hashlib.sha256(g.tobytes()).hexdigest(),
                    microbatch=args.microbatch, validation_batch_size=args.validation_batch_size,
                    device=args.device)
    summary = dict(contract=contract, macros=len(g), prefix_macros=prefix, tail_scale=scale,
                   g_histogram={str(k):int((g==k).sum()) for k in np.unique(g)},
                   terminal_T=len(g)*args.lr, tokens=needed-1,
                   optimizer=dict(name='SGD', momentum=0, weight_decay=0, clipping=False, norm_control=False))
    manifest = out/'config.json'
    if manifest.exists() and json.loads(manifest.read_text()) != summary:
        raise ValueError('output directory belongs to a different experiment')
    manifest.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    if args.preflight:
        return
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    model = GPT(GPTConfig(n_layer=args.layers, n_head=args.heads, n_embd=args.width)).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0, weight_decay=0)
    arms = ['prefix','lr','batch'] if args.arm == 'all' else [args.arm]
    for arm in arms:
        checkpoint = out/f'{arm}.pt'
        source = checkpoint if checkpoint.exists() else (out/'prefix.pt' if arm != 'prefix' else None)
        macro = position = updates = 0
        if source is not None:
            state = torch.load(source, map_location=args.device, weights_only=False)
            if state['contract'] != contract:
                raise ValueError('checkpoint contract mismatch')
            model.load_state_dict(state['model'])
            macro, position, updates = state['macro'], state['position'], state['updates']
            torch.set_rng_state(state['rng'].cpu())
            if args.device == 'cuda':
                torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
            del state
            if arm != 'prefix' and macro < prefix:
                raise ValueError('shared prefix is incomplete')
        end = prefix if arm == 'prefix' else len(g)
        trace = out/f'{arm}.jsonl'
        # Resume truncates only records after the saved checkpoint.
        if trace.exists():
            rows = [json.loads(s) for s in trace.read_text().splitlines()]
            trace.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['macro'] <= macro))
        start_time = time.monotonic()
        def record(loss=None, grad=None):
            norm = float(sum(p.detach().float().square().sum() for p in model.parameters()).sqrt())
            row = dict(arm=arm, macro=macro, T=macro*args.lr, contexts=position,
                       tokens=position*args.context, updates=updates,
                       validation_loss=evaluate(model,val,args.validation_contexts,args.validation_batch_size,args.device),
                       train_loss=loss, gradient_norm=grad, parameter_norm=norm,
                       seconds=time.monotonic()-start_time)
            if macro:
                value = int(g[macro-1])
                row.update(lr=args.lr if arm=='batch' else args.lr/value,
                           batch=args.batch*value if arm=='batch' else args.batch,
                           ratio=args.batch*value/args.lr)
            with trace.open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
        if macro == (0 if arm == 'prefix' else prefix):
            record()
        for index in range(macro,end):
            value = int(g[index])
            repeats = 1 if arm == 'batch' else value
            size = args.batch*value if arm == 'batch' else args.batch
            lr = args.lr if arm == 'batch' else args.lr/value
            for _ in range(repeats):
                micro = size if args.microbatch is None else args.microbatch
                loss, grad = update(model,optimizer,train,position,size,lr,micro,args.device)
                position += size
                updates += 1
            macro = index+1
            milestone = save_prefix_milestone(out,model,arm,macro,position,updates,contract)
            if milestone is not None:
                print(f'Preserved milestone checkpoint: {milestone}', flush=True)
            if macro % args.eval_every == 0 or macro == end:
                record(loss,grad)
                save(checkpoint,model,macro,position,updates,contract)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--train',default='F:/datasets/fineweb10B-gpt2/raw/fineweb_train_*.bin')
    p.add_argument('--val',default='F:/datasets/fineweb10B-gpt2/raw/fineweb_val_*.bin')
    p.add_argument('--output',required=True)
    p.add_argument('--updates',type=int,required=True)
    p.add_argument('--lr',type=float,default=0.01)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--microbatch',type=int,default=None,
                   help='Optional cap per forward/backward; omitted uses the full current global batch')
    p.add_argument('--validation-batch-size',type=int,default=32)
    p.add_argument('--context',type=int,default=1024)
    p.add_argument('--width',type=int,default=768)
    p.add_argument('--layers',type=int,default=12)
    p.add_argument('--heads',type=int,default=6)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--validation-contexts',type=int,default=512)
    p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--arm',choices=['prefix','lr','batch','all'],default='all')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--preflight',action='store_true')
    args=p.parse_args()
    if min(args.batch,args.validation_batch_size,args.context,args.eval_every,args.validation_contexts) <= 0 or args.lr <= 0 or (args.microbatch is not None and args.microbatch <= 0):
        p.error('sizes, cadence, and learning rate must be positive')
    run(args)
