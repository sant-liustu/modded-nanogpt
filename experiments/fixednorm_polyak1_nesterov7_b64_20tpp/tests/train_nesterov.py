"""Fixednorm Polyak for matrices and RMSNorm scales, single GPU, exact tape resume."""
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
from dataclasses import dataclass
from torch import nn
import torch.nn.functional as F

# Model copied from the 2026-11-03 w768 fixednorm script.
class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = self.q_norm(q), self.k_norm(k) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_norm = RMSNorm(config.n_embd)
        self.mlp_norm = RMSNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 3 # head dim 128 suggested by @Grad62304977
    n_embd : int = 384
    init_std : float = 0.02
    scale_emb : float = 1.0
    scale_base_model : int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.width_multiplier = config.n_embd / config.scale_base_model

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        self.final_norm = RMSNorm(config.n_embd)
        self.apply(self._init_mup_weights)
        # Final effective initialization; tied embedding stays N(0, 0.02^2).
        hidden_std = config.init_std / math.sqrt(self.width_multiplier)
        residual_std = config.init_std / math.sqrt(2 * config.n_layer * self.width_multiplier)
        for pn, p in self.named_parameters():
            if pn.startswith('transformer.h.') and pn.endswith('.weight') and p.ndim == 2:
                std = 1.0 / math.sqrt(p.shape[1])  # fixednorm entry: original fanin_fixed branch
                torch.nn.init.normal_(p, mean=0.0, std=std)

    def _init_mup_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = x * self.config.scale_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.final_norm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x) / self.width_multiplier
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) / self.width_multiplier # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader



OPTIMIZER = 'nesterov'

class MomentumOptimizer(torch.optim.Optimizer):
    """Unnormalized heavy-ball or explicit lookahead NAG; fixednorm is external."""
    def __init__(self, groups, lr, momentum):
        super().__init__(groups, dict(lr=lr, momentum=momentum))
        self.at_lookahead = False
        self._base = {}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Use explicit accumulated backward; closures are unsupported')
        if self.at_lookahead:
            raise RuntimeError('Restore base parameters before step')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('Dense gradients required')
                state = self.state[p]
                if OPTIMIZER == 'polyak':
                    if 'previous_parameter' not in state:
                        state['previous_parameter'] = p.detach().clone()
                    previous = state['previous_parameter']
                    delta = (p - previous).mul_(group['momentum'])
                    delta.add_(p.grad, alpha=-group['lr'])
                    previous.copy_(p)
                    p.add_(delta)
                else:
                    if 'velocity' not in state:
                        state['velocity'] = torch.zeros_like(p)
                    velocity = state['velocity']
                    velocity.mul_(group['momentum']).add_(p.grad, alpha=-group['lr'])
                    p.add_(velocity)

    @contextlib.contextmanager
    def lookahead(self):
        if self.at_lookahead:
            raise RuntimeError('Nested lookahead is unsupported')
        self.at_lookahead = True
        try:
            with torch.no_grad():
                for group in self.param_groups:
                    for p in group['params']:
                        state = self.state[p]
                        if 'velocity' not in state:
                            state['velocity'] = torch.zeros_like(p)
                        self._base[p] = p.detach().clone()
                        p.add_(state['velocity'], alpha=group['momentum'])
            yield
        finally:
            with torch.no_grad():
                for p, value in self._base.items():
                    p.copy_(value)
            self._base.clear()
            self.at_lookahead = False


def build_optimizer(model, args):
    groups = [dict(params=[p for p in model.parameters() if p.ndim == 2]),
              dict(params=[p for p in model.parameters() if p.ndim != 2])]
    if OPTIMIZER == 'sgd':
        return torch.optim.SGD(groups, lr=args.lr, momentum=0, weight_decay=0)
    return MomentumOptimizer(groups, lr=args.lr, momentum=args.momentum)




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

    def batch(self, position, size, device, valid_tokens=None):
        valid_tokens = size*self.context if valid_tokens is None else valid_tokens
        if not 0 < valid_tokens <= size*self.context:
            raise ValueError('invalid token count')
        start = position*self.context
        end = start + valid_tokens + 1
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
        if valid_tokens == size*self.context:
            return values[:-1].view(size, self.context), values[1:].view(size, self.context)
        x = torch.zeros(size*self.context, dtype=torch.long, device=device)
        y = torch.full_like(x, -1)
        x[:valid_tokens] = values[:-1]
        y[:valid_tokens] = values[1:]
        return x.view(size, self.context), y.view(size, self.context)


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


def resolve_training_plan(args):
    # Unique parameters: tied embedding/head once; two QK gammas per block.
    width, head_dim = args.width, args.width // args.heads
    args.parameter_count = args.vocab_size*width + args.layers*(
        12*width*width + 2*width + 2*head_dim) + width
    args.target_train_tokens = 20 * args.parameter_count
    args.tokens_per_update = args.batch * args.context
    args.updates = math.ceil(args.target_train_tokens / args.tokens_per_update)
    args.warmup_updates = max(1, math.ceil(args.updates * args.warmup_fraction))
    args.last_update_tokens = args.target_train_tokens - (args.updates-1)*args.tokens_per_update


def boundaries(args):
    return (args.warmup_updates, args.updates)


def base_lr(args, index):
    return args.lr * min(1., (index+1)/args.warmup_updates)


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
    step_tokens = min(args.batch*args.context, args.target_train_tokens-position*args.context)
    rows = math.ceil(step_tokens/args.context)
    lookahead = optimizer.lookahead() if OPTIMIZER == 'nesterov' else contextlib.nullcontext()
    with lookahead:
        for offset in range(0, rows, micro):
            count = min(micro, rows-offset)
            valid = min(count*args.context, step_tokens-offset*args.context)
            if valid != count*args.context:
                x,y = tape.batch(position+offset, count, args.device, valid_tokens=valid)
            else:
                x,y = tape.batch(position+offset, count, args.device)
            with amp(args.device):
                _,loss = model(x,y,return_logits=False)
            value = float(loss.detach())
            if not math.isfinite(value):
                raise FloatingPointError('nonfinite training loss')
            (loss * (valid/step_tokens)).backward()
            total += value*valid/step_tokens
    # Clip the complete accumulated gradient, including RMSNorm gamma.
    # The returned norm (and logged gradient_norm) is measured BEFORE clipping.
    grad = float(torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=50.0, norm_type=2.0, error_if_nonfinite=True))
    if not math.isfinite(grad):
        raise FloatingPointError('nonfinite gradient')
    optimizer.step()
    project(model, targets, next_scale)
    return total,grad


def save(path, model, targets, updates, contract, cumulative_lr, optimizer):
    tmp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), norm_targets=targets, updates=updates,
                    position=min(updates*contract['tokens_per_update'], contract['target_train_tokens']), contract=contract,
                    cumulative_lr=cumulative_lr, rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if contract['device']=='cuda' else []),tmp)
    os.replace(tmp,path)


def run(args):
    if not math.isfinite(args.effective_lr) or args.effective_lr <= 0:
        raise ValueError('effective LR must be positive and finite')
    if not math.isfinite(args.momentum) or not 0 <= args.momentum < 1:
        raise ValueError('momentum must be finite and in [0,1)')
    if OPTIMIZER == 'sgd' and args.momentum != 0:
        raise ValueError('SGD baseline requires momentum=0')
    args.lr = args.effective_lr * (1 - args.momentum)
    if min(args.vocab_size,args.batch,args.context,args.width,args.layers,args.heads,
           args.validation_contexts,args.validation_batch_size,args.eval_every,
           args.checkpoint_every,args.dense_every) <= 0 or args.lr <= 0:
        raise ValueError('positive sizes and LR required')
    if args.microbatch is not None and args.microbatch <= 0:
        raise ValueError('microbatch must be positive')
    if args.width % args.heads or (args.width//args.heads)%2:
        raise ValueError('width must support an even RoPE head dimension')
    if not 0 < args.warmup_fraction < 1:
        raise ValueError('warmup fraction must be in (0,1)')
    resolve_training_plan(args)
    warm, _ = boundaries(args)
    train,val = Tape(args.train,args.context),Tape(args.val,args.context)
    if args.target_train_tokens+1 > train.tokens:
        raise ValueError('insufficient training tokens; no implicit repetition')
    if args.validation_contexts*args.context+1 > val.tokens:
        raise ValueError('insufficient validation tokens')
    contract = {k:v for k,v in vars(args).items() if k not in ('output','preflight')}
    for key,tape in [('train',train),('val',val)]:
        contract[key]=[[str(Path(f).resolve()),Path(f).stat().st_size,Path(f).stat().st_mtime_ns] for f in tape.files]
    contract['model_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    contract['policy']='fixednorm_all_parameters_'+OPTIMIZER+'_embedded_fanin_warmup_constant_20tpp_v3'
    contract['optimizer']=OPTIMIZER
    contract['gradient_clipping']=dict(max_norm=50.0,norm_type=2.0,scope='all_trainable_parameters')
    contract['training_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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
    model=GPT(GPTConfig(vocab_size=args.vocab_size,n_layer=args.layers,n_head=args.heads,n_embd=args.width)).to(args.device)
    if sum(p.numel() for p in model.parameters()) != args.parameter_count:
        raise RuntimeError('parameter-count formula disagrees with actual unique parameters')
    targets=targets_for(model)
    optimizer=build_optimizer(model,args)
    completed=0;cumulative_lr=0.
    checkpoint=out/'latest.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location=args.device,weights_only=False)
        if state['contract'] != contract:
            raise ValueError('checkpoint contract mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if set(state['norm_targets']) != set(targets) or any(not math.isfinite(v) or v<=0 for v in state['norm_targets'].values()):
            raise ValueError('invalid restored targets')
        targets=state['norm_targets'];completed=state['updates']
        if state['position'] != min(completed*args.tokens_per_update,args.target_train_tokens) or not 0<=completed<=args.updates:
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
            row=dict(updates=completed,tokens=min(completed*args.tokens_per_update,args.target_train_tokens),
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
            steps.write(json.dumps(dict(updates=completed,tokens=min(completed*args.tokens_per_update,args.target_train_tokens),
                train_loss=loss,gradient_norm=grad,base_lr=lr,matrix_lr=matrix_lr,gamma_lr=lr,
                optimizer=OPTIMIZER,momentum=args.momentum,effective_lr=lr/(1-args.momentum),
                loss_evaluation_point='lookahead' if OPTIMIZER=='nesterov' else 'base',
                norm_scale_used=scale,norm_scale_after=next_scale,
                matrix_lr_over_scale=matrix_lr/scale,matrix_lr_over_scale2=matrix_lr/scale**2,
                cumulative_matrix_lr=cumulative_lr,batch=args.batch))+'\n')
            if completed%100==0:steps.flush()
            special=completed in (warm,args.updates)
            dense=args.dense_window > 0 and any(abs(completed-b)<=args.dense_window for b in (warm,))
            if completed%args.eval_every==0 or special or (dense and completed%args.dense_every==0):
                measure()
            if completed%args.checkpoint_every==0 or special or completed==200000:
                steps.flush();metrics.flush()
                save(checkpoint,model,targets,completed,contract,cumulative_lr,optimizer)
                if completed==warm:
                    save(out/'warmup_end.pt',model,targets,completed,contract,cumulative_lr,optimizer)
                if completed==args.updates:
                    save(out/'final.pt',model,targets,completed,contract,cumulative_lr,optimizer)
                if completed==200000:
                    save(out/'step200000.pt',model,targets,completed,contract,cumulative_lr,optimizer)



def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--train',default='F:/datasets/fineweb10B-gpt2/raw/fineweb_train_*.bin')
    p.add_argument('--val',default='F:/datasets/fineweb10B-gpt2/raw/fineweb_val_*.bin')
    p.add_argument('--output',default='runs/polyak_elr0p24')
    p.add_argument('--arm',choices=['fixed'],default='fixed')
    for flag,default in [('vocab-size',50304),('batch',64),('context',1024),('width',768),('layers',12),('heads',6),('seed',0),('validation-contexts',256),('validation-batch-size',32),('eval-every',500),('checkpoint-every',1000),('dense-window',0),('dense-every',20)]:
        p.add_argument('--'+flag,type=int,default=default)
    p.add_argument('--microbatch',type=int,default=64)
    p.add_argument('--effective-lr',type=float,choices=[0.24],default=0.24,help='Fixed eta/(1-beta) for this trial; distinct from RMSNorm gamma')
    p.add_argument('--momentum',type=float,choices=[0.9],default=0.9)
    p.add_argument('--warmup-fraction',type=float,default=.05)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--preflight',action='store_true')
    return p


if __name__=='__main__':
    run(parser().parse_args())
