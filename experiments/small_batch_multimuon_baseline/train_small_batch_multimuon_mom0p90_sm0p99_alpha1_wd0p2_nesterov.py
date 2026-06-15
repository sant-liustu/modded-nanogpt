import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import json
import uuid
import glob
import time
import math
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class MultiMuon(torch.optim.Optimizer):
    """
    MultiMuon - Multi-Momentum Orthogonalized by Newton-schulz

    MultiMuon internally runs fast/slow SGD-style momentum, and then performs an orthogonalization
    post-processing step, in which each 2D parameter's update is replaced with the nearest
    orthogonal matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration,
    which has the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        slow_momentum: The final slow momentum beta.
        slow_alpha: The final raw mixing weight for slow momentum.
        slow_alpha_warmup_steps: The linear warmup length for slow_alpha.
        slow_momentum_warmup_steps: The half-life warmup length for slow_momentum.
        weight_decay: Decoupled weight decay applied directly to the weights.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr, momentum, slow_momentum, slow_alpha,
                  slow_alpha_warmup_steps, slow_momentum_warmup_steps,
                  weight_decay=0.0, nesterov=True, backend='newtonschulz5', backend_steps=5,
                  rank=0, world_size=1):
        defaults = dict(lr=lr, momentum=momentum, slow_momentum=slow_momentum, slow_alpha=slow_alpha,
                        slow_alpha_warmup_steps=slow_alpha_warmup_steps,
                        slow_momentum_warmup_steps=slow_momentum_warmup_steps,
                        weight_decay=weight_decay, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size
        assert 0 < momentum < 1
        assert 0 < slow_momentum < 1
        assert slow_alpha_warmup_steps > 0
        assert slow_momentum_warmup_steps > 0
        assert weight_decay >= 0

    def step(self):

        for group in self.param_groups:

            group['step'] = group.get('step', 0) + 1
            step = group['step']
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            slow_momentum = group['slow_momentum']
            slow_alpha = group['slow_alpha']
            slow_alpha_warmup_steps = group['slow_alpha_warmup_steps']
            slow_momentum_warmup_steps = group['slow_momentum_warmup_steps']
            alpha_t = slow_alpha * min(step / slow_alpha_warmup_steps, 1.0)
            beta3_warmup = min(step / slow_momentum_warmup_steps, 1.0)
            h1 = math.log(0.5) / math.log(momentum) - 1
            h3 = math.log(0.5) / math.log(slow_momentum) - 1
            slow_momentum_t = 0.5 ** (1 / ((1 - beta3_warmup) * h1 + beta3_warmup * h3 + 1))
            group['slow_alpha_t'] = alpha_t
            group['slow_momentum_t'] = slow_momentum_t
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if 'fast_momentum_buffer' not in state:
                        state['fast_momentum_buffer'] = torch.zeros_like(g)
                    if 'slow_momentum_buffer' not in state:
                        state['slow_momentum_buffer'] = torch.zeros_like(g)
                    fast_buf = state['fast_momentum_buffer']
                    slow_buf = state['slow_momentum_buffer']
                    fast_buf.mul_(momentum).add_(g)
                    slow_buf.mul_(slow_momentum_t).add_(g)
                    buf = fast_buf.add(slow_buf, alpha=alpha_t)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(g.size(0), g.size(1))**0.5 # scale to have update.square().mean() == 1
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                if weight_decay != 0 and p.grad is not None:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

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
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

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

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# EMA weights for eval-time model swapping

class EMASet:
    """
    Track multiple exponential moving averages of trainable model parameters.

    This class is intentionally separate from the optimizer: call update() after a
    full optimizer step, then use apply_to()/restore() to temporarily evaluate an
    EMA weight set without changing the raw training trajectory.
    """

    def __init__(self, model, half_lives, device=None):
        self.params = self._trainable_params(model)
        if not self.params:
            raise ValueError("EMASet requires at least one trainable parameter")

        self.device = device
        self.half_lives = [float(h) for h in half_lives]
        if any(h <= 0 for h in self.half_lives):
            raise ValueError("EMA half-lives must be positive")

        self.names = [self._name_for_half_life(h) for h in self.half_lives]
        if len(set(self.names)) != len(self.names):
            raise ValueError("EMA half-lives must produce unique names")

        self.decays = {
            name: 0.5 ** (1.0 / half_life)
            for name, half_life in zip(self.names, self.half_lives)
        }
        self.shadows = {
            name: [p.detach().clone().to(device=device) for p in self.params]
            for name in self.names
        }
        self._backup = None

    @staticmethod
    def _trainable_params(model):
        return [p for p in model.parameters() if p.requires_grad]

    @staticmethod
    def _name_for_half_life(half_life):
        if float(half_life).is_integer():
            return f"ema_h{int(half_life)}"
        return f"ema_h{half_life:g}"

    @torch.no_grad()
    def update(self, model=None):
        if self._backup is not None:
            raise RuntimeError("Cannot update EMA while EMA weights are applied")
        params = self.params if model is None else self._trainable_params(model)
        if len(params) != len(self.params):
            raise ValueError("Model parameter count changed since EMASet initialization")

        for name in self.names:
            decay = self.decays[name]
            for shadow, param in zip(self.shadows[name], params):
                value = param.detach()
                if value.device != shadow.device or value.dtype != shadow.dtype:
                    value = value.to(device=shadow.device, dtype=shadow.dtype)
                shadow.lerp_(value, 1.0 - decay)

    @torch.no_grad()
    def apply_to(self, model, name):
        if self._backup is not None:
            raise RuntimeError("EMA weights are already applied; call restore() first")
        if name not in self.shadows:
            raise KeyError(f"Unknown EMA name: {name}")

        params = self._trainable_params(model)
        if len(params) != len(self.params):
            raise ValueError("Model parameter count changed since EMASet initialization")

        self._backup = [p.detach().clone() for p in params]
        for param, shadow in zip(params, self.shadows[name]):
            value = shadow
            if value.device != param.device or value.dtype != param.dtype:
                value = value.to(device=param.device, dtype=param.dtype)
            param.copy_(value)

    @torch.no_grad()
    def restore(self, model):
        if self._backup is None:
            raise RuntimeError("No EMA weights are currently applied")

        params = self._trainable_params(model)
        if len(params) != len(self._backup):
            raise ValueError("Model parameter count changed since apply_to()")

        for param, backup in zip(params, self._backup):
            param.copy_(backup)
        self._backup = None

    def description(self):
        return {
            name: dict(half_life=half_life, decay=self.decays[name])
            for name, half_life in zip(self.names, self.half_lives)
        }

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 128 # batch size, in sequences, across all devices
    device_batch_size : int = 16 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 20400 # number of iterations to run
    learning_rate : float = 0.0036
    warmup_iters : int = 0
    warmdown_iters : int = 5800 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    adamw_weight_decay : float = 0
    muon_weight_decay : float = 0.2
    muon_momentum : float = 0.9
    muon_slow_momentum : float = 0.99
    muon_slow_alpha : float = 1
    muon_nesterov : bool = True
    muon_backend : str = 'newtonschulz5'
    # evaluation and logging hyperparams
    val_loss_every : int = 500 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    ema_halflife_steps : str = '32,128' # comma-separated EMA half-lives, in optimizer steps
    save_every : int = 400 # every how many steps to save the checkpoint? 0 for only at the end
    compile_model : int = 1 # compile the model with torch.compile
    tensor_norm_every : int = 4 # every how many steps to log tensor norms? 0 disables
    optimizer_update_norm_every : int = 4 # every how many optimizer steps to log applied update norms? 0 disables
    multimuon_buffer_norm_every : int = 4 # every how many optimizer steps to log MultiMuon buffers? 0 disables
args = Hyperparameters()

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
use_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
if use_ddp:
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
if args.compile_model:
    model = torch.compile(model)
# here we wrap model into DDP container
if use_ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module # always contains the "raw" unwrapped model
else:
    raw_model = model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
ema_half_lives = [float(x) for x in args.ema_halflife_steps.split(',') if x.strip()]
ema_set = EMASet(raw_model, ema_half_lives)

# init the optimizer(s)
optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.adamw_weight_decay, fused=True)
optimizer2 = MultiMuon(raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, momentum=args.muon_momentum,
                       rank=ddp_rank, world_size=ddp_world_size,
                       slow_momentum=args.muon_slow_momentum, slow_alpha=args.muon_slow_alpha,
                       slow_alpha_warmup_steps=args.num_iterations,
                       slow_momentum_warmup_steps=args.num_iterations,
                       weight_decay=args.muon_weight_decay,
                       nesterov=args.muon_nesterov,
                       backend=args.muon_backend)
optimizers = [optimizer1, optimizer2]
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

def evaluate_val_loss():
    model.eval()
    val_loader.reset()
    val_loss = torch.zeros((), device=device)
    for _ in range(val_steps):
        x_val, y_val = val_loader.next_batch()
        with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
            _, loss = model(x_val, y_val, return_logits=False)
            val_loss += loss.detach()
            del loss
    if use_ddp:
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    return val_loss / val_steps

# begin logging
run_id = str(uuid.uuid4()) if master_process else None
run_id_box = [run_id]
if use_ddp:
    dist.broadcast_object_list(run_id_box, src=0)
run_id = run_id_box[0]
logdir = 'logs/%s/' % run_id
os.makedirs(logdir, exist_ok=True)
logfile = 'logs/%s.txt' % run_id
if master_process:
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')
        f.write(f'EMA: {ema_set.description()}\n')
    print(f"EMA tracking: {ema_set.description()}")

def tensor_metadata_records(model):
    records = []
    for name, tensor in model.named_parameters():
        records.append(dict(
            name=name,
            shape=list(tensor.shape),
            ndim=tensor.ndim,
            numel=tensor.numel(),
            dtype=str(tensor.dtype),
            trainable=tensor.requires_grad,
        ))
    return records

def write_tensor_metadata():
    if not master_process:
        return
    metadata_path = os.path.join(logdir, 'tensor_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(tensor_metadata_records(raw_model), f, indent=2)

def tensor_norm_fields(tensor, prefix=''):
    x = tensor.detach().float()
    sq_sum = x.square().sum()
    fro_norm = torch.sqrt(sq_sum).item()
    return {
        f'{prefix}fro_norm': fro_norm,
        f'{prefix}rms_norm': fro_norm / (x.numel() ** 0.5),
    }

def tensor_norm_record(step, name, tensor):
    record = dict(step=step, name=name, shape=list(tensor.shape), ndim=tensor.ndim)
    record.update(tensor_norm_fields(tensor))
    return record

def maybe_log_tensor_norms(step):
    if not master_process or args.tensor_norm_every <= 0:
        return
    if step % args.tensor_norm_every != 0 and step != args.num_iterations:
        return
    history_path = os.path.join(logdir, 'tensor_norm_history.jsonl')
    with torch.no_grad(), open(history_path, 'a') as f:
        for name, tensor in raw_model.named_parameters():
            if not tensor.requires_grad:
                continue
            f.write(json.dumps(tensor_norm_record(step, name, tensor)) + '\n')

def optimizer_parameter_hparams():
    hparams = {}
    for optimizer_index, optimizer in enumerate(optimizers):
        optimizer_type = type(optimizer).__name__
        for param_group_index, group in enumerate(optimizer.param_groups):
            for param in group['params']:
                hparams[id(param)] = dict(
                    optimizer_index=optimizer_index,
                    optimizer_type=optimizer_type,
                    param_group_index=param_group_index,
                    lr=float(group['lr']),
                    weight_decay=float(group.get('weight_decay', 0.0)),
                    momentum=float(group['momentum']) if 'momentum' in group else None,
                    slow_momentum=float(group['slow_momentum']) if 'slow_momentum' in group else None,
                    slow_alpha=float(group['slow_alpha']) if 'slow_alpha' in group else None,
                    slow_alpha_t=float(group['slow_alpha_t']) if 'slow_alpha_t' in group else None,
                    slow_momentum_t=float(group['slow_momentum_t']) if 'slow_momentum_t' in group else None,
                    slow_alpha_warmup_steps=int(group['slow_alpha_warmup_steps']) if 'slow_alpha_warmup_steps' in group else None,
                    slow_momentum_warmup_steps=int(group['slow_momentum_warmup_steps']) if 'slow_momentum_warmup_steps' in group else None,
                    nesterov=bool(group['nesterov']) if 'nesterov' in group else None,
                    backend=group.get('backend'),
                    backend_steps=group.get('backend_steps'),
                )
    return hparams

def should_log_optimizer_update_norms(update_step):
    if not master_process or args.optimizer_update_norm_every <= 0:
        return False
    return update_step % args.optimizer_update_norm_every == 0 or update_step == args.num_iterations

def maybe_capture_optimizer_update_state(update_step):
    if not should_log_optimizer_update_norms(update_step):
        return None
    hparams = optimizer_parameter_hparams()
    snapshots = {}
    for tensor in raw_model.parameters():
        param_id = id(tensor)
        if tensor.requires_grad and param_id in hparams:
            snapshots[param_id] = tensor.detach().clone()
    return dict(hparams=hparams, snapshots=snapshots)

def optimizer_update_norm_record(step, name, tensor, tensor_before, param_hparams):
    lr = param_hparams['lr']
    if lr == 0:
        raise RuntimeError(f"cannot infer optimizer update for {name} at step {step}: lr is 0")
    before = tensor_before.float()
    after = tensor.detach().float()
    applied_update = (before - after) / lr
    record = dict(
        step=step,
        name=name,
        shape=list(tensor.shape),
        ndim=tensor.ndim,
        lr=lr,
        weight_decay=param_hparams['weight_decay'],
        optimizer_index=param_hparams['optimizer_index'],
        optimizer_type=param_hparams['optimizer_type'],
        param_group_index=param_hparams['param_group_index'],
        momentum=param_hparams['momentum'],
        slow_momentum=param_hparams['slow_momentum'],
        slow_alpha=param_hparams['slow_alpha'],
        slow_alpha_t=param_hparams['slow_alpha_t'],
        slow_momentum_t=param_hparams['slow_momentum_t'],
        nesterov=param_hparams['nesterov'],
        backend=param_hparams['backend'],
        backend_steps=param_hparams['backend_steps'],
    )
    record.update(tensor_norm_fields(before, prefix='param_before_'))
    record.update(tensor_norm_fields(after, prefix='param_after_'))
    record.update(tensor_norm_fields(applied_update, prefix='applied_update_'))
    return record

def maybe_log_optimizer_update_norms(update_step, update_state):
    if update_state is None:
        return
    history_path = os.path.join(logdir, 'optimizer_update_norm_history.jsonl')
    hparams = update_state['hparams']
    snapshots = update_state['snapshots']
    with torch.no_grad(), open(history_path, 'a') as f:
        for name, tensor in raw_model.named_parameters():
            if not tensor.requires_grad:
                continue
            param_id = id(tensor)
            if param_id not in hparams:
                raise RuntimeError(f"missing optimizer hyperparameters for {name}")
            if param_id not in snapshots:
                raise RuntimeError(f"missing parameter snapshot for {name}")
            tensor_before = snapshots.pop(param_id)
            f.write(json.dumps(optimizer_update_norm_record(
                update_step,
                name,
                tensor,
                tensor_before,
                hparams[param_id],
            )) + '\n')
            del tensor_before
    if snapshots:
        raise RuntimeError(f"unused optimizer update snapshots: {len(snapshots)}")

def write_multimuon_schedule_metadata_for_next_step():
    for group in optimizer2.param_groups:
        if 'slow_momentum' not in group:
            continue
        next_step = group.get('step', 0) + 1
        momentum = group['momentum']
        slow_momentum = group['slow_momentum']
        slow_alpha = group['slow_alpha']
        slow_alpha_warmup_steps = group['slow_alpha_warmup_steps']
        slow_momentum_warmup_steps = group['slow_momentum_warmup_steps']
        alpha_t = slow_alpha * min(next_step / slow_alpha_warmup_steps, 1.0)
        beta3_warmup = min(next_step / slow_momentum_warmup_steps, 1.0)
        h1 = math.log(0.5) / math.log(momentum) - 1
        h3 = math.log(0.5) / math.log(slow_momentum) - 1
        slow_momentum_t = 0.5 ** (1 / ((1 - beta3_warmup) * h1 + beta3_warmup * h3 + 1))
        group['slow_alpha_t'] = alpha_t
        group['slow_momentum_t'] = slow_momentum_t

def should_log_multimuon_buffer_norms(update_step):
    if args.multimuon_buffer_norm_every <= 0:
        return False
    return update_step % args.multimuon_buffer_norm_every == 0 or update_step == args.num_iterations

def multimuon_buffer_norm_record(step, name, tensor, buffer_kind, buffer_tensor, param_hparams, param_index):
    record = dict(
        step=step,
        name=name,
        shape=list(tensor.shape),
        ndim=tensor.ndim,
        buffer_kind=buffer_kind,
        rank=ddp_rank,
        owner_rank=ddp_rank,
        world_size=ddp_world_size,
        param_index=param_index,
        optimizer_index=param_hparams['optimizer_index'],
        optimizer_type=param_hparams['optimizer_type'],
        param_group_index=param_hparams['param_group_index'],
        momentum=param_hparams['momentum'],
        slow_momentum=param_hparams['slow_momentum'],
        slow_alpha=param_hparams['slow_alpha'],
        slow_alpha_t=param_hparams['slow_alpha_t'],
        slow_momentum_t=param_hparams['slow_momentum_t'],
        nesterov=param_hparams['nesterov'],
        weight_decay=param_hparams['weight_decay'],
        backend=param_hparams['backend'],
        backend_steps=param_hparams['backend_steps'],
    )
    record.update(tensor_norm_fields(buffer_tensor, prefix=f'{buffer_kind}_'))
    return record

def multimuon_buffer_history_path():
    if ddp_world_size == 1:
        return os.path.join(logdir, 'multimuon_buffer_norm_history.jsonl')
    return os.path.join(logdir, f'multimuon_buffer_norm_history_rank{ddp_rank:02d}.jsonl')

def maybe_log_multimuon_buffer_norms(update_step):
    if not should_log_multimuon_buffer_norms(update_step):
        return
    hparams = optimizer_parameter_hparams()
    names_by_id = {id(tensor): name for name, tensor in raw_model.named_parameters()}
    history_path = multimuon_buffer_history_path()
    with torch.no_grad(), open(history_path, 'a') as f:
        for group in optimizer2.param_groups:
            for param_index, tensor in enumerate(group['params']):
                if param_index % ddp_world_size != ddp_rank:
                    continue
                if not tensor.requires_grad:
                    continue
                param_id = id(tensor)
                name = names_by_id.get(param_id)
                if name is None:
                    raise RuntimeError("missing parameter name for MultiMuon buffer logging")
                param_hparams = hparams.get(param_id)
                if param_hparams is None or param_hparams['optimizer_type'] != 'MultiMuon':
                    raise RuntimeError(f"missing MultiMuon optimizer hyperparameters for {name}")
                state = optimizer2.state.get(tensor)
                if state is None:
                    raise RuntimeError(f"missing MultiMuon state for {name} at step {update_step}")
                for buffer_kind in ('fast_momentum_buffer', 'slow_momentum_buffer'):
                    if buffer_kind not in state:
                        raise RuntimeError(f"missing {buffer_kind} for {name} at step {update_step}")
                    f.write(json.dumps(multimuon_buffer_norm_record(
                        update_step,
                        name,
                        tensor,
                        buffer_kind,
                        state[buffer_kind],
                        param_hparams,
                        param_index,
                    )) + '\n')

write_tensor_metadata()

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    maybe_log_tensor_norms(step)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches on raw weights and each EMA weight set
        val_losses = {"raw": evaluate_val_loss()}
        for ema_name in ema_set.names:
            ema_set.apply_to(raw_model, ema_name)
            val_losses[ema_name] = evaluate_val_loss()
            ema_set.restore(raw_model)
        # log val loss to console and to logfile
        if master_process:
            raw_val_loss = val_losses["raw"]
            val_loss_text = ' '.join(f'val_loss/{name}:{loss:.4f}' for name, loss in val_losses.items())
            print(f'step:{step}/{args.num_iterations} val_loss:{raw_val_loss:.4f} {val_loss_text} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{raw_val_loss:.4f} {val_loss_text} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync() if use_ddp else nullcontext(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # step the optimizers and schedulers
    update_step = step + 1
    write_multimuon_schedule_metadata_for_next_step()
    optimizer_update_state = maybe_capture_optimizer_update_state(update_step)
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    maybe_log_optimizer_update_norms(update_step, optimizer_update_state)
    maybe_log_multimuon_buffer_norms(update_step)
    ema_set.update(raw_model)
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
if use_ddp:
    dist.destroy_process_group()
