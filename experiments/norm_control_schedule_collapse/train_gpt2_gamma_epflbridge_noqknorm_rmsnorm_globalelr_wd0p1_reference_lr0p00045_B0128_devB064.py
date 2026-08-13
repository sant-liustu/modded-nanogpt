
# EPFL-style global effective-LR bridge experiment.
# Architecture: noqknorm_rmsnorm; mode: reference; reference WD: 0.1.
# Shared peak LR 4.5e-4, WSD schedule, B=128, 20400 optimizer steps.
import os
import random
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import contextlib
import fnmatch
import hashlib
import json
import uuid
import glob
import time
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

GLOBAL_ELR_ARCHITECTURE = 'noqknorm_rmsnorm'
GLOBAL_ELR_MODE = 'reference'
GLOBAL_ELR_REFERENCE_WEIGHT_DECAY = 0.1

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
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
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

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

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
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5,
                 rank=0, world_size=1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
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
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
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
        c_proj_std = config.init_std / math.sqrt(2 * config.n_layer)
        torch.nn.init.normal_(self.c_proj.weight, mean=0.0, std=c_proj_std)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
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
        c_proj_std = config.init_std / math.sqrt(2 * config.n_layer)
        torch.nn.init.normal_(self.c_proj.weight, mean=0.0, std=c_proj_std)

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
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768
    init_std : float = 0.02

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
        self.final_norm = RMSNorm(config.n_embd)

    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.final_norm(x)

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
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 128 # batch size, in sequences, across all devices
    device_batch_size : int = 64 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 20400 # number of iterations to run
    embed_learning_rate : float = 0.00045 # shared peak LR for embedding, blocks, and norm gamma
    warmup_iters : int = 1000
    warmdown_iters : int = 5800 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0.1 # reference matrices+tied embedding; matched scripts use 0
    # evaluation and logging hyperparams
    val_loss_every : int = 500 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    compile_model : int = 1 # compile the model with torch.compile
    tensor_norm_every : int = 4 # every how many steps to log tensor norm history? 0 disables
    adamw_update_norm_every : int = 4 # every how many optimizer steps to log AdamW effective update norms? 0 disables
    activation_probe_every : int = 0 # every how many steps to log fixed-probe activation RMS ratios? 0 disables
    spectral_norm_estimate_enabled : int = 1 # whether to estimate 2D spectral norms in tensor/update norm histories
    activation_probe_eps : float = 1e-12 # denominator epsilon for activation RMS ratios
    seed : int = 0
    norm_control_config : str = 'experiments/norm_control_schedule_collapse/delayed_constant_all_matrices_start1000.json' # matrix-selection manifest only; projection is disabled
    global_elr_reference_path : str = 'experiments/norm_control_schedule_collapse/reference_trajectories/noqknorm_rmsnorm_wd0p1_lr0p00045.jsonl'
args = Hyperparameters()
def parse_hparam_value(name, value):
    current = getattr(args, name)
    if isinstance(current, bool):
        return value.lower() in ('1', 'true', 'yes', 'on')
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value

for arg in sys.argv[1:]:
    if not arg.startswith('--') or '=' not in arg:
        raise ValueError(f"expected --name=value argument, got: {arg}")
    name, value = arg[2:].split('=', 1)
    if not hasattr(args, name):
        raise ValueError(f"unknown command line argument: {arg}")
    setattr(args, name, parse_hparam_value(name, value))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

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

DEFAULT_NORM_CONTROL_MODE = 'delayed_captured_constant'
DEFAULT_NORM_CONTROL_START_STEP = 1000

def load_norm_control_config(path):
    if not path:
        return dict(enabled=False, mode='disabled', targets=[], eps=1e-12, log_every=1, start_step=None)
    with open(path, 'r', encoding='utf-8-sig') as f:
        spec = json.load(f)
    if not spec.get('enabled', True):
        return dict(
            enabled=False,
            mode='disabled',
            targets=[],
            eps=float(spec.get('eps', 1e-12)),
            log_every=int(spec.get('log_every', 1)),
            start_step=None,
        )
    if spec.get('norm_type', 'rms') != 'rms':
        raise ValueError("norm_control_config only supports norm_type='rms'")
    mode = spec.get('mode', DEFAULT_NORM_CONTROL_MODE)
    mode_aliases = {
        'specified_target': 'specified_target',
        'constant_from_start': 'specified_target',
        'immediate': 'specified_target',
        'immediate_target': 'specified_target',
        'delayed_captured_constant': 'delayed_captured_constant',
        'delayed_start_captured_constant': 'delayed_captured_constant',
        'warmup_then_constant': 'delayed_captured_constant',
    }
    if mode not in mode_aliases:
        raise ValueError(f"unknown norm-control mode: {mode}")
    mode = mode_aliases[mode]

    targets = spec.get('targets')
    if targets is None and 'controlled_patterns' in spec:
        targets = [
            {'pattern': pattern} if isinstance(pattern, str) else pattern
            for pattern in spec['controlled_patterns']
        ]
    if not isinstance(targets, list) or not targets:
        raise ValueError("enabled norm_control_config requires a non-empty targets list")
    for target in targets:
        if 'pattern' not in target:
            raise ValueError("each norm-control target requires a pattern")
        if mode == 'specified_target':
            target_rms = float(target.get('target_rms', 0.0))
            if not (target_rms > 0.0):
                raise ValueError(f"target_rms must be positive for pattern {target['pattern']}")
            target['target_rms'] = target_rms
        elif 'target_rms' in target:
            raise ValueError(
                "delayed_captured_constant captures target_rms at start_step; "
                f"remove target_rms for pattern {target['pattern']}"
            )
    start_step = None
    if mode == 'delayed_captured_constant':
        start_step = int(spec.get('start_step', DEFAULT_NORM_CONTROL_START_STEP))
        if start_step < 0:
            raise ValueError("norm-control start_step must be non-negative")
    return dict(
        enabled=True,
        mode=mode,
        targets=targets,
        eps=float(spec.get('eps', 1e-12)),
        log_every=int(spec.get('log_every', 1)),
        start_step=start_step,
    )

def canonical_param_name(name):
    if name.startswith('_orig_mod.'):
        return name[len('_orig_mod.'):]
    return name

def pattern_matches_name(pattern, name):
    canonical_name = canonical_param_name(name)
    return (
        pattern == name
        or pattern == canonical_name
        or fnmatch.fnmatchcase(name, pattern)
        or fnmatch.fnmatchcase(canonical_name, pattern)
    )

def is_allowed_norm_control_parameter(name):
    canonical_name = canonical_param_name(name)
    return (
        canonical_name.startswith('transformer.h.')
        or canonical_name == 'transformer.wte.weight'
    )

def build_norm_control_state(raw_model, path):
    spec = load_norm_control_config(path)
    if not spec['enabled']:
        return dict(
            enabled=False,
            mode=spec['mode'],
            params=[],
            eps=spec['eps'],
            log_every=spec['log_every'],
            targets=[],
            start_step=spec['start_step'],
        )
    named_parameters = list(raw_model.named_parameters())
    matched_by_name = {}
    target_records = []
    for target in spec['targets']:
        pattern = target['pattern']
        target_rms = target.get('target_rms')
        matches = [(name, p) for name, p in named_parameters if pattern_matches_name(pattern, name)]
        if not matches:
            raise ValueError(f"norm-control pattern matched no parameters: {pattern}")
        for name, p in matches:
            if name in matched_by_name:
                previous = matched_by_name[name]['pattern']
                raise ValueError(f"norm-control parameter {name} matched multiple patterns: {previous}, {pattern}")
            if not p.requires_grad:
                raise ValueError(f"norm-control target is not trainable: {name}")
            if not torch.is_floating_point(p):
                raise ValueError(f"norm-control target is not floating point: {name}")
            if p.ndim < 2:
                raise ValueError(f"norm-control target must have ndim >= 2: {name}")
            matched_by_name[name] = dict(
                name=name,
                param=p,
                pattern=pattern,
                target_rms=target_rms,
                captured=False,
            )
        record = dict(pattern=pattern, matched_names=[name for name, _ in matches])
        if target_rms is not None:
            record['target_rms'] = target_rms
        target_records.append(record)
    controlled = [matched_by_name[name] for name in sorted(matched_by_name)]
    return dict(
        enabled=True,

        mode=spec['mode'],
        params=controlled,
        eps=spec['eps'],
        log_every=spec['log_every'],
        targets=target_records,
        start_step=spec['start_step'],
    )

@torch.no_grad()
def apply_rms_norm_control(norm_control_state, step, event):
    if not norm_control_state['enabled']:
        return
    mode = norm_control_state['mode']
    start_step = norm_control_state['start_step']
    if mode == 'specified_target':
        phase = 'specified_target'
    elif step < start_step:
        phase = 'pre_start'
    elif step == start_step:
        phase = 'capture'
    else:
        phase = 'post_start'
    should_log = (
        master_process
        and norm_control_state['log_every'] > 0
        and (
            step % norm_control_state['log_every'] == 0
            or step == args.num_iterations
            or event == 'initial'
            or phase == 'capture'
        )
    )
    history_path = os.path.join(logdir, 'norm_control_history.jsonl') if should_log else None
    f = open(history_path, 'a') if should_log else None
    captured_any = False
    try:
        for entry in norm_control_state['params']:
            p = entry['param']
            rms_before = p.detach().float().square().mean().sqrt()
            base_target_rms = entry['target_rms']
            target_rms = base_target_rms
            ratio = 1.0
            projected = phase in ('specified_target', 'post_start')
            captured = phase == 'capture'
            if captured:
                target_rms = rms_before.item()
                entry['target_rms'] = target_rms
                base_target_rms = target_rms
                entry['captured'] = True
                captured_any = True
            if projected:
                if base_target_rms is None:
                    raise RuntimeError(f"missing captured target RMS for norm-control parameter {entry['name']}")
                ratio = schedule_ratio(step)
                target_rms = base_target_rms * ratio
                if rms_before > norm_control_state['eps']:
                    target = torch.tensor(target_rms, dtype=torch.float32, device=p.device)
                    scale = target / rms_before
                    p.mul_(scale.to(dtype=p.dtype, device=p.device))
                else:
                    scale = torch.ones((), dtype=torch.float32, device=p.device)
            else:
                scale = torch.ones((), dtype=torch.float32, device=p.device)
            rms_after = p.detach().float().square().mean().sqrt()
            if f is not None:
                relative_error = None
                if target_rms is not None:
                    target = torch.tensor(target_rms, dtype=torch.float32, device=p.device)
                    relative_error = ((rms_after - target).abs() / target).item()
                f.write(json.dumps(dict(
                    step=step,
                    event=event,
                    mode=mode,
                    phase=phase,
                    name=entry['name'],
                    pattern=entry['pattern'],
                    base_target_rms=base_target_rms,
                    schedule_ratio=ratio,
                    target_rms=target_rms,
                    pre_control_rms=rms_before.item(),
                    post_control_rms=rms_after.item(),
                    scale=scale.item(),
                    relative_error=relative_error,
                    projected=projected,
                    captured=captured,
                    start_step=start_step,
                    weight_decay=0.0,
                )) + '\n')
    finally:
        if f is not None:
            f.close()
    if captured_any:
        write_norm_control_targets(norm_control_state)

def write_norm_control_targets(norm_control_state):
    if not master_process or not norm_control_state['enabled']:
        return
    records = []
    for entry in norm_control_state['params']:
        if entry['target_rms'] is not None:
            records.append(dict(
                name=entry['name'],
                pattern=entry['pattern'],
                target_rms=entry['target_rms'],
                captured=entry['captured'],
                start_step=norm_control_state['start_step'],
                mode=norm_control_state['mode'],
            ))
    with open(os.path.join(logdir, 'norm_control_targets.json'), 'w') as f:
        json.dump(records, f, indent=2)

def write_norm_control_metadata(norm_control_state):
    if not master_process or not norm_control_state['enabled']:
        return
    metadata = dict(
        enabled=True,
        mode=norm_control_state['mode'],
        norm_type='rms',
        start_step=norm_control_state['start_step'],
        eps=norm_control_state['eps'],
        log_every=norm_control_state['log_every'],
        targets=norm_control_state['targets'],
        controlled_parameters=[
            dict(
                name=entry['name'],
                pattern=entry['pattern'],
                target_rms=entry['target_rms'],
                captured=entry['captured'],
                shape=list(entry['param'].shape),
                ndim=entry['param'].ndim,
                weight_decay=0.0,
            )
            for entry in norm_control_state['params']
        ],
    )
    with open(os.path.join(logdir, 'norm_control_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

norm_control_state = build_norm_control_state(raw_model, args.norm_control_config)
controlled_param_ids = {id(entry['param']) for entry in norm_control_state['params']}
def is_rmsnorm_gamma_name(name):
    canonical_name = canonical_param_name(name)
    return (
        canonical_name == 'final_norm.weight'
        or canonical_name.endswith('.q_norm.weight')
        or canonical_name.endswith('.k_norm.weight')
        or canonical_name.endswith('.attn_norm.weight')
        or canonical_name.endswith('.mlp_norm.weight')
    )

block_named_parameters = [
    (name, p)
    for name, p in raw_model.named_parameters()
    if canonical_param_name(name).startswith('transformer.h.')
    and not is_rmsnorm_gamma_name(name)
]
controlled_block_parameters = [p for _, p in block_named_parameters if id(p) in controlled_param_ids]
uncontrolled_block_parameters = [p for _, p in block_named_parameters if id(p) not in controlled_param_ids]
rmsnorm_gamma_parameters = [
    p
    for name, p in raw_model.named_parameters()
    if is_rmsnorm_gamma_name(name)
]
unexpected_controlled = [
    entry['name']
    for entry in norm_control_state['params']
    if not is_allowed_norm_control_parameter(entry['name'])
]
if unexpected_controlled:
    raise ValueError(
        "norm-control targets must be transformer block parameters or the tied embedding "
        f"parameter transformer.wte.weight in this implementation: {unexpected_controlled}"
    )

# One standard AdamW optimizer. All groups share one scalar LR; only norm gamma has WD=0.
matrix_parameters = uncontrolled_block_parameters + controlled_block_parameters
optimizer_groups = [
    dict(params=list(raw_model.lm_head.parameters()), weight_decay=args.weight_decay),
    dict(params=matrix_parameters, weight_decay=args.weight_decay),
]
if rmsnorm_gamma_parameters:
    optimizer_groups.append(dict(params=rmsnorm_gamma_parameters, weight_decay=0.0))
optimizer = torch.optim.AdamW(optimizer_groups, lr=args.embed_learning_rate, betas=(0.9, 0.95), fused=True)
optimizers = [optimizer]
# learning rate decay scheduler (linear warmup and warmdown)
def schedule_ratio(it):
    return 1.0

def get_wsd_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        wsd_ratio = (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        wsd_ratio = 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        wsd_ratio = decay_ratio
    return wsd_ratio

def get_lr(it):
    wsd_ratio = get_wsd_lr(it)
    return wsd_ratio * schedule_ratio(it)
optimizer_lr_lambdas = [get_wsd_lr for _ in optimizer.param_groups]
schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, optimizer_lr_lambdas)]

if len(norm_control_state['params']) != 73:
    raise RuntimeError(f"expected 73 matrices in global ELR denominator, got {len(norm_control_state['params'])}")

def current_global_matrix_norm():
    total_sq = torch.zeros((), dtype=torch.float32, device=norm_control_state['params'][0]['param'].device)
    for entry in norm_control_state['params']:
        total_sq.add_(entry['param'].detach().float().square().sum())
    return total_sq.sqrt()

global_elr_reference_rows = None
global_elr_reference_file = args.global_elr_reference_path

def initialize_global_elr_experiment():
    global global_elr_reference_rows
    if GLOBAL_ELR_MODE == 'matched':
        if not os.path.isfile(global_elr_reference_file):
            raise FileNotFoundError(f"missing reference trajectory: {global_elr_reference_file}; run the corresponding WD reference script first")
        rows = {}
        with open(global_elr_reference_file) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows[int(row['step'])] = row
        missing = [step for step in range(args.num_iterations) if step not in rows]
        if missing:
            raise RuntimeError(f"reference trajectory incomplete: {len(missing)} missing; first={missing[0]}")
        global_elr_reference_rows = rows
    elif GLOBAL_ELR_MODE == 'reference':
        if master_process:
            parent = os.path.dirname(global_elr_reference_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(global_elr_reference_file, 'w'):
                pass
            with open(global_elr_reference_file + '.meta.json', 'w') as f:
                json.dump(dict(architecture=GLOBAL_ELR_ARCHITECTURE, mode=GLOBAL_ELR_MODE,
                    weight_decay=args.weight_decay, peak_lr=args.embed_learning_rate,
                    norm_definition='sqrt(sum_i ||W_i||_F^2)',
                    controlled_matrix_count=len(norm_control_state['params']),
                    controlled_names=[entry['name'] for entry in norm_control_state['params']],
                    norm_gamma_in_denominator=False), f, indent=2)
    else:
        raise ValueError(f"unsupported GLOBAL_ELR_MODE={GLOBAL_ELR_MODE}")
    if use_ddp:
        dist.barrier()

def prepare_global_elr_step(step):
    current_norm = current_global_matrix_norm()
    if not torch.isfinite(current_norm) or current_norm <= 0:
        raise RuntimeError(f"invalid global matrix norm at step {step}: {current_norm.item()}")
    base_lr = args.embed_learning_rate * get_wsd_lr(step)
    if GLOBAL_ELR_MODE == 'reference':
        reference_norm, reference_lr = current_norm.item(), base_lr
        target_elr, applied_lr = reference_lr / reference_norm, reference_lr
    else:
        row = global_elr_reference_rows[step]
        reference_norm, reference_lr = float(row['global_norm']), float(row['lr'])
        target_elr = float(row['lr_over_global_norm'])
        applied_lr = target_elr * current_norm.item()
    for group in optimizer.param_groups:
        group['lr'] = applied_lr
    actual_elr = applied_lr / current_norm.item()
    record = dict(step=step, architecture=GLOBAL_ELR_ARCHITECTURE, mode=GLOBAL_ELR_MODE,
        weight_decay=args.weight_decay, reference_weight_decay=GLOBAL_ELR_REFERENCE_WEIGHT_DECAY,
        base_lr=base_lr, lr=applied_lr, global_norm=current_norm.item(),
        lr_over_global_norm=actual_elr, target_lr_over_global_norm=target_elr,
        reference_lr=reference_lr, reference_global_norm=reference_norm,
        relative_alignment_error=abs(actual_elr-target_elr)/max(abs(target_elr),1e-30))
    if master_process:
        with open(os.path.join(logdir, 'global_elr_history.jsonl'), 'a') as f:
            f.write(json.dumps(record) + '\n')
        if GLOBAL_ELR_MODE == 'reference':
            with open(global_elr_reference_file, 'a') as f:
                f.write(json.dumps(record) + '\n')
    return record

# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
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

initialize_global_elr_experiment()

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

ACTIVATION_PROBE_FIELDS = (
    'rms_h_pre',
    'attn_residual_ratio',
    'attn_branch_ratio',
    'mlp_residual_ratio',
    'mlp_branch_ratio',
)

def token_rms(x):
    return torch.sqrt(x.detach().float().square().mean(dim=-1))

def summarize_activation_values(step, layer, field, values):
    x = values.detach().float()
    flat = x.reshape(-1)
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    if finite_values.numel() == 0:
        stats = dict(mean=float('nan'), std=float('nan'), p05=float('nan'), p50=float('nan'), p95=float('nan'), min=float('nan'), max=float('nan'))
    else:
        quantiles = torch.quantile(finite_values, torch.tensor([0.05, 0.5, 0.95], device=finite_values.device))
        stats = dict(
            mean=finite_values.mean().item(),
            std=finite_values.std(unbiased=False).item(),
            p05=quantiles[0].item(),
            p50=quantiles[1].item(),
            p95=quantiles[2].item(),
            min=finite_values.min().item(),
            max=finite_values.max().item(),
        )
    return dict(
        step=step,
        layer=layer,
        field=field,
        shape=list(values.shape),
        nan_count=torch.isnan(flat).sum().item(),
        inf_count=torch.isinf(flat).sum().item(),
        **stats,
    )

class ActivationProbeCapture:
    def __init__(self, model, eps):
        self.blocks = list(model.transformer.h)
        self.eps = eps
        self.handles = []

        self.h_pre = [None] * len(self.blocks)
        self.h_mid = [None] * len(self.blocks)
        self.records = {field: [None] * len(self.blocks) for field in ACTIVATION_PROBE_FIELDS}

    def __enter__(self):
        for layer, block in enumerate(self.blocks):
            self.handles.append(block.register_forward_pre_hook(self._block_pre_hook(layer)))
            self.handles.append(block.attn.register_forward_hook(self._attn_hook(layer)))
            self.handles.append(block.mlp.register_forward_hook(self._mlp_hook(layer)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _block_pre_hook(self, layer):
        def hook(module, inputs):
            h_pre = inputs[0].detach()
            self.h_pre[layer] = h_pre
            self.records['rms_h_pre'][layer] = token_rms(h_pre)
        return hook

    def _attn_hook(self, layer):
        def hook(module, inputs, output):
            h_pre = self.h_pre[layer]
            if h_pre is None:
                raise RuntimeError(f"missing h_l for activation probe layer {layer}")
            attn_out = output.detach()
            h_mid = h_pre + attn_out
            rms_h_pre = self.records['rms_h_pre'][layer]
            rms_h_mid = token_rms(h_mid)
            self.h_mid[layer] = h_mid
            self.records['attn_residual_ratio'][layer] = rms_h_mid / (rms_h_pre + self.eps)
            self.records['attn_branch_ratio'][layer] = token_rms(attn_out) / (rms_h_pre + self.eps)
        return hook

    def _mlp_hook(self, layer):
        def hook(module, inputs, output):
            h_mid = self.h_mid[layer]
            if h_mid is None:
                raise RuntimeError(f"missing h_l+0.5 for activation probe layer {layer}")
            mlp_out = output.detach()
            rms_h_mid = token_rms(h_mid)
            rms_h_post = token_rms(h_mid + mlp_out)
            self.records['mlp_residual_ratio'][layer] = rms_h_post / (rms_h_mid + self.eps)
            self.records['mlp_branch_ratio'][layer] = token_rms(mlp_out) / (rms_h_mid + self.eps)
        return hook

    def stacked_records(self):
        stacked = {}
        for field, values_by_layer in self.records.items():
            missing = [layer for layer, value in enumerate(values_by_layer) if value is None]
            if missing:
                raise RuntimeError(f"missing activation probe field {field} for layers {missing}")
            stacked[field] = torch.stack(values_by_layer, dim=0).detach().cpu()
        return stacked

def should_log_activation_probe(step):
    if not master_process or args.activation_probe_every <= 0:
        return False
    return step % args.activation_probe_every == 0 or step == args.num_iterations

def build_activation_probe_batch():
    if not master_process or args.activation_probe_every <= 0:
        return None
    val_loader.reset()
    x_probe, _ = val_loader.next_batch()
    val_loader.reset()
    return x_probe.detach().clone()

def activation_probe_token_hash(x_probe):
    token_bytes = x_probe.detach().cpu().numpy().astype(np.int64).tobytes()
    return hashlib.sha256(token_bytes).hexdigest()

def write_activation_probe_metadata(x_probe):
    if x_probe is None:
        return
    metadata = dict(
        probe_source='first validation batch after val_loader.reset()',
        probe_input_val_bin=args.input_val_bin,
        probe_batch_shape=list(x_probe.shape),
        probe_token_sha256=activation_probe_token_hash(x_probe),
        eps=args.activation_probe_eps,
        layer_count=len(raw_model.transformer.h),
        recorded_fields=list(ACTIVATION_PROBE_FIELDS),
        array_layout='[layer, batch, seq]',
        logging_cadence=args.activation_probe_every,
        model_config=dict(
            n_layer=raw_model.config.n_layer,
            n_head=raw_model.config.n_head,
            n_embd=raw_model.config.n_embd,
            vocab_size=raw_model.config.vocab_size,
        ),
    )
    metadata_path = os.path.join(logdir, 'activation_probe_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def maybe_log_activation_probe(step, x_probe):
    if x_probe is None or not should_log_activation_probe(step):
        return
    was_training = raw_model.training
    raw_model.eval()
    with torch.no_grad(), ctx, ActivationProbeCapture(raw_model, args.activation_probe_eps) as capture:
        raw_model(x_probe, targets=None, return_logits=False)
        arrays = capture.stacked_records()
    if was_training:
        raw_model.train()
    arrays_dir = os.path.join(logdir, 'activation_probe_arrays')
    os.makedirs(arrays_dir, exist_ok=True)
    array_path = os.path.join(arrays_dir, f'step_{step:06d}.pt')
    torch.save(dict(step=step, **arrays), array_path)
    summary_path = os.path.join(logdir, 'activation_probe_summary.jsonl')
    with open(summary_path, 'a') as f:
        for field, values in arrays.items():
            for layer in range(values.shape[0]):
                f.write(json.dumps(summarize_activation_values(step, layer, field, values[layer])) + '\n')

SPECTRAL_NORM_ESTIMATE_BLOCK_SIZE = 48
SPECTRAL_NORM_ESTIMATE_ITERS = 10
SPECTRAL_NORM_ESTIMATE_METHOD = "batched_power_q48_i10"
spectral_norm_generator = torch.Generator(device=device)
spectral_norm_generator.manual_seed(20260525)

def batched_spectral_norm_estimate(matrices):
    batch, _, cols = matrices.shape
    vectors = torch.randn(
        (batch, cols, SPECTRAL_NORM_ESTIMATE_BLOCK_SIZE),
        device=matrices.device,
        dtype=matrices.dtype,
        generator=spectral_norm_generator,
    )
    vectors = vectors / torch.linalg.vector_norm(vectors, dim=1, keepdim=True).clamp_min(1e-12)
    for _ in range(SPECTRAL_NORM_ESTIMATE_ITERS):
        left_vectors = torch.bmm(matrices, vectors)
        left_vectors = left_vectors / torch.linalg.vector_norm(left_vectors, dim=1, keepdim=True).clamp_min(1e-12)
        vectors = torch.bmm(matrices.transpose(1, 2), left_vectors)
        vectors = vectors / torch.linalg.vector_norm(vectors, dim=1, keepdim=True).clamp_min(1e-12)
    projections = torch.bmm(matrices, vectors)
    return torch.linalg.vector_norm(projections, dim=1).max(dim=1).values

def spectral_norm_estimates_by_name(named_tensors):
    grouped = defaultdict(list)
    for name, tensor in named_tensors:
        x = tensor.detach().float()
        if x.ndim == 2:
            grouped[tuple(x.shape)].append((name, x))
    estimates = {}
    for items in grouped.values():
        names = [name for name, _ in items]
        matrices = torch.stack([x for _, x in items], dim=0).contiguous()
        values = batched_spectral_norm_estimate(matrices)
        for name, value in zip(names, values):
            estimates[name] = value.item()
    return estimates

def tensor_norm_fields(tensor, prefix='', spectral_norm_estimate=None):
    x = tensor.detach().float()
    if x.ndim == 0:
        return {
            f'{prefix}abs_value': x.abs().item(),
            f'{prefix}rms_norm': x.abs().item(),
        }
    sq = x.square()
    fields = {
        f'{prefix}fro_norm': torch.sqrt(sq.sum()).item(),
        f'{prefix}rms_norm': torch.sqrt(sq.mean()).item(),
    }
    if x.ndim == 2 and args.spectral_norm_estimate_enabled > 0:
        if spectral_norm_estimate is None:
            spectral_norm_estimate = batched_spectral_norm_estimate(x.unsqueeze(0).contiguous())[0].item()
        fields[f'{prefix}spectral_norm_estimate'] = spectral_norm_estimate
        fields[f'{prefix}spectral_norm_estimate_method'] = SPECTRAL_NORM_ESTIMATE_METHOD
        fields[f'{prefix}spectral_norm_estimate_block_size'] = SPECTRAL_NORM_ESTIMATE_BLOCK_SIZE
        fields[f'{prefix}spectral_norm_estimate_iters'] = SPECTRAL_NORM_ESTIMATE_ITERS
    return fields

def tensor_norm_record(step, name, tensor, spectral_norm_estimate=None):
    record = dict(step=step, name=name, shape=list(tensor.shape), ndim=tensor.ndim)
    record.update(tensor_norm_fields(tensor, spectral_norm_estimate=spectral_norm_estimate))
    return record

def optimizer_parameter_hparams():
    hparams = {}
    for optimizer_index, opt in enumerate(optimizers):
        for param_group_index, group in enumerate(opt.param_groups):
            for p in group['params']:
                hparams[id(p)] = dict(
                    optimizer_index=optimizer_index,
                    param_group_index=param_group_index,
                    lr=float(group['lr']),
                    weight_decay=float(group.get('weight_decay', 0.0)),
                )
    return hparams

def should_log_adamw_update_norms(update_step):
    if not master_process or args.adamw_update_norm_every <= 0:
        return False
    return update_step % args.adamw_update_norm_every == 0 or update_step == args.num_iterations

def maybe_capture_adamw_update_state(update_step):
    if not should_log_adamw_update_norms(update_step):
        return None
    hparams = optimizer_parameter_hparams()
    snapshots = {}
    with torch.no_grad():
        for _, p in raw_model.named_parameters():
            if p.requires_grad:
                snapshots[id(p)] = p.detach().clone()
    return dict(hparams=hparams, snapshots=snapshots)

def adamw_update_tensor(tensor, tensor_before, param_hparams, step, name):
    lr = param_hparams['lr']
    if lr == 0:
        raise RuntimeError(f"cannot infer AdamW update for {name} at step {step}: lr is 0")
    weight_decay = param_hparams['weight_decay']
    before = tensor_before.float()
    after = tensor.detach().float()
    delta = after - before
    return -(delta + lr * weight_decay * before) / lr

def adamw_update_norm_record(step, name, tensor, adamw_update, param_hparams, spectral_norm_estimate=None):
    lr = param_hparams['lr']
    weight_decay = param_hparams['weight_decay']
    record = dict(
        step=step,
        name=name,
        shape=list(tensor.shape),
        ndim=tensor.ndim,
        lr=lr,
        weight_decay=weight_decay,
        optimizer_index=param_hparams['optimizer_index'],
        param_group_index=param_hparams['param_group_index'],
    )
    record.update(tensor_norm_fields(
        adamw_update,
        prefix='adamw_update_',
        spectral_norm_estimate=spectral_norm_estimate,
    ))
    return record

def maybe_log_adamw_update_norms(update_step, update_state):
    if update_state is None:
        return
    history_path = os.path.join(logdir, 'adamw_update_norm_history.jsonl')
    hparams = update_state['hparams']
    snapshots = update_state['snapshots']
    pending_records = []
    pending_2d_updates = []
    with torch.no_grad(), open(history_path, 'a') as f:
        for name, tensor in raw_model.named_parameters():
            if not tensor.requires_grad:
                continue
            param_id = id(tensor)
            if param_id not in hparams:
                raise RuntimeError(f"missing optimizer param group mapping for {name}")
            if param_id not in snapshots:
                raise RuntimeError(f"missing parameter snapshot for {name}")
            tensor_before = snapshots.pop(param_id)
            adamw_update = adamw_update_tensor(tensor, tensor_before, hparams[param_id], update_step, name)
            if args.spectral_norm_estimate_enabled > 0 and adamw_update.ndim == 2:
                pending_2d_updates.append((name, adamw_update))
                pending_records.append((name, tensor, adamw_update, hparams[param_id]))
            else:
                record = adamw_update_norm_record(update_step, name, tensor, adamw_update, hparams[param_id])
                f.write(json.dumps(record) + '\n')
                del record
            del tensor_before
        spectral_estimates = spectral_norm_estimates_by_name(pending_2d_updates)
        for name, tensor, adamw_update, param_hparams in pending_records:
            record = adamw_update_norm_record(
                update_step,
                name,
                tensor,
                adamw_update,
                param_hparams,
                spectral_norm_estimate=spectral_estimates[name],
            )
            f.write(json.dumps(record) + '\n')
            del record
    if snapshots:
        raise RuntimeError(f"unused AdamW update snapshots: {len(snapshots)}")

def maybe_log_tensor_norms(step):
    if not master_process or args.tensor_norm_every <= 0:
        return
    if step % args.tensor_norm_every != 0 and step != args.num_iterations:
        return
    history_path = os.path.join(logdir, 'tensor_norm_history.jsonl')
    named_parameters = list(raw_model.named_parameters())
    spectral_estimates = (
        spectral_norm_estimates_by_name(named_parameters)
        if args.spectral_norm_estimate_enabled > 0
        else {}
    )
    with torch.no_grad(), open(history_path, 'a') as f:
        for name, tensor in named_parameters:
            f.write(json.dumps(tensor_norm_record(
                step,

                name,
                tensor,
                spectral_norm_estimate=spectral_estimates.get(name),
            )) + '\n')

activation_probe_x = build_activation_probe_batch()
write_tensor_metadata()
write_activation_probe_metadata(activation_probe_x)

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
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
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        if use_ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
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

    maybe_log_tensor_norms(step)
    maybe_log_activation_probe(step, activation_probe_x)

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
            no_sync = model.no_sync() if use_ddp else contextlib.nullcontext()
            with no_sync: # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # step the optimizers and schedulers
    update_step = step + 1
    prepare_global_elr_step(step)
    adamw_update_state = maybe_capture_adamw_update_state(update_step)
    for opt in optimizers:
        opt.step()
    maybe_log_adamw_update_norms(update_step, adamw_update_state)
    for sched in schedulers:
        sched.step()
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
