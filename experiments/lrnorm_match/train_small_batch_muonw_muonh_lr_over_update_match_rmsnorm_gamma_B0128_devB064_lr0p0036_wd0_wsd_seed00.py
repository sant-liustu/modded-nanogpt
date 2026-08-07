# Copied from experiment/batch-size-norm-dynamics train_gpt2.py.
# Purpose: WSD MuonW wd0 experiment with learnable RMSNorm gamma at B=128, T=1024, 20400 steps.
# Variant: Hyperball-compatible Muon initialization plus MuonH LR-over-raw-update matching.
# Token budget matches the B=512, T=1024, 5100-step large-batch setup.
# Config: batch_size=128, device_batch_size=64, sequence_length=1024, num_iterations=20400, lr=0.0036, warmup+stable+warmdown, block_weight_decay=0.0, seed=0
import os
import random
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import contextlib
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

class MuonW(torch.optim.Optimizer):
    """
    MuonW - Muon with decoupled weight decay for 2D transformer block weights.

    Embedding/head and RMSNorm gamma parameters should stay on AdamW; this optimizer assumes that
    every parameter it receives is a 2D matrix.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, weight_decay=0.0, nesterov=True,
                 backend='newtonschulz5', backend_steps=5,
                 rank=0, world_size=1):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            backend=backend,
            backend_steps=backend_steps,
        )
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size
        self.last_raw_update_fro_norm = None
        self.last_raw_update_fro_norm_by_parameter_id = {}
        assert 0 < momentum < 1
        assert weight_decay >= 0

    def step(self):

        self.last_raw_update_fro_norm_by_parameter_id = {}

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
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
                        curr_idx += p.numel()
                        continue
                    if g.ndim != 2:
                        raise RuntimeError(f"MuonW expects 2D parameters, got shape {tuple(g.shape)}")
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
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
            raw_update_sq = 0.0
            lr_by_parameter_id = group.get('lr_by_parameter_id', {})
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                raw_update_norm = torch.linalg.vector_norm(g.float()).item()
                self.last_raw_update_fro_norm_by_parameter_id[id(p)] = raw_update_norm
                raw_update_sq += raw_update_norm * raw_update_norm
                parameter_lr = lr_by_parameter_id.get(id(p), lr)
                if weight_decay != 0 and p.grad is not None:
                    p.data.mul_(1 - parameter_lr * weight_decay)
                p.data.add_(g, alpha=-parameter_lr)
                curr_idx += p.numel()
            self.last_raw_update_fro_norm = math.sqrt(raw_update_sq)

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
        # Hyperball-compatible initialization for every Muon matrix:
        # W_ij ~ N(0, 1/d_in), i.e. std(W) = 1/sqrt(d_in).
        for linear in (self.c_q, self.c_k, self.c_v, self.c_proj):
            torch.nn.init.normal_(linear.weight, mean=0.0, std=1.0 / math.sqrt(linear.in_features))
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
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        # Same initialization for the MLP matrices; d_in is each Linear's input width.
        for linear in (self.c_fc, self.c_proj):
            torch.nn.init.normal_(linear.weight, mean=0.0, std=1.0 / math.sqrt(linear.in_features))

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
    embed_learning_rate : float = 0.0036
    muon_learning_rate : float = 0.00036 # original Muon recipe: 0.1 * embed_learning_rate
    warmup_iters : int = 1000
    warmdown_iters : int = 5800 # number of iterations of linear warmup/warmdown for trapezoidal WSD schedule
    weight_decay : float = 0.0 # weight decay for block weights and tied wte/lm_head; RMSNorm gamma is excluded
    muonh_lr_over_update_reference_npz : str = 'experiments/lrnorm_match/reference_rmsnorm_gamma_qknorm_adamh_muonh_wd0_lr_over_raw_update.npz'
    # evaluation and logging hyperparams
    val_loss_every : int = 500 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    compile_model : int = 1 # compile the model with torch.compile
    tensor_norm_every : int = 1 # every how many steps to log tensor norm history? 0 disables
    muonw_update_norm_every : int = 1 # every how many optimizer steps to log MuonW effective update norms? 0 disables
    activation_probe_every : int = 0 # every how many steps to log fixed-probe activation RMS ratios? 0 disables
    spectral_norm_estimate_enabled : int = 1 # whether to estimate 2D spectral norms in tensor/update norm histories
    activation_probe_eps : float = 1e-12 # denominator epsilon for activation RMS ratios
    seed : int = 0
args = Hyperparameters()

def load_muonh_lr_over_update_targets(path):
    with np.load(path, allow_pickle=False) as reference:
        update_steps = reference['update_steps'].copy()
        block_names = tuple(str(name) for name in reference['block_names'])
        block_reference_lrs = reference['block_reference_lrs'].copy()
        block_reference_raw_updates = reference['block_reference_raw_updates'].copy()
        block_target_lr_over_updates = reference['block_target_lr_over_updates'].copy()
        tied_reference_lrs = reference['tied_reference_lrs'].copy()
        tied_reference_raw_updates = reference['tied_reference_raw_updates'].copy()
        tied_target_lr_over_updates = reference['tied_target_lr_over_updates'].copy()
    if list(update_steps) != list(range(1, args.num_iterations + 1)):
        raise ValueError(
            f"expected consecutive MuonH reference steps 1..{args.num_iterations} in {path}"
        )
    return dict(
        block_index_by_name={name: index for index, name in enumerate(block_names)},
        block_reference_lrs=block_reference_lrs,
        block_reference_raw_updates=block_reference_raw_updates,
        block_target_lr_over_updates=block_target_lr_over_updates,
        tied_reference_lrs=tied_reference_lrs,
        tied_reference_raw_updates=tied_reference_raw_updates,
        tied_target_lr_over_updates=tied_target_lr_over_updates,
    )

muonh_lr_over_update_targets = load_muonh_lr_over_update_targets(
    args.muonh_lr_over_update_reference_npz
)
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
raw_model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
raw_model = raw_model.cuda()
model = raw_model
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
if args.compile_model:
    model = torch.compile(model)
# here we wrap the executable model into DDP; raw_model remains the original GPT for stable parameter names/checkpoints
if use_ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer(s)
def is_rmsnorm_gamma_name(name):
    if name.startswith('_orig_mod.'):
        name = name[len('_orig_mod.'):]
    return (
        name == 'final_norm.weight'
        or name.endswith('.q_norm.weight')
        or name.endswith('.k_norm.weight')
        or name.endswith('.attn_norm.weight')
        or name.endswith('.mlp_norm.weight')
    )

block_weight_parameters_by_name = {
    (name[len('_orig_mod.'):] if name.startswith('_orig_mod.') else name): p
    for name, p in raw_model.named_parameters()
    if (name[len('_orig_mod.'):] if name.startswith('_orig_mod.') else name).startswith('transformer.h.')
    and not is_rmsnorm_gamma_name(name)
}
block_weight_parameters = list(block_weight_parameters_by_name.values())
rmsnorm_gamma_parameters = [
    p
    for name, p in raw_model.named_parameters()
    if is_rmsnorm_gamma_name(name)
]
optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.embed_learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)
tied_embedding_parameter = raw_model.transformer.wte.weight
if not any(parameter is tied_embedding_parameter for parameter in optimizer1.param_groups[0]['params']):
    raise RuntimeError('AdamW optimizer does not own the tied embedding/lm_head weight')
optimizer2 = MuonW([
    dict(params=block_weight_parameters, weight_decay=args.weight_decay, lrnorm_match_group=True)
], lr=args.muon_learning_rate, momentum=0.95, weight_decay=args.weight_decay,
   nesterov=True, rank=ddp_rank, world_size=ddp_world_size)

@torch.no_grad()
def initial_block_radius_total():
    total_sq = torch.zeros((), device=device, dtype=torch.float64)
    for parameter in block_weight_parameters:
        total_sq.add_(parameter.detach().double().square().sum())
    return torch.sqrt(total_sq).item()

@torch.no_grad()
def current_block_parameter_fro_norm_total():
    total_sq = torch.zeros((), device=device, dtype=torch.float64)
    for parameter in block_weight_parameters:
        total_sq.add_(parameter.detach().double().square().sum())
    return torch.sqrt(total_sq).item()

@torch.no_grad()
def current_block_parameter_fro_norms_by_name():
    return {
        name: torch.linalg.vector_norm(parameter.detach().float()).item()
        for name, parameter in block_weight_parameters_by_name.items()
    }

last_adamw_raw_update_fro_norm = None
if master_process:
    print(
        "Loaded "
        f"{len(muonh_lr_over_update_targets['tied_target_lr_over_updates'])} "
        "MuonH LR-over-update reference steps"
    )
optimizers = [optimizer1, optimizer2]
if rmsnorm_gamma_parameters:
    optimizer3 = torch.optim.AdamW([
        dict(params=rmsnorm_gamma_parameters, weight_decay=0.0, lrnorm_match_group=False)
    ], lr=0.5*args.embed_learning_rate, betas=(0.9, 0.95), fused=True)
    optimizers.append(optimizer3)
# learning rate decay scheduler (linear warmup, stable phase, linear warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) stable LR for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
# The tied embedding and block LRs are overwritten by the controller before
# every step; their schedulers must therefore be no-ops.
schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer1, lambda _: 1.0),
               torch.optim.lr_scheduler.LambdaLR(optimizer2, lambda _: 1.0)]
if rmsnorm_gamma_parameters:
    schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimizer3, get_lr))

# begin logging
if master_process:
    run_id = os.environ.get('RUN_ID_OVERRIDE') or str(uuid.uuid4())
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
                    lr=float(group.get('lr_by_parameter_id', {}).get(id(p), group['lr'])),
                    weight_decay=float(group.get('weight_decay', 0.0)),
                )
    return hparams

@torch.no_grad()
def current_tied_embedding_fro_norm():
    return torch.linalg.vector_norm(tied_embedding_parameter.detach().float()).item()

@torch.no_grad()
def current_adamw_raw_update_fro_norm():
    """Recover the bias-corrected AdamW direction from fused AdamW state."""
    state = optimizer1.state[tied_embedding_parameter]
    if not state:
        return None
    step = int(state['step'].item())
    if step <= 0:
        return None
    beta1, beta2 = optimizer1.param_groups[0]['betas']
    eps = optimizer1.param_groups[0]['eps']
    exp_avg = state['exp_avg'].detach().float()
    exp_avg_sq = state['exp_avg_sq'].detach().float()
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2_sqrt = math.sqrt(1.0 - beta2 ** step)
    denominator = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps)
    direction = exp_avg.div(bias_correction1).div_(denominator)
    return torch.linalg.vector_norm(direction).item()

def apply_muonh_lr_over_update_match(update_step):
    """Match each parameter's LR/norm to its MuonH LR/raw-update ratio."""
    row = update_step - 1
    if row < 0 or row >= args.num_iterations:
        raise RuntimeError(f"invalid MuonH reference update_step={update_step}")
    current_block_parameter_fro_norms = current_block_parameter_fro_norms_by_name()
    current_tied_embedding_norm = current_tied_embedding_fro_norm()
    block_index_by_name = muonh_lr_over_update_targets['block_index_by_name']
    if set(block_index_by_name) != set(block_weight_parameters_by_name):
        raise RuntimeError('MuonH reference block names do not match this model')
    block_target_lr_over_update_values = (
        muonh_lr_over_update_targets['block_target_lr_over_updates'][row]
    )
    adjusted_muonw_lr_by_name = {
        name: (
            float(block_target_lr_over_update_values[block_index_by_name[name]])
            * current_block_parameter_fro_norms[name]
        )
        for name in block_weight_parameters_by_name
    }
    tied_target_lr_over_update = float(
        muonh_lr_over_update_targets['tied_target_lr_over_updates'][row]
    )
    adjusted_embed_lr = tied_target_lr_over_update * current_tied_embedding_norm
    for group in optimizer1.param_groups:
        group['lr'] = adjusted_embed_lr
    for group in optimizer2.param_groups:
        group['lr_by_parameter_id'] = {
            id(block_weight_parameters_by_name[name]): adjusted_muonw_lr_by_name[name]
            for name in block_weight_parameters_by_name
        }
    return dict(
        update_step=update_step,
        block_reference_lr=float(muonh_lr_over_update_targets['block_reference_lrs'][row]),
        block_reference_raw_update_fro_norm_by_name={
            name: float(
                muonh_lr_over_update_targets['block_reference_raw_updates'][
                    row, block_index_by_name[name]
                ]
            )
            for name in block_weight_parameters_by_name
        },
        current_block_parameter_fro_norms_by_name=current_block_parameter_fro_norms,
        block_target_lr_over_update_by_name={
            name: float(block_target_lr_over_update_values[block_index_by_name[name]])
            for name in block_weight_parameters_by_name
        },
        adjusted_muonw_lr_by_name=adjusted_muonw_lr_by_name,
        current_tied_embedding_fro_norm=current_tied_embedding_norm,
        tied_reference_lr=float(muonh_lr_over_update_targets['tied_reference_lrs'][row]),
        tied_reference_raw_update_fro_norm=float(
            muonh_lr_over_update_targets['tied_reference_raw_updates'][row]
        ),
        tied_target_lr_over_update=tied_target_lr_over_update,
        adjusted_embed_lr=adjusted_embed_lr,
    )

def maybe_log_muonh_lr_over_update_match(controller_record, current_adamw_raw_update_fro_norm):
    if not master_process:
        return
    current_raw_update_fro_norm_by_parameter_id = optimizer2.last_raw_update_fro_norm_by_parameter_id
    if not current_raw_update_fro_norm_by_parameter_id:
        raise RuntimeError('MuonW step did not expose a raw update norm')
    if current_adamw_raw_update_fro_norm is None:
        raise RuntimeError('AdamW step did not expose a raw update norm')
    controller_record = dict(controller_record)
    controller_record.update(
        current_muonw_raw_update_fro_norm_by_name={
            name: current_raw_update_fro_norm_by_parameter_id[id(parameter)]
            for name, parameter in block_weight_parameters_by_name.items()
        },
        current_adamw_raw_update_fro_norm=current_adamw_raw_update_fro_norm,
        actual_block_lr_over_parameter_norm_by_name={
            name: (
                controller_record['adjusted_muonw_lr_by_name'][name]
                / controller_record['current_block_parameter_fro_norms_by_name'][name]
            )
            for name in block_weight_parameters_by_name
        },
        actual_tied_lr_over_parameter_norm=(
            controller_record['adjusted_embed_lr']
            / controller_record['current_tied_embedding_fro_norm']
        ),
    )
    with open(os.path.join(logdir, 'muonh_lr_over_update_match_history.jsonl'), 'a') as f:
        f.write(json.dumps(controller_record) + '\n')

def should_log_muonw_update_norms(update_step):
    if not master_process or args.muonw_update_norm_every <= 0:
        return False
    return update_step % args.muonw_update_norm_every == 0 or update_step == args.num_iterations

def maybe_capture_muonw_update_state(update_step):
    if not should_log_muonw_update_norms(update_step):
        return None
    hparams = optimizer_parameter_hparams()
    snapshots = {}
    with torch.no_grad():
        for _, p in raw_model.named_parameters():
            if p.requires_grad:
                snapshots[id(p)] = p.detach().clone()
    return dict(hparams=hparams, snapshots=snapshots)

def muonw_update_tensor(tensor, tensor_before, param_hparams, step, name):
    lr = param_hparams['lr']
    if lr == 0:
        raise RuntimeError(f"cannot infer MuonW update for {name} at step {step}: lr is 0")
    weight_decay = param_hparams['weight_decay']
    before = tensor_before.float()
    after = tensor.detach().float()
    delta = after - before
    return -(delta + lr * weight_decay * before) / lr

def muonw_update_with_wd_tensor(tensor, tensor_before, param_hparams, step, name):
    lr = param_hparams['lr']
    if lr == 0:
        raise RuntimeError(f"cannot infer MuonW update with WD for {name} at step {step}: lr is 0")
    before = tensor_before.float()
    after = tensor.detach().float()
    delta = after - before
    return -delta / lr

def muonw_update_norm_record(
    step,
    name,
    tensor,
    muonw_update,
    muonw_update_with_wd,
    param_hparams,
    spectral_norm_estimate=None,
    spectral_norm_estimate_with_wd=None,
):
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
        muonw_update,
        prefix='muonw_update_',
        spectral_norm_estimate=spectral_norm_estimate,
    ))
    record.update(tensor_norm_fields(
        muonw_update_with_wd,
        prefix='muonw_update_with_wd_',
        spectral_norm_estimate=spectral_norm_estimate_with_wd,
    ))
    return record

def maybe_log_muonw_update_norms(update_step, update_state):
    if update_state is None:
        return
    history_path = os.path.join(logdir, 'muonw_update_norm_history.jsonl')
    hparams = update_state['hparams']
    snapshots = update_state['snapshots']
    pending_records = []
    pending_2d_updates = []
    pending_2d_updates_with_wd = []
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
            muonw_update = muonw_update_tensor(tensor, tensor_before, hparams[param_id], update_step, name)
            muonw_update_with_wd = muonw_update_with_wd_tensor(tensor, tensor_before, hparams[param_id], update_step, name)
            if args.spectral_norm_estimate_enabled > 0 and muonw_update.ndim == 2:
                pending_2d_updates.append((name, muonw_update))
                pending_2d_updates_with_wd.append((name, muonw_update_with_wd))
                pending_records.append((name, tensor, muonw_update, muonw_update_with_wd, hparams[param_id]))
            else:
                record = muonw_update_norm_record(update_step, name, tensor, muonw_update, muonw_update_with_wd, hparams[param_id])
                f.write(json.dumps(record) + '\n')
                del record
            del tensor_before
        spectral_estimates = spectral_norm_estimates_by_name(pending_2d_updates)
        spectral_estimates_with_wd = spectral_norm_estimates_by_name(pending_2d_updates_with_wd)
        for name, tensor, muonw_update, muonw_update_with_wd, param_hparams in pending_records:
            record = muonw_update_norm_record(
                update_step,
                name,
                tensor,
                muonw_update,
                muonw_update_with_wd,
                param_hparams,
                spectral_norm_estimate=spectral_estimates[name],
                spectral_norm_estimate_with_wd=spectral_estimates_with_wd[name],
            )
            f.write(json.dumps(record) + '\n')
            del record
    if snapshots:
        raise RuntimeError(f"unused MuonW update snapshots: {len(snapshots)}")

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
    muonh_match_record = apply_muonh_lr_over_update_match(update_step)
    muonw_update_state = maybe_capture_muonw_update_state(update_step)
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    maybe_log_muonw_update_norms(update_step, muonw_update_state)
    last_adamw_raw_update_fro_norm = current_adamw_raw_update_fro_norm()
    maybe_log_muonh_lr_over_update_match(muonh_match_record, last_adamw_raw_update_fro_norm)
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
