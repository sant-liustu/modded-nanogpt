# Copied from experiment/batch-size-norm-dynamics train_gpt2.py.
# Purpose: WSD AdamH + MuonH wd0 experiment with learnable RMSNorm gamma at B=128, T=1024, 20400 steps.
# Variant: dynamically match MuonH LR/raw-update to the per-tensor MuonW LR/norm reference,
#          while using the exact MuonW reference initialization for every block matrix.
# Token budget matches the B=512, T=1024, 5100-step large-batch setup.
# Config: batch_size=128, device_batch_size=64, sequence_length=1024, num_iterations=20400, lr=0.0036, warmup+stable+warmdown, block_weight_decay=0.0, seed=0; controller target=MuonW wd0 WSD; init=MuonW reference.
import os
import random
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import contextlib
import hashlib
import gzip
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

class Adam(torch.optim.Optimizer):
    """Small reference Adam implementation used as the base for AdamH."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta values: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _update_adam_state(self, p, grad, state, group, apply_weight_decay=True):
        if grad.is_sparse:
            raise RuntimeError("Adam does not support sparse gradients")
        if apply_weight_decay and group['weight_decay'] != 0:
            grad = grad.add(p, alpha=group['weight_decay'])
        if 'exp_avg' not in state:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p)
            state['exp_avg_sq'] = torch.zeros_like(p)

        state['step'] += 1
        beta1, beta2 = group['betas']
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        # torch.optim.Adam uses lerp_ for the first moment; keep the same
        # operation order so this reference implementation matches it closely.
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2_sqrt = math.sqrt(1 - beta2 ** state['step'])
        # Match torch.optim.Adam exactly: bias-correct the second moment before
        # adding epsilon, then leave the learning rate to the caller.
        denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(group['eps'])
        return exp_avg, denom, bias_correction1

    def _adam_direction(self, p, grad, state, group, apply_weight_decay=True):
        exp_avg, denom, bias_correction1 = self._update_adam_state(
            p, grad, state, group, apply_weight_decay=apply_weight_decay
        )
        direction = exp_avg / denom
        direction.div_(bias_correction1)
        return direction

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                exp_avg, denom, bias_correction1 = self._update_adam_state(
                    p, p.grad, self.state[p], group
                )
                # Keep the same fused-in-place update form as torch.optim.Adam.
                p.addcdiv_(exp_avg, denom, value=-group['lr'] / bias_correction1)
        return loss

class AdamH(Adam):
    """Adam Hyperball for 2D matrices, following the MuonH projection pattern."""

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.last_hyperball_records = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.last_hyperball_records = {}

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise RuntimeError(f"AdamH expects 2D parameters, got shape {tuple(p.shape)}")
                state = self.state[p]
                if 'hyperball_R' not in state:
                    state['hyperball_R'] = torch.linalg.vector_norm(p.detach().float()).item()
                radius = state['hyperball_R']

                # Adam direction before Hyperball normalization; the decoupled decay,
                # if enabled, is applied below just as in MuonW.
                direction = self._adam_direction(
                    p, p.grad, state, group, apply_weight_decay=False
                )
                weight_norm_before = torch.linalg.vector_norm(p.data.float()).item()
                raw_update_fro_norm = torch.linalg.vector_norm(direction.float()).item()
                target_lr_over_norm = group.get('target_lr_over_norm_by_parameter_id', {}).get(id(p))
                parameter_lr = lr if target_lr_over_norm is None else target_lr_over_norm * raw_update_fro_norm
                direction_norm = max(raw_update_fro_norm, 1e-12)
                direction.mul_(radius / direction_norm)
                normalized_update_fro_norm = torch.linalg.vector_norm(direction.float()).item()

                if weight_decay != 0:
                    p.data.mul_(1 - parameter_lr * weight_decay)
                p.data.add_(direction, alpha=-parameter_lr)
                weight_norm_after_update = torch.linalg.vector_norm(p.data.float()).item()
                weight_norm = max(weight_norm_after_update, 1e-12)
                p.data.mul_(radius / weight_norm)
                weight_norm_after_projection = torch.linalg.vector_norm(p.data.float()).item()

                self.last_hyperball_records[id(p)] = dict(
                    lr=float(parameter_lr),
                    weight_decay=float(weight_decay),
                    radius=float(radius),
                    raw_update_fro_norm=float(raw_update_fro_norm),
                    normalized_update_fro_norm=float(normalized_update_fro_norm),
                    weight_norm_before=float(weight_norm_before),
                    weight_norm_after_update=float(weight_norm_after_update),
                    weight_norm_after_projection=float(weight_norm_after_projection),
                    has_grad=True,
                )
        return loss

class MuonW(torch.optim.Optimizer):
    """
    MuonW - Muon with decoupled weight decay for 2D transformer block weights.

    Embedding/head and RMSNorm gamma parameters should stay on Adam or AdamH; this optimizer assumes that
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
        self.last_hyperball_records = {}
        assert 0 < momentum < 1
        assert weight_decay >= 0

    def step(self):

        # Rank 0 keeps exact per-matrix Hyperball telemetry from this optimizer
        # step. Other ranks still execute the same update, but do not retain a
        # duplicate copy of the logging records.
        self.last_hyperball_records = {}

        for group in self.param_groups:

            momentum = group['momentum']
            weight_decay = group['weight_decay']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                state = self.state[p]
                if 'hyperball_R' not in state:
                    state['hyperball_R'] = torch.linalg.vector_norm(p.detach().float()).item()
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        curr_idx += p.numel()
                        continue
                    if g.ndim != 2:
                        raise RuntimeError(f"MuonW expects 2D parameters, got shape {tuple(g.shape)}")
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

            # Hyperball: normalize the update and the updated weight to the
            # matrix's initial F-norm radius.
            curr_idx = 0
            for p in group['params']:
                state = self.state[p]
                radius = state['hyperball_R']
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                weight_norm_before = torch.linalg.vector_norm(p.data.float()).item()
                raw_update_fro_norm = torch.linalg.vector_norm(g.float()).item()
                target_lr_over_norm = group.get('target_lr_over_norm_by_parameter_id', {}).get(id(p))
                parameter_lr = (
                    group['lr'] if target_lr_over_norm is None
                    else target_lr_over_norm * raw_update_fro_norm
                )
                update_norm = max(raw_update_fro_norm, 1e-12)
                g.mul_(radius / update_norm)
                normalized_update_fro_norm = torch.linalg.vector_norm(g.float()).item()
                if weight_decay != 0 and p.grad is not None:
                    p.data.mul_(1 - parameter_lr * weight_decay)
                p.data.add_(g, alpha=-parameter_lr)
                weight_norm_after_update = torch.linalg.vector_norm(p.data.float()).item()
                weight_norm = max(weight_norm_after_update, 1e-12)
                p.data.mul_(radius / weight_norm)
                weight_norm_after_projection = torch.linalg.vector_norm(p.data.float()).item()
                if self.rank == 0:
                    self.last_hyperball_records[id(p)] = dict(
                        lr=float(parameter_lr),
                        weight_decay=float(weight_decay),
                        radius=float(radius),
                        raw_update_fro_norm=float(raw_update_fro_norm),
                        normalized_update_fro_norm=float(normalized_update_fro_norm),
                        weight_norm_before=float(weight_norm_before),
                        weight_norm_after_update=float(weight_norm_after_update),
                        weight_norm_after_projection=float(weight_norm_after_projection),
                        has_grad=p.grad is not None,
                    )
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
        # Match the reference MuonW initialization exactly: Q/K/V retain
        # nn.Linear's default initialization and only the residual-writing
        # projection receives GPT depth scaling.
        c_proj_std = config.init_std / math.sqrt(2 * config.n_layer)
        torch.nn.init.normal_(self.c_proj.weight, mean=0.0, std=c_proj_std)
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
        # Match the reference MuonW initialization exactly: c_fc retains its
        # nn.Linear default and only the residual-writing projection is scaled.
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
    embed_learning_rate : float = 0.0036
    hyperball_learning_rate : float = 0.01 # relative Hyperball step size eta
    warmup_iters : int = 1000
    warmdown_iters : int = 5800 # number of iterations of linear warmup/warmdown for trapezoidal WSD schedule
    weight_decay : float = 0.0 # weight decay for block weights and tied wte/lm_head; RMSNorm gamma is excluded
    muonw_lrnorm_reference_json : str = 'experiments/lrnorm_match/reference_rmsnorm_gamma_qknorm_muonw_wd0_wsd_per_tensor_lr_over_norm.jsonl.gz'
    # evaluation and logging hyperparams
    val_loss_every : int = 500 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    compile_model : int = 1 # compile the model with torch.compile
    tensor_norm_every : int = 1 # every how many steps to log tensor norm history? 0 disables
    muonw_update_norm_every : int = 1 # every how many optimizer steps to log exact Hyperball norms? 0 disables
    activation_probe_every : int = 0 # every how many steps to log fixed-probe activation RMS ratios? 0 disables
    spectral_norm_estimate_enabled : int = 1 # whether to estimate 2D spectral norms in tensor/update norm histories
    activation_probe_eps : float = 1e-12 # denominator epsilon for activation RMS ratios
    seed : int = 0
args = Hyperparameters()

def load_muonw_lrnorm_targets(path):
    rows = []
    opener = gzip.open if path.endswith('.gz') else open
    mode = 'rt' if path.endswith('.gz') else 'r'
    with opener(path, mode, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            update_step = int(record['update_step'])
            if update_step != len(rows) + 1:
                raise RuntimeError(
                    f"expected MuonW reference step {len(rows) + 1}, got {update_step} at {path}:{line_number}"
                )
            targets = {
                str(name): float(value)
                for name, value in record['target_lr_over_tensor_norms'].items()
            }
            rows.append(dict(
                reference_block_lr=float(record['reference_block_lr']),
                reference_embed_lr=float(record['reference_embed_lr']),
                reference_tensor_fro_norms={
                    str(name): float(value)
                    for name, value in record['reference_tensor_fro_norms'].items()
                },
                target_lr_over_tensor_norms=targets,
            ))
    if len(rows) != args.num_iterations:
        raise RuntimeError(
            f"expected {args.num_iterations} MuonW reference steps in {path}, got {len(rows)}"
        )
    return rows

muonw_lrnorm_targets = load_muonw_lrnorm_targets(args.muonw_lrnorm_reference_json)
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
# The tied wte/lm_head is one 2D parameter, so it receives one AdamH update.
optimizer1 = AdamH(raw_model.lm_head.parameters(), lr=args.hyperball_learning_rate, betas=(0.9, 0.95),
                   weight_decay=args.weight_decay)
tied_embedding_parameter = raw_model.transformer.wte.weight
if not any(parameter is tied_embedding_parameter for parameter in optimizer1.param_groups[0]['params']):
    raise RuntimeError('AdamH optimizer does not own the tied embedding/lm_head weight')
optimizer2 = MuonW([
    dict(params=block_weight_parameters, weight_decay=args.weight_decay, lrnorm_match_group=True)
], lr=args.hyperball_learning_rate, momentum=0.95, weight_decay=args.weight_decay,
   nesterov=True, rank=ddp_rank, world_size=ddp_world_size)
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
# The tied embedding and block LRs are selected inside AdamH/MuonH from the
# current raw update norms, so their schedulers must be no-ops.
schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer1, lambda _: 1.0),
              torch.optim.lr_scheduler.LambdaLR(optimizer2, lambda _: 1.0)]
if rmsnorm_gamma_parameters:
    schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimizer3, get_lr))

def apply_muonw_lrnorm_match(update_step):
    """Give MuonH the MuonW per-tensor LR/norm targets for this update."""
    row = update_step - 1
    if row < 0 or row >= args.num_iterations:
        raise RuntimeError(f'invalid MuonW reference update_step={update_step}')
    reference = muonw_lrnorm_targets[row]
    targets = reference['target_lr_over_tensor_norms']
    expected_names = set(block_weight_parameters_by_name) | {'transformer.wte.weight'}
    if set(targets) != expected_names:
        raise RuntimeError('MuonW per-tensor reference names do not match this model')
    for group in optimizer1.param_groups:
        group['target_lr_over_norm_by_parameter_id'] = {
            id(tied_embedding_parameter): targets['transformer.wte.weight']
        }
    for group in optimizer2.param_groups:
        group['target_lr_over_norm_by_parameter_id'] = {
            id(block_weight_parameters_by_name[name]): targets[name]
            for name in block_weight_parameters_by_name
        }
    return dict(
        update_step=update_step,
        reference_block_lr=reference['reference_block_lr'],
        reference_embed_lr=reference['reference_embed_lr'],
        reference_tensor_fro_norms=reference['reference_tensor_fro_norms'],
        target_lr_over_tensor_norms=targets,
    )

def maybe_log_muonw_lrnorm_match(controller_record, hyperball_records):
    if not master_process:
        return
    if hyperball_records is None:
        raise RuntimeError('missing MuonH telemetry on the master rank')
    controller_record = dict(controller_record)
    controller_record.update(
        current_muonh_raw_update_fro_norm_by_name={
            name: float(hyperball_records[id(parameter)]['raw_update_fro_norm'])
            for name, parameter in block_weight_parameters_by_name.items()
        },
        current_adamh_raw_update_fro_norm=float(
            hyperball_records[id(tied_embedding_parameter)]['raw_update_fro_norm']
        ),
        actual_block_lr_over_raw_update_by_name={
            name: (
                float(hyperball_records[id(parameter)]['lr'])
                / float(hyperball_records[id(parameter)]['raw_update_fro_norm'])
            )
            for name, parameter in block_weight_parameters_by_name.items()
        },
        actual_tied_lr_over_raw_update=(
            float(hyperball_records[id(tied_embedding_parameter)]['lr'])
            / float(hyperball_records[id(tied_embedding_parameter)]['raw_update_fro_norm'])
        ),
    )
    with open(os.path.join(logdir, 'muonw_lrnorm_match_history.jsonl'), 'a') as f:
        f.write(json.dumps(controller_record) + '\n')

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

def should_log_muonhyperball_norms(update_step):
    if not master_process or args.muonw_update_norm_every <= 0:
        return False
    return update_step % args.muonw_update_norm_every == 0 or update_step == args.num_iterations

def maybe_log_muonhyperball_norms(update_step, hyperball_records):
    if not should_log_muonhyperball_norms(update_step):
        return
    if hyperball_records is None:
        raise RuntimeError('missing MuonHyperball telemetry on the master rank')
    history_path = os.path.join(logdir, 'muonhyperball_norm_history.jsonl')
    with torch.no_grad(), open(history_path, 'a') as f:
        for name, tensor in raw_model.named_parameters():
            param_record = hyperball_records.get(id(tensor))
            if param_record is None:
                continue
            record = dict(
                step=update_step,
                name=name,
                shape=list(tensor.shape),
                ndim=tensor.ndim,
                **param_record,
            )
            f.write(json.dumps(record) + '\n')

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
    controller_record = apply_muonw_lrnorm_match(update_step)
    hyperball_records = {}
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        if isinstance(opt, (MuonW, AdamH)):
            hyperball_records.update(opt.last_hyperball_records)
        sched.step()
    maybe_log_muonhyperball_norms(update_step, hyperball_records or None)
    maybe_log_muonw_lrnorm_match(controller_record, hyperball_records or None)
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
