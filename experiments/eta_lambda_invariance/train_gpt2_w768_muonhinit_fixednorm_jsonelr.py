import os
# Configure the allocator before importing torch or initializing CUDA.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import random
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import contextlib
import hashlib
import json
import uuid
import glob
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

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
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768
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
        # Match the existing MuonH-initialized fixed-norm runner:
        # every block matrix has variance 1 / fan_in, including c_proj.
        # Keep tied embedding initialization and RMSNorm gamma unchanged.
        for pn, p in self.named_parameters():
            if pn.startswith("transformer.h.") and pn.endswith(".weight") and p.ndim == 2:
                torch.nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(p.shape[1]))

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
    batch_size : int = 128 # global batch: 2 ranks x 64 sequences
    device_batch_size : int = 64 # sequences per GPU; 65536 tokens at sequence_length=1024
    expected_world_size : int = 2
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 20400 # same total training tokens as 5100 updates at global batch512
    embed_learning_rate : float = 0.0036 # matrix LR is overwritten by JSON ELR
    gamma_learning_rate : float = 0.0018 # original absolute LR; warmup then constant
    warmup_iters : int = 1000 # same warmup token budget as 250 updates at batch512
    warmdown_iters : int = 5800 # legacy scheduler setting; actual tensor ELR comes from JSON
    weight_decay : float = 0.0 # fixed-initial-RMS arm: no weight decay
    seed : int = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 0 # periodic validation disabled; set positive to enable
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    compile_model : int = 1 # compile the model with torch.compile
    tensor_norm_every : int = 1 # every how many steps to log tensor norm history? 0 disables
    adamw_update_norm_every : int = 1 # every how many optimizer steps to log AdamW effective update norms? 0 disables
args = Hyperparameters()

import argparse
from fnmatch import fnmatchcase
from pathlib import Path
PREVIOUS_RUNNER_SOURCE_SHA256 = 'b470ac1a787072e7015a6d9f8b8540b02900b79ef515ea469730ef00bd585339'
RETIRED_CONFIG_FIELDS = ('activation_probe_every', 'spectral_norm_estimate_enabled', 'activation_probe_eps', 'lrnorm_match_enabled', 'lrnorm_reference_json', 'lrnorm_reference_global_batch_size', 'lrnorm_match_start_update', 'lrnorm_match_log_every', 'simpson3_every', 'lca_probe_batches', 'lca_include_vectors', 'lca_fraction_eps', 'save_every', 'muon_learning_rate', 'peak_rms_elr')
parser = argparse.ArgumentParser(description='Per-update tensor RMS-ELR JSON with complete-state resume')
parser.add_argument('--schedule-json', type=Path, required=True)
parser.add_argument('--resume', type=Path)
parser.add_argument('--checkpoint-dir', type=Path)
parser.add_argument('--save-at', type=int, help='completed update at which to save; defaults to final update')
parser.add_argument('--stop-after-save', action='store_true')
cli = parser.parse_args()
cli.branch = 'json'
cli.fork_step = 0  # filled from checkpoint, never used for schedule indexing
if cli.resume:
    manifest = json.loads((cli.resume / 'complete.json').read_text())
    cli.fork_step = manifest['step']
if not isinstance(cli.fork_step, int) or not 0 <= cli.fork_step <= args.num_iterations:
    parser.error('invalid checkpoint step')
if (cli.save_at is not None or cli.stop_after_save) and cli.checkpoint_dir is None:
    parser.error('--save-at/--stop-after-save requires --checkpoint-dir')
if cli.checkpoint_dir:
    cli.save_at = args.num_iterations if cli.save_at is None else cli.save_at
    if not cli.fork_step < cli.save_at <= args.num_iterations:
        parser.error('save step must be after the starting checkpoint and within the run')
    if cli.checkpoint_dir.exists():
        parser.error('checkpoint output directory already exists')

def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result

schedule_text = cli.schedule_json.read_text(encoding='utf-8-sig')
schedule_document = json.loads(schedule_text, object_pairs_hook=reject_duplicate_keys)
schedule_sha256 = hashlib.sha256(schedule_text.encode()).hexdigest()

def load_tensor_elr_schedule(document, names, completed_step, total_steps):
    if not isinstance(document, dict) or set(document) != {'version', 'steps'} or type(document['version']) is not int or document['version'] != 1:
        raise ValueError('schedule requires version=1 and steps only')
    if not isinstance(document['steps'], list):
        raise ValueError('steps must be a list')
    table = {}
    seen = set()
    def elr(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'ELR must be a finite nonnegative number: {value!r}')
        return float(value)
    for row in document['steps']:
        if not isinstance(row, dict) or set(row)-{'update_step', 'default_elr', 'overrides'}:
            raise ValueError('invalid schedule row keys')
        step = row.get('update_step')
        if type(step) is not int or not 1 <= step <= total_steps or step in seen:
            raise ValueError(f'invalid or duplicate update_step: {step}')
        seen.add(step)
        # The checkpoint already contains these updates. Only future ELRs matter.
        if step <= completed_step:
            continue
        values = {n: elr(row['default_elr']) for n in names} if 'default_elr' in row else {}
        overrides = row.get('overrides', {})
        if not isinstance(overrides, dict):
            raise ValueError('overrides must map tensor names/globs to ELR')
        assigned = set()
        for pattern, value in overrides.items():
            matches = {n for n in names if fnmatchcase(n, pattern)}
            if not matches:
                raise ValueError(f'no tensor matches: {pattern}')
            if assigned & matches:
                raise ValueError(f'overlapping overrides at step {step}: {pattern}')
            assigned.update(matches)
            target = elr(value)
            values.update({n: target for n in matches})
        if set(values) != set(names):
            raise ValueError(f'incomplete tensor coverage at step {step}')
        if step > completed_step:
            table[step] = values
    missing = set(range(completed_step+1, total_steps+1))-set(table)
    if missing:
        raise ValueError(f'missing future schedule steps, first: {min(missing)}')
    return table

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
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
if ddp_world_size != args.expected_world_size:
    raise RuntimeError(
        f'this training runner requires {args.expected_world_size} GPUs, got {ddp_world_size}; '
        'launch with: torchrun --standalone --nproc_per_node=2 <script>'
    )

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
width_multiplier = model.width_multiplier
model = model.cuda()
# Keep clean parameter names while the execution model is compiled and wrapped.
raw_model = model
tensor_elr_schedule = load_tensor_elr_schedule(schedule_document,
    [n for n, p in raw_model.named_parameters() if p.requires_grad and p.ndim >= 2], cli.fork_step, args.num_iterations)
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
if args.compile_model:
    model = torch.compile(model)
# here we wrap model into DDP container
if use_ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer(s)
def is_rmsnorm_gamma_name(name):
    return name.endswith('.weight') and (
        name == 'final_norm.weight'
        or '.q_norm.' in name
        or '.k_norm.' in name
        or '.attn_norm.' in name
        or '.mlp_norm.' in name
    )

rmsnorm_gamma_parameters = [
    p for name, p in raw_model.named_parameters()
    if is_rmsnorm_gamma_name(name)
]
rmsnorm_gamma_param_ids = {id(p) for p in rmsnorm_gamma_parameters}
block_parameters = [
    p for name, p in raw_model.named_parameters()
    if name.startswith('transformer.h.') and id(p) not in rmsnorm_gamma_param_ids
]
optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.embed_learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)
optimizer2_groups = []
for p in block_parameters:
    optimizer2_groups.append(dict(params=[p], weight_decay=args.weight_decay))
for p in rmsnorm_gamma_parameters:
    optimizer2_groups.append(dict(params=[p], lr=args.gamma_learning_rate, weight_decay=0.0))
optimizer2 = torch.optim.AdamW(optimizer2_groups, lr=0.5 * args.embed_learning_rate / width_multiplier, betas=(0.9, 0.95),
                               fused=True)
optimizers = [optimizer1, optimizer2]

# learning rate decay scheduler (linear warmup, constant plateau, linear warmdown)
def get_lr(it):
    return (it+1) / args.warmup_iters if it < args.warmup_iters else 1.0

schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

@torch.no_grad()
def apply_tensor_rms_elr(optimizers, update_step, history_path=None):
    """Matrices use JSON ELR; gamma uses absolute LR, independent of its RMS."""
    records = []
    for opt in optimizers:
        for group in opt.param_groups:
            if len(group['params']) != 1:
                raise RuntimeError('per-tensor RMS-ELR requires one tensor per optimizer group')
            p = group['params'][0]
            rms = p.detach().float().square().mean().sqrt().item()
            if not math.isfinite(rms) or rms <= 0:
                raise RuntimeError('per-tensor RMS-ELR requires finite positive RMS')
            is_gamma = id(p) in rmsnorm_gamma_param_ids
            target = None if is_gamma else tensor_elr_schedule[update_step][tensor_name_by_id[id(p)]]
            # Use the global update index so resume starts at the correct LR.
            group['lr'] = (args.gamma_learning_rate * get_lr(update_step - 1)
                           if is_gamma else target * rms)
            if history_path is not None:
                records.append(dict(step=update_step, name=tensor_name_by_id[id(p)],
                                    target_rms_elr=target, pre_update_rms=rms,
                                    lr_policy='absolute_gamma_lr' if is_gamma else 'json_rms_elr',
                                    lr=group['lr'], actual_rms_elr=group['lr']/rms))
    if history_path is not None:
        with open(history_path, 'a') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')

tensor_name_by_id = {id(p): name for name, p in raw_model.named_parameters() if p.requires_grad}

# begin logging
if master_process:
    run_id = f'w768_muonhinit_fixed_initial_rms_adamw_jsonelr_{schedule_sha256[:12]}_wd0_s{args.seed}_' + str(uuid.uuid4())
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

def tensor_norm_fields(tensor, prefix=''):
    x = tensor.detach().float()
    return {f'{prefix}fro_norm': x.square().sum().sqrt().item(),
            f'{prefix}rms_norm': x.square().mean().sqrt().item()}


def tensor_norm_record(step, name, tensor):
    return dict(step=step, name=name, shape=list(tensor.shape), ndim=tensor.ndim,
                **tensor_norm_fields(tensor))


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
        if not torch.equal(tensor.detach(), tensor_before):
            raise RuntimeError(f'unexpected AdamW parameter change at zero LR: {name}')
        return None  # The normalized direction cannot be inferred from zero displacement.
    weight_decay = param_hparams['weight_decay']
    before = tensor_before.float()
    after = tensor.detach().float()
    delta = after - before
    return -(delta + lr * weight_decay * before) / lr

def adamw_update_norm_record(step, name, tensor, adamw_update, param_hparams):
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
    if adamw_update is None:
        record.update({key: None for key in tensor_norm_fields(torch.zeros_like(tensor), prefix='adamw_update_')})
        record.update(adamw_update_inferred=False, raw_parameter_delta_norm=0.0)
        return record
    record.update(tensor_norm_fields(
        adamw_update,
        prefix='adamw_update_',
    ))
    return record

def maybe_log_adamw_update_norms(update_step, update_state):
    if update_state is None:
        return
    snapshots = update_state['snapshots']
    with torch.no_grad(), open(os.path.join(logdir, 'adamw_update_norm_history.jsonl'), 'a') as f:
        for name, tensor in raw_model.named_parameters():
            if not tensor.requires_grad:
                continue
            param_id = id(tensor)
            tensor_before = snapshots.pop(param_id)
            hparams = update_state['hparams'][param_id]
            update = adamw_update_tensor(tensor, tensor_before, hparams, update_step, name)
            f.write(json.dumps(adamw_update_norm_record(update_step, name, tensor, update, hparams)) + '\n')
    if snapshots:
        raise RuntimeError(f'unused AdamW update snapshots: {len(snapshots)}')


def maybe_log_tensor_norms(step):
    if not master_process or args.tensor_norm_every <= 0:
        return
    if step % args.tensor_norm_every != 0 and step != args.num_iterations:
        return
    with torch.no_grad(), open(os.path.join(logdir, 'tensor_norm_history.jsonl'), 'a') as f:
        for name, tensor in raw_model.named_parameters():
            f.write(json.dumps(tensor_norm_record(step, name, tensor)) + '\n')


def build_fixed_norm_state(model):
    """Capture each matrix's actual initial RMS after DDP parameter broadcast.

    Matches fixed_initial_norm_all_matrices_start0: block matrices plus tied
    embedding; RMSNorm gamma remains trainable and is not projected.
    """
    entries = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and (name.startswith('transformer.h.') or name == 'transformer.wte.weight'):
            target = p.detach().float().square().mean().sqrt()
            if not torch.isfinite(target) or target <= 0:
                raise RuntimeError(f'invalid initial RMS for {name}')
            entries.append(dict(name=name, param=p, target_rms=target.clone()))
    expected = 6 * model.config.n_layer + 1
    if len(entries) != expected:
        raise RuntimeError(f'expected {expected} fixed-norm matrices, got {len(entries)}')
    return entries


@torch.no_grad()
def apply_fixed_norm_control(entries, update_step, history_path=None):
    # Same hard-RMS projection as the existing constant-norm runner.
    records = []
    for entry in entries:
        p, target = entry['param'], entry['target_rms']
        before = p.float().square().mean().sqrt()
        if not torch.isfinite(before) or before <= 0:
            raise RuntimeError(f'invalid current RMS for {entry["name"]}')
        p.mul_((target / before).to(dtype=p.dtype))
        if history_path is not None:
            after = p.float().square().mean().sqrt()
            records.append(dict(step=update_step, name=entry['name'],
                                target_rms=target.item(), rms_before=before.item(),
                                rms_after=after.item(),
                                relative_error=((after-target).abs()/target).item()))
    if history_path is not None:
        with open(history_path, 'a') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')


fixed_norm_state = build_fixed_norm_state(raw_model)
fixed_norm_history_path = os.path.join(logdir, 'norm_control_history.jsonl') if master_process else None
write_tensor_metadata()

def checkpoint_tree(value, target_device='cpu'):
    if isinstance(value, torch.Tensor):
        return value.detach().to(target_device).clone()
    if isinstance(value, dict):
        return {k: checkpoint_tree(v, target_device) for k, v in value.items()}
    if isinstance(value, list):
        return [checkpoint_tree(v, target_device) for v in value]
    if isinstance(value, tuple):
        return tuple(checkpoint_tree(v, target_device) for v in value)
    return value

def loader_state(loader):
    return dict(files=[str(Path(f).resolve()) for f in loader.files],
                sizes=[os.path.getsize(f) for f in loader.files],
                modified_ns=[os.stat(f).st_mtime_ns for f in loader.files],
                shard=loader.current_shard, position=loader.current_position,
                B=loader.B, T=loader.T, rank=loader.process_rank, world=loader.num_processes)

def restore_loader(loader, state):
    current = loader_state(loader)
    for key in ('files', 'sizes', 'modified_ns', 'B', 'T', 'rank', 'world'):
        if current[key] != state[key]:
            raise RuntimeError(f'checkpoint data loader mismatch: {key}')
    loader.current_shard = state['shard']
    loader.current_position = state['position']
    loader.tokens = _load_data_shard(loader.files[loader.current_shard])

def capture_training_state(step, x, y):
    if any(p.grad is not None for p in raw_model.parameters()):
        raise RuntimeError('checkpoint must be between complete optimizer updates')
    return checkpoint_tree(dict(
        version=1, step=step, branch=cli.branch, fork_step=cli.fork_step, schedule_sha256=schedule_sha256,
        config=vars(args), model_config=vars(raw_model.config),
        parameter_names=[n for n, _ in raw_model.named_parameters()],
        model=raw_model.state_dict(), optimizers=[o.state_dict() for o in optimizers],
        schedulers=[s.state_dict() for s in schedulers],
        targets={e['name']: e['target_rms'] for e in fixed_norm_state},
        train_loader=loader_state(train_loader), val_loader=loader_state(val_loader),
        next_batch=(x, y),
        rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()),
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        source_sha256=hashlib.sha256(code.encode()).hexdigest()))

def restore_training_state(state):
    if state['config'].get('gamma_learning_rate') != args.gamma_learning_rate:
        raise RuntimeError('checkpoint uses the old gamma ELR policy; regenerate the shared warmup with absolute gamma LR')
    previous_source = state['source_sha256'] == PREVIOUS_RUNNER_SOURCE_SHA256
    saved_config = state['config']
    if previous_source:
        saved_config = {k: v for k, v in saved_config.items() if k not in RETIRED_CONFIG_FIELDS}
    if state['version'] != 1 or saved_config != vars(args) or state['model_config'] != vars(raw_model.config):
        raise RuntimeError('checkpoint configuration mismatch')
    if not previous_source and state['source_sha256'] != hashlib.sha256(code.encode()).hexdigest():
        raise RuntimeError('unsupported checkpoint source')
    if state['torch_version'] != str(torch.__version__) or state['cuda_version'] != torch.version.cuda:
        raise RuntimeError('checkpoint PyTorch/CUDA version mismatch')
    if state['step'] != cli.fork_step:
        raise RuntimeError('checkpoint and manifest step mismatch')
    if state['parameter_names'] != [n for n, _ in raw_model.named_parameters()]:
        raise RuntimeError('checkpoint optimizer parameter order mismatch')
    raw_model.load_state_dict(state['model'], strict=True)
    for opt, saved in zip(optimizers, state['optimizers'], strict=True):
        opt.load_state_dict(saved)
    for sched, saved in zip(schedulers, state['schedulers'], strict=True):
        sched.load_state_dict(saved)
    if set(state['targets']) != {e['name'] for e in fixed_norm_state}:
        raise RuntimeError('checkpoint norm targets mismatch')
    for entry in fixed_norm_state:
        entry['target_rms'] = state['targets'][entry['name']].to(device)
    restore_loader(train_loader, state['train_loader'])
    restore_loader(val_loader, state['val_loader'])
    # Restore RNG after model, optimizers and loaders.
    random.setstate(state['rng']['python'])
    np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch'])
    torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state['step'], tuple(t.to(device) for t in state['next_batch'])

def save_fork_checkpoint(step, x, y):
    destination = cli.checkpoint_dir
    if master_process:
        destination.mkdir(parents=True, exist_ok=False)
    if use_ddp:
        dist.barrier()
    rank_file = destination / f'rank{ddp_rank:05d}.pt'
    temporary = rank_file.with_suffix('.tmp')
    torch.save(capture_training_state(step, x, y), temporary)
    os.replace(temporary, rank_file)
    if use_ddp:
        dist.barrier()
    if master_process:
        manifest = dict(version=1, step=step, world_size=ddp_world_size,
                        files=[f'rank{r:05d}.pt' for r in range(ddp_world_size)])
        (destination / 'complete.json').write_text(json.dumps(manifest, indent=2))
    if use_ddp:
        dist.barrier()

start_step = 0
if cli.resume:
    manifest = json.loads((cli.resume / 'complete.json').read_text())
    if manifest.get('version') != 1 or manifest.get('files') != [f'rank{r:05d}.pt' for r in range(ddp_world_size)]:
        raise RuntimeError('invalid checkpoint manifest')
    if manifest['world_size'] != ddp_world_size:
        raise RuntimeError('resume requires the same number of ranks')
    payload = torch.load(cli.resume / f'rank{ddp_rank:05d}.pt', map_location='cpu', weights_only=False)
    start_step, (x, y) = restore_training_state(payload)
    del payload
    if master_process:
        (Path(logdir) / 'resume_metadata.json').write_text(json.dumps(dict(
            checkpoint=str(cli.resume.resolve()), next_update=start_step+1,
            branch=cli.branch, schedule_sha256=schedule_sha256), indent=2))

if master_process:
    (Path(logdir) / 'schedule.json').write_text(schedule_text, encoding='utf-8')
    (Path(logdir) / 'schedule_metadata.json').write_text(json.dumps(dict(
        schedule_sha256=schedule_sha256, completed_step=start_step,
        next_update=start_step+1, total_updates=args.num_iterations,
        scope='all trainable tensors; LR = JSON target * pre-update RMS'), indent=2))

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
# The initial next_batch already advanced the cursor. Do not reset it here.
for step in range(start_step, args.num_iterations + 1):
    if cli.checkpoint_dir and step == cli.save_at:
        save_fork_checkpoint(step, x, y)
        if cli.stop_after_save:
            break
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if args.val_loss_every > 0 and (last_step or step % args.val_loss_every == 0):
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

    maybe_log_tensor_norms(step)

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
    apply_tensor_rms_elr(optimizers, update_step,
                         os.path.join(logdir, 'tensor_rms_elr_history.jsonl') if master_process else None)
    adamw_update_state = maybe_capture_adamw_update_state(update_step)
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    maybe_log_adamw_update_norms(update_step, adamw_update_state)
    # Log raw AdamW update norms before fixed-RMS projection.
    apply_fixed_norm_control(fixed_norm_state, update_step, fixed_norm_history_path)
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
