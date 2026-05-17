import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import argparse
import json
import socket
import uuid
import glob
import time
from dataclasses import asdict, dataclass, fields

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

    def state_dict(self):
        return dict(
            current_shard=self.current_shard,
            current_position=self.current_position,
            files=self.files,
            B=self.B,
            T=self.T,
            process_rank=self.process_rank,
            num_processes=self.num_processes,
        )

    def load_state_dict(self, state):
        assert state["files"] == self.files, "data shard list changed between checkpoint and resume"
        assert state["B"] == self.B and state["T"] == self.T, "batch shape changed between checkpoint and resume"
        assert state["process_rank"] == self.process_rank, "DDP rank changed between checkpoint and resume"
        assert state["num_processes"] == self.num_processes, "DDP world size changed between checkpoint and resume"
        self.current_shard = int(state["current_shard"])
        self.current_position = int(state["current_position"])
        self.tokens = _load_data_shard(self.files[self.current_shard])

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    device_batch_size : int = 64 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 5100 # number of iterations to run
    embed_learning_rate : float = 0.0036
    muon_learning_rate : float = 0.02
    warmup_iters : int = 250
    warmdown_iters : int = 1450 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    # run/checkpoint hyperparams
    run_name : str = ''
    out_dir : str = 'logs'
    resume_from : str = ''
    resume_mode : str = 'none' # none | exact | fork
    parent_run_id : str = ''


def parse_args():
    defaults = Hyperparameters()
    parser = argparse.ArgumentParser()
    for field in fields(Hyperparameters):
        default = getattr(defaults, field.name)
        parser.add_argument(f"--{field.name}", type=type(default), default=argparse.SUPPRESS)
    cli_overrides = vars(parser.parse_args())
    values = asdict(defaults)
    values.update(cli_overrides)
    args = Hyperparameters(**values)
    assert args.resume_mode in ("none", "exact", "fork"), "resume_mode must be none, exact, or fork"
    assert args.resume_mode == "none" or args.resume_from, "resume_from is required when resume_mode is exact or fork"
    return args, cli_overrides


def load_resume_checkpoint(args):
    if args.resume_mode == "none":
        return None
    return torch.load(args.resume_from, map_location="cpu", weights_only=False)


def merge_resume_config(args, cli_overrides, checkpoint):
    if checkpoint is None:
        return args
    valid_keys = {field.name for field in fields(Hyperparameters)}
    values = asdict(Hyperparameters())
    values.update({k: v for k, v in checkpoint.get("config", {}).items() if k in valid_keys})
    values.update(cli_overrides)
    return Hyperparameters(**values)


def get_rng_state():
    return dict(
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        numpy=np.random.get_state(),
    )


def set_rng_state(state):
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])


def ddp_broadcast_object(obj):
    objects = [obj]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]


def init_run(args, checkpoint, master_process):
    parent_metadata = checkpoint.get("run_metadata", {}) if checkpoint is not None else {}
    parent_run_id = args.parent_run_id
    if not parent_run_id and args.resume_mode == "fork":
        parent_run_id = parent_metadata.get("run_id", "")

    if args.resume_mode == "exact":
        run_id = parent_metadata.get("run_id") or args.run_name or os.path.basename(os.path.dirname(args.resume_from))
        logdir = parent_metadata.get("logdir") or os.path.dirname(os.path.abspath(args.resume_from))
        logfile = parent_metadata.get("logfile") or os.path.join(logdir, "train.log")
        os.makedirs(logdir, exist_ok=True)
        log_mode = "a"
    else:
        run_id = args.run_name or str(uuid.uuid4())
        logdir = os.path.join(args.out_dir, run_id)
        if os.path.exists(logdir):
            raise FileExistsError(f"run directory already exists: {logdir}")
        os.makedirs(logdir, exist_ok=False)
        logfile = os.path.join(logdir, "train.log")
        log_mode = "w"

    run_metadata = dict(
        run_id=run_id,
        run_name=args.run_name or run_id,
        parent_run_id=parent_run_id,
        resume_mode=args.resume_mode,
        resume_from=os.path.abspath(args.resume_from) if args.resume_from else "",
        logdir=os.path.abspath(logdir),
        logfile=os.path.abspath(logfile),
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        hostname=socket.gethostname(),
    )

    if master_process:
        with open(os.path.join(logdir, "config.json"), "w") as f:
            json.dump(asdict(args), f, indent=2, sort_keys=True)
        with open(os.path.join(logdir, "metadata.json"), "w") as f:
            json.dump(run_metadata, f, indent=2, sort_keys=True)
        with open(os.path.join(logdir, "code_snapshot.py"), "w") as f:
            f.write(code)
        with open(logfile, log_mode) as f:
            f.write("="*100 + "\n")
            f.write(code)
            f.write("="*100 + "\n")
            f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\n")
            if torch.cuda.is_available():
                import subprocess
                result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                f.write(f"nvidia-smi:\n{result.stdout}\n")
            f.write(f"run_metadata: {json.dumps(run_metadata, sort_keys=True)}\n")
            f.write("="*100 + "\n")
    return run_metadata


args, cli_overrides = parse_args()
checkpoint = load_resume_checkpoint(args)
args = merge_resume_config(args, cli_overrides, checkpoint)

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

if master_process:
    run_metadata = init_run(args, checkpoint, master_process)
else:
    run_metadata = None
run_metadata = ddp_broadcast_object(run_metadata)
run_id = run_metadata["run_id"]
logdir = run_metadata["logdir"]
logfile = run_metadata["logfile"]

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

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer(s)
optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.embed_learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)
optimizer2 = torch.optim.AdamW(raw_model.transformer.h.parameters(), lr=0.5*args.embed_learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)
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

global_step = 0
if checkpoint is not None:
    required_keys = ("model", "optimizers", "schedulers", "train_loader", "val_loader", "rng_state")
    missing_keys = [key for key in required_keys if key not in checkpoint]
    assert not missing_keys, f"checkpoint is missing required resume keys: {missing_keys}"
    raw_model.load_state_dict(checkpoint["model"])
    for opt, opt_state in zip(optimizers, checkpoint["optimizers"]):
        opt.load_state_dict(opt_state)
    for sched, sched_state in zip(schedulers, checkpoint["schedulers"]):
        sched.load_state_dict(sched_state)
    if "rank_states" in checkpoint:
        rank_states = checkpoint["rank_states"]
        assert len(rank_states) == ddp_world_size, "DDP world size changed between checkpoint and resume"
        rank_state = rank_states[ddp_rank]
        train_loader.load_state_dict(rank_state["train_loader"])
        val_loader.load_state_dict(rank_state["val_loader"])
        set_rng_state(rank_state["rng_state"])
    else:
        train_loader.load_state_dict(checkpoint["train_loader"])
        val_loader.load_state_dict(checkpoint["val_loader"])
        set_rng_state(checkpoint["rng_state"])
    global_step = int(checkpoint.get("global_step", checkpoint.get("step", 0)))
    if master_process:
        print(f"resumed from {args.resume_from} at global_step {global_step} with mode {args.resume_mode}")


def save_checkpoint(global_step):
    rank_state = dict(
        train_loader=train_loader.state_dict(),
        val_loader=val_loader.state_dict(),
        rng_state=get_rng_state(),
    )
    gathered_rank_states = [None for _ in range(ddp_world_size)] if master_process else None
    dist.gather_object(rank_state, gathered_rank_states, dst=0)

    if not master_process:
        return
    checkpoint_data = dict(
        global_step=global_step,
        step=global_step,
        code=code,
        config=asdict(args),
        run_metadata=run_metadata,
        model=raw_model.state_dict(),
        optimizers=[opt.state_dict() for opt in optimizers],
        schedulers=[sched.state_dict() for sched in schedulers],
        train_loader=rank_state["train_loader"],
        val_loader=rank_state["val_loader"],
        rng_state=rank_state["rng_state"],
        rank_states=gathered_rank_states,
    )
    torch.save(checkpoint_data, os.path.join(logdir, f"state_step{global_step:06d}.pt"))


def maybe_eval_and_checkpoint(global_step, training_time_ms, t0):
    last_step = (global_step == args.num_iterations)
    timed_steps = float("nan") if global_step <= 11 else (global_step - 10) + 1
    do_val = last_step or (args.val_loss_every > 0 and global_step % args.val_loss_every == 0)
    do_save = last_step or (args.save_every > 0 and global_step % args.save_every == 0)

    if do_val:
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        model.eval()
        val_loader.reset()
        val_loss = torch.zeros((), device=device)
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx:
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        if master_process:
            denom = timed_steps - 1
            step_avg = training_time_ms / denom if denom > 0 else float("nan")
            line = f"step:{global_step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{step_avg:.2f}ms"
            print(line)
            with open(logfile, "a") as f:
                f.write(line + "\n")
        torch.cuda.synchronize()
        t0 = time.time()

    if do_save:
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        save_checkpoint(global_step)
        torch.cuda.synchronize()
        t0 = time.time()

    return training_time_ms, t0


training_time_ms = 0
torch.cuda.synchronize()
t0 = time.time()

if checkpoint is None:
    training_time_ms, t0 = maybe_eval_and_checkpoint(global_step, training_time_ms, t0)

while global_step < args.num_iterations:
    if global_step == 10:
        training_time_ms = 0
        t0 = time.time()

    model.train()
    for i in range(1, train_accumulation_steps+1):
        x, y = train_loader.next_batch()
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad /= train_accumulation_steps
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    model.zero_grad(set_to_none=True)
    global_step += 1

    if master_process:
        timed_steps = float("nan") if global_step <= 11 else (global_step - 10) + 1
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        step_avg = approx_time / timed_steps if timed_steps > 0 else float("nan")
        line = f"step:{global_step}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{step_avg:.2f}ms"
        print(line)
        with open(logfile, "a") as f:
            f.write(line + "\n")

    training_time_ms, t0 = maybe_eval_and_checkpoint(global_step, training_time_ms, t0)

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
