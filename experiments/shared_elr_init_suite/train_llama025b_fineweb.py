"""Standalone PyTorch LLaMA / AdamW / distributed pseudo-label Fisher.

Paper: arxiv.org/abs/2502.19002, Figure 3(b). Model and schedule follow
wang-jinbo/BlockwiseLearningRate/{modelling_llama_new,train_llama_bw}.py.
Data: existing GPT-2-tokenized FineWeb shards, read sequentially in shard order.
Default: 50k-step cosine, CUT at 20k completed updates (9.8304B tokens).
No Transformers, norm control, ELR scaling, weight tying or blockwise LR.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import argparse
import contextlib
from dataclasses import dataclass, asdict
import glob
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class LlamaConfig:
    vocab_size: int = 50304
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    initializer_range: float = 0.02


class RMSNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c.hidden_size))
        self.eps = c.rms_norm_eps

    def forward(self, x):
        # HF LLaMA: variance in FP32, normalized activation cast back before gain.
        y = x.float()
        return self.weight * (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype)


def rotary(q, k, theta):
    dim, length = q.shape[-1], q.shape[-2]
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim))
    with torch.autocast(q.device.type, enabled=False):
        phase = torch.outer(torch.arange(length, device=q.device, dtype=torch.float32), inv)
        phase = torch.cat((phase, phase), -1)[None, None]
        cos, sin = phase.cos().to(q.dtype), phase.sin().to(q.dtype)
    def apply(x):
        a, b = x.chunk(2, -1)
        return x * cos + torch.cat((-b, a), -1) * sin
    return apply(q), apply(k)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.num_attention_heads
        self.theta = c.rope_theta
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self, name, nn.Linear(c.hidden_size, c.hidden_size, bias=False))

    def forward(self, x):
        b, t, width = x.shape
        q, k, v = [proj(x).view(b, t, self.heads, width // self.heads).transpose(1, 2)
                   for proj in (self.q_proj, self.k_proj, self.v_proj)]
        q, k = rotary(q, k, self.theta)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        return self.o_proj(y.transpose(1, 2).contiguous().view(b, t, width))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = Attention(c)
        self.mlp = MLP(c)
        self.input_layernorm = RMSNorm(c)
        self.post_attention_layernorm = RMSNorm(c)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Backbone(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([Layer(c) for _ in range(c.num_hidden_layers)])
        self.norm = RMSNorm(c)

    def forward(self, x):
        h = self.embed_tokens(x)
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


class Llama(nn.Module):
    supports_fisher_hidden = True

    def __init__(self, c):
        super().__init__()
        if c.hidden_size % c.num_attention_heads or (c.hidden_size // c.num_attention_heads) % 2:
            raise ValueError('Head dimension must be an even integer')
        self.config = c
        self.model = Backbone(c)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, x, targets=None, return_hidden=False):
        hidden = self.model(x)
        if return_hidden:
            return hidden
        logits = self.lm_head(hidden)
        # Explicit shifted dataset targets; never shift labels a second time.
        loss = None if targets is None else F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss


def block_type(name):
    if name == 'model.embed_tokens.weight': return 'Emb'
    if name == 'lm_head.weight': return 'Head'
    if 'norm' in name: return 'Norm'
    if '.q_proj.' in name or '.k_proj.' in name: return 'QK'
    if '.v_proj.' in name or '.o_proj.' in name: return 'VO'
    if '.mlp.' in name: return 'FFN'
    raise ValueError(f'Unclassified parameter: {name}')


def learning_rate(iteration, peak=8e-4, warmup=1000, horizon=50000):
    # Exact official get_lr(alpha=1): first update uses iteration=0, LR=0.
    if iteration <= warmup:
        return peak * iteration / warmup
    ratio = min(1.0, (iteration - warmup) / (horizon - warmup))
    return peak / 20 + 0.5 * (1 + math.cos(math.pi * ratio)) * (peak - peak / 20)


class TokenData:
    """Sequential rank-strided reader matching the previous training loader.

    Each rank reads a disjoint slice of the current global microbatch.
    All ranks switch shards together, dropping the short tail; wrap only
    after the last shard. Fisher and validation have independent cursors.
    """
    def __init__(self, pattern, length, vocab_size, rank=0, world=1):
        self.rank, self.world = rank, world
        if not 0 <= rank < world: raise ValueError('Invalid data rank/world')
        self.files = sorted(glob.glob(pattern))
        if not self.files: raise FileNotFoundError(pattern)
        self.maps, counts = [], []
        self.manifest = []
        for path in self.files:
            with open(path, 'rb') as f:
                h = np.frombuffer(f.read(1024), dtype=np.int32)
            if len(h) != 256 or tuple(h[:2]) != (20240520, 1):
                raise ValueError(f'Expected v1 GPT-2 uint16 shard: {path}')
            count = int(h[2])
            if count < 1 or os.path.getsize(path) != 1024 + 2 * count:
                raise ValueError(f'Bad shard length: {path}')
            self.maps.append(np.memmap(path, dtype=np.uint16, mode='r', offset=1024, shape=(count,)))
            counts.append(count)
            self.manifest.append(dict(path=str(Path(path).resolve()), tokens=count, bytes=os.path.getsize(path)))
        self.counts = counts
        self.total = sum(counts)
        self.length, self.vocab_size = length, vocab_size
        self.reset()

    def reset(self):
        self.shard, self.position, self.epoch = 0, 0, 0
        self.batch_size = None

    def state_dict(self):
        return dict(shard=self.shard, position=self.position, epoch=self.epoch,
                    batch_size=self.batch_size, rank=self.rank, world=self.world)

    def load_state_dict(self, state):
        if state['rank'] != self.rank or state['world'] != self.world:
            raise ValueError('Data cursor rank/world mismatch')
        if not 0 <= state['shard'] < len(self.maps) or state['position'] < 0:
            raise ValueError('Invalid data cursor')
        self.shard, self.position, self.epoch = state['shard'], state['position'], state['epoch']
        self.batch_size = state['batch_size']

    def batch(self, batch_size):
        if self.batch_size is not None and batch_size != self.batch_size:
            raise ValueError('Sequential loader requires a fixed microbatch size')
        self.batch_size = batch_size
        local_tokens = batch_size * self.length
        stride = local_tokens * self.world
        if min(self.counts) < stride + 1:
            raise ValueError('Every shard must hold at least one global microbatch plus one target token')
        start = self.position + self.rank * local_tokens
        tokens = torch.from_numpy(np.array(self.maps[self.shard][start:start+local_tokens+1], dtype=np.int64))
        if tokens.max().item() >= self.vocab_size:
            raise ValueError('Token ID outside vocabulary; use GPT-2-tokenized shards, do not remap IDs')
        self.position += stride
        if self.position + stride + 1 > self.counts[self.shard]:
            self.position = 0
            self.shard += 1
            if self.shard == len(self.maps):
                self.shard = 0
                self.epoch += 1
        return tokens[:-1].view(batch_size, self.length), tokens[1:].view(batch_size, self.length)


def cpu_tree(obj):
    if isinstance(obj, torch.Tensor): return obj.detach().cpu().clone()
    if isinstance(obj, dict): return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return type(obj)(cpu_tree(v) for v in obj)
    return obj


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_json(path, obj):
    with open(path, 'a', encoding='utf8') as f:
        f.write(json.dumps(obj, allow_nan=False) + '\n')


@torch.no_grad()
def tensor_gradient_norms(model):
    """Read post-DDP, accumulated gradients BEFORE global clipping; no mutation.

    Stack scalars before the CPU transfer to avoid one device sync per tensor.
    """
    params = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
    norms = torch.stack([torch.linalg.vector_norm(p.grad.detach()) for _, p in params]).double().cpu().tolist() if params else []
    total_squared = sum(value * value for value in norms)
    return {name: dict(group=block_type(name), shape=list(p.shape), numel=p.numel(),
                       grad_norm=value, grad_rms=value / math.sqrt(p.numel()),
                       squared_norm_fraction=value * value / total_squared if total_squared else 0.0)
            for (name, p), value in zip(params, norms)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-bin', default='data/fineweb10B/fineweb_train_*.bin')
    p.add_argument('--input-val-bin', default='data/fineweb10B/fineweb_val_*.bin')
    p.add_argument('--output', type=Path, default=Path('logs/llama025b_fineweb'))
    p.add_argument('--stop-step', type=int, default=20000)
    p.add_argument('--schedule-steps', type=int, default=50000)
    p.add_argument('--warmup-steps', type=int, default=1000)
    p.add_argument('--learning-rate', type=float, default=8e-4)
    p.add_argument('--batch-size', type=int, default=480)
    p.add_argument('--micro-batch-size', type=int, default=8)
    p.add_argument('--sequence-length', type=int, default=1024)
    p.add_argument('--expected-world-size', type=int, default=2)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--seed', type=int, default=41)
    p.add_argument('--resume', type=Path)
    p.add_argument('--save-every', type=int, default=0)  # full resume state at cutoff; model snapshots at every probe
    p.add_argument('--val-every', type=int, default=1000)
    p.add_argument('--val-batches', type=int, default=20)
    p.add_argument('--fisher-every', type=int, default=1000)
    p.add_argument('--fisher-sequences', type=int, default=1024)
    p.add_argument('--fisher-micro-batch-size', type=int, default=64)
    p.add_argument('--fisher-token-chunk-size', type=int, default=1024)
    p.add_argument('--fisher-draws', type=int, default=2)
    p.add_argument('--threads', type=int, default=2)
    # Only for local small-parameter tests; production defaults are paper 0.25B.
    p.add_argument('--hidden-size', type=int, default=1024)
    p.add_argument('--intermediate-size', type=int, default=4096)
    p.add_argument('--num-layers', type=int, default=8)
    p.add_argument('--num-heads', type=int, default=16)
    p.add_argument('--vocab-size', type=int, default=50304)
    a = p.parse_args()
    world, rank = int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', 0))
    if world != a.expected_world_size: p.error(f'Expected {a.expected_world_size} ranks, got {world}')
    if not 0 <= a.stop_step <= a.schedule_steps or not 0 < a.warmup_steps < a.schedule_steps:
        p.error('Require 0<=stop<=schedule and 0<warmup<schedule')
    if min(a.batch_size, a.micro_batch_size, a.sequence_length, a.fisher_sequences,
           a.fisher_micro_batch_size, a.fisher_token_chunk_size, a.fisher_draws, a.val_batches) < 1:
        p.error('Batch sizes, lengths and draws must be positive')
    if min(a.fisher_every, a.save_every, a.val_every) < 0: p.error('Intervals cannot be negative')
    if a.batch_size % (world * a.micro_batch_size) or a.fisher_sequences % world:
        p.error('Global training batch must divide world*micro; Fisher sequences must divide world')
    device = torch.device(f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}' if a.device == 'cuda' else 'cpu')
    if device.type == 'cuda': torch.cuda.set_device(device)
    if world > 1: dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    c = LlamaConfig(a.vocab_size, a.hidden_size, a.intermediate_size, a.num_layers, a.num_heads)
    raw = Llama(c).to(device)
    model = DDP(raw, device_ids=[device.index] if device.type == 'cuda' else None) if world > 1 else raw
    groups = [{'params': [v for n, v in raw.named_parameters() if block_type(n) == name], 'name': name}
              for name in ('Emb', 'Head', 'Norm', 'QK', 'VO', 'FFN')]
    optimizer = torch.optim.AdamW(groups, lr=a.learning_rate, betas=(0.9, 0.95), eps=1e-8,
                                 weight_decay=0.1, fused=device.type == 'cuda')
    train = TokenData(a.input_bin, a.sequence_length, a.vocab_size, rank, world)
    validation = TokenData(a.input_val_bin, a.sequence_length, a.vocab_size, rank, world) if a.val_every else None
    # Fisher uses an independent sequential cursor, never the training cursor.
    if a.fisher_every:
        data = TokenData(a.input_bin, a.sequence_length, a.vocab_size)
        fx, fy = data.batch(a.fisher_sequences)
        input_hash = hashlib.sha256(fx.numpy().tobytes() + fy.numpy().tobytes()).hexdigest()
        count = a.fisher_sequences // world
        fx, fy = fx[rank*count:(rank+1)*count], fy[rank*count:(rank+1)*count]
    else: input_hash = None
    initial_rms = parameter_rms(raw)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = dict(vars(a), output=str(a.output), resume=None, model=asdict(c), world_size=world,
                  torch_version=str(torch.__version__), source_sha256=source_hash,
                  training_manifest=train.manifest, validation_manifest=None if validation is None else validation.manifest,
                  fisher_input_sha256=input_hash, planned_training_tokens=a.stop_step*a.batch_size*a.sequence_length,
                  schedule_horizon_tokens=a.schedule_steps*a.batch_size*a.sequence_length,
                  sampling='sequential shards; rank-strided microbatches; short shard tails dropped', norm_control=False)
    # Resuming may extend stop-step but must retain all dynamics and sampling settings.
    compatibility = {k: v for k, v in config.items() if k not in ('output', 'resume', 'stop_step', 'planned_training_tokens')}
    step = 0
    if a.resume:
        state = torch.load(a.resume / f'rank{rank}.pt', map_location='cpu', weights_only=False)
        manifest = json.loads((a.resume / 'complete.json').read_text())
        if manifest['world_size'] != world or manifest['step'] != state['step']:
            raise ValueError('Incomplete or mismatched checkpoint')
        if state['compatibility'] != compatibility: raise ValueError('Resume configuration/source/data mismatch')
        raw.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        initial_rms = state['initial_rms']
        train.load_state_dict(state['data_state'])
        torch.set_rng_state(state['torch_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        if device.type == 'cuda': torch.cuda.set_rng_state(state['cuda_rng'], device)
        step = state['step']
        if step > a.stop_step: raise ValueError('Checkpoint is beyond stop-step')
    if rank == 0:
        a.output.mkdir(parents=True, exist_ok=False)
        (a.output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
        (a.output/'fisher_initial_rms.json').write_text(json.dumps(initial_rms, indent=2), encoding='utf8')
        print(json.dumps(dict(parameters=sum(v.numel() for v in raw.parameters()),
                              accumulation=a.batch_size//(world*a.micro_batch_size), **config)), flush=True)
    if world > 1: dist.barrier()
    def checkpoint(completed):
        dest = a.output/'checkpoints'/f'step{completed:06d}'
        state = dict(format='llama_full_resume_v1', step=completed, compatibility=compatibility,
                     model=cpu_tree(raw.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                     initial_rms=initial_rms, data_state=train.state_dict(),
                     torch_rng=torch.get_rng_state(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                     cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
        atomic_save(state, dest/f'rank{rank}.pt')
        if world > 1: dist.barrier()
        if rank == 0:
            tmp = dest/'complete.json.tmp'
            tmp.write_text(json.dumps(dict(step=completed, world_size=world)), encoding='utf8')
            os.replace(tmp, dest/'complete.json')
        if world > 1: dist.barrier()
    accumulation = a.batch_size // (world*a.micro_batch_size)
    while True:
        if a.fisher_every and (step in (0, 100, a.stop_step) or step % a.fisher_every == 0):
            if rank == 0: print(f'Fisher start step={step}', flush=True)
            record = probe(raw, fx, fy, a.fisher_draws, 1729, initial_rms,
                           a.fisher_micro_batch_size, world > 1, a.fisher_token_chunk_size)
            if rank == 0:
                relative = Path('fisher_models')/f'step{step:06d}.pt'
                atomic_save(dict(format='llama_model_only_v1', step=step, model=cpu_tree(raw.state_dict()),
                                 config=config, initial_rms=initial_rms), a.output/relative)
                record.update(step=step, input_sha256=input_hash, model_checkpoint=relative.as_posix())
                append_json(a.output/'fisher_history.jsonl', record)
                print('Fisher', step, {k: v['mean_fisher'] for k,v in record['groups'].items()}, flush=True)
        if validation and (step % a.val_every == 0 or step == a.stop_step):
            raw.eval()
            validation.reset()
            value = torch.zeros((), device=device)
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                for _ in range(a.val_batches):
                    x, y = validation.batch(a.micro_batch_size)
                    logits, loss = raw(x.to(device), y.to(device))
                    value += loss / a.val_batches
                    del logits, loss, x, y
            if world > 1: dist.all_reduce(value)
            if rank == 0: append_json(a.output/'validation.jsonl', dict(step=step, loss=value.item()/world))
        if step == a.stop_step or (a.save_every and step > 0 and step % a.save_every == 0):
            checkpoint(step)
        if step == a.stop_step: break
        raw.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        started = time.perf_counter()
        for micro in range(accumulation):
            x, y = train.batch(a.micro_batch_size)
            sync = model.no_sync() if world > 1 and micro+1 < accumulation else contextlib.nullcontext()
            with sync:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    logits, loss = model(x.to(device), y.to(device))
                    scaled = loss / accumulation
                scaled.backward()
            loss_sum += scaled.detach()
            del logits, loss, scaled, x, y
        per_tensor_grads = tensor_gradient_norms(raw) if rank == 0 else None
        grad_norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0, error_if_nonfinite=True)
        # Same coefficient as PyTorch clip_grad_norm_, including its epsilon.
        clip_coef = (1.0 / (grad_norm + 1e-6)).clamp(max=1.0).item()
        lr = learning_rate(step, a.learning_rate, a.warmup_steps, a.schedule_steps)
        for group in optimizer.param_groups: group['lr'] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if world > 1: dist.all_reduce(loss_sum)
        if rank == 0:
            append_json(a.output/'tensor_grad_norm_history.jsonl',
                        dict(step=step, timing='after_DDP_and_accumulation_before_global_clipping',
                             global_grad_norm=grad_norm.item(), clip_threshold=1.0,
                             clip_coef=clip_coef, clipped=clip_coef < 1.0,
                             parameters=per_tensor_grads))
            row = dict(step=step, loss=loss_sum.item()/world, lr=lr, grad_norm=grad_norm.item(),
                       clip_coef=clip_coef, clipped=clip_coef < 1.0,
                       tokens=step*a.batch_size*a.sequence_length, seconds=time.perf_counter()-started)
            append_json(a.output/'train_history.jsonl', row)
            print(json.dumps(row), flush=True)
    if world > 1: dist.destroy_process_group()


# Fisher implementation below is copied from the validated shared estimator,
# with independent Emb/Head classification and NO width-based logits scaling.

@torch.no_grad()
def parameter_rms(model):
    """RMS (not Frobenius norm), counted once for tied parameters."""
    return {n: torch.linalg.vector_norm(p.detach(), dtype=torch.float64).item()
            / math.sqrt(p.numel()) for n, p in model.named_parameters() if p.requires_grad}


def should_measure(step, start_step, final_step, every):
    return every > 0 and (step in (0, 100, start_step, final_step) or step % every == 0)


def _accumulate_hidden_head(model, xb, yb, params, accum, N, generator, token_chunk_size):
    """Chain rule with a bounded vocabulary head; no full [B,T,V] tensor.

    The detached leaf separates head graphs from the retained trunk graph.
    Direct head and trunk/embedding contributions go to separate parameter
    buffers. This is the exact loss gradient for the sampled labels.
    """
    hidden = model(xb, return_hidden=True)
    leaf = hidden.detach().reshape(-1, hidden.shape[-1])
    hidden_grad = torch.zeros_like(leaf)
    head_index = next(i for i, (_, p) in enumerate(params) if p is model.lm_head.weight)
    mask = (yb.reshape(-1) != -1)
    for offset in range(0, len(leaf), token_chunk_size):
        valid = mask[offset:offset+token_chunk_size]
        if not valid.any():
            continue
        h = leaf[offset:offset+token_chunk_size].detach().requires_grad_(True)
        z = model.lm_head(h)
        # Sampling buffers are bounded and released BEFORE CE backward buffers.
        with torch.no_grad():
            probs = z.float().softmax(-1)
            labels = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            labels[~valid] = -1
            del probs
        loss = F.cross_entropy(z.float(), labels, ignore_index=-1, reduction='sum') / N
        head_grad, h_grad = torch.autograd.grad(loss, (model.lm_head.weight, h))
        accum[head_index].add_(head_grad.detach())
        hidden_grad[offset:offset+len(h)].copy_(h_grad.detach())
        del loss, z, labels, head_grad, h_grad, h
    grads = torch.autograd.grad(hidden, [p for _, p in params],
                                grad_outputs=hidden_grad.reshape_as(hidden), allow_unused=True)
    for total, grad in zip(accum, grads):
        if grad is not None:
            total.add_(grad.detach())


def probe(model, x, y, draws, seed, initial_rms=None, micro_batch_size=64, distributed=False, token_chunk_size=1024, use_bf16=True):
    """Each independent draw accumulates token-mean gradients, THEN squares.

    Inputs may be CPU resident. y supplies the valid-position mask, never true CE labels.
    Private gradient buffers never write training parameter .grad.
    """
    if draws < 1 or micro_batch_size < 1 or token_chunk_size < 1:
        raise ValueError('Expected positive draws and micro-batch size')
    if distributed and not dist.is_initialized():
        raise ValueError('Distributed Fisher requires an initialized process group')
    world = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if initial_rms is not None and set(initial_rms) != {n for n, _ in params}:
        raise ValueError('Initial RMS parameter names do not match model')
    if any(p.dtype != torch.float32 for _, p in params):
        raise ValueError('Fisher diagnostic expects FP32 master parameters')
    mapping = {n: block_type(n) for n, _ in params}
    aliases = {}
    for n, p in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(p), []).append(n)
    modes = [(m, m.training) for m in model.modules()]
    device = params[0][1].device
    cuda = device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    before_bytes = torch.cuda.memory_allocated(device) if cuda else 0
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    current_rms = parameter_rms(model)
    counts_global = torch.tensor([int((y != -1).sum().item()), len(x)], device=device, dtype=torch.int64)
    if distributed:
        dist.all_reduce(counts_global, op=dist.ReduceOp.SUM)
    N, global_sequences = counts_global.tolist()
    if not N:
        raise ValueError('No valid Fisher tokens')
    rows = []
    all_draw_seeds = [[seed + draw*world + r for r in range(world)] for draw in range(draws)]
    draw_seeds = [seeds[rank] for seeds in all_draw_seeds]
    try:
        model.eval()
        with torch.random.fork_rng(devices=[device.index] if cuda else []), torch.enable_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda and use_bf16):
            for draw_seed in draw_seeds:
                accum = [torch.zeros_like(p) for _, p in params]
                generator = torch.Generator(device=device).manual_seed(draw_seed)
                for offset in range(0, len(x), micro_batch_size):
                    xb = x[offset:offset+micro_batch_size].to(device)
                    yb = y[offset:offset+micro_batch_size].to(device)
                    if not (yb != -1).any():
                        continue
                    _accumulate_hidden_head(model, xb, yb, params, accum, N, generator, token_chunk_size)
                    del xb, yb
                if distributed:
                    # SUM global token-mean gradients BEFORE squaring. No DDP averaging.
                    for total in accum:
                        dist.all_reduce(total, op=dist.ReduceOp.SUM)
                traces = torch.stack([N*g.double().square().sum() for g in accum]).cpu().tolist()
                if not all(math.isfinite(t) for t in traces):
                    raise FloatingPointError('Nonfinite Fisher estimate')
                rows.append(dict(zip(mapping, traces)))
                del accum
    finally:
        for module, mode in modes:
            module.training = mode
    if cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-start
    if distributed:
        elapsed_tensor = torch.tensor(elapsed, device=device, dtype=torch.float64)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        elapsed = elapsed_tensor.item()
    counts = {n: p.numel() for n, p in params}
    def stats(names):
        n = sum(counts[k] for k in names)
        values = [sum(row[k] for k in names)/n for row in rows]
        mean = sum(values)/draws
        se = (math.sqrt(sum((v-mean)**2 for v in values)/(draws-1)/draws)
              if draws > 1 else None)
        return dict(numel=n, mean_fisher=mean, trace=mean*n,
                    mc_standard_error=se, mean_fisher_draws=values)
    peak = torch.cuda.max_memory_allocated(device) if cuda else 0
    return dict(estimator='N_times_accumulated_pseudolabel_mean_gradient_squared',
                valid_tokens=N, sequences=global_sequences, local_sequences=len(x),
                world_size=world, gradient_reduction='SUM_before_square' if distributed else 'none',
                sequence_length=x.shape[1],
                micro_batch_size=micro_batch_size,
                accumulation_steps=math.ceil(global_sequences/micro_batch_size),
                local_accumulation_steps=math.ceil(len(x)/micro_batch_size),
                draws=draws, seed=seed, draw_seeds=draw_seeds, draw_seeds_by_rank=all_draw_seeds, seconds=elapsed, precision='bf16_autocast_fp32_accumulation_fp64_square_sum' if cuda and use_bf16 else 'fp32_forward_accumulation_fp64_square_sum',
                token_chunk_size=token_chunk_size,
                vocabulary_head='chunked_hidden_chain_rule',
                peak_allocated_bytes=peak, extra_peak_allocated_bytes=max(0,peak-before_bytes),
                uncertainty='pseudo-label Monte Carlo only, conditional on fixed inputs; two-draw SE is noisy' if draws > 1 else 'single estimate; Monte Carlo standard error unavailable',
                embedding_policy='untied input Emb and output Head measured separately',
                groups={g:stats([n for n in mapping if mapping[n]==g]) for g in sorted(set(mapping.values()))},
                norm_definition='sqrt(sum(parameter**2)/numel); no normalization applied to Fisher',
                parameters={n:dict(stats([n]), group=mapping[n], aliases=aliases[id(p)],
                                   shape=list(p.shape), rms=current_rms[n],
                                   initial_rms=None if initial_rms is None else initial_rms[n]) for n,p in params})


if __name__ == '__main__':
    main()
