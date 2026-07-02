import argparse
import contextlib
import glob
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_TOP_KS = (5, 10, 50)


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    learnable_rmsnorm: bool = False


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if self.seq_len_cached != seq_len or self.cos_cached is None or self.cos_cached.device != x.device:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(x.device)).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, learnable=False, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if learnable else None

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
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()
        self.q_norm = RMSNorm(self.head_dim, learnable=config.learnable_rmsnorm)
        self.k_norm = RMSNorm(self.head_dim, learnable=config.learnable_rmsnorm)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        bsz, seq, channels = x.size()
        q = self.c_q(x).view(bsz, seq, self.n_head, self.head_dim)
        k = self.c_k(x).view(bsz, seq, self.n_head, self.head_dim)
        v = self.c_v(x).view(bsz, seq, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq, channels)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_norm = RMSNorm(config.n_embd, learnable=config.learnable_rmsnorm)
        self.mlp_norm = RMSNorm(config.n_embd, learnable=config.learnable_rmsnorm)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.final_norm = RMSNorm(config.n_embd, learnable=config.learnable_rmsnorm)

    def forward(self, idx, targets=None):
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x).float()
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss


def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        if header[0] != 20240520:
            raise ValueError(f"magic number mismatch in data shard: {filename}")
        if header[1] != 1:
            raise ValueError(f"unsupported data shard version in: {filename}")
        return int(header[2])


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        if header[0] != 20240520:
            raise ValueError(f"magic number mismatch in data shard: {filename}")
        if header[1] != 1:
            raise ValueError(f"unsupported data shard version in: {filename}")
        ntok = int(header[2])
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    if len(tokens) != ntok:
        raise ValueError(f"token count mismatch in {filename}: expected {ntok}, got {len(tokens)}")
    return tokens


class EvalDataLoader:
    def __init__(self, filename_pattern, batch_size, sequence_length):
        self.files = sorted(glob.glob(filename_pattern))
        if not self.files:
            raise FileNotFoundError(f"no validation shards matched {filename_pattern}")
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            if shard_ntok < batch_size * sequence_length + 1:
                raise ValueError(f"not enough tokens in {fname}: {shard_ntok}")
            self.ntok_total += shard_ntok
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard += 1
        if self.current_shard >= len(self.files):
            raise StopIteration
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        bsz = self.batch_size
        seq = self.sequence_length
        if self.current_position + bsz * seq + 1 > len(self.tokens):
            self.advance()
        buf = self.tokens[self.current_position : self.current_position + bsz * seq + 1]
        self.current_position += bsz * seq
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf[:-1].view(bsz, seq)
        y = buf[1:].view(bsz, seq)
        return x, y

    def max_full_batches(self):
        return sum(max(0, (_peek_data_shard(fname) - 1) // (self.batch_size * self.sequence_length)) for fname in self.files)


def canonical_state_dict(state_dict):
    canonical = {}
    for key, value in state_dict.items():
        name = key
        if name.startswith("module."):
            name = name[len("module.") :]
        if name.startswith("_orig_mod."):
            name = name[len("_orig_mod.") :]
        canonical[name] = value
    return canonical


def state_dict_from_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"unsupported checkpoint object in {path}: {type(checkpoint)}")
    return checkpoint if isinstance(checkpoint, dict) else {}, canonical_state_dict(state_dict)


def checkpoint_has_learnable_rmsnorm(state_dict):
    return (
        "final_norm.weight" in state_dict
        or any(key.endswith(".attn_norm.weight") for key in state_dict)
        or any(key.endswith(".q_norm.weight") for key in state_dict)
    )


def infer_config(checkpoint_path, state_dict, n_head_override=None):
    vocab_size, n_embd = state_dict["transformer.wte.weight"].shape
    layer_ids = []
    for key in state_dict:
        if key.startswith("transformer.h.") and ".attn.c_q.weight" in key:
            layer_ids.append(int(key.split(".")[2]))
    n_layer = max(layer_ids) + 1 if layer_ids else 0
    n_head = n_head_override

    metadata_path = Path(checkpoint_path).parent / "activation_probe_metadata.json"
    if n_head is None and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        model_config = metadata.get("model_config", {})
        n_head = model_config.get("n_head")
    if n_head is None:
        n_head = 6

    return GPTConfig(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=int(n_head),
        n_embd=n_embd,
        learnable_rmsnorm=checkpoint_has_learnable_rmsnorm(state_dict),
    )


def load_model_from_checkpoint(checkpoint_path, device, n_head_override=None):
    checkpoint, state_dict = state_dict_from_checkpoint(checkpoint_path)
    config = infer_config(checkpoint_path, state_dict, n_head_override=n_head_override)
    model = GPT(config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return checkpoint, model, config


def expand_paths(items, extra_glob=None):
    paths = []
    for item in items:
        matches = sorted(glob.glob(item))
        paths.extend(matches if matches else [item])
    if extra_glob:
        paths.extend(sorted(glob.glob(extra_glob)))
    unique = []
    seen = set()
    for path in paths:
        normalized = str(Path(path))
        if normalized not in seen:
            unique.append(Path(path))
            seen.add(normalized)
    if not unique:
        raise ValueError("no candidate checkpoints provided")
    return unique


def checkpoint_slug(path):
    parent = Path(path).parent.name
    stem = Path(path).stem
    return f"{parent}_{stem}".replace(os.sep, "_")


def token_sha256_update(hasher, x):
    hasher.update(x.detach().cpu().numpy().astype(np.int64).tobytes())


def autocast_context(device, dtype_name):
    if device.type != "cuda" or dtype_name == "fp32":
        return contextlib.nullcontext()
    if dtype_name == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype_name == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"unsupported dtype: {dtype_name}")


class RunningMean:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, values):
        x = values.detach().float()
        self.total += x.sum().item()
        self.count += x.numel()

    def update_scalar(self, value, count):
        self.total += float(value) * int(count)
        self.count += int(count)

    def mean(self):
        return self.total / self.count if self.count else float("nan")


class MetricAccumulator:
    def __init__(self, top_ks):
        self.top_ks = tuple(top_ks)
        self.metrics = {
            "full_vocab_js_divergence": RunningMean(),
            "top1_agreement": RunningMean(),
        }
        for k in self.top_ks:
            self.metrics[f"top{k}_overlap"] = RunningMean()
            self.metrics[f"top{k}_union_js_renorm"] = RunningMean()
        self.reference_loss = RunningMean()
        self.candidate_loss = RunningMean()
        self.tokens = 0
        self.batches = 0

    def update_losses(self, reference_loss, candidate_loss, token_count):
        self.reference_loss.update_scalar(reference_loss.item(), token_count)
        self.candidate_loss.update_scalar(candidate_loss.item(), token_count)

    def as_dict(self):
        out = {name: metric.mean() for name, metric in self.metrics.items()}
        out["reference_loss"] = self.reference_loss.mean()
        out["candidate_loss"] = self.candidate_loss.mean()
        out["loss_abs_diff"] = abs(out["candidate_loss"] - out["reference_loss"])
        out["tokens"] = self.tokens
        out["batches"] = self.batches
        return out


def js_from_probs(p, q, eps):
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp_min(eps))
    kl_pm = torch.where(p > 0, p * (torch.log(p.clamp_min(eps)) - log_m), torch.zeros_like(p)).sum(dim=-1)
    kl_qm = torch.where(q > 0, q * (torch.log(q.clamp_min(eps)) - log_m), torch.zeros_like(q)).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def update_distribution_metrics(acc, reference_logits, candidate_logits, eps):
    ref = reference_logits.detach().float().reshape(-1, reference_logits.size(-1))
    cand = candidate_logits.detach().float().reshape(-1, candidate_logits.size(-1))
    token_count, vocab_size = ref.shape
    max_k = max(acc.top_ks)
    if max_k > vocab_size:
        raise ValueError(f"top-k={max_k} exceeds vocab size={vocab_size}")

    logp = F.log_softmax(ref, dim=-1)
    logq = F.log_softmax(cand, dim=-1)
    p = logp.exp()
    q = logq.exp()
    m = 0.5 * (p + q)
    logm = torch.log(m.clamp_min(eps))
    full_js = 0.5 * (p * (logp - logm)).sum(dim=-1) + 0.5 * (q * (logq - logm)).sum(dim=-1)
    acc.metrics["full_vocab_js_divergence"].update(full_js)

    ref_top = torch.topk(ref, max_k, dim=-1).indices
    cand_top = torch.topk(cand, max_k, dim=-1).indices
    acc.metrics["top1_agreement"].update((ref_top[:, 0] == cand_top[:, 0]).float())

    for k in acc.top_ks:
        ref_mask = torch.zeros((token_count, vocab_size), device=ref.device, dtype=torch.bool)
        ref_mask.scatter_(1, ref_top[:, :k], True)
        overlap = ref_mask.gather(1, cand_top[:, :k]).float().sum(dim=-1) / k
        acc.metrics[f"top{k}_overlap"].update(overlap)

        union_mask = ref_mask
        union_mask.scatter_(1, cand_top[:, :k], True)
        p_union = torch.where(union_mask, p, torch.zeros_like(p))
        q_union = torch.where(union_mask, q, torch.zeros_like(q))
        p_union = p_union / p_union.sum(dim=-1, keepdim=True).clamp_min(eps)
        q_union = q_union / q_union.sum(dim=-1, keepdim=True).clamp_min(eps)
        acc.metrics[f"top{k}_union_js_renorm"].update(js_from_probs(p_union, q_union, eps))

    acc.tokens += token_count
    acc.batches += 1


def compare_checkpoints(reference_model, candidate_model, data_loader, args, device):
    acc = MetricAccumulator(args.top_k)
    hasher = hashlib.sha256()
    dtype_context = autocast_context(device, args.dtype)
    data_loader.reset()
    num_batches = args.num_batches
    if num_batches == 0:
        num_batches = data_loader.max_full_batches()

    for _ in range(num_batches):
        try:
            x, y = data_loader.next_batch()
        except StopIteration:
            break
        token_sha256_update(hasher, x)
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad(), dtype_context:
            reference_logits, reference_loss = reference_model(x, y)
            candidate_logits, candidate_loss = candidate_model(x, y)
        token_count = x.numel()
        acc.update_losses(reference_loss, candidate_loss, token_count)
        update_distribution_metrics(acc, reference_logits, candidate_logits, args.eps)
        del reference_logits, candidate_logits, reference_loss, candidate_loss, x, y

    result = acc.as_dict()
    result["probe_token_sha256"] = hasher.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare checkpoint output distributions on validation tokens.")
    parser.add_argument("--reference-checkpoint", required=True, help="Reference checkpoint path.")
    parser.add_argument("--checkpoint", action="append", default=[], help="Candidate checkpoint path or glob. Can be repeated.")
    parser.add_argument("--checkpoint-glob", default=None, help="Additional candidate checkpoint glob.")
    parser.add_argument("--input-val-bin", default="data/local_debug/fineweb_val_*.bin", help="Validation token shard glob.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--num-batches", type=int, default=32, help="Number of validation batches. Use 0 for one full pass over shards.")
    parser.add_argument("--n-head", type=int, default=None, help="Override attention head count; defaults to metadata or 6.")
    parser.add_argument("--top-k", type=int, action="append", default=None, help="Top-k values for overlap and union JS. Defaults to 5,10,50.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.top_k is None:
        args.top_k = list(DEFAULT_TOP_KS)
    args.top_k = sorted(set(args.top_k))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = expand_paths(args.checkpoint, args.checkpoint_glob)
    device = torch.device(args.device)
    data_loader = EvalDataLoader(args.input_val_bin, args.batch_size, args.sequence_length)

    reference_checkpoint, reference_model, reference_config = load_model_from_checkpoint(
        args.reference_checkpoint,
        device,
        n_head_override=args.n_head,
    )
    reference_step = reference_checkpoint.get("step") if isinstance(reference_checkpoint, dict) else None

    metadata = {
        "reference_checkpoint": str(args.reference_checkpoint),
        "reference_step": reference_step,
        "candidate_checkpoints": [str(path) for path in candidates],
        "input_val_bin": args.input_val_bin,
        "validation_files": data_loader.files,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "num_batches": args.num_batches,
        "top_k": args.top_k,
        "metrics": [
            "full_vocab_js_divergence",
            "top1_agreement",
            *[f"top{k}_overlap" for k in args.top_k],
            *[f"top{k}_union_js_renorm" for k in args.top_k],
        ],
        "device": str(device),
        "dtype": args.dtype,
        "eps": args.eps,
        "reference_model_config": reference_config.__dict__,
    }
    (output_dir / "function_collapse_probe_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary_path = output_dir / "function_collapse_probe_summary.jsonl"
    if summary_path.exists():
        summary_path.unlink()

    index = []
    for candidate_path in candidates:
        candidate_checkpoint, candidate_model, candidate_config = load_model_from_checkpoint(
            candidate_path,
            device,
            n_head_override=args.n_head,
        )
        candidate_step = candidate_checkpoint.get("step") if isinstance(candidate_checkpoint, dict) else None
        metrics = compare_checkpoints(reference_model, candidate_model, data_loader, args, device)
        record = {
            "reference_checkpoint": str(args.reference_checkpoint),
            "reference_step": reference_step,
            "candidate_checkpoint": str(candidate_path),
            "candidate_step": candidate_step,
            "candidate_slug": checkpoint_slug(candidate_path),
            "candidate_model_config": candidate_config.__dict__,
            **metrics,
        }
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        index.append(record)
        del candidate_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (output_dir / "function_collapse_probe_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "candidates": len(index), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
