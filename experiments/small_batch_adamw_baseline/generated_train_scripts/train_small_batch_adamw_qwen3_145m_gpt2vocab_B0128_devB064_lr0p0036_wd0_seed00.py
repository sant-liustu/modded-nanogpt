# Copied from the RMSNorm-gamma AdamW baseline in this directory.
# Purpose: 145M Qwen3-style AdamW run with the existing GPT-2 tokenizer/data.
# Token budget matches the B=512, T=1024, 5100-step large-batch setup.
# Config: 12 layers, hidden_size=768, Q/KV heads=12/6, head_dim=128,
#         vocab_size=50304, batch_size=128, device_batch_size=64,
#         sequence_length=1024, num_iterations=20400, lr=0.0036,
#         block_weight_decay=0.0, seed=0
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
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch._inductor.config as config
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
# Dense Qwen3 model (standalone pure-PyTorch training path)
#
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Inlined from model_qwen3.py so this experiment script has no local model import.

@dataclass
class Qwen3Config:
    """Configuration for the dense Qwen3-0.6B architecture.

    Defaults match the official checkpoint config.  For the existing local
    GPT-2-tokenized FineWeb shards, use ``Qwen3Config.local_gpt2_vocab()``.
    """

    vocab_size: int = 151_936
    hidden_size: int = 1_024
    intermediate_size: int = 3_072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 40_960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
    bos_token_id: int = 151_643
    eos_token_id: int = 151_645
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None

    def __post_init__(self) -> None:
        integer_fields = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in integer_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by "
                f"num_key_value_heads, got {self.num_attention_heads} and "
                f"{self.num_key_value_heads}"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        if self.hidden_act != "silu":
            raise ValueError(f"this standalone implementation supports only silu, got {self.hidden_act!r}")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError(f"attention_dropout must be in [0, 1), got {self.attention_dropout}")
        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be positive, got {self.rms_norm_eps}")
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")
        if self.initializer_range <= 0:
            raise ValueError(f"initializer_range must be positive, got {self.initializer_range}")
        if self.use_sliding_window or self.sliding_window is not None:
            raise ValueError("this standalone implementation supports full attention only")

    @classmethod
    def local_gpt2_vocab(cls) -> "Qwen3Config":
        """Official 0.6B backbone dimensions with the local GPT-2 vocabulary."""

        return cls(vocab_size=50_304, bos_token_id=50_256, eos_token_id=50_256)


class Qwen3RMSNorm(nn.Module):
    """Qwen3 RMSNorm with a learned per-coordinate scale."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3RotaryEmbedding(nn.Module):
    """Default (non-scaled) Qwen3 RoPE for one shared set of positions."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2:
            raise ValueError(
                f"position_ids must have shape [batch, sequence], got {tuple(position_ids.shape)}"
            )

        # Compute frequencies in fp32, matching the numerical intent of the
        # official implementation even when the model runs under autocast.
        positions = position_ids.to(device=hidden_states.device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=hidden_states.device, dtype=torch.float32)
        freqs = positions.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        embeddings = torch.cat((freqs, freqs), dim=-1)
        return embeddings.cos().to(hidden_states.dtype), embeddings.sin().to(hidden_states.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two contiguous halves of the last dimension."""

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3 RoPE to [batch, heads, sequence, head_dim] tensors."""

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_states = query_states * cos + rotate_half(query_states) * sin
    key_states = key_states * cos + rotate_half(key_states) * sin
    return query_states, key_states


def repeat_kv(hidden_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """Expand KV heads to query-head count without allocating before reshape."""

    if num_key_value_groups == 1:
        return hidden_states
    batch, num_key_value_heads, sequence_length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        num_key_value_groups,
        sequence_length,
        head_dim,
    )
    return hidden_states.reshape(
        batch,
        num_key_value_heads * num_key_value_groups,
        sequence_length,
        head_dim,
    )


class Qwen3MLP(nn.Module):
    """Qwen3 SwiGLU feed-forward block."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen3Attention(nn.Module):
    """Qwen3 grouped-query self-attention with learned QK normalization."""

    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = config.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch, sequence_length, _ = hidden_states.shape
        query_shape = (batch, sequence_length, self.num_attention_heads, self.head_dim)
        key_value_shape = (batch, sequence_length, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(query_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(key_value_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(key_value_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
            scale=self.scaling,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(
            batch,
            sequence_length,
            self.num_attention_heads * self.head_dim,
        )
        return self.o_proj(attention_output)


class Qwen3DecoderLayer(nn.Module):
    """One pre-norm Qwen3 decoder layer."""

    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


def _initialize_qwen3_module(module: nn.Module, initializer_range: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()
    elif isinstance(module, Qwen3RMSNorm):
        nn.init.ones_(module.weight)


class Qwen3Model(nn.Module):
    """Token embedding, decoder stack, and final RMSNorm."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.apply(lambda module: _initialize_qwen3_module(module, config.initializer_range))

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}")
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must have dtype torch.long, got {input_ids.dtype}")

        batch, sequence_length = input_ids.shape
        if sequence_length > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )

        hidden_states = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        elif position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[-1] != sequence_length or position_ids.shape[0] not in (1, batch):
            raise ValueError(
                "position_ids must have shape [1, sequence] or [batch, sequence], "
                f"got {tuple(position_ids.shape)} for input shape {tuple(input_ids.shape)}"
            )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings)
        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    """Dense Qwen3 backbone plus a tied language-model readout."""

    def __init__(self, config: Optional[Qwen3Config] = None) -> None:
        super().__init__()
        self.config = config if config is not None else Qwen3Config()
        self.model = Qwen3Model(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        _initialize_qwen3_module(self.lm_head, self.config.initializer_range)
        if self.config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_logits: bool = True,
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run the model using the existing nanoGPT training contract.

        When targets are supplied, they must already be next-token shifted and
        have the same shape as input_ids.  When targets are absent, only the
        final-position logits are materialized, matching the existing trainer.
        """

        hidden_states = self.model(input_ids, position_ids=position_ids)
        loss = None

        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError(
                    f"targets must match input_ids shape, got {tuple(targets.shape)} "
                    f"and {tuple(input_ids.shape)}"
                )
            logits_for_loss = self.lm_head(hidden_states).float()
            loss = F.cross_entropy(
                logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
            logits = logits_for_loss if return_logits else None
        elif return_logits:
            logits = self.lm_head(hidden_states[:, [-1], :]).float()
        else:
            logits = None

        return logits, loss

    def num_parameters(self, *, exclude_embeddings: bool = False) -> int:
        total = sum(parameter.numel() for parameter in self.parameters())
        if exclude_embeddings:
            total -= self.model.embed_tokens.weight.numel()
        return total

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
# Qwen3 model definitions live in the repository-level model_qwen3.py module.

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
    warmup_iters : int = 1000
    warmdown_iters : int = 5800 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0.0 # weight decay for block weights and tied wte/lm_head; RMSNorm gamma is excluded
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
args = Hyperparameters()
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

# Keep the existing GPT-2 tokenizer/data and tied embedding while using the
# agreed 12-layer, 144,824,064-parameter Qwen3-style architecture.
qwen3_config = Qwen3Config(
    vocab_size=50_304,
    hidden_size=768,
    intermediate_size=2_304,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=6,
    head_dim=128,
    max_position_embeddings=1_024,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000.0,
    attention_bias=False,
    attention_dropout=0.0,
    initializer_range=0.02,
    hidden_act="silu",
    tie_word_embeddings=True,
    bos_token_id=50_256,
    eos_token_id=50_256,
)
model = Qwen3ForCausalLM(qwen3_config)
if master_process:
    print(f"Qwen3 parameter count: {model.num_parameters():,}")
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
model_for_structure = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer(s)
def canonical_param_name(name):
    if name.startswith('_orig_mod.'):
        return name[len('_orig_mod.'):]
    return name

def is_qwen3_rmsnorm_gamma_name(name):
    name = canonical_param_name(name)
    return (
        name == 'model.norm.weight'
        or name.endswith('.self_attn.q_norm.weight')
        or name.endswith('.self_attn.k_norm.weight')
        or name.endswith('.input_layernorm.weight')
        or name.endswith('.post_attention_layernorm.weight')
    )

block_weight_parameters = [
    p
    for name, p in raw_model.named_parameters()
    if canonical_param_name(name).startswith('model.layers.')
    and p.ndim == 2
]
rmsnorm_gamma_parameters = [
    p
    for name, p in raw_model.named_parameters()
    if is_qwen3_rmsnorm_gamma_name(name)
]
lm_head_parameters = list(model_for_structure.lm_head.parameters())

expected_block_matrix_count = qwen3_config.num_hidden_layers * 7
expected_norm_gamma_count = qwen3_config.num_hidden_layers * 4 + 1
if len(block_weight_parameters) != expected_block_matrix_count:
    raise RuntimeError(
        f"expected {expected_block_matrix_count} Qwen3 block matrices, "
        f"found {len(block_weight_parameters)}"
    )
if len(rmsnorm_gamma_parameters) != expected_norm_gamma_count:
    raise RuntimeError(
        f"expected {expected_norm_gamma_count} Qwen3 RMSNorm gamma tensors, "
        f"found {len(rmsnorm_gamma_parameters)}"
    )

assigned_parameters = lm_head_parameters + block_weight_parameters + rmsnorm_gamma_parameters
assigned_ids = [id(p) for p in assigned_parameters]
if len(assigned_ids) != len(set(assigned_ids)):
    raise RuntimeError("a trainable parameter was assigned to more than one optimizer group")
trainable_by_id = {id(p): canonical_param_name(name) for name, p in raw_model.named_parameters() if p.requires_grad}
missing_optimizer_parameters = [
    name for param_id, name in trainable_by_id.items() if param_id not in set(assigned_ids)
]
if missing_optimizer_parameters:
    raise RuntimeError(f"trainable parameters missing from AdamW groups: {missing_optimizer_parameters}")

optimizer1 = torch.optim.AdamW(lm_head_parameters, lr=args.embed_learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)
optimizer2_groups = [dict(params=block_weight_parameters, weight_decay=args.weight_decay, lrnorm_match_group=True)]
if rmsnorm_gamma_parameters:
    optimizer2_groups.append(dict(params=rmsnorm_gamma_parameters, weight_decay=0.0, lrnorm_match_group=False))
optimizer2 = torch.optim.AdamW(optimizer2_groups, lr=0.5*args.embed_learning_rate, betas=(0.9, 0.95),
                               fused=True)
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
        model = model._orig_mod if hasattr(model, '_orig_mod') else model
        self.blocks = list(model.model.layers)
        self.eps = eps
        self.handles = []
        self.h_pre = [None] * len(self.blocks)
        self.h_mid = [None] * len(self.blocks)
        self.records = {field: [None] * len(self.blocks) for field in ACTIVATION_PROBE_FIELDS}

    def __enter__(self):
        for layer, block in enumerate(self.blocks):
            self.handles.append(block.register_forward_pre_hook(self._block_pre_hook(layer)))
            self.handles.append(block.self_attn.register_forward_hook(self._attn_hook(layer)))
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
        layer_count=len(model_for_structure.model.layers),
        recorded_fields=list(ACTIVATION_PROBE_FIELDS),
        array_layout='[layer, batch, seq]',
        logging_cadence=args.activation_probe_every,
        model_config=dict(
            num_hidden_layers=model_for_structure.config.num_hidden_layers,
            num_attention_heads=model_for_structure.config.num_attention_heads,
            num_key_value_heads=model_for_structure.config.num_key_value_heads,
            head_dim=model_for_structure.config.head_dim,
            hidden_size=model_for_structure.config.hidden_size,
            intermediate_size=model_for_structure.config.intermediate_size,
            vocab_size=model_for_structure.config.vocab_size,
            rope_theta=model_for_structure.config.rope_theta,
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
    adamw_update_state = maybe_capture_adamw_update_state(update_step)
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    maybe_log_adamw_update_norms(update_step, adamw_update_state)
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
