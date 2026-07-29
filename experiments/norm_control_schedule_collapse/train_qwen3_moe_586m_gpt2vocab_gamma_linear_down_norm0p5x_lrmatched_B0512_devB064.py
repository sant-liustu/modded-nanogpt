# Qwen3-MoE 586M linear-down norm-control experiment.
# Capture tied embedding, block matrices, router matrices, and each expert matrix slice after warmup update 500, then linearly scale from 1x to 0.5x.
# Match tied embedding and controlled-matrix LR to the same schedule ratio; keep RMSNorm gamma on base WSD.
# Backbone: 12 layers, hidden_size=768, Q/KV heads=12/4, head_dim=128.
# MoE: 32 experts, top-4 routing, expert intermediate_size=576, all 12 layers sparse.
# Expert dispatch: MegaBlocks grouped GEMM (gg.ops.gmm) with @torch.compiler.disable,
#   treated as opaque boundary by torch.compile.
# Compilation: torch.compile(fullgraph=False) on full model.
#   - Attention, norms, router, LM head, loss: compiled by Inductor
#   - Grouped GEMM expert block: skipped (already optimized CUDA kernel)
# Activation checkpoint: DISABLED (testing if B64 fits without checkpointing)
# Training: vocab_size=50304, batch_size=512, device_batch_size=64 (8×64=512, no accum),
#           sequence_length=1024, num_iterations=10200, lr=0.0036,
#           block_weight_decay=0.0, router_aux_loss_coef=0.001, seed=0.
import os
import random
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import contextlib
import hashlib
import json
import math
import uuid
import glob
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from megablocks import grouped_gemm_util as gg


# -----------------------------------------------------------------------------
# Qwen3-MoE model (standalone pure-PyTorch training path)
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
# Inlined from model_qwen3_moe.py so this experiment script has no local model import.

@dataclass
class Qwen3MoeConfig:
    """Configuration for the standalone Qwen3-MoE model.

    Defaults mirror Qwen3-30B-A3B-Base. The ``output_router_logits`` default
    also mirrors the checkpoint metadata, so callers that want the auxiliary
    routing loss must enable it explicitly (the local preset does so).
    """

    vocab_size: int = 151_936
    hidden_size: int = 2_048
    intermediate_size: int = 6_144
    moe_intermediate_size: int = 768
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    num_experts: int = 128
    num_experts_per_tok: int = 8
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()
    norm_topk_prob: bool = True
    router_aux_loss_coef: float = 0.001
    output_router_logits: bool = False
    max_position_embeddings: int = 32_768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    bos_token_id: int = 151_643
    eos_token_id: int = 151_643
    ignore_index: int = -1
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None

    def __post_init__(self) -> None:
        positive_integer_fields = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "decoder_sparse_step": self.decoder_sparse_step,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in positive_integer_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if self.num_experts < 0:
            raise ValueError(f"num_experts must be non-negative, got {self.num_experts}")
        if self.num_experts > 0 and not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError(
                "num_experts_per_tok must be between 1 and num_experts, got "
                f"{self.num_experts_per_tok} and {self.num_experts}"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads, got "
                f"{self.num_attention_heads} and {self.num_key_value_heads}"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        if self.hidden_act != "silu":
            raise ValueError(
                f"this standalone implementation supports only silu, got {self.hidden_act!r}"
            )
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError(
                f"attention_dropout must be in [0, 1), got {self.attention_dropout}"
            )
        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be positive, got {self.rms_norm_eps}")
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")
        if self.initializer_range <= 0:
            raise ValueError(
                f"initializer_range must be positive, got {self.initializer_range}"
            )
        if self.router_aux_loss_coef < 0:
            raise ValueError(
                "router_aux_loss_coef must be non-negative, got "
                f"{self.router_aux_loss_coef}"
            )
        if self.use_sliding_window or self.sliding_window is not None:
            raise ValueError("this standalone implementation supports full attention only")

        self.mlp_only_layers = tuple(self.mlp_only_layers)
        if len(set(self.mlp_only_layers)) != len(self.mlp_only_layers):
            raise ValueError("mlp_only_layers must not contain duplicate layer indices")
        invalid_dense_layers = [
            layer_idx
            for layer_idx in self.mlp_only_layers
            if layer_idx < 0 or layer_idx >= self.num_hidden_layers
        ]
        if invalid_dense_layers:
            raise ValueError(
                "mlp_only_layers contains indices outside the decoder stack: "
                f"{invalid_dense_layers}"
            )

    @classmethod
    def tiny(cls, **overrides: object) -> "Qwen3MoeConfig":
        """Small configuration for CPU/CUDA routing and backward smoke tests."""

        values: dict[str, object] = {
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "max_position_embeddings": 128,
            "tie_word_embeddings": True,
            "bos_token_id": 0,
            "eos_token_id": 0,
            "output_router_logits": True,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def local_586m_top4(cls, **overrides: object) -> "Qwen3MoeConfig":
        """Use the local 12-layer backbone with the 586M top-4 MoE preset.

        ``moe_intermediate_size=576`` with top-4 routing gives the same active
        expert width per token as a dense intermediate size of 2304. The total
        model has 586,307,328 unique parameters, of which 140,400,384 are
        active per token under the Qwen model-card counting convention.
        """

        values: dict[str, object] = {
            "vocab_size": 50_304,
            "hidden_size": 768,
            "intermediate_size": 2_304,
            "moe_intermediate_size": 576,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_experts": 32,
            "num_experts_per_tok": 4,
            "max_position_embeddings": 1_024,
            "tie_word_embeddings": True,
            "bos_token_id": 50_256,
            "eos_token_id": 50_256,
            "output_router_logits": True,
        }
        values.update(overrides)
        return cls(**values)


@dataclass
class Qwen3MoeModelOutput:
    """Backbone output plus optional per-sparse-layer router logits."""

    last_hidden_state: torch.Tensor
    router_logits: Optional[tuple[torch.Tensor, ...]]


@dataclass
class Qwen3MoeCausalLMOutput:
    """Explicit MoE training output for inspecting the two loss terms."""

    logits: Optional[torch.Tensor]
    loss: Optional[torch.Tensor]
    lm_loss: Optional[torch.Tensor]
    aux_loss: Optional[torch.Tensor]
    router_logits: Optional[tuple[torch.Tensor, ...]]


class Qwen3MoeRMSNorm(nn.Module):
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


class Qwen3MoeRotaryEmbedding(nn.Module):
    """Default, non-scaled Qwen3 RoPE."""

    def __init__(self, config: Qwen3MoeConfig) -> None:
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
                "position_ids must have shape [batch, sequence], got "
                f"{tuple(position_ids.shape)}"
            )

        positions = position_ids.to(device=hidden_states.device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=hidden_states.device, dtype=torch.float32)
        freqs = positions.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        embeddings = torch.cat((freqs, freqs), dim=-1)
        return (
            embeddings.cos().to(hidden_states.dtype),
            embeddings.sin().to(hidden_states.dtype),
        )


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
    """Expand KV heads to the query-head count."""

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


class Qwen3MoeMLP(nn.Module):
    """Dense Qwen3 SwiGLU block used on explicitly dense decoder layers."""

    def __init__(
        self,
        config: Qwen3MoeConfig,
        intermediate_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class Qwen3MoeExperts(nn.Module):
    """Local experts stored as two stacked 3D parameter tensors."""

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )

    @torch.compiler.disable  # Keep grouped GEMM as opaque boundary for torch.compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Grouped GEMM dispatch: sort tokens by expert, batched GEMM, scatter back.

        This region is intentionally excluded from torch.compile because:
        1. grouped_gemm is already a compiled CUDA kernel (no further fusion benefit)
        2. Dynamic token counts per expert would cause graph breaks / large unrolled graphs
        3. Allows the rest of the model (attention, norms, router, LM head) to be compiled
        """

        if hidden_states.ndim != 2:
            raise ValueError(
                f"hidden_states must have shape [tokens, hidden], got {tuple(hidden_states.shape)}"
            )
        if top_k_index.shape != top_k_weights.shape:
            raise ValueError(
                "top_k_index and top_k_weights must have the same shape, got "
                f"{tuple(top_k_index.shape)} and {tuple(top_k_weights.shape)}"
            )
        if top_k_index.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "routing tensors must contain one row per token, got "
                f"{top_k_index.shape[0]} rows for {hidden_states.shape[0]} tokens"
            )

        num_tokens = hidden_states.shape[0]
        top_k = top_k_index.shape[1]

        # Flatten token-expert pairs and sort by expert assignment.
        flat_experts = top_k_index.reshape(-1)                 # [T*K]
        flat_weights = top_k_weights.reshape(-1, 1)            # [T*K, 1]

        sort_idx = flat_experts.argsort()
        sorted_experts = flat_experts[sort_idx]
        sorted_weights = flat_weights[sort_idx]
        sorted_hidden = hidden_states.repeat_interleave(top_k, dim=0)[sort_idx].contiguous()

        # Count tokens per expert; filter out experts with zero tokens.
        counts = torch.bincount(sorted_experts, minlength=self.num_experts)
        nonzero_mask = counts > 0
        nonzero_counts = counts[nonzero_mask].cpu().to(torch.long)
        nonzero_gate_up = self.gate_up_proj[nonzero_mask].contiguous()
        nonzero_down = self.down_proj[nonzero_mask].contiguous()

        # Ensure bfloat16: grouped_gemm backend requires strict bf16.
        work_dtype = torch.bfloat16
        sorted_hidden = sorted_hidden.to(work_dtype)
        nonzero_gate_up = nonzero_gate_up.to(work_dtype)
        nonzero_down = nonzero_down.to(work_dtype)

        # Grouped GEMM: gate-up projection → SwiGLU → down projection.
        gate_up_out = gg.ops.gmm(sorted_hidden, nonzero_gate_up, nonzero_counts, trans_b=True)
        gate, up = gate_up_out.chunk(2, dim=-1)
        intermediate = F.silu(gate) * up
        expert_output = gg.ops.gmm(intermediate, nonzero_down, nonzero_counts, trans_b=True)
        expert_output = expert_output * sorted_weights

        # Scatter back to original token order using differentiable index_add.
        token_idx = sort_idx // top_k
        result = torch.index_add(
            torch.zeros(num_tokens, self.hidden_dim, device=hidden_states.device,
                        dtype=work_dtype),
            0, token_idx, expert_output,
        )
        return result.to(hidden_states.dtype)


class Qwen3MoeTopKRouter(nn.Module):
    """Bias-free Qwen3 router with float32 softmax and top-k selection."""

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float32, dim=-1)
        routing_weights, selected_experts = torch.topk(
            router_probs,
            self.top_k,
            dim=-1,
        )

        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1,
                keepdim=True,
            )

        routing_weights = routing_weights.to(router_logits.dtype)
        return router_logits, routing_weights, selected_experts


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Router plus local expert dispatch for one sparse decoder layer."""

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.experts = Qwen3MoeExperts(config)
        self.gate = Qwen3MoeTopKRouter(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, hidden_dim)
        router_logits, routing_weights, selected_experts = self.gate(flat_hidden_states)
        output = self.experts(
            flat_hidden_states,
            selected_experts,
            routing_weights,
        )
        return output.reshape(batch_size, sequence_length, hidden_dim), router_logits


class Qwen3MoeAttention(nn.Module):
    """Qwen3 grouped-query self-attention with learned QK normalization."""

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
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
        self.q_norm = Qwen3MoeRMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3MoeRMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch, sequence_length, _ = hidden_states.shape
        query_shape = (
            batch,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        )
        key_value_shape = (
            batch,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(query_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(key_value_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(key_value_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )
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


class Qwen3MoeDecoderLayer(nn.Module):
    """One pre-norm Qwen3 decoder layer with dense or sparse SwiGLU."""

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Qwen3MoeAttention(config, layer_idx)
        self.is_sparse = (
            layer_idx not in config.mlp_only_layers
            and config.num_experts > 0
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        )
        if self.is_sparse:
            self.mlp: nn.Module = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_sparse:
            hidden_states, router_logits = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            router_logits = None
        return residual + hidden_states, router_logits


def _initialize_qwen3_moe_module(
    module: nn.Module,
    initializer_range: float,
) -> None:
    if isinstance(module, Qwen3MoeExperts):
        nn.init.normal_(module.gate_up_proj, mean=0.0, std=initializer_range)
        nn.init.normal_(module.down_proj, mean=0.0, std=initializer_range)
    elif isinstance(module, Qwen3MoeTopKRouter):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()
    elif isinstance(module, Qwen3MoeRMSNorm):
        nn.init.ones_(module.weight)


def load_balancing_loss_func(
    gate_logits: Optional[tuple[torch.Tensor, ...]],
    num_experts: int,
    top_k: int,
) -> Optional[torch.Tensor]:
    """Compute the Qwen/Switch-style router load-balancing objective.

    For uniform routing the value is approximately ``top_k`` rather than one,
    matching the Hugging Face Qwen3-MoE implementation.
    """

    if gate_logits is None or len(gate_logits) == 0:
        return None

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat(
        [layer_logits.to(compute_device).float() for layer_logits in gate_logits],
        dim=0,
    )
    routing_weights = F.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).float()

    tokens_per_expert = expert_mask.mean(dim=0)
    router_prob_per_expert = routing_weights.mean(dim=0)
    overall_loss = torch.sum(
        tokens_per_expert * router_prob_per_expert.unsqueeze(0)
    )
    return overall_loss * num_experts


class Qwen3MoeModel(nn.Module):
    """Token embedding, decoder stack, final RMSNorm, and router collection."""

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config)
        self.sparse_layer_indices = tuple(
            layer_idx for layer_idx, layer in enumerate(self.layers) if layer.is_sparse
        )
        self.apply(
            lambda module: _initialize_qwen3_moe_module(
                module,
                config.initializer_range,
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        output_router_logits: Optional[bool] = None,
    ) -> Qwen3MoeModelOutput:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}"
            )
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must have dtype torch.long, got {input_ids.dtype}")

        batch, sequence_length = input_ids.shape
        if sequence_length > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )
        output_router_logits = (
            self.config.output_router_logits
            if output_router_logits is None
            else output_router_logits
        )

        hidden_states = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(
                sequence_length,
                device=input_ids.device,
            ).unsqueeze(0)
        elif position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[-1] != sequence_length or position_ids.shape[0] not in (
            1,
            batch,
        ):
            raise ValueError(
                "position_ids must have shape [1, sequence] or [batch, sequence], "
                f"got {tuple(position_ids.shape)} for input shape {tuple(input_ids.shape)}"
            )

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        collected_router_logits: list[torch.Tensor] = []
        for decoder_layer in self.layers:
            hidden_states, layer_router_logits = decoder_layer(
                hidden_states,
                position_embeddings,
            )  # BENCHMARK: no activation checkpoint for B64 throughput test
            if output_router_logits and layer_router_logits is not None:
                collected_router_logits.append(layer_router_logits)

        hidden_states = self.norm(hidden_states)
        router_logits = (
            tuple(collected_router_logits) if output_router_logits else None
        )
        return Qwen3MoeModelOutput(
            last_hidden_state=hidden_states,
            router_logits=router_logits,
        )


class Qwen3MoeForCausalLM(nn.Module):
    """Qwen3-MoE backbone plus the local nanoGPT-style language-model API."""

    def __init__(self, config: Optional[Qwen3MoeConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else Qwen3MoeConfig()
        self.model = Qwen3MoeModel(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
        )
        _initialize_qwen3_moe_module(
            self.lm_head,
            self.config.initializer_range,
        )
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
        output_router_logits: Optional[bool] = None,
        return_dict: bool = False,
    ) -> (
        tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
        | Qwen3MoeCausalLMOutput
    ):
        """Run the model with already-shifted targets.

        The default return value is ``(logits, total_loss)`` so this class can
        replace the existing dense standalone model with minimal trainer
        changes. Set ``return_dict=True`` to inspect ``lm_loss``, ``aux_loss``,
        and the per-sparse-layer router logits separately.
        """

        output_router_logits = (
            self.config.output_router_logits
            if output_router_logits is None
            else output_router_logits
        )
        model_output = self.model(
            input_ids,
            position_ids=position_ids,
            output_router_logits=output_router_logits,
        )
        hidden_states = model_output.last_hidden_state
        router_logits = model_output.router_logits
        aux_loss = load_balancing_loss_func(
            router_logits,
            num_experts=self.config.num_experts,
            top_k=self.config.num_experts_per_tok,
        )

        logits: Optional[torch.Tensor]
        lm_loss: Optional[torch.Tensor] = None
        total_loss: Optional[torch.Tensor] = None

        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError(
                    f"targets must match input_ids shape, got {tuple(targets.shape)} "
                    f"and {tuple(input_ids.shape)}"
                )
            # Chunked logits + cross_entropy to avoid materializing the full
            # (B*T, vocab_size) ~13 GiB float32 logits tensor on each GPU.
            # Each chunk computes logits independently; CE is per-sample so
            # reduction='sum' over chunks then divide by non-ignored count is
            # equivalent to the original reduction='mean'.
            CHUNK_SIZE = 1024
            hidden_flat = hidden_states.reshape(-1, hidden_states.size(-1))
            targets_flat = targets.reshape(-1)
            ce_sum = 0.0
            ce_count = 0
            for i in range(0, hidden_flat.size(0), CHUNK_SIZE):
                h_chunk = hidden_flat[i:i + CHUNK_SIZE]
                t_chunk = targets_flat[i:i + CHUNK_SIZE]
                valid = (t_chunk != self.config.ignore_index)
                if valid.sum() == 0:
                    continue
                logits_chunk = self.lm_head(h_chunk).float()
                ce_sum += F.cross_entropy(
                    logits_chunk, t_chunk,
                    ignore_index=self.config.ignore_index,
                    reduction='sum',
                )
                ce_count += valid.sum()
            lm_loss = ce_sum / ce_count if ce_count > 0 else ce_sum * 0
            total_loss = lm_loss
            if aux_loss is not None:
                total_loss = total_loss + self.config.router_aux_loss_coef * aux_loss
            logits = logits_for_loss if return_logits else None
        elif return_logits:
            logits = self.lm_head(hidden_states[:, [-1], :]).float()
        else:
            logits = None

        if return_dict:
            return Qwen3MoeCausalLMOutput(
                logits=logits,
                loss=total_loss,
                lm_loss=lm_loss,
                aux_loss=aux_loss,
                router_logits=router_logits,
            )
        return logits, total_loss

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
# Qwen3-MoE model definitions are inlined above from model_qwen3_moe.py.

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
    batch_size : int = 512 # 8 GPUs x 64 sequences/GPU; gradient accumulation is 1
    device_batch_size : int = 64 # batch size, in sequences, per device (BENCHMARK: B64)
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 10200 # full training run
    embed_learning_rate : float = 0.0036
    warmup_iters : int = 500
    warmdown_iters : int = 2900 # 2x the prior 250/1450 large-batch schedule
    weight_decay : float = 0.0 # weight decay for block weights and tied wte/lm_head; RMSNorm gamma is excluded
    norm_control_start_step : int = 500 # capture reference RMS after warmup optimizer update
    norm_control_log_every : int = 10
    norm_control_eps : float = 1e-12
    # evaluation and logging hyperparams
    run_validation : int = 0 # training-only run; set to 1 to construct/use the validation loader
    val_loss_every : int = 0 # positive cadence is used only when run_validation=1
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    # Heavy tensor/update monitors stay off; norm-control writes its own compact history.
    tensor_norm_every : int = 1 # log tensor norms every step
    adamw_update_norm_every : int = 0 # BENCHMARK: disabled for throughput measurement
    activation_probe_every : int = 0 # every how many steps to log fixed-probe activation RMS ratios? 0 disables
    spectral_norm_estimate_enabled : int = 0 # heavy spectral estimates stay off for training-only observation
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
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = None
val_steps = 0
if args.run_validation:
    assert args.val_loss_every > 0
    assert args.val_tokens % (B * T * ddp_world_size) == 0
    val_steps = args.val_tokens // (B * T * ddp_world_size)
    val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    if val_loader is not None:
        print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# Keep the existing GPT-2 tokenizer/data, tied embedding, optimizer schedule,
# and active expert width while using finer-grained 32-expert top-4 Qwen3-MoE.
# output_router_logits must stay enabled: Qwen3MoeForCausalLM then returns
# total_loss = lm_loss + router_aux_loss_coef * load_balancing_loss.
qwen3_moe_config = Qwen3MoeConfig(
    vocab_size=50_304,
    hidden_size=768,
    intermediate_size=2_304,
    moe_intermediate_size=576,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,
    head_dim=128,
    num_experts=32,
    num_experts_per_tok=4,
    decoder_sparse_step=1,
    mlp_only_layers=(),
    norm_topk_prob=True,
    router_aux_loss_coef=0.001,
    output_router_logits=True,
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
    ignore_index=-1,
)
if not qwen3_moe_config.output_router_logits:
    raise RuntimeError("Qwen3-MoE training requires output_router_logits=True for the auxiliary loss")
if qwen3_moe_config.router_aux_loss_coef <= 0:
    raise RuntimeError("Qwen3-MoE training requires a positive router_aux_loss_coef")
model = Qwen3MoeForCausalLM(qwen3_moe_config)
expected_total_parameters = 586_307_328
actual_total_parameters = model.num_parameters()
if actual_total_parameters != expected_total_parameters:
    raise RuntimeError(
        f"expected {expected_total_parameters:,} parameters for the 586M preset, "
        f"found {actual_total_parameters:,}"
    )
if master_process:
    print(f"Qwen3-MoE total parameter count: {actual_total_parameters:,}")
    print("Qwen3-MoE active parameter count: 140,400,384")
    print(f"Router auxiliary loss: coefficient={qwen3_moe_config.router_aux_loss_coef}, "
          f"experts={qwen3_moe_config.num_experts}, top_k={qwen3_moe_config.num_experts_per_tok}")
    print("Qwen3-MoE expert backend: MegaBlocks grouped GEMM (gg.ops.gmm, @torch.compiler.disable)")
model = model.cuda()
# Keep a reference to the uncompiled model for parameter group construction.
# torch.compile wraps the model; DDP then wraps the compiled model.
# We need `raw_model` pointing to the ORIGINAL for parameter access.
raw_model = model  # uncompiled original, used for parameter group setup
# Apply torch.compile with fullgraph=False:
if master_process:
    print("Applying torch.compile(fullgraph=False) ...")
compiled_model = torch.compile(model, fullgraph=False)
if master_process:
    print("torch.compile applied (fullgraph=False, expert block disabled).")
# Wrap compiled model in DDP
if use_ddp:
    model = DDP(compiled_model, device_ids=[ddp_local_rank])
else:
    model = compiled_model
model_for_structure = raw_model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer(s)
def canonical_param_name(name):
    return name

def is_qwen3_moe_rmsnorm_gamma_name(name):
    name = canonical_param_name(name)
    return (
        name == 'model.norm.weight'
        or name.endswith('.self_attn.q_norm.weight')
        or name.endswith('.self_attn.k_norm.weight')
        or name.endswith('.input_layernorm.weight')
        or name.endswith('.post_attention_layernorm.weight')
    )

# Four attention matrices plus router/expert tensors (or three dense-MLP matrices)
# per layer. Stacked expert parameters are 3D and must remain in AdamW.
block_weight_parameters = [
    p
    for name, p in raw_model.named_parameters()
    if canonical_param_name(name).startswith('model.layers.')
    and p.ndim in (2, 3)
]
rmsnorm_gamma_parameters = [
    p
    for name, p in raw_model.named_parameters()
    if is_qwen3_moe_rmsnorm_gamma_name(name)
]
lm_head_parameters = list(model_for_structure.lm_head.parameters())

expected_block_tensor_count = qwen3_moe_config.num_hidden_layers * 7
expected_norm_gamma_count = qwen3_moe_config.num_hidden_layers * 4 + 1
if len(block_weight_parameters) != expected_block_tensor_count:
    raise RuntimeError(
        f"expected {expected_block_tensor_count} Qwen3-MoE block tensors, "
        f"found {len(block_weight_parameters)}"
    )
if len(rmsnorm_gamma_parameters) != expected_norm_gamma_count:
    raise RuntimeError(
        f"expected {expected_norm_gamma_count} Qwen3-MoE RMSNorm gamma tensors, "
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

if model_for_structure.lm_head.weight is not model_for_structure.model.embed_tokens.weight:
    raise RuntimeError("norm-control experiment requires tied embedding/lm_head weights")
if args.weight_decay != 0.0:
    raise RuntimeError("this norm-control experiment requires AdamW weight_decay=0.0")

NORM_CONTROL_SCHEDULE = 'linear_down_1_to_0p5'

def norm_control_tensor(entry):
    parameter = entry['param']
    slice_index = entry.get('slice_index')
    if slice_index is None:
        return parameter
    return parameter[slice_index]

def build_all_matrix_norm_control_state(model):
    controlled = []
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.ndim not in (2, 3):
            continue
        base_name = canonical_param_name(name)
        if parameter.ndim == 2:
            key = (id(parameter), None)
            if key in seen:
                raise RuntimeError(f"duplicate 2D norm-control parameter: {base_name}")
            seen.add(key)
            controlled.append(dict(
                name=base_name,
                parent_name=base_name,
                param=parameter,
                slice_index=None,
                target_rms=None,
                captured=False,
            ))
        else:
            for expert_idx in range(parameter.shape[0]):
                key = (id(parameter), expert_idx)
                if key in seen:
                    raise RuntimeError(f"duplicate expert norm-control slice: {base_name}[{expert_idx}]")
                seen.add(key)
                controlled.append(dict(
                    name=f"{base_name}[expert_{expert_idx:02d}]",
                    parent_name=base_name,
                    param=parameter,
                    slice_index=expert_idx,
                    target_rms=None,
                    captured=False,
                ))

    expected_count = 1 + qwen3_moe_config.num_hidden_layers * (5 + 2 * qwen3_moe_config.num_experts)
    if len(controlled) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} unique norm-control matrices/slices, found {len(controlled)}"
        )

    controlled_parent_ids = {id(entry['param']) for entry in controlled}
    expected_parent_ids = {id(parameter) for parameter in lm_head_parameters + block_weight_parameters}
    if controlled_parent_ids != expected_parent_ids:
        missing = [
            canonical_param_name(name)
            for name, parameter in model.named_parameters()
            if id(parameter) in expected_parent_ids - controlled_parent_ids
        ]
        unexpected = sorted({entry['parent_name'] for entry in controlled if id(entry['param']) not in expected_parent_ids})
        raise RuntimeError(
            f"norm-control routing mismatch: missing={missing}, unexpected={unexpected}"
        )

    return dict(
        enabled=True,
        mode='warmup_end_captured_per_matrix_schedule',
        schedule=NORM_CONTROL_SCHEDULE,
        start_step=args.norm_control_start_step,
        eps=args.norm_control_eps,
        log_every=args.norm_control_log_every,
        params=controlled,
    )

norm_control_state = build_all_matrix_norm_control_state(raw_model)

optimizer1 = torch.optim.AdamW(lm_head_parameters, lr=args.embed_learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)
optimizer2_groups = [dict(params=block_weight_parameters, weight_decay=args.weight_decay, norm_control_group=True)]
if rmsnorm_gamma_parameters:
    optimizer2_groups.append(dict(params=rmsnorm_gamma_parameters, weight_decay=0.0, norm_control_group=False))
optimizer2 = torch.optim.AdamW(optimizer2_groups, lr=0.5*args.embed_learning_rate, betas=(0.9, 0.95),
                               fused=True)
optimizers = [optimizer1, optimizer2]

# learning rate decay scheduler (linear warmup and warmdown)
def schedule_ratio(update_step):
    start_step = norm_control_state['start_step']
    if start_step is None or update_step <= start_step:
        return 1.0
    if NORM_CONTROL_SCHEDULE == 'constant':
        return 1.0
    if NORM_CONTROL_SCHEDULE == 'cosine_wave_period10000_amp0p5':
        period_steps = 10000
        phase = 2.0 * math.pi * (update_step - start_step) / period_steps
        return 1.0 + 0.5 * math.cos(phase - 0.5 * math.pi)
    progress = (update_step - start_step) / max(1, args.num_iterations - start_step)
    progress = min(1.0, max(0.0, progress))
    if NORM_CONTROL_SCHEDULE == 'linear_up_1_to_2':
        return 1.0 + progress
    if NORM_CONTROL_SCHEDULE == 'linear_down_1_to_0p5':
        return 1.0 - 0.5 * progress
    raise RuntimeError(f"unknown norm-control schedule: {NORM_CONTROL_SCHEDULE}")

def get_wsd_lr(it):
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

def get_lr(it):
    return get_wsd_lr(it) * schedule_ratio(it)

optimizer2_lr_lambdas = [get_lr for _ in optimizer2.param_groups]
if rmsnorm_gamma_parameters:
    optimizer2_lr_lambdas[-1] = get_wsd_lr
schedulers = [
    torch.optim.lr_scheduler.LambdaLR(optimizer1, get_lr),
    torch.optim.lr_scheduler.LambdaLR(optimizer2, optimizer2_lr_lambdas),
]

@torch.no_grad()
def apply_rms_norm_control(state, step, event):
    start_step = state['start_step']
    if step < start_step:
        phase = 'pre_start'
    elif step == start_step:
        phase = 'capture'
    else:
        phase = 'post_start'

    should_log = (
        master_process
        and state['log_every'] > 0
        and (
            step % state['log_every'] == 0
            or step == args.num_iterations
            or event == 'initial'
            or phase == 'capture'
        )
    )
    history_path = os.path.join(logdir, 'norm_control_history.jsonl') if should_log else None
    history_file = open(history_path, 'a') if should_log else None
    captured_any = False
    try:
        for entry in state['params']:
            parameter = norm_control_tensor(entry)
            rms_before = parameter.detach().float().square().mean().sqrt()
            base_target_rms = entry['target_rms']
            target_rms = base_target_rms
            ratio = 1.0
            projected = phase == 'post_start'
            captured = phase == 'capture'

            if captured:
                target_rms = rms_before.item()
                entry['target_rms'] = target_rms
                entry['captured'] = True
                base_target_rms = target_rms
                captured_any = True

            if projected:
                if base_target_rms is None:
                    raise RuntimeError(f"missing captured RMS for norm-control tensor {entry['name']}")
                ratio = schedule_ratio(step)
                target_rms = base_target_rms * ratio
                if rms_before > state['eps']:
                    target = torch.tensor(target_rms, dtype=torch.float32, device=parameter.device)
                    scale = target / rms_before
                    parameter.mul_(scale.to(dtype=parameter.dtype, device=parameter.device))
                else:
                    scale = torch.ones((), dtype=torch.float32, device=parameter.device)
            else:
                scale = torch.ones((), dtype=torch.float32, device=parameter.device)

            rms_after = parameter.detach().float().square().mean().sqrt()
            if history_file is not None:
                relative_error = None
                if target_rms is not None:
                    target = torch.tensor(target_rms, dtype=torch.float32, device=parameter.device)
                    relative_error = ((rms_after - target).abs() / target).item()
                history_file.write(json.dumps(dict(
                    step=step,
                    event=event,
                    mode=state['mode'],
                    schedule=state['schedule'],
                    phase=phase,
                    name=entry['name'],
                    parent_name=entry['parent_name'],
                    slice_index=entry['slice_index'],
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
        if history_file is not None:
            history_file.close()

    if captured_any:
        write_norm_control_targets(state)

def write_norm_control_targets(state):
    if not master_process:
        return
    records = [
        dict(
            name=entry['name'],
            parent_name=entry['parent_name'],
            slice_index=entry['slice_index'],
            target_rms=entry['target_rms'],
            captured=entry['captured'],
            start_step=state['start_step'],
            mode=state['mode'],
            schedule=state['schedule'],
        )
        for entry in state['params']
        if entry['target_rms'] is not None
    ]
    with open(os.path.join(logdir, 'norm_control_targets.json'), 'w') as target_file:
        json.dump(records, target_file, indent=2)

def write_norm_control_metadata(state):
    if not master_process:
        return
    metadata = dict(
        enabled=True,
        mode=state['mode'],
        schedule=state['schedule'],
        norm_type='rms',
        start_step=state['start_step'],
        eps=state['eps'],
        log_every=state['log_every'],
        controlled_count=len(state['params']),
        controlled_parent_tensor_count=len({id(entry['param']) for entry in state['params']}),
        expert_slices_are_controlled_individually=True,
        includes_tied_embedding=True,
        includes_moe_experts=True,
        controlled_parameters=[
            dict(
                name=entry['name'],
                parent_name=entry['parent_name'],
                slice_index=entry['slice_index'],
                target_rms=entry['target_rms'],
                captured=entry['captured'],
                shape=list(norm_control_tensor(entry).shape),
                parent_shape=list(entry['param'].shape),
                ndim=norm_control_tensor(entry).ndim,
                parent_ndim=entry['param'].ndim,
                weight_decay=0.0,
            )
            for entry in state['params']
        ],
    )
    with open(os.path.join(logdir, 'norm_control_metadata.json'), 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

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
            mlp_hidden = output[0] if isinstance(output, tuple) else output
            mlp_out = mlp_hidden.detach()
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
    if val_loader is None:
        raise RuntimeError("activation probes require run_validation=1 and a validation loader")
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
            moe_intermediate_size=model_for_structure.config.moe_intermediate_size,
            num_experts=model_for_structure.config.num_experts,
            num_experts_per_tok=model_for_structure.config.num_experts_per_tok,
            decoder_sparse_step=model_for_structure.config.decoder_sparse_step,
            router_aux_loss_coef=model_for_structure.config.router_aux_loss_coef,
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
apply_rms_norm_control(norm_control_state, step=0, event='initial')
write_norm_control_metadata(norm_control_state)
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
    if args.run_validation and (
        last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
    ):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx:
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

    # The extra sentinel step saves the final checkpoint and final training-only
    # telemetry, then exits without running an evaluation forward.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    train_loss = torch.zeros((), device=device, dtype=torch.float32)
    train_lm_loss = torch.zeros((), device=device, dtype=torch.float32)
    train_aux_loss = torch.zeros((), device=device, dtype=torch.float32)
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            model_output = model(
                x,
                y,
                return_logits=False,
                return_dict=True,
            )
            if (
                model_output.loss is None
                or model_output.lm_loss is None
                or model_output.aux_loss is None
            ):
                raise RuntimeError("MoE training requires LM, auxiliary, and total losses")
            loss = model_output.loss
            train_loss += loss.detach() / train_accumulation_steps
            train_lm_loss += model_output.lm_loss.detach() / train_accumulation_steps
            train_aux_loss += model_output.aux_loss.detach() / train_accumulation_steps
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            no_sync = model.no_sync() if use_ddp else contextlib.nullcontext()
            with no_sync: # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
        del model_output, loss
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # Apply AdamW first, then project controlled matrices, then advance LR schedules.
    update_step = step + 1
    adamw_update_state = maybe_capture_adamw_update_state(update_step)
    for opt in optimizers:
        opt.step()
    maybe_log_adamw_update_norms(update_step, adamw_update_state)
    apply_rms_norm_control(norm_control_state, step=update_step, event='post_step')
    for sched in schedulers:
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(
            f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} "
            f"train_lm_loss:{train_lm_loss.item():.4f} "
            f"train_aux_loss:{train_aux_loss.item():.4f} "
            f"train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms"
        )
        with open(logfile, "a") as f:
            f.write(
                f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} "
                f"train_lm_loss:{train_lm_loss.item():.4f} "
                f"train_aux_loss:{train_aux_loss.item():.4f} "
                f"train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n"
            )

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
if use_ddp:
    dist.destroy_process_group()
