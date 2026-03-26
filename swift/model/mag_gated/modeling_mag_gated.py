# Copyright (c) 2024. All rights reserved.
# ResidualGate Transformer Model Implementation
"""
ResidualGate Transformer: Qwen3-compatible decoder-only model with dual-gated
magnitude-aware residual connections.

Core mechanism (replaces standard h = h + o):
    h_new = α(h, o) ⊙ h + β(h, o) ⊙ o

where α and β are independently computed per-dimension gates informed by
DoRA-inspired magnitude/direction signals.

All linear layers are standard nn.Linear — the ONLY architectural difference
from Qwen3 is the ResidualGate at each residual connection.

Supports:
- Flash Attention 2/3 via transformers ALL_ATTENTION_FUNCTIONS dispatch
- Packing/padding_free training via FlashAttentionKwargs (cu_seq_lens_q/k)
- SDPA, eager, flex_attention backends
"""

import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.generation import GenerationMixin
from transformers.utils import logging

try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
except ImportError:
    FlashAttentionKwargs = None

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:
    ALL_ATTENTION_FUNCTIONS = None

try:
    from transformers.masking_utils import create_causal_mask
except ImportError:
    create_causal_mask = None

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    GradientCheckpointingLayer = nn.Module

try:
    from transformers.processing_utils import Unpack
except ImportError:
    Unpack = None

try:
    from transformers.utils import TransformersKwargs
except ImportError:
    TransformersKwargs = None

from .configuration_mag_gated import MagGatedConfig

logger = logging.get_logger(__name__)




# ==============================================================================
# ResidualGate — the ONLY architectural innovation
# ==============================================================================

class ResidualGate(nn.Module):
    """Dual-gate magnitude-aware residual connection.

    h_new = α(h, o) ⊙ h + β(h, o) ⊙ o

    where:
        α ∈ (0,1)^d = retain gate (how much old info to keep per dimension)
        β ∈ (0,1)^d = accept gate (how much new info to accept per dimension)

    Gate input is concat([h, o, mag_ratio, dir_agree]) → 4d, compressed through
    a shared low-rank bottleneck (Gate_A: 4d→rank) then split into two independent
    projections (Gate_B_alpha: rank→d, Gate_B_beta: rank→d).

    DoRA-inspired signals (detached, no gradient):
        mag_ratio = |h| / (|h| + |o| + ε)   — relative strength per dimension
        dir_agree = (h·o) / (|h|·|o| + ε)   — directional agreement per dimension

    init_bias=5.0 → sigmoid(5)≈0.993 → initial h_new ≈ 0.993h + 0.993o ≈ h + o
    """

    def __init__(self, hidden_size: int, rank: int = 16, init_bias: float = 5.0):
        super().__init__()
        self.hidden_size = hidden_size

        # Gate input: concat(h, o, mag_ratio, dir_agree) → 4d
        gate_input_size = hidden_size * 4

        # Shared low-rank projection (parameter efficient)
        self.gate_A = nn.Linear(gate_input_size, rank, bias=False)
        # Two independent output projections for α and β
        self.gate_B_alpha = nn.Linear(rank, hidden_size, bias=True)
        self.gate_B_beta = nn.Linear(rank, hidden_size, bias=True)

        # Careful initialization:
        # - gate_B weights ≈ 0 (small random noise for symmetry breaking)
        # - gate_B bias = init_bias → sigmoid(5.0) ≈ 0.993 (near-identity start)
        # - gate_A weights: moderate std for gradient flow
        nn.init.normal_(self.gate_B_alpha.weight, std=0.01)
        nn.init.constant_(self.gate_B_alpha.bias, init_bias)
        nn.init.normal_(self.gate_B_beta.weight, std=0.01)
        nn.init.constant_(self.gate_B_beta.bias, init_bias)
        nn.init.normal_(self.gate_A.weight, std=0.02)

        # Gate monitoring (lightweight, no gradient)
        self._gate_stats: Optional[dict] = None

    def forward(self, residual: torch.Tensor, new_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residual: (B, T, d) - the residual stream (h)
            new_output: (B, T, d) - output from attention/FFN sub-layer (o)
        Returns:
            updated: (B, T, d) - dual-gated combination α⊙h + β⊙o
        """
        # DoRA-inspired signals (DETACHED — read-only physics sensors, no gradient)
        eps = 1e-6
        h_mag = residual.detach().abs()
        o_mag = new_output.detach().abs()
        mag_ratio = h_mag / (h_mag + o_mag + eps)                              # ∈ [0, 1]
        dir_agree = (residual.detach() * new_output.detach()) / (h_mag * o_mag + eps)  # ∈ [-1, 1]

        # Gate input: h and o carry gradient; mag/dir are detached hints
        gate_input = torch.cat([residual, new_output, mag_ratio, dir_agree], dim=-1)  # (B, T, 4d)

        # Shared low-rank compression → independent α and β
        gate_hidden = self.gate_A(gate_input)                                  # (B, T, rank)
        alpha = torch.sigmoid(self.gate_B_alpha(gate_hidden))                  # (B, T, d)
        beta = torch.sigmoid(self.gate_B_beta(gate_hidden))                    # (B, T, d)

        # Record gate statistics for monitoring
        if self.training:
            with torch.no_grad():
                a = alpha.detach().float()
                b = beta.detach().float()
                self._gate_stats = {
                    "mean": a.mean().item(),
                    "std": a.std().item(),
                    "min": a.min().item(),
                    "max": a.max().item(),
                    "sparsity": (a < 0.1).float().mean().item(),
                    "saturation": (a > 0.9).float().mean().item(),
                    "forget_ratio": (a < 0.5).float().mean().item(),
                    "dim_mean": a.mean(dim=(0, 1)).cpu(),
                    "beta_mean": b.mean().item(),
                    "beta_std": b.std().item(),
                    "beta_min": b.min().item(),
                    "beta_max": b.max().item(),
                    "beta_sparsity": (b < 0.1).float().mean().item(),
                    "beta_saturation": (b > 0.9).float().mean().item(),
                }

        return alpha * residual + beta * new_output


# ==============================================================================
# RMSNorm
# ==============================================================================

class MagGatedRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ==============================================================================
# Rotary Position Embedding
# ==============================================================================

class MagGatedRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ==============================================================================
# Eager attention fallback
# ==============================================================================

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# ==============================================================================
# Attention (standard nn.Linear, identical to Qwen3)
# ==============================================================================

class MagGatedAttention(nn.Module):
    """Multi-head attention with standard nn.Linear projections.

    Supports flash_attention_2/3, sdpa, eager, flex_attention
    via transformers ALL_ATTENTION_FUNCTIONS dispatch.
    """

    def __init__(self, config: MagGatedConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = MagGatedRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if ALL_ATTENTION_FUNCTIONS is not None and self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


# ==============================================================================
# MLP — SwiGLU (standard nn.Linear, identical to Qwen3)
# ==============================================================================

class MagGatedMLP(nn.Module):
    """SwiGLU MLP with standard nn.Linear projections."""

    def __init__(self, config: MagGatedConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ==============================================================================
# Decoder Layer
# ==============================================================================

class MagGatedDecoderLayer(GradientCheckpointingLayer):
    """Transformer decoder layer with ResidualGate.

    Structure:
        h → norm1 → Attn → ResidualGate(h, attn_out) → h'
        h'→ norm2 → MLP  → ResidualGate(h', ffn_out) → h''
    """

    def __init__(self, config: MagGatedConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MagGatedAttention(config=config, layer_idx=layer_idx)
        self.mlp = MagGatedMLP(config=config)
        self.input_layernorm = MagGatedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MagGatedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ResidualGate (the ONLY difference from Qwen3)
        self.use_residual_gate = config.use_residual_gate
        if self.use_residual_gate:
            init_bias = config.residual_gate_init_bias
            rank = config.residual_gate_rank
            self.attn_residual_gate = ResidualGate(config.hidden_size, rank=rank, init_bias=init_bias)
            self.ffn_residual_gate = ResidualGate(config.hidden_size, rank=rank, init_bias=init_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple] = None,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

        # Residual connection: gated or standard
        if self.use_residual_gate:
            hidden_states = self.attn_residual_gate(residual, attn_output)
        else:
            hidden_states = residual + attn_output

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        ffn_output = self.mlp(hidden_states)

        if self.use_residual_gate:
            hidden_states = self.ffn_residual_gate(residual, ffn_output)
        else:
            hidden_states = residual + ffn_output

        return hidden_states


# ==============================================================================
# Full Model
# ==============================================================================

class MagGatedPreTrainedModel(PreTrainedModel):
    config_class = MagGatedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MagGatedDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        # Skip init for internal linear layers of ResidualGate as they are carefully initialized
        if hasattr(module, "_is_residual_gate_linear"):
            return
            
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MagGatedModel(MagGatedPreTrainedModel):
    """ResidualGate Transformer model (decoder-only, no LM head)."""

    def __init__(self, config: MagGatedConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MagGatedDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MagGatedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MagGatedRotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.gradient_checkpointing = False

        # Mark all ResidualGate linears BEFORE post_init() so they are
        # skipped by _init_weights (which is called inside post_init).
        for name, module in self.named_modules():
            if isinstance(module, ResidualGate):
                module.gate_A._is_residual_gate_linear = True
                module.gate_B_alpha._is_residual_gate_linear = True
                module.gate_B_beta._is_residual_gate_linear = True

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if create_causal_mask is not None:
            causal_mask = create_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        else:
            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            if isinstance(layer_outputs, torch.Tensor):
                hidden_states = layer_outputs
            else:
                hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values):
        """Fallback causal mask for older transformers versions."""
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = (
            cache_position[-1].item() + 1
            if past_key_values is not None
            else sequence_length
        )

        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=target_length - sequence_length + 1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.dtype)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class MagGatedForCausalLM(MagGatedPreTrainedModel, GenerationMixin):
    """ResidualGate Transformer with causal language model head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MagGatedConfig):
        super().__init__(config)
        self.model = MagGatedModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if input_ids.shape[1] != 1:
                input_ids = input_ids[:, -1:]

        if cache_position is None:
            past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_length, past_length + input_ids.shape[1], device=input_ids.device
            )

        position_ids = cache_position.unsqueeze(0)

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
        }
        return model_inputs

    # ==================================================================
    # Gate Monitoring API
    # ==================================================================

    def get_gate_stats(self) -> dict:
        """Collect ResidualGate statistics from all layers."""
        stats = {}
        res_alpha_means = []
        res_beta_means = []
        res_sparsities = []
        res_saturations = []
        res_forget_ratios = []

        for layer_idx, layer in enumerate(self.model.layers):
            if not layer.use_residual_gate:
                continue

            for gate_name in ["attn_residual_gate", "ffn_residual_gate"]:
                gate = getattr(layer, gate_name, None)
                if isinstance(gate, ResidualGate) and gate._gate_stats is not None:
                    gs = gate._gate_stats
                    prefix = f"gate/layer{layer_idx}_{gate_name}"

                    # α (retain gate) stats
                    for k in ["mean", "std", "min", "max", "sparsity", "saturation"]:
                        if k in gs:
                            stats[f"{prefix}_{k}"] = gs[k]

                    # β (accept gate) stats
                    prefix_beta = f"gate/layer{layer_idx}_{gate_name}_beta"
                    for k in ["beta_mean", "beta_std", "beta_min", "beta_max", "beta_sparsity", "beta_saturation"]:
                        if k in gs:
                            short_k = k.replace("beta_", "")
                            stats[f"{prefix_beta}_{short_k}"] = gs[k]

                    # Collect for global summary
                    res_alpha_means.append(gs["mean"])
                    if "beta_mean" in gs:
                        res_beta_means.append(gs["beta_mean"])
                    if "sparsity" in gs:
                        res_sparsities.append(gs["sparsity"])
                    if "saturation" in gs:
                        res_saturations.append(gs["saturation"])
                    if "forget_ratio" in gs:
                        res_forget_ratios.append(gs["forget_ratio"])

        # Global summary
        if res_alpha_means:
            stats["gate/residual_alpha_global_mean"] = sum(res_alpha_means) / len(res_alpha_means)
        if res_beta_means:
            stats["gate/residual_beta_global_mean"] = sum(res_beta_means) / len(res_beta_means)
        if res_sparsities:
            stats["gate/residual_global_sparsity"] = sum(res_sparsities) / len(res_sparsities)
        if res_saturations:
            stats["gate/residual_global_saturation"] = sum(res_saturations) / len(res_saturations)
        if res_forget_ratios:
            stats["gate/residual_global_forget_ratio"] = sum(res_forget_ratios) / len(res_forget_ratios)

        return stats

    def reinit_gates(self, residual_gate_init_bias: Optional[float] = None) -> int:
        """Re-initialize all ResidualGate parameters."""
        if residual_gate_init_bias is None:
            residual_gate_init_bias = self.config.residual_gate_init_bias

        count = 0
        for module in self.modules():
            if isinstance(module, ResidualGate):
                nn.init.normal_(module.gate_B_alpha.weight, std=0.01)
                nn.init.constant_(module.gate_B_alpha.bias, residual_gate_init_bias)
                nn.init.normal_(module.gate_B_beta.weight, std=0.01)
                nn.init.constant_(module.gate_B_beta.bias, residual_gate_init_bias)
                nn.init.normal_(module.gate_A.weight, std=0.02)
                count += 1

        logger.info(
            f"[ResidualGate] ✓ Re-initialized {count} gate modules "
            f"(init_bias={residual_gate_init_bias})"
        )
        return count

    def verify_gate_init(self) -> dict:
        """Verify that gate parameters are correctly initialized."""
        expected_bias = self.config.residual_gate_init_bias
        gate_biases = []
        gate_weight_stds = []

        for module in self.modules():
            if isinstance(module, ResidualGate):
                gate_biases.append(module.gate_B_alpha.bias.data.float().mean().item())
                gate_weight_stds.append(module.gate_B_alpha.weight.data.float().std().item())

        result = {
            "gate_count": len(gate_biases),
            "expected_bias": expected_bias,
        }

        if gate_biases:
            avg_bias = sum(gate_biases) / len(gate_biases)
            avg_w_std = sum(gate_weight_stds) / len(gate_weight_stds)
            result["actual_bias_mean"] = avg_bias
            result["actual_weight_std"] = avg_w_std
            result["bias_ok"] = abs(avg_bias - expected_bias) < max(expected_bias * 0.3, 0.3)
            result["weight_ok"] = avg_w_std < 0.05
            result["all_ok"] = result["bias_ok"] and result["weight_ok"]
        else:
            result["all_ok"] = True

        return result
