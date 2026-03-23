# Copyright (c) 2024. All rights reserved.
# Magnitude-Gated Transformer Model Implementation
"""
MagGated Transformer: Magnitude-Direction decomposition with dynamic gating.

Key idea: Replace nn.Linear with MagGatedLinear:
    y = m ⊙ g(x) ⊙ (V̂·x)
where:
    V̂ = direction matrix (row-normalized weights)
    m = static magnitude (per output dim)
    g(x) = sigmoid(B(A(x))) = dynamic input-dependent gate

This allows selective dimension activation, enabling smaller hidden sizes.
"""

import math
from typing import List, Optional, Tuple, Union

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

from .configuration_mag_gated import MagGatedConfig

logger = logging.get_logger(__name__)


# ==============================================================================
# Core: MagGatedLinear — replaces nn.Linear
# ==============================================================================

class MagGatedLinear(nn.Module):
    """Magnitude-Gated Linear layer: y = m ⊙ g(x) ⊙ (V·x)

    Decomposes a linear transformation into:
    - V: Direction matrix (learned, standard linear weights)
    - m: Static magnitude per output dimension
    - g(x): Dynamic gate computed via low-rank projection

    When g_j(x) → 0, the j-th output dimension is silenced.
    This enables dimension-level routing / time-multiplexing.
    """

    def __init__(self, d_in: int, d_out: int, rank: int = 16,
                 bias: bool = False, gate_init_bias: float = 3.0,
                 gate_floor: float = 0.05,
                 use_weight_norm: bool = False, use_gate_norm: bool = True):
        super().__init__()
        self.use_weight_norm = use_weight_norm
        self.use_gate_norm = use_gate_norm
        self.gate_floor = gate_floor

        # === Direction: standard linear (acts as V̂) ===
        self.V = nn.Linear(d_in, d_out, bias=bias)

        # === Magnitude: per-output-dim static scale ===
        # Calibrate so initial output scale ≈ standard Linear
        # With gate_init_bias=3.0: sigmoid(3)≈0.953
        # With gate_floor=0.05: effective_gate = 0.953*(1-0.05)+0.05 ≈ 0.955
        # So m should be ≈ 1/0.955 ≈ 1.047 to compensate
        if use_weight_norm:
            std = 0.02
            initial_m = std * math.sqrt(d_in) / self._initial_effective_gate(gate_init_bias, gate_floor)
        else:
            initial_m = 1.0 / self._initial_effective_gate(gate_init_bias, gate_floor)
        self.m = nn.Parameter(torch.full((d_out,), initial_m))

        # === Dynamic Gate: low-rank projection + sigmoid ===
        # g(x) = σ(B(Norm(A(x))) + b)
        self.gate_A = nn.Linear(d_in, rank, bias=False)
        self.gate_norm = MagGatedRMSNorm(rank) if use_gate_norm else nn.Identity()
        self.gate_B = nn.Linear(rank, d_out, bias=True)

        # Initialize gate: sigmoid(3.0) ≈ 0.95 → most dims initially ON
        nn.init.zeros_(self.gate_B.weight)
        nn.init.constant_(self.gate_B.bias, gate_init_bias)
        # Small init for gate_A to keep initial gates near sigmoid(bias)
        nn.init.normal_(self.gate_A.weight, std=0.01)

    @staticmethod
    def _initial_effective_gate(gate_init_bias: float, gate_floor: float) -> float:
        """Compute effective gate value at initialization for m calibration."""
        raw = torch.sigmoid(torch.tensor(gate_init_bias)).item()
        return raw * (1.0 - gate_floor) + gate_floor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_in)
        Returns:
            y: (batch, seq_len, d_out)
        """
        if self.use_weight_norm:
            # L2 normalize each row of V (dimension 1 is d_in for Linear weights [d_out, d_in])
            V_weight = F.normalize(self.V.weight, p=2, dim=1)
            direction = F.linear(x, V_weight, self.V.bias)
        else:
            direction = self.V(x)                       # (B, T, d_out)

        gate_raw = torch.sigmoid(self.gate_B(self.gate_norm(self.gate_A(x))))  # (B, T, d_out)
        # P0: gate floor prevents dead dimensions — minimum 5% signal always passes
        gate = gate_raw * (1.0 - self.gate_floor) + self.gate_floor  # (B, T, d_out) ∈ [0.05, 1.0]
        return self.m * gate * direction            # (B, T, d_out)

    @property
    def weight(self):
        """For compatibility with code that accesses .weight"""
        return self.V.weight

    @property
    def in_features(self):
        return self.V.in_features

    @property
    def out_features(self):
        return self.V.out_features


def _make_linear(d_in: int, d_out: int, config: MagGatedConfig,
                 use_gate: bool, bias: bool = False) -> nn.Module:
    """Factory: create MagGatedLinear or standard nn.Linear based on config."""
    if config.use_mag_gate and use_gate:
        return MagGatedLinear(
            d_in, d_out,
            rank=config.gate_rank,
            bias=bias,
            gate_init_bias=config.gate_init_bias,
            gate_floor=getattr(config, "gate_floor", 0.05),
            use_weight_norm=getattr(config, "use_weight_norm", False),
            use_gate_norm=getattr(config, "use_gate_norm", True),
        )
    else:
        return nn.Linear(d_in, d_out, bias=bias)


# ==============================================================================
# Residual Forgetting Gate
# ==============================================================================

class ResidualGate(nn.Module):
    """Forgetting gate for residual connections.

    f(h) ∈ (0,1)^d, applied as: h_new = f(h) ⊙ h + o
    where o is the output from attention or FFN.

    This allows the model to selectively forget/retain dimensions in the
    residual stream, enabling more efficient dimension utilization.
    """

    def __init__(self, hidden_size: int, rank: int = 16, init_bias: float = 4.0):
        super().__init__()
        # Low-rank gate: g(x) = σ(B(A(x)) + b)
        self.gate_A = nn.Linear(hidden_size, rank, bias=False)
        self.gate_B = nn.Linear(rank, hidden_size, bias=True)

        # P0-2: Initialize with high bias so forget gate starts near 1.0
        # sigmoid(4.0) ≈ 0.982 → conservative retention
        # With 28 layers × 2 gates: 0.982^56 ≈ 0.36 (reasonable information retention)
        nn.init.zeros_(self.gate_B.weight)
        nn.init.constant_(self.gate_B.bias, init_bias)
        nn.init.normal_(self.gate_A.weight, std=0.01)

    def forward(self, residual: torch.Tensor, new_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residual: (B, T, d) - the residual stream
            new_output: (B, T, d) - output from attention/FFN sub-layer
        Returns:
            updated: (B, T, d) - gated residual + new output
        """
        forget_gate = torch.sigmoid(self.gate_B(self.gate_A(residual)))
        return forget_gate * residual + new_output


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
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ==============================================================================
# Attention
# ==============================================================================

def _should_gate(name: str, positions: str) -> bool:
    """Decide whether a given projection should use MagGated."""
    if positions == "none":
        return False
    if positions == "all":
        return True
    if positions == "bottleneck":
        # Only gate the write-back projections: o_proj and down_proj
        return name in ("o_proj", "down_proj")
    return False


class MagGatedAttention(nn.Module):
    """Multi-head attention with optional MagGated projections."""

    def __init__(self, config: MagGatedConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.attention_dropout = config.attention_dropout

        pos = config.mag_gate_positions

        self.q_proj = _make_linear(
            self.hidden_size, self.num_heads * self.head_dim, config,
            use_gate=_should_gate("q_proj", pos), bias=config.attention_bias)
        self.k_proj = _make_linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, config,
            use_gate=_should_gate("k_proj", pos), bias=config.attention_bias)
        self.v_proj = _make_linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, config,
            use_gate=_should_gate("v_proj", pos), bias=config.attention_bias)
        self.o_proj = _make_linear(
            self.num_heads * self.head_dim, self.hidden_size, config,
            use_gate=_should_gate("o_proj", pos), bias=config.attention_bias)

        self.rotary_emb = MagGatedRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # GQA: repeat kv heads
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # SDPA
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None


# ==============================================================================
# MLP (SwiGLU)
# ==============================================================================

class MagGatedMLP(nn.Module):
    """SwiGLU MLP with optional MagGated projections.

    Note: The gate_proj in SwiGLU already provides gating, so we don't
    add MagGated to it (would be redundant). Only up_proj and down_proj
    get MagGated treatment when configured.
    """

    def __init__(self, config: MagGatedConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        pos = config.mag_gate_positions

        # gate_proj: SwiGLU's own gate — no MagGated (already gated)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)

        # up_proj: can be MagGated
        self.up_proj = _make_linear(
            self.hidden_size, self.intermediate_size, config,
            use_gate=_should_gate("up_proj", pos), bias=config.mlp_bias)

        # down_proj: ★ key bottleneck — writes back to residual
        self.down_proj = _make_linear(
            self.intermediate_size, self.hidden_size, config,
            use_gate=_should_gate("down_proj", pos), bias=config.mlp_bias)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ==============================================================================
# Decoder Layer
# ==============================================================================

class MagGatedDecoderLayer(nn.Module):
    """Transformer decoder layer with MagGated Linear and residual gating.

    Structure:
        h → norm1 → MagGated_Attn → [residual_gate] → h + attn_out
        h → norm2 → MagGated_MLP  → [residual_gate] → h + ffn_out
    """

    def __init__(self, config: MagGatedConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MagGatedAttention(config=config, layer_idx=layer_idx)
        self.mlp = MagGatedMLP(config=config)
        self.input_layernorm = MagGatedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MagGatedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Residual forgetting gates
        self.use_residual_gate = config.use_residual_gate
        if self.use_residual_gate:
            residual_bias = getattr(config, 'residual_gate_init_bias', 4.0)
            self.attn_residual_gate = ResidualGate(
                config.hidden_size, rank=config.residual_gate_rank, init_bias=residual_bias)
            self.ffn_residual_gate = ResidualGate(
                config.hidden_size, rank=config.residual_gate_rank, init_bias=residual_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        attn_output, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        # Residual with forgetting gate
        if self.use_residual_gate:
            hidden_states = self.attn_residual_gate(residual, attn_output)
        else:
            hidden_states = residual + attn_output

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        ffn_output = self.mlp(hidden_states)

        # Residual with forgetting gate
        if self.use_residual_gate:
            hidden_states = self.ffn_residual_gate(residual, ffn_output)
        else:
            hidden_states = residual + ffn_output

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# ==============================================================================
# Full Model
# ==============================================================================

class MagGatedPreTrainedModel(PreTrainedModel):
    config_class = MagGatedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MagGatedDecoderLayer"]
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, MagGatedLinear):
            nn.init.normal_(module.V.weight, mean=0.0, std=std)
            if module.V.bias is not None:
                nn.init.zeros_(module.V.bias)
            # P1-3: m is calibrated in MagGatedLinear.__init__ to compensate gate
            # Don't override here — the __init__ already computed the correct value
            # gate_A and gate_B are also initialized in MagGatedLinear.__init__
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MagGatedModel(MagGatedPreTrainedModel):
    """MagGated Transformer model (decoder-only, no LM head)."""

    def __init__(self, config: MagGatedConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MagGatedDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MagGatedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Handle cache
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create causal mask
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values
        )

        hidden_states = inputs_embeds

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
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None
        if return_legacy_cache and next_cache is not None:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values):
        """Create 4D causal mask from 2D attention_mask."""
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = (
            cache_position[-1].item() + 1
            if past_key_values is not None
            else sequence_length
        )

        # Create causal mask
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
    """MagGated Transformer with causal language model head."""

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
            # Shift so that tokens < n predict n
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
        # If we have cache, only take the last token
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
