# Copyright (c) 2024. All rights reserved.
# Attention Hidden-Size Transformer Model Implementation (V12 — Independent Synaptic Gating)
"""
Attention Hidden-Size Transformer: Qwen3-compatible decoder-only model with
Independent Synaptic Gating at each residual connection.

Core mechanism (replaces standard h = h + o):
    gate_forget  = σ(w_forget ⊙ RMSNorm(h) + b_forget)
    gate_acquire = σ(w_acquire ⊙ RMSNorm(o) + b_acquire)
    h_new = gate_forget ⊙ h + gate_acquire ⊙ o

The RMSNorm inside the gate is parameter-free (no learnable weight), used
solely to normalize activation magnitudes before gate computation. This
prevents sigmoid saturation as hidden-state norms grow in deeper layers
(the PreNorm dilution problem). The final combination still uses raw h and o.

Biological inspiration — Independent Synaptic Scaling:
    Each hidden dimension j ∈ {1…d} has its own independent gate parameters.
    Dimension #42's "forget/acquire" decision is made solely based on the signal
    flowing through dimension #42 and its own learned "synaptic sensitivity"
    (w and b). It does NOT need to know what dimension #99 is doing.

    This mimics how a biological neuron's synapse independently decides
    signal strength based on local activity history — no global coordination.

Key design principles:
    1. Pure element-wise operations (⊙): NO matrix multiplication in the gate
    2. Only 4d learnable parameters per gate: w_forget, b_forget, w_acquire, b_acquire
    3. Zero-loss fallback: init w=0, b=+4.0 → σ(4.0)≈0.982 → h_new ≈ h + o
    4. No feature manifold shattering: no cross-channel mixing
    5. SFT-friendly: 4×1024 = 4096 scalars per gate, trivially trainable
    6. No anti-entropy loss needed, no complex routing architecture

Supports:
- Flash Attention 2/3 via transformers ALL_ATTENTION_FUNCTIONS dispatch
- Packing/padding_free training via FlashAttentionKwargs (cu_seq_lens_q/k)
- SDPA, eager, flex_attention backends
- torch.compile (all ops are standard PyTorch)
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

from .configuration_attn_hidden import AttnHiddenConfig

logger = logging.get_logger(__name__)


# ==============================================================================
# SynapticGate — Independent Per-Dimension Synaptic Gating (V12)
# ==============================================================================

class SynapticGate(nn.Module):
    """Independent Synaptic Gate over the hidden-size dimension.

    Core formula (pure element-wise, per dimension j):
        gate_forget_j  = σ(w_forget_j · RMSNorm(h)_j + b_forget_j)
        gate_acquire_j = σ(w_acquire_j · RMSNorm(o)_j + b_acquire_j)
        h_new_j = gate_forget_j · h_j + gate_acquire_j · o_j

    The RMSNorm inside the gate is **parameter-free** (no learnable weight),
    used solely to normalize activation magnitudes before gate computation.
    This prevents sigmoid saturation as hidden-state norms grow in deeper
    layers (the PreNorm dilution problem). The final combination still uses
    the raw (un-normalized) h and o.

    Biological analogy:
        Each dimension is an independent "synapse" that decides:
        - How much of its current signal (h) to retain (gate_forget)
        - How much of the new signal (o) to accept (gate_acquire)
        Based solely on the magnitude of the signal flowing through it
        and its own learned sensitivity (w, b).

    Two initialization modes:
        - "sft" (default): w=zeros, b=+init_bias.
          σ(w·RMSNorm(x) + b) = σ(0 + 4.0) = σ(4.0) ≈ 0.982
          → h_new ≈ 0.982·h + 0.982·o ≈ h + o (exact standard residual).
          Perfect for weight transfer from Qwen3 — zero training disruption.

        - "pretrain": w=small random (std=0.01), b=+init_bias.
          σ(ε·RMSNorm(x) + 4.0) ≈ 0.982 + tiny noise
          → h_new ≈ h + o with tiny perturbation.

    Key properties:
        1. Initial h_new ≈ h + o (zero-loss fallback, no training disruption)
        2. Each dimension independently learns to forget or acquire
        3. No cross-channel interaction (no matrix multiplication)
        4. No feature manifold shattering
        5. Only 4d parameters per gate (extremely lightweight)
        6. Gradient highway: ∂h_new/∂h = gate_forget ≈ 1 initially
        7. RMSNorm stabilizes gate inputs regardless of depth

    Parameters per gate: 4 × d  (RMSNorm is parameter-free)
        With d=1024: 4,096 scalars per gate, 8,192 per layer (attn + ffn)
    """

    def __init__(self, hidden_size: int, init_bias: float = 4.0,
                 init_mode: str = "sft", rms_norm_eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.init_bias = init_bias
        self.init_mode = init_mode
        self.rms_norm_eps = rms_norm_eps

        # === Forget gate parameters (controls retention of h) ===
        self.w_forget = nn.Parameter(torch.zeros(hidden_size))
        self.b_forget = nn.Parameter(torch.full((hidden_size,), init_bias))

        # === Acquire gate parameters (controls acceptance of o) ===
        self.w_acquire = nn.Parameter(torch.zeros(hidden_size))
        self.b_acquire = nn.Parameter(torch.full((hidden_size,), init_bias))

        # === Initialization ===
        if init_mode == "pretrain":
            # Small random weights for slight initial diversity
            nn.init.normal_(self.w_forget, std=0.01)
            nn.init.normal_(self.w_acquire, std=0.01)
        # else "sft": w=zeros already set above

        # Mark all child modules to skip _init_weights
        for child in self.modules():
            if child is not self:
                child._skip_init = True

        # Gate monitoring
        self._gate_raw: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    @staticmethod
    def _rms_norm_no_weight(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Parameter-free RMS normalization (signal conditioning only).

        Normalizes the magnitude of x so that gate computation σ(w⊙x+b)
        stays in a well-behaved range regardless of hidden-state scale.
        No learnable weight — preserves the '4d parameters per gate' property.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def forward(self, residual: torch.Tensor, new_output: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            residual: (*, d) - the residual stream (h)
            new_output: (*, d) - output from attention/FFN sub-layer (o)
        Returns:
            h_new: (*, d) - gate_forget⊙h + gate_acquire⊙o
        """
        # Normalize inputs for gate computation (parameter-free RMSNorm)
        # This prevents sigmoid saturation when hidden-state norms grow
        normed_h = self._rms_norm_no_weight(residual, self.rms_norm_eps)
        normed_o = self._rms_norm_no_weight(new_output, self.rms_norm_eps)

        # Compute gates: σ(w ⊙ RMSNorm(x) + b) — pure element-wise, no matrix mult
        gate_forget = torch.sigmoid(self.w_forget * normed_h + self.b_forget)
        gate_acquire = torch.sigmoid(self.w_acquire * normed_o + self.b_acquire)

        # Gated residual update (uses raw h and o, NOT normalized versions)
        result = gate_forget * residual + gate_acquire * new_output

        # Save gate values for monitoring (training only)
        if self.training:
            self._gate_raw = (gate_forget.detach(), gate_acquire.detach())

        return result

    @property
    def _gate_stats(self) -> Optional[dict]:
        """Compute gate statistics from raw tensors."""
        if self._gate_raw is None:
            return None
        gate_forget, gate_acquire = self._gate_raw
        with torch.no_grad():
            gf = gate_forget.float()
            ga = gate_acquire.float()
            return {
                "forget_mean": gf.mean().item(),
                "forget_std": gf.std().item(),
                "forget_min": gf.min().item(),
                "forget_max": gf.max().item(),
                "acquire_mean": ga.mean().item(),
                "acquire_std": ga.std().item(),
                "acquire_min": ga.min().item(),
                "acquire_max": ga.max().item(),
                # How many dimensions are actively forgetting (gate < 0.9)
                "forget_active": (gf < 0.9).float().mean().item(),
                # How many dimensions are actively blocking (gate < 0.5)
                "forget_blocking": (gf < 0.5).float().mean().item(),
                # How many dimensions are actively suppressing new info (gate < 0.9)
                "acquire_active": (ga < 0.9).float().mean().item(),
                # How many dimensions are actively blocking new info (gate < 0.5)
                "acquire_blocking": (ga < 0.5).float().mean().item(),
            }

    @_gate_stats.setter
    def _gate_stats(self, value):
        if value is None:
            self._gate_raw = None


# Legacy alias for backward compatibility
ResidualGate = SynapticGate


# ==============================================================================
# RMSNorm (standard, same as Qwen3)
# ==============================================================================

class AttnHiddenRMSNorm(nn.Module):
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
# Rotary Position Embedding (standard, same as Qwen3)
# ==============================================================================

class AttnHiddenRotaryEmbedding(nn.Module):
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
# Token-Sequence Attention (standard nn.Linear, identical to Qwen3)
# ==============================================================================

class AttnHiddenAttention(nn.Module):
    """Standard multi-head token-sequence attention with nn.Linear projections.

    This is the standard Qwen3 self-attention (token × token), unchanged.
    The hidden-size dimension gating is in SynapticGate, not here.
    """

    def __init__(self, config: AttnHiddenConfig, layer_idx: int):
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

        # QK-Norm (same as Qwen3): RMSNorm on Q and K per head
        self.q_norm = AttnHiddenRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = AttnHiddenRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = AttnHiddenRotaryEmbedding(
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

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
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

class AttnHiddenMLP(nn.Module):
    """Standard SwiGLU MLP with nn.Linear projections (identical to Qwen3)."""

    def __init__(self, config: AttnHiddenConfig):
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
# Decoder Layer — with Independent Synaptic Gate
# ==============================================================================

class AttnHiddenDecoderLayer(GradientCheckpointingLayer):
    """Transformer decoder layer with Independent Synaptic Gating.

    Structure:
        h → norm1 → TokenAttn → SynapticGate(h, attn_out) → h'
        h'→ norm2 → MLP       → SynapticGate(h', ffn_out) → h''

    The SynapticGate uses independent per-dimension element-wise gating:
        gate_forget  = σ(w_forget ⊙ RMSNorm(h) + b_forget)
        gate_acquire = σ(w_acquire ⊙ RMSNorm(o) + b_acquire)
        h_new = gate_forget ⊙ h + gate_acquire ⊙ o

    RMSNorm inside the gate is parameter-free (signal conditioning only).
    """

    def __init__(self, config: AttnHiddenConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = AttnHiddenAttention(config=config, layer_idx=layer_idx)
        self.mlp = AttnHiddenMLP(config=config)
        self.input_layernorm = AttnHiddenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = AttnHiddenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # SynapticGate (the ONLY difference from Qwen3)
        self.use_synaptic_gate = config.use_synaptic_gate
        if self.use_synaptic_gate:
            init_bias = getattr(config, 'synaptic_gate_init_bias', 4.0)
            init_mode = getattr(config, 'synaptic_gate_init_mode', 'sft')
            self.attn_synaptic_gate = SynapticGate(
                config.hidden_size, init_bias=init_bias, init_mode=init_mode,
                rms_norm_eps=config.rms_norm_eps,
            )
            self.ffn_synaptic_gate = SynapticGate(
                config.hidden_size, init_bias=init_bias, init_mode=init_mode,
                rms_norm_eps=config.rms_norm_eps,
            )

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

        # Residual connection: synaptic gate or standard
        if self.use_synaptic_gate:
            hidden_states = self.attn_synaptic_gate(residual, attn_output)
        else:
            hidden_states = residual + attn_output

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        ffn_output = self.mlp(hidden_states)

        if self.use_synaptic_gate:
            hidden_states = self.ffn_synaptic_gate(residual, ffn_output)
        else:
            hidden_states = residual + ffn_output

        return hidden_states


# ==============================================================================
# Full Model — Attention Hidden-Size Transformer
# ==============================================================================

class AttnHiddenPreTrainedModel(PreTrainedModel):
    config_class = AttnHiddenConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AttnHiddenDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        # SynapticGate handles its own initialization in __init__.
        # Skip any module that is or belongs to a SynapticGate.
        if isinstance(module, SynapticGate):
            return
        if getattr(module, '_skip_init', False):
            return
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_gate_param_groups(self, base_lr: float) -> list:
        """Return optimizer parameter groups with scaled LR for gate parameters."""
        gate_lr_scale = getattr(self.config, 'synaptic_gate_lr_scale', 5.0)
        gate_params = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'synaptic_gate' in name:
                gate_params.append(param)
            else:
                other_params.append(param)

        groups = [
            {"params": other_params, "lr": base_lr},
            {"params": gate_params, "lr": base_lr * gate_lr_scale,
             "weight_decay": 0.0},
        ]
        if gate_params:
            logger.info(
                f"[AttnHidden] Synaptic Gate LR scale: {gate_lr_scale}x "
                f"({len(gate_params)} gate params, {len(other_params)} other params)"
            )
        return groups


class AttnHiddenModel(AttnHiddenPreTrainedModel):
    """Attention Hidden-Size Transformer (decoder-only, no LM head)."""

    def __init__(self, config: AttnHiddenConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [AttnHiddenDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = AttnHiddenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = AttnHiddenRotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

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
            elif isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
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


class AttnHiddenForCausalLM(AttnHiddenPreTrainedModel, GenerationMixin):
    """Attention Hidden-Size Transformer with causal language model head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: AttnHiddenConfig):
        super().__init__(config)
        self.model = AttnHiddenModel(config)
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

    # NOTE: prepare_inputs_for_generation is NOT overridden here.
    # We rely on GenerationMixin's default implementation which correctly
    # handles cache_position, past_key_values, and input truncation for
    # transformers >= 4.57.

    # ==================================================================
    # Gate Monitoring API
    # ==================================================================

    def get_gate_stats(self) -> dict:
        """Collect SynapticGate statistics from all layers."""
        stats = {}
        forget_means = []
        acquire_means = []

        for layer_idx, layer in enumerate(self.model.layers):
            if not getattr(layer, 'use_synaptic_gate', False):
                continue

            for gate_name in ["attn_synaptic_gate", "ffn_synaptic_gate"]:
                gate = getattr(layer, gate_name, None)
                if isinstance(gate, SynapticGate) and gate._gate_stats is not None:
                    gs = gate._gate_stats
                    prefix = f"gate/layer{layer_idx}_{gate_name}"

                    for k, v in gs.items():
                        stats[f"{prefix}_{k}"] = v

                    if "forget_mean" in gs:
                        forget_means.append(gs["forget_mean"])
                    if "acquire_mean" in gs:
                        acquire_means.append(gs["acquire_mean"])

        if forget_means:
            stats["gate/global_forget_mean"] = sum(forget_means) / len(forget_means)
        if acquire_means:
            stats["gate/global_acquire_mean"] = sum(acquire_means) / len(acquire_means)

        return stats
