# Copyright (c) 2024. All rights reserved.
# Attention Residual Gate Transformer Model Implementation
"""
AttnResGate Transformer: Qwen3-compatible decoder-only model with grouped
attention-based residual gates at each residual connection.

Core mechanism (replaces standard h = h + o):
    h_new = (1 - forget) ⊙ h + accept ⊙ o

where forget and accept are computed via grouped attention over h and o,
following self-attention principles applied to the hidden-size dimension:
    Q: per-group learned query w_q (n_groups × group_size), init small random
    K: per-dim projections key_h = w_kh ⊙ h, key_o = w_ko ⊙ o
    Score: (w_q * key_g).sum(group_dim) / (τ·√group_size)  ← true dot product
    forget = sigmoid(score_h + b_forget)  ← h-based, initially ≈0
    accept = sigmoid(score_o + b_accept)  ← o-based, initially ≈1
    → h_new ≈ h + o at init (near-identity), ∂h_new/∂h ≈ 1 (no grad vanish)

Key design choices for gate trainability:
    - init_bias=1.0 (default): σ(±1) ≈ 0.27/0.73, σ'(±1) ≈ 0.197
      Provides good gradient flow while still being near-identity.
      Avoid init_bias ≥ 3.0 which causes sigmoid saturation (σ'(±3) ≈ 0.045).
    - w_q initialized with small random values (std=0.01) to break symmetry
      and provide non-zero scores from the start.
    - Gate parameters get higher learning rate (gate_lr_scale) to compensate
      for sigmoid gradient attenuation.

All linear layers are standard nn.Linear — the ONLY architectural difference
from Qwen3 is the ResidualGate at each residual connection.

Supports:
- Flash Attention 2/3 via transformers ALL_ATTENTION_FUNCTIONS dispatch
- Packing/padding_free training via FlashAttentionKwargs (cu_seq_lens_q/k)
- SDPA, eager, flex_attention backends
- torch.compile (all ops are standard PyTorch element-wise/reduction)
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
    """Softmax-normalized residual gate over the hidden-size dimension.

    Applies softmax attention between h (residual stream) and o (sub-layer output)
    to selectively aggregate information with learned, content-dependent weights.
    Inspired by Attention Residuals (AttnRes) which uses softmax over depth;
    this applies the same principle over the hidden-size dimension.

    Architecture:
        Q (query):   per-group learned pseudo-query w_q ∈ R^{n_groups × group_size}
        K (keys):    RMSNorm(w_kh ⊙ h), RMSNorm(w_ko ⊙ o) — normalized per group
        Score:       Q·K / (τ·√group_size) — grouped dot product
        Gate:        softmax([score_h, score_o]) → [α_h, α_o], α_h + α_o = 1

    Output:
        h_new = α_h ⊙ h + α_o ⊙ o    (softmax-normalized weighted sum)

    Key properties:
        1. Softmax normalization: α_h + α_o = 1 per group → bounded hidden state magnitude
        2. Competitive selection: increasing α_h necessarily decreases α_o
        3. RMSNorm on keys: prevents magnitude differences from biasing softmax
           (critical because h grows with depth in PreNorm, while o is fixed-scale)
        4. Content-dependent: gate values depend on actual h and o content

    Why softmax (not sigmoid/tanh):
        - AttnRes paper Table 4: softmax > sigmoid (1.737 vs 1.741)
        - Softmax's competitive normalization forces sharper selection
        - Bounded output magnitude prevents cumulative signal decay/growth
        - AttnRes Figure 5(b): softmax keeps output magnitude bounded across depth

    Initialization (w_q = ZEROS):
        w_q = 0 → score_h = score_o = 0 → softmax([0,0]) = [0.5, 0.5]
        → h_new = 0.5·h + 0.5·o = 0.5·(h + o)
        This is NOT h + o, but AttnRes paper proves that softmax-normalized
        equal-weight averaging is BETTER than standard residual from the start.
        softmax gradient at uniform: ∂α/∂score = 0.25 (good gradient flow)

    Parameters per gate: 2d + n_groups·group_size + 1 ≈ 3d
    """

    def __init__(self, hidden_size: int, n_groups: int = 16, init_bias: float = 0.0):
        super().__init__()
        if hidden_size % n_groups != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_groups ({n_groups})"
            )
        self.hidden_size = hidden_size
        self.n_groups = n_groups
        self.group_size = hidden_size // n_groups

        # Separate Q for h and o — allows independent score evolution
        # w_qh = 0, w_qo = small random: initial softmax slightly biased toward o
        # This gives the gate a small initial preference for new info (accept > retain),
        # which is the correct inductive bias (new layer output should be incorporated).
        # The small random init also breaks symmetry immediately.
        self.w_qh = nn.Parameter(torch.zeros(n_groups, self.group_size))
        self.w_qo = nn.Parameter(torch.randn(n_groups, self.group_size) * 0.1)

        # K: per-dim key projections for h and o
        self.w_kh = nn.Parameter(torch.ones(hidden_size))   # init=1: key_h = h
        self.w_ko = nn.Parameter(torch.ones(hidden_size))   # init=1: key_o = o

        # RMSNorm epsilon for key normalization
        self.rms_eps = 1e-6

        # Learnable temperature τ
        self.log_tau = nn.Parameter(torch.zeros(1))  # τ = exp(0) = 1

        # Scaling factor (like √d_k in attention)
        self.register_buffer('_scale', torch.tensor(self.group_size ** -0.5, dtype=torch.float32))

        # Gate monitoring
        self._gate_raw: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def forward(self, residual: torch.Tensor, new_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residual: (*, d) - the residual stream (h)
            new_output: (*, d) - output from attention/FFN sub-layer (o)
        Returns:
            updated: (*, d) - α_h⊙h + α_o⊙o where α_h + α_o = 1 (softmax)

        Optimized for efficiency:
        - RMSNorm stays in input dtype (no float32 conversion for variance)
        - Only softmax computed in float32 (2-element softmax, minimal cost)
        - All view ops are zero-copy
        - torch.compile friendly (all standard ops)
        """
        orig_shape = residual.shape

        # K: per-dim key projection (element-wise, very fast)
        key_h = self.w_kh * residual      # (*, d)
        key_o = self.w_ko * new_output    # (*, d)

        # Reshape to groups: (*, n_groups, group_size) — zero-copy view
        key_h_g = key_h.view(*orig_shape[:-1], self.n_groups, self.group_size)
        key_o_g = key_o.view(*orig_shape[:-1], self.n_groups, self.group_size)

        # Inline RMSNorm on keys — stays in input dtype for speed
        # Prevents h's growing magnitude from dominating softmax
        # (same principle as AttnRes paper: φ(q,k) = exp(q^T · RMSNorm(k)))
        var_h = key_h_g.pow(2).mean(-1, keepdim=True)
        key_h_g = key_h_g * torch.rsqrt(var_h + self.rms_eps)
        var_o = key_o_g.pow(2).mean(-1, keepdim=True)
        key_o_g = key_o_g * torch.rsqrt(var_o + self.rms_eps)

        # Q·K / (τ·√group_size): grouped dot product with scaling
        scale = self._scale / (self.log_tau.exp() + 1e-8)

        score_h = (self.w_qh * key_h_g).sum(dim=-1) * scale  # (*, n_groups)
        score_o = (self.w_qo * key_o_g).sum(dim=-1) * scale  # (*, n_groups)

        # Softmax over [h, o] — competitive selection, α_h + α_o = 1
        # Only softmax in float32 for numerical precision (2-element, minimal cost)
        logits = torch.stack([score_h, score_o], dim=-1)  # (*, n_groups, 2)
        weights = F.softmax(logits.float(), dim=-1).to(residual.dtype)
        alpha_h = weights[..., 0]  # (*, n_groups), init = 0.5
        alpha_o = weights[..., 1]  # (*, n_groups), init = 0.5

        # Weighted sum: α_h⊙h + α_o⊙o — zero-copy views for grouping
        residual_g = residual.view(*orig_shape[:-1], self.n_groups, self.group_size)
        output_g = new_output.view(*orig_shape[:-1], self.n_groups, self.group_size)
        result = (alpha_h.unsqueeze(-1) * residual_g + alpha_o.unsqueeze(-1) * output_g).view(orig_shape)

        # Save gate for monitoring
        if self.training:
            self._gate_raw = (alpha_h.detach(), alpha_o.detach())

        return result

    @property
    def _gate_stats(self) -> Optional[dict]:
        """Lazily compute gate statistics from raw tensors (triggers GPU-CPU sync only when accessed)."""
        if self._gate_raw is None:
            return None
        alpha_h, alpha_o = self._gate_raw  # both ∈ [0, 1], sum to 1
        with torch.no_grad():
            ah = alpha_h.float()  # retain weight
            ao = alpha_o.float()  # accept weight
            return {
                "mean": ah.mean().item(),           # retain weight mean (0.5 = balanced)
                "std": ah.std().item(),
                "min": ah.min().item(),
                "max": ah.max().item(),
                "sparsity": (ah < 0.2).float().mean().item(),    # groups strongly favoring new info
                "saturation": (ah > 0.8).float().mean().item(),  # groups strongly retaining old info
                "forget_ratio": (ah < 0.4).float().mean().item(),  # groups favoring new over old
                "beta_mean": ao.mean().item(),           # accept weight mean
                "beta_std": ao.std().item(),
                "beta_min": ao.min().item(),
                "beta_max": ao.max().item(),
                "beta_sparsity": (ao < 0.2).float().mean().item(),
                "beta_saturation": (ao > 0.8).float().mean().item(),
            }

    @_gate_stats.setter
    def _gate_stats(self, value):
        """Allow setting _gate_stats to None for reset."""
        if value is None:
            self._gate_raw = None


# ==============================================================================
# RMSNorm (standard, same as Qwen3)
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
# Rotary Position Embedding (standard, same as Qwen3)
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
# Token-Sequence Attention (standard nn.Linear, identical to Qwen3)
# Note: "Attention" here means the standard token-sequence self-attention,
# NOT the hidden-size attention gate. The hidden-size attention is in ResidualGate.
# ==============================================================================

class MagGatedAttention(nn.Module):
    """Standard multi-head token-sequence attention with nn.Linear projections.

    This is the standard Qwen3 self-attention (token × token), unchanged.
    The hidden-size dimension attention gate is in ResidualGate, not here.

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
    """Standard SwiGLU MLP with nn.Linear projections (identical to Qwen3)."""

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
# Decoder Layer — with Attention Hidden-Size Residual Gate
# ==============================================================================

class MagGatedDecoderLayer(GradientCheckpointingLayer):
    """Transformer decoder layer with Attention Hidden-Size ResidualGate.

    Structure:
        h → norm1 → TokenAttn → ResidualGate(h, attn_out) → h'
        h'→ norm2 → MLP       → ResidualGate(h', ffn_out) → h''

    The ResidualGate uses grouped attention over the hidden-size dimension:
        Q·K score per group → forget/accept sigmoid gates
        h_new = (1-forget)⊙h + accept⊙o
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
            n_groups = config.residual_gate_n_groups
            self.attn_residual_gate = ResidualGate(config.hidden_size, n_groups=n_groups, init_bias=init_bias)
            self.ffn_residual_gate = ResidualGate(config.hidden_size, n_groups=n_groups, init_bias=init_bias)

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
# Full Model — Attention Hidden-Size Residual Gate Transformer
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
        # ResidualGate uses nn.Parameter (not nn.Linear), so _init_weights
        # naturally skips its parameters. No special handling needed.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_gate_param_groups(self, base_lr: float) -> list:
        """Return optimizer parameter groups with scaled LR for gate parameters.

        Gate parameters need higher learning rate to compensate for sigmoid
        gradient attenuation. This method returns two param groups:
        1. Non-gate parameters with base_lr
        2. Gate parameters with base_lr * gate_lr_scale

        Usage:
            param_groups = model.get_gate_param_groups(lr=1e-4)
            optimizer = torch.optim.AdamW(param_groups)
        """
        gate_lr_scale = getattr(self.config, 'residual_gate_lr_scale', 5.0)
        gate_params = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'residual_gate' in name:
                gate_params.append(param)
            else:
                other_params.append(param)

        groups = [
            {"params": other_params, "lr": base_lr},
            {"params": gate_params, "lr": base_lr * gate_lr_scale,
             "weight_decay": 0.0},  # no weight decay for gate params
        ]
        if gate_params:
            logger.info(
                f"[ResidualGate] Gate LR scale: {gate_lr_scale}x "
                f"({len(gate_params)} gate params, {len(other_params)} other params)"
            )
        return groups


class MagGatedModel(MagGatedPreTrainedModel):
    """Attention Hidden-Size Residual Gate Transformer (decoder-only, no LM head).

    A Qwen3-compatible model where each residual connection h = h + o is
    replaced by a grouped attention-based gate:
        h_new = (1 - forget) ⊙ h + accept ⊙ o
    where forget and accept are computed via grouped Q·K attention over
    the hidden-size dimension.
    """

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
        # V4 ResidualGate uses nn.Parameter (not nn.Linear), so _init_weights
        # naturally skips gate parameters. No marking needed.
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
    """Attention Hidden-Size Residual Gate Transformer with causal language model head.

    Qwen3-compatible decoder-only LM where every residual connection h = h + o
    is replaced by a grouped attention-based gate (ResidualGate):
        h_new = (1 - forget) ⊙ h + accept ⊙ o
    """

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

                    # Retain (1-forget) stats
                    for k in ["mean", "std", "min", "max", "sparsity", "saturation", "forget_ratio"]:
                        if k in gs:
                            stats[f"{prefix}_{k}"] = gs[k]

                    # Accept stats
                    prefix_beta = f"gate/layer{layer_idx}_{gate_name}_beta"
                    for k in ["beta_mean", "beta_std", "beta_min", "beta_max", "beta_sparsity", "beta_saturation"]:
                        if k in gs:
                            short_k = k.replace("beta_", "")
                            stats[f"{prefix_beta}_{short_k}"] = gs[k]

                    # Temperature τ
                    tau = gate.log_tau.exp().item()
                    stats[f"{prefix}_tau"] = tau

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
        """Re-initialize all ResidualGate parameters (softmax gate)."""
        count = 0
        for module in self.modules():
            if isinstance(module, ResidualGate):
                # w_qh=0, w_qo=random(std=0.1) — breaks symmetry
                nn.init.zeros_(module.w_qh)
                nn.init.normal_(module.w_qo, mean=0.0, std=0.1)
                nn.init.ones_(module.w_kh)
                nn.init.ones_(module.w_ko)
                nn.init.zeros_(module.log_tau)
                count += 1

        logger.info(
            f"[ResidualGate] ✓ Re-initialized {count} gate modules "
            f"(w_q=zeros, softmax gate with RMSNorm keys)"
        )
        return count

    def verify_gate_init(self) -> dict:
        """Verify that gate parameters are correctly initialized (softmax gate)."""
        gate_count = 0
        query_maxabs = []

        for module in self.modules():
            if isinstance(module, ResidualGate):
                gate_count += 1
                query_maxabs.append(max(
                    module.w_qh.data.float().abs().max().item(),
                    module.w_qo.data.float().abs().max().item()
                ))

        result = {
            "gate_count": gate_count,
        }

        if query_maxabs:
            avg_q_max = sum(query_maxabs) / len(query_maxabs)
            result["query_maxabs"] = avg_q_max
            result["accept_bias_mean"] = 0.0  # no bias in softmax gate
            # w_qh=0, w_qo=N(0,0.1) — max abs can be up to ~0.4
            result["query_ok"] = avg_q_max < 1.0
            result["accept_ok"] = True
            result["all_ok"] = result["query_ok"]
        else:
            result["all_ok"] = True

        return result
