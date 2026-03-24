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
# Gate Linear marker — skipped by _init_weights to preserve gate initialization
# ==============================================================================

class _GateLinear(nn.Linear):
    """Marker subclass of nn.Linear for gate projections (gate_A, gate_B).

    _init_weights() will skip this class, preserving the careful initialization
    done in MagGatedLinear.__init__ and ResidualGate.__init__.
    """
    pass


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
                 bias: bool = False, gate_init_bias: float = 0.0,
                 gate_floor: float = 0.05,
                 use_weight_norm: bool = False, use_gate_norm: bool = True,
                 gate_mode: str = "softmax",
                 gate_temperature: float = 1.0,
                 gate_loss_type: str = "none",
                 gate_target_sparsity: float = 0.4,
                 gate_grad_scale: float = 1.0):
        super().__init__()
        self.d_out = d_out
        self.use_weight_norm = use_weight_norm
        self.use_gate_norm = use_gate_norm
        self.gate_floor = gate_floor
        self.gate_mode = gate_mode
        self.gate_temperature = gate_temperature
        self.gate_loss_type = gate_loss_type
        self.gate_target_sparsity = gate_target_sparsity
        self.gate_grad_scale = gate_grad_scale

        # === Direction: standard linear (acts as V̂) ===
        self.V = nn.Linear(d_in, d_out, bias=bias)

        # === Magnitude: per-output-dim static scale ===
        # For softmax mode: initial gate ≈ 1.0 (uniform softmax × d_out / d_out = 1.0)
        # So m should be ≈ 1.0 to keep output scale ≈ standard Linear
        if gate_mode == "softmax":
            initial_m = 1.0
        else:
            # sigmoid mode: calibrate based on gate_init_bias
            if use_weight_norm:
                std = 0.02
                initial_m = std * math.sqrt(d_in) / self._initial_effective_gate(gate_init_bias, gate_floor)
            else:
                initial_m = 1.0 / self._initial_effective_gate(gate_init_bias, gate_floor)
        self.m = nn.Parameter(torch.full((d_out,), initial_m))

        # === Dynamic Gate: low-rank projection ===
        # g(x) = gate_fn(B(Norm(A(x))) + b)
        # Use _GateLinear so _init_weights() won't override our careful init
        self.gate_A = _GateLinear(d_in, rank, bias=False)
        self.gate_norm = MagGatedRMSNorm(rank) if use_gate_norm else nn.Identity()
        self.gate_B = _GateLinear(rank, d_out, bias=True)

        if gate_mode == "softmax":
            # For softmax mode: small random init for both weight and bias
            # This gives near-uniform softmax initially (all logits ≈ 0)
            # but with enough variation for gradient signal
            nn.init.normal_(self.gate_B.weight, std=0.01)
            nn.init.zeros_(self.gate_B.bias)  # Zero bias → uniform softmax
            nn.init.normal_(self.gate_A.weight, std=0.1)
        else:
            # sigmoid mode: moderate bias for good gradient flow
            nn.init.normal_(self.gate_B.weight, std=0.01)
            nn.init.constant_(self.gate_B.bias, gate_init_bias)
            nn.init.normal_(self.gate_A.weight, std=0.1)

        # Gate monitoring: lightweight stats, no gradient, no extra memory
        self._gate_stats: Optional[dict] = None
        # Gate auxiliary loss (computed in forward, aggregated in model forward)
        self._gate_aux_loss: Optional[torch.Tensor] = None

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
            V_weight = F.normalize(self.V.weight, p=2, dim=1)
            direction = F.linear(x, V_weight, self.V.bias)
        else:
            direction = self.V(x)                       # (B, T, d_out)

        if self.gate_mode == "none":
            # No gate — just magnitude-scaled linear: y = m ⊙ V(x)
            # Gate parameters exist but are not used in forward pass.
            # This is the recommended mode: MagGatedLinear gate is redundant with V.
            # All gating is done by ResidualGate (which compares residual vs output).
            return self.m * direction

        gate_logits = self.gate_B(self.gate_norm(self.gate_A(x)))  # (B, T, d_out)

        if self.gate_mode == "softmax":
            gate = self.d_out * F.softmax(gate_logits / self.gate_temperature, dim=-1)
        else:
            # Sigmoid mode (legacy)
            if self.training and self.gate_grad_scale != 1.0:
                gate_logits = gate_logits * self.gate_grad_scale - gate_logits.detach() * (self.gate_grad_scale - 1.0)
            gate_raw = torch.sigmoid(gate_logits)
            gate = gate_raw * (1.0 - self.gate_floor) + self.gate_floor

            if self.training and self.gate_loss_type != "none":
                self._gate_aux_loss = self._compute_gate_aux_loss(gate_raw)

        # Record gate statistics
        if self.training:
            with torch.no_grad():
                gate_flat = gate.detach().float()
                self._gate_stats = {
                    "mean": gate_flat.mean().item(),
                    "std": gate_flat.std().item(),
                    "min": gate_flat.min().item(),
                    "max": gate_flat.max().item(),
                    "sparsity": (gate_flat < (0.5 if self.gate_mode == "softmax" else 0.1)).float().mean().item(),
                    "saturation": (gate_flat > (1.5 if self.gate_mode == "softmax" else 0.9)).float().mean().item(),
                    "dim_mean": gate_flat.mean(dim=(0, 1)).cpu(),
                }

        return self.m * gate * direction

    def _compute_gate_aux_loss(self, gate_raw: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary loss for sigmoid mode (not needed for softmax)."""
        if self.gate_loss_type == "l1_target":
            dim_mean = gate_raw.mean(dim=(0, 1))
            return (dim_mean.mean() - self.gate_target_sparsity) ** 2
        elif self.gate_loss_type == "neg_entropy":
            eps = 1e-6
            entropy = -(gate_raw * torch.log(gate_raw + eps) +
                       (1 - gate_raw) * torch.log(1 - gate_raw + eps))
            return entropy.mean()
        elif self.gate_loss_type == "l1_sparse":
            return gate_raw.mean()
        else:
            return torch.tensor(0.0, device=gate_raw.device)

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
            gate_mode=getattr(config, "gate_mode", "softmax"),
            gate_temperature=getattr(config, "gate_temperature", 1.0),
            gate_loss_type=getattr(config, "gate_loss_type", "none"),
            gate_target_sparsity=getattr(config, "gate_target_sparsity", 0.4),
            gate_grad_scale=getattr(config, "gate_grad_scale", 1.0),
        )
    else:
        return nn.Linear(d_in, d_out, bias=bias)


# ==============================================================================
# Residual Forgetting Gate
# ==============================================================================

class ResidualGate(nn.Module):
    """Dual-gate magnitude-aware residual connection.

    h_new = α(h, o) ⊙ h + β(h, o) ⊙ o

    where:
        α ∈ (0,1)^d = retain gate (how much old info to keep per dimension)
        β ∈ (0,1)^d = accept gate (how much new info to accept per dimension)

    Key design principles:
    1. Dual gates: α and β are INDEPENDENT — model can keep old AND accept new,
       or discard old AND reject new, per dimension.
    2. Gate sees BOTH residual and output (can compare importance).
    3. DoRA-inspired magnitude awareness: gate input includes per-dimension
       magnitude ratio |h|/(|h|+|o|), giving the gate an immediate signal
       about relative information strength.
    4. init_bias=3.0 for both → sigmoid(3)≈0.95 → initial behavior ≈ standard
       residual h + o. Gate learns to deviate from this as needed.

    Possible behaviors per dimension:
        α≈1, β≈1: standard residual (both important)
        α≈1, β≈0: retain old, reject new (dimension already has good info)
        α≈0, β≈1: forget old, accept new (release dimension for new knowledge)
        α≈0, β≈0: suppress dimension entirely (dimension is noise)
    """

    def __init__(self, hidden_size: int, rank: int = 16, init_bias: float = 3.0):
        super().__init__()
        self.hidden_size = hidden_size

        # Gate input: concat(residual, output) → 2d
        # Raw values preserve full gradient flow (no abs/div that cause instability)
        # The gate network learns magnitude/direction features internally
        gate_input_size = hidden_size * 2

        # Shared low-rank projection for both gates (parameter efficient)
        # Use _GateLinear so _init_weights() won't override our careful init
        self.gate_A = _GateLinear(gate_input_size, rank, bias=False)
        # Two separate output projections: one for α, one for β
        self.gate_B_alpha = _GateLinear(rank, hidden_size, bias=True)
        self.gate_B_beta = _GateLinear(rank, hidden_size, bias=True)

        # init_bias=3.0 → sigmoid(3)≈0.953
        # Initial: α≈0.95, β≈0.95 → h_new ≈ 0.95*h + 0.95*o ≈ h + o
        # This makes initial behavior very close to standard residual connection,
        # so the model starts learning normally. Gates then gradually learn to
        # deviate: lowering α for redundant dims, lowering β for noisy dims.
        nn.init.normal_(self.gate_B_alpha.weight, std=0.01)
        nn.init.constant_(self.gate_B_alpha.bias, init_bias)
        nn.init.normal_(self.gate_B_beta.weight, std=0.01)
        nn.init.constant_(self.gate_B_beta.bias, init_bias)
        nn.init.normal_(self.gate_A.weight, std=0.02)

        # Gate monitoring
        self._gate_stats: Optional[dict] = None

    def forward(self, residual: torch.Tensor, new_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residual: (B, T, d) - the residual stream
            new_output: (B, T, d) - output from attention/FFN sub-layer
        Returns:
            updated: (B, T, d) - dual-gated combination
        """
        # Gate input: raw residual and output values
        # The gate network learns to extract magnitude/direction features internally
        # Raw values preserve clean gradient flow (no abs/div operations)
        gate_input = torch.cat([residual, new_output], dim=-1)  # (B, T, 2d)

        # Shared low-rank compression
        gate_hidden = self.gate_A(gate_input)  # (B, T, rank)

        # Independent α and β
        alpha = torch.sigmoid(self.gate_B_alpha(gate_hidden))  # (B, T, d) retain gate
        beta = torch.sigmoid(self.gate_B_beta(gate_hidden))    # (B, T, d) accept gate

        # Record gate statistics
        if self.training:
            with torch.no_grad():
                a = alpha.detach().float()
                b = beta.detach().float()
                self._gate_stats = {
                    "mean": a.mean().item(),       # α mean (retain)
                    "std": a.std().item(),
                    "min": a.min().item(),
                    "max": a.max().item(),
                    "sparsity": (a < 0.1).float().mean().item(),
                    "saturation": (a > 0.9).float().mean().item(),
                    "forget_ratio": (a < 0.5).float().mean().item(),
                    "dim_mean": a.mean(dim=(0, 1)).cpu(),
                    # β stats (accept gate)
                    "beta_mean": b.mean().item(),
                    "beta_std": b.std().item(),
                    "beta_sparsity": (b < 0.1).float().mean().item(),
                    "beta_saturation": (b > 0.9).float().mean().item(),
                }

        # Dual-gate combination: α ⊙ residual + β ⊙ output
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
        if isinstance(module, _GateLinear):
            # Skip! Gate projections have careful init in MagGatedLinear/ResidualGate.__init__
            # (gate_B.weight=0, gate_B.bias=init_bias, gate_A.weight~N(0,0.01))
            pass
        elif isinstance(module, MagGatedLinear):
            nn.init.normal_(module.V.weight, mean=0.0, std=std)
            if module.V.bias is not None:
                nn.init.zeros_(module.V.bias)
            # P1-3: m is calibrated in MagGatedLinear.__init__ to compensate gate
            # Don't override here — the __init__ already computed the correct value
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

            # === Gate auxiliary loss: direct gradient signal for gate differentiation ===
            gate_loss_weight = getattr(self.config, 'gate_loss_weight', 0.1)
            gate_loss_type = getattr(self.config, 'gate_loss_type', 'l1_target')
            if gate_loss_weight > 0 and gate_loss_type != 'none':
                gate_aux_losses = []
                for module in self.modules():
                    if isinstance(module, MagGatedLinear) and module._gate_aux_loss is not None:
                        gate_aux_losses.append(module._gate_aux_loss)
                if gate_aux_losses:
                    total_gate_loss = torch.stack(gate_aux_losses).mean()
                    loss = loss + gate_loss_weight * total_gate_loss

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

    # ==================================================================
    # Gate Monitoring API
    # ==================================================================

    def get_gate_stats(self) -> dict:
        """Collect gate statistics from all MagGatedLinear and ResidualGate modules.

        Returns a dict suitable for logging to TensorBoard/WandB, e.g.:
        {
            "gate/layer0_q_proj_mean": 0.85,
            "gate/layer0_q_proj_sparsity": 0.02,
            "gate/layer0_attn_residual_mean": 0.97,
            ...
            "gate/global_mean": 0.82,        # average across ALL mag gates
            "gate/global_sparsity": 0.05,    # fraction of mag gates < 0.1
            "gate/global_saturation": 0.70,  # fraction of mag gates > 0.9
            "gate/global_std": 0.15,         # average std across all mag gates
            "gate/residual_global_mean": 0.95,  # average across all residual gates
            "gate/residual_global_sparsity": 0.0,
            "gate/residual_global_saturation": 0.8,
            "gate/dim_active_ratio": 0.45,   # fraction of dims with mean gate > 0.5
            "gate/dim_reuse_score": 0.72,    # cross-layer dimension reuse metric
            "gate/health": "ok",             # "ok" / "warning_all_on" / "warning_all_off" / "warning_no_differentiation"
        }
        """
        import torch as _torch

        stats = {}

        # === MagGatedLinear gate stats ===
        mag_means = []
        mag_sparsities = []
        mag_saturations = []
        mag_stds = []
        mag_dim_means = []  # list of (d_out,) tensors for dimension reuse analysis

        # === Magnitude (m) stats ===
        m_means = []
        m_stds = []
        m_mins = []
        m_maxs = []

        # === ResidualGate stats (separate tracking) ===
        res_means = []
        res_sparsities = []
        res_saturations = []
        res_forget_ratios = []

        # Per-layer summary for detailed logging
        layer_summaries = []

        def _collect_mag_linear_stats(proj, prefix, layer_mag_means, layer_mag_stds):
            """Helper to collect stats from a MagGatedLinear projection."""
            for k, v in proj._gate_stats.items():
                if k != "dim_mean":  # dim_mean is a tensor, log separately
                    stats[f"{prefix}_{k}"] = v
            mag_means.append(proj._gate_stats["mean"])
            mag_sparsities.append(proj._gate_stats["sparsity"])
            mag_saturations.append(proj._gate_stats["saturation"])
            mag_stds.append(proj._gate_stats["std"])
            layer_mag_means.append(proj._gate_stats["mean"])
            layer_mag_stds.append(proj._gate_stats["std"])
            if "dim_mean" in proj._gate_stats:
                mag_dim_means.append(proj._gate_stats["dim_mean"])

            # Collect magnitude (m) statistics
            m_data = proj.m.data.detach().float()
            m_mean = m_data.mean().item()
            m_std = m_data.std().item()
            m_min = m_data.min().item()
            m_max = m_data.max().item()
            stats[f"{prefix}_m_mean"] = m_mean
            stats[f"{prefix}_m_std"] = m_std
            stats[f"{prefix}_m_min"] = m_min
            stats[f"{prefix}_m_max"] = m_max
            m_means.append(m_mean)
            m_stds.append(m_std)
            m_mins.append(m_min)
            m_maxs.append(m_max)

        for layer_idx, layer in enumerate(self.model.layers):
            layer_mag_means = []
            layer_mag_stds = []

            # MagGated projections in attention
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                proj = getattr(layer.self_attn, proj_name, None)
                if isinstance(proj, MagGatedLinear) and proj._gate_stats is not None:
                    prefix = f"gate/layer{layer_idx}_{proj_name}"
                    _collect_mag_linear_stats(proj, prefix, layer_mag_means, layer_mag_stds)

            # MagGated projections in MLP
            for proj_name in ["up_proj", "down_proj"]:
                proj = getattr(layer.mlp, proj_name, None)
                if isinstance(proj, MagGatedLinear) and proj._gate_stats is not None:
                    prefix = f"gate/layer{layer_idx}_{proj_name}"
                    _collect_mag_linear_stats(proj, prefix, layer_mag_means, layer_mag_stds)

            # Residual gates — now fully tracked with sparsity/saturation
            if layer.use_residual_gate:
                for gate_name in ["attn_residual_gate", "ffn_residual_gate"]:
                    gate = getattr(layer, gate_name, None)
                    if isinstance(gate, ResidualGate) and gate._gate_stats is not None:
                        prefix = f"gate/layer{layer_idx}_{gate_name}"
                        for k, v in gate._gate_stats.items():
                            if k != "dim_mean":
                                stats[f"{prefix}_{k}"] = v
                        res_means.append(gate._gate_stats["mean"])
                        if "sparsity" in gate._gate_stats:
                            res_sparsities.append(gate._gate_stats["sparsity"])
                        if "saturation" in gate._gate_stats:
                            res_saturations.append(gate._gate_stats["saturation"])
                        if "forget_ratio" in gate._gate_stats:
                            res_forget_ratios.append(gate._gate_stats["forget_ratio"])

            # Per-layer summary
            if layer_mag_means:
                layer_mean = sum(layer_mag_means) / len(layer_mag_means)
                layer_std = sum(layer_mag_stds) / len(layer_mag_stds)
                stats[f"gate/layer{layer_idx}_summary_mean"] = layer_mean
                stats[f"gate/layer{layer_idx}_summary_std"] = layer_std
                layer_summaries.append((layer_idx, layer_mean, layer_std))

        # === Global MagGatedLinear summary ===
        if mag_means:
            global_mean = sum(mag_means) / len(mag_means)
            stats["gate/global_mean"] = global_mean
        if mag_sparsities:
            global_sparsity = sum(mag_sparsities) / len(mag_sparsities)
            stats["gate/global_sparsity"] = global_sparsity
        if mag_saturations:
            global_saturation = sum(mag_saturations) / len(mag_saturations)
            stats["gate/global_saturation"] = global_saturation
        if mag_stds:
            global_std = sum(mag_stds) / len(mag_stds)
            stats["gate/global_std"] = global_std

        # === Global Magnitude (m) summary ===
        if m_means:
            stats["mag/global_m_mean"] = sum(m_means) / len(m_means)
            stats["mag/global_m_std"] = sum(m_stds) / len(m_stds)
            stats["mag/global_m_min"] = min(m_mins)
            stats["mag/global_m_max"] = max(m_maxs)
            # m differentiation: if m_std across dims is high, model learned dim importance
            stats["mag/m_differentiation"] = sum(m_stds) / len(m_stds)

        # === Global ResidualGate summary (separate from MagGatedLinear) ===
        if res_means:
            stats["gate/residual_global_mean"] = sum(res_means) / len(res_means)
        if res_sparsities:
            stats["gate/residual_global_sparsity"] = sum(res_sparsities) / len(res_sparsities)
        if res_saturations:
            stats["gate/residual_global_saturation"] = sum(res_saturations) / len(res_saturations)
        if res_forget_ratios:
            stats["gate/residual_global_forget_ratio"] = sum(res_forget_ratios) / len(res_forget_ratios)

        # === Dimension reuse analysis ===
        # Analyze how dimensions are used across layers (key metric for the paper thesis)
        if mag_dim_means:
            try:
                # Stack all dim_mean tensors: shape (num_gates, d_out_varies)
                # Group by output dimension size for meaningful analysis
                dim_groups = {}
                for dm in mag_dim_means:
                    d = dm.shape[0]
                    if d not in dim_groups:
                        dim_groups[d] = []
                    dim_groups[d].append(dm)

                total_active = 0
                total_dims = 0
                all_dim_stds = []

                for d, tensors in dim_groups.items():
                    stacked = _torch.stack(tensors)  # (num_gates_with_this_d, d)
                    # Active ratio: fraction of dims with mean gate > 0.5
                    dim_avg = stacked.mean(dim=0)  # (d,) average across all gates
                    active = (dim_avg > 0.5).float().sum().item()
                    total_active += active
                    total_dims += d

                    # Dimension reuse: std across gates for each dim
                    # High std = dimension used differently by different layers = good reuse
                    if stacked.shape[0] > 1:
                        dim_std = stacked.std(dim=0)  # (d,) std across gates
                        all_dim_stds.append(dim_std.mean().item())

                if total_dims > 0:
                    stats["gate/dim_active_ratio"] = total_active / total_dims

                if all_dim_stds:
                    # Reuse score: higher std across layers = more differentiated usage
                    stats["gate/dim_reuse_score"] = sum(all_dim_stds) / len(all_dim_stds)
            except Exception:
                pass  # Dimension analysis is best-effort

        # === Health check: detect degenerate gates ===
        if mag_means:
            if global_saturation > 0.95:
                stats["gate/health"] = "warning_all_on"  # Gates nearly all 1 → no gating effect
            elif global_sparsity > 0.95:
                stats["gate/health"] = "warning_all_off"  # Gates nearly all 0 → model dead
            elif global_sparsity < 0.01 and global_saturation < 0.01:
                # NEW: Detect undifferentiated gates — all gates clustered around mean
                # This means the gate mechanism is not learning to differentiate dimensions
                stats["gate/health"] = "warning_no_differentiation"
            else:
                stats["gate/health"] = "ok"

        return stats

    def reinit_gates(self, gate_init_bias: Optional[float] = None,
                     residual_gate_init_bias: Optional[float] = None) -> int:
        """Re-initialize all gate parameters to their intended initial values.

        This is useful when loading from a checkpoint that was saved with
        incorrect gate initialization (e.g., gate_B.bias=0 instead of 3.0).

        Args:
            gate_init_bias: Bias for MagGatedLinear gates. If None, uses config value.
            residual_gate_init_bias: Bias for ResidualGate. If None, uses config value.

        Returns:
            Number of gate modules re-initialized.
        """
        if gate_init_bias is None:
            gate_init_bias = getattr(self.config, 'gate_init_bias', 0.5)
        if residual_gate_init_bias is None:
            residual_gate_init_bias = getattr(self.config, 'residual_gate_init_bias', 2.0)

        count = 0
        for module in self.modules():
            if isinstance(module, MagGatedLinear):
                nn.init.normal_(module.gate_B.weight, std=0.01)
                nn.init.constant_(module.gate_B.bias, gate_init_bias)
                nn.init.normal_(module.gate_A.weight, std=0.1)
                # Recalibrate m to match the gate init
                gate_floor = getattr(module, 'gate_floor', 0.05)
                effective_gate = module._initial_effective_gate(gate_init_bias, gate_floor)
                if module.use_weight_norm:
                    initial_m = 0.02 * math.sqrt(module.V.in_features) / effective_gate
                else:
                    initial_m = 1.0 / effective_gate
                module.m.data.fill_(initial_m)
                count += 1
            elif isinstance(module, ResidualGate):
                if hasattr(module, 'gate_B_alpha'):
                    nn.init.normal_(module.gate_B_alpha.weight, std=0.01)
                    nn.init.constant_(module.gate_B_alpha.bias, residual_gate_init_bias)
                    nn.init.normal_(module.gate_B_beta.weight, std=0.01)
                    nn.init.constant_(module.gate_B_beta.bias, residual_gate_init_bias)
                elif hasattr(module, 'gate_B'):
                    nn.init.normal_(module.gate_B.weight, std=0.01)
                    nn.init.constant_(module.gate_B.bias, residual_gate_init_bias)
                nn.init.normal_(module.gate_A.weight, std=0.02)
                count += 1

        logger.info(
            f"[MagGated] ✓ Re-initialized {count} gate modules "
            f"(gate_init_bias={gate_init_bias}, residual_gate_init_bias={residual_gate_init_bias})"
        )
        return count

    def verify_gate_init(self) -> dict:
        """Verify that gate parameters are correctly initialized.

        Checks if gate_B.bias values match the expected init values from config.
        Returns a dict with verification results.

        This should be called after model loading to detect corrupted checkpoints.
        """
        gate_init_bias = getattr(self.config, 'gate_init_bias', 3.0)
        residual_gate_init_bias = getattr(self.config, 'residual_gate_init_bias', 4.0)

        mag_gate_biases = []
        res_gate_biases = []
        mag_gate_weight_stds = []
        res_gate_weight_stds = []

        for module in self.modules():
            if isinstance(module, MagGatedLinear):
                mag_gate_biases.append(module.gate_B.bias.data.float().mean().item())
                mag_gate_weight_stds.append(module.gate_B.weight.data.float().std().item())
            elif isinstance(module, ResidualGate):
                # Support both old (gate_B) and new (gate_B_alpha/gate_B_beta) ResidualGate
                if hasattr(module, 'gate_B_alpha'):
                    res_gate_biases.append(module.gate_B_alpha.bias.data.float().mean().item())
                    res_gate_weight_stds.append(module.gate_B_alpha.weight.data.float().std().item())
                elif hasattr(module, 'gate_B'):
                    res_gate_biases.append(module.gate_B.bias.data.float().mean().item())
                    res_gate_weight_stds.append(module.gate_B.weight.data.float().std().item())

        result = {
            "mag_gate_count": len(mag_gate_biases),
            "res_gate_count": len(res_gate_biases),
            "expected_mag_bias": gate_init_bias,
            "expected_res_bias": residual_gate_init_bias,
        }

        if mag_gate_biases:
            avg_mag_bias = sum(mag_gate_biases) / len(mag_gate_biases)
            avg_mag_w_std = sum(mag_gate_weight_stds) / len(mag_gate_weight_stds)
            result["actual_mag_bias_mean"] = avg_mag_bias
            result["actual_mag_weight_std"] = avg_mag_w_std
            # Check if bias is close to expected (within 10%)
            result["mag_bias_ok"] = abs(avg_mag_bias - gate_init_bias) < max(gate_init_bias * 0.3, 0.2)
            # Check if weight std is reasonable (should be < 0.05 for fresh init)
            result["mag_weight_ok"] = avg_mag_w_std < 0.05

        if res_gate_biases:
            avg_res_bias = sum(res_gate_biases) / len(res_gate_biases)
            avg_res_w_std = sum(res_gate_weight_stds) / len(res_gate_weight_stds)
            result["actual_res_bias_mean"] = avg_res_bias
            result["actual_res_weight_std"] = avg_res_w_std
            result["res_bias_ok"] = abs(avg_res_bias - residual_gate_init_bias) < max(residual_gate_init_bias * 0.3, 0.3)
            result["res_weight_ok"] = avg_res_w_std < 0.05

        # Overall health
        all_ok = result.get("mag_bias_ok", True) and result.get("res_bias_ok", True)
        result["all_ok"] = all_ok

        if not all_ok:
            logger.warning(
                f"[MagGated] ⚠️  Gate initialization CORRUPTED!\n"
                f"  MagGatedLinear gate_B.bias: expected={gate_init_bias:.1f}, "
                f"actual={result.get('actual_mag_bias_mean', 'N/A'):.4f} "
                f"({'OK' if result.get('mag_bias_ok', True) else 'WRONG'})\n"
                f"  MagGatedLinear gate_B.weight std: {result.get('actual_mag_weight_std', 'N/A'):.6f} "
                f"({'OK' if result.get('mag_weight_ok', True) else 'WRONG (should be ~0)'})\n"
                f"  ResidualGate gate_B.bias: expected={residual_gate_init_bias:.1f}, "
                f"actual={result.get('actual_res_bias_mean', 'N/A'):.4f} "
                f"({'OK' if result.get('res_bias_ok', True) else 'WRONG'})\n"
                f"  → Call model.reinit_gates() to fix this before training!"
            )
        else:
            logger.info(
                f"[MagGated] ✓ Gate initialization verified OK "
                f"(mag_bias={result.get('actual_mag_bias_mean', 'N/A'):.2f}, "
                f"res_bias={result.get('actual_res_bias_mean', 'N/A'):.2f})"
            )

        return result

    def get_gate_param_groups(self, gate_lr_multiplier: float = 5.0) -> list:
        """Return parameter groups with separate learning rate for gate parameters.

        This enables adaptive gate learning — gate parameters (gate_A, gate_B)
        get a higher learning rate to encourage faster differentiation.

        Usage with HuggingFace Trainer:
            # In your training script, after model creation:
            param_groups = model.get_gate_param_groups(gate_lr_multiplier=5.0)
            optimizer = AdamW(param_groups, lr=base_lr)

        Args:
            gate_lr_multiplier: Multiplier for gate parameter learning rate.
                Default 5.0 means gate params learn 5x faster than other params.

        Returns:
            List of param group dicts for optimizer construction.
        """
        gate_params = []
        gate_param_names = []
        other_params = []
        other_param_names = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Gate parameters: gate_A, gate_B in both MagGatedLinear and ResidualGate
            if "gate_A" in name or "gate_B" in name:
                gate_params.append(param)
                gate_param_names.append(name)
            else:
                other_params.append(param)
                other_param_names.append(name)

        logger.info(
            f"[MagGated] Gate param groups: {len(gate_params)} gate params "
            f"(lr × {gate_lr_multiplier}), {len(other_params)} other params (base lr)"
        )

        return [
            {"params": other_params, "lr_multiplier": 1.0},
            {"params": gate_params, "lr_multiplier": gate_lr_multiplier,
             "_gate_param_names": gate_param_names},
        ]
