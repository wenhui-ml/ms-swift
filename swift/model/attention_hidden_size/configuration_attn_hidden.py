# Copyright (c) 2024. All rights reserved.
# Attention Hidden-Size Transformer Configuration (V12 — Independent Synaptic Gating)
"""Configuration for Attention Hidden-Size Transformer model.

Qwen3-compatible architecture with Independent Synaptic Gating:
    gate_forget  = σ(w_forget ⊙ RMSNorm(h) + b_forget)
    gate_acquire = σ(w_acquire ⊙ RMSNorm(o) + b_acquire)
    h_new = gate_forget ⊙ h + gate_acquire ⊙ o

RMSNorm inside the gate is parameter-free (no learnable weight), used solely
to normalize activation magnitudes before gate computation. This prevents
sigmoid saturation as hidden-state norms grow in deeper layers.

Each hidden dimension j ∈ {1…d} has its own independent scalar gate parameters,
mimicking biological synaptic scaling: every "neuron" independently decides
how much to retain (forget) and how much to accept (acquire) based solely
on the signal flowing through itself.

Key properties:
    - Pure element-wise operations (no matrix multiplication in the gate)
    - Only 4d parameters per gate (w_forget, b_forget, w_acquire, b_acquire)
    - Parameter-free RMSNorm stabilizes gate inputs regardless of depth
    - Zero-loss fallback: init w=0, b=+4.0 → σ(4.0)≈0.98 → h_new ≈ h + o
    - No cross-layer context needed
    - No multi-head needed
    - Biologically inspired: independent synaptic scaling per dimension
"""

from transformers import PretrainedConfig


class AttnHiddenConfig(PretrainedConfig):
    """Configuration class for Attention Hidden-Size Transformer.

    This model is identical to Qwen3 except for the SynapticGate mechanism
    that replaces the standard additive residual connection (h + o) with
    independent element-wise gating: gate_forget⊙h + gate_acquire⊙o.

    The gate uses independent per-dimension synaptic scaling:
    - w_forget, b_forget ∈ ℝ^d — controls retention of residual stream h
    - w_acquire, b_acquire ∈ ℝ^d — controls acceptance of new output o
    - gate = σ(w ⊙ RMSNorm(x) + b) — sigmoid ensures gate values in [0, 1]
    - RMSNorm is parameter-free (signal conditioning only, no learnable weight)
    - No cross-channel interaction: each dimension is fully independent

    Args:
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of the residual stream (d).
        intermediate_size: Dimensionality of the FFN intermediate layer.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads for token-sequence attention.
        num_key_value_heads: Number of KV heads for GQA.
        head_dim: Dimension per attention head.
        hidden_act: Activation function for FFN.
        max_position_embeddings: Maximum sequence length.
        initializer_range: Std for weight initialization.
        rms_norm_eps: Epsilon for RMSNorm.
        use_cache: Whether to use KV cache.
        tie_word_embeddings: Whether to tie input/output embeddings.
        rope_theta: Base frequency for RoPE.
        use_synaptic_gate: Whether to use SynapticGate (True) or standard add (False).
        synaptic_gate_init_bias: Initial bias value for gate parameters.
            σ(+4.0) ≈ 0.982 → h_new ≈ 0.98·h + 0.98·o ≈ h + o.
            Higher values → closer to standard residual at init.
        synaptic_gate_lr_scale: Learning rate multiplier for gate parameters.
        synaptic_gate_init_mode: Initialization mode for gate weights.
            - "sft": w=zeros, b=+init_bias → σ(bias)≈1 → h_new ≈ h+o exactly.
              Use when transferring weights from a pretrained Qwen3 model.
            - "pretrain": w=small random, b=+init_bias → h_new ≈ h+o + tiny noise.
              Use when training from scratch.
        attention_bias: Whether to use bias in attention projections.
        mlp_bias: Whether to use bias in MLP projections.
        attention_dropout: Dropout rate for attention.
    """

    model_type = "attn_hidden"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=40960,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        # === Independent Synaptic Gate specific ===
        use_synaptic_gate=True,
        synaptic_gate_init_bias=4.0,
        synaptic_gate_lr_scale=5.0,
        synaptic_gate_init_mode="sft",
        # === Qwen3-compatible metadata ===
        max_window_layers=28,
        sliding_window=None,
        use_sliding_window=False,
        rope_scaling=None,
        torch_dtype=None,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        # === Legacy compatibility (ignored, kept for loading old configs) ===
        use_residual_gate=None,
        residual_gate_num_heads=None,
        residual_gate_context_dim=None,
        residual_gate_lr_scale=None,
        residual_gate_init_mode=None,
        residual_gate_init_remove_scale=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        # Independent Synaptic Gate
        # Support legacy config: if use_residual_gate is set but use_synaptic_gate not explicitly given
        if use_residual_gate is not None and use_synaptic_gate is True:
            self.use_synaptic_gate = use_residual_gate
        else:
            self.use_synaptic_gate = use_synaptic_gate
        self.synaptic_gate_init_bias = synaptic_gate_init_bias
        self.synaptic_gate_lr_scale = synaptic_gate_lr_scale
        self.synaptic_gate_init_mode = synaptic_gate_init_mode
        # Qwen3-compatible metadata
        self.max_window_layers = max_window_layers
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            torch_dtype=torch_dtype,
            **kwargs,
        )
