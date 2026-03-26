# Copyright (c) 2024. All rights reserved.
# AttnResGate Transformer Configuration
"""Configuration for Attention Residual Gate Transformer model.

Pure Qwen3-compatible architecture with one core enhancement:
    ResidualGate — grouped attention-based residual gate.
    h_new = (1 - forget) ⊙ h + accept ⊙ o

Gate values are computed via grouped query-key attention over h and o,
following self-attention principles applied to the hidden-size dimension:
    Q: per-group learned query w_q
    K: per-dim projections key_h = w_kh ⊙ h, key_o = w_ko ⊙ o
    Score: grouped dot product Q·K / (τ·√group_size)
    forget = sigmoid(score_h + b_forget)  ← based on h's features
    accept = sigmoid(score_o + b_accept)  ← based on o's features

All linear layers are standard nn.Linear.
"""

from transformers import PretrainedConfig


class MagGatedConfig(PretrainedConfig):
    """Configuration class for Attention Residual Gate Transformer.

    This model is identical to Qwen3 except for the ResidualGate mechanism
    that replaces the standard additive residual connection (h + o) with
    a grouped attention-gated combination:
        h_new = (1 - forget) ⊙ h + accept ⊙ o

    The gate uses grouped attention principles (Q·K/√d_k, temperature scaling)
    to compute data-dependent, cross-dimension-aware forget/accept gates.

    Args:
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of the residual stream (d).
        intermediate_size: Dimensionality of the FFN intermediate layer.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        head_dim: Dimension per attention head.
        hidden_act: Activation function for FFN.
        max_position_embeddings: Maximum sequence length.
        initializer_range: Std for weight initialization.
        rms_norm_eps: Epsilon for RMSNorm.
        use_cache: Whether to use KV cache.
        tie_word_embeddings: Whether to tie input/output embeddings.
        rope_theta: Base frequency for RoPE.
        use_residual_gate: Whether to use ResidualGate (True) or standard add (False).
        residual_gate_n_groups: Number of dimension groups for grouped attention gate.
            Each group has (hidden_size // n_groups) dimensions. The gate score
            is computed as a dot product over all dimensions within each group,
            providing true cross-dimension interaction.
        residual_gate_init_bias: Sigmoid bias for gate initialization.
            Controls forget/accept initial values:
            - b_forget = -init_bias → sigmoid(-init_bias) ≈ 0 (no forgetting)
            - b_accept = +init_bias → sigmoid(+init_bias) ≈ 1 (full acceptance)
            - h_new ≈ h + o (near-identity standard residual)
            - ∂h_new/∂h ≈ 1 (no gradient vanishing at initialization)
            Default: 3.0 → sigmoid(±3) ≈ 0.047/0.953 (faster gate divergence).
            Alternative: 5.0 → sigmoid(±5) ≈ 0.007/0.993 (tighter near-identity, slower divergence).
        attention_bias: Whether to use bias in attention projections.
        mlp_bias: Whether to use bias in MLP projections.
        attention_dropout: Dropout rate for attention.
    """

    model_type = "mag_gated"

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
        # === ResidualGate (Attention Hidden-Size) specific ===
        use_residual_gate=True,
        residual_gate_n_groups=16,
        residual_gate_init_bias=3.0,
        # === Qwen3-compatible metadata ===
        max_window_layers=28,
        sliding_window=None,
        use_sliding_window=False,
        rope_scaling=None,
        torch_dtype=None,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
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
        # ResidualGate (Attention Hidden-Size)
        self.use_residual_gate = use_residual_gate
        self.residual_gate_n_groups = residual_gate_n_groups
        self.residual_gate_init_bias = residual_gate_init_bias
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
