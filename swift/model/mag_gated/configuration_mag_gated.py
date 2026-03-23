# Copyright (c) 2024. All rights reserved.
# Magnitude-Gated Transformer Configuration
"""Configuration for MagGated Transformer model."""

from transformers import PretrainedConfig


class MagGatedConfig(PretrainedConfig):
    """Configuration class for MagGated Transformer.

    This model implements the Magnitude-Gated Linear architecture where each
    linear layer decomposes weights into magnitude (m), direction (V_hat), and
    a dynamic input-dependent gate g(x). This allows dimensions to be
    selectively activated per input, enabling smaller hidden sizes.

    Args:
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of the residual stream (d).
        intermediate_size: Dimensionality of the FFN intermediate layer (d_mid).
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads (n_h).
        num_key_value_heads: Number of KV heads for GQA.
        head_dim: Dimension per attention head. If None, computed as hidden_size // num_attention_heads.
        hidden_act: Activation function for FFN.
        max_position_embeddings: Maximum sequence length.
        initializer_range: Std for weight initialization.
        rms_norm_eps: Epsilon for RMSNorm.
        use_cache: Whether to use KV cache.
        tie_word_embeddings: Whether to tie input/output embeddings.
        rope_theta: Base frequency for RoPE.
        gate_rank: Rank of the low-rank gate projection (r for LoRA-style gate).
        use_mag_gate: Whether to use MagGated Linear layers (True) or standard Linear (False).
        mag_gate_positions: Which linear layers get MagGated treatment.
            Options: 'all', 'bottleneck', 'none'.
            'all': All 6 projections (q,k,v,o,up,down) get MagGated.
            'bottleneck': Only o_proj and down_proj (write-back to residual).
            'none': Standard transformer (baseline).
        use_residual_gate: Whether to use residual forgetting gates.
        residual_gate_rank: Rank for residual gate projections.
        gate_init_bias: Initial bias for gate sigmoid. 3.0 => sigmoid(3)≈0.95.
        gate_floor: Minimum gate value to prevent dead dimensions. Default 0.05.
        attention_bias: Whether to use bias in attention projections.
        mlp_bias: Whether to use bias in MLP projections.
        attention_dropout: Dropout rate for attention.
    """

    model_type = "mag_gated"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3584,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        # === MagGated specific ===
        gate_rank=16,
        use_mag_gate=True,
        mag_gate_positions="all",
        use_residual_gate=True,
        residual_gate_rank=16,
        gate_init_bias=3.0,
        gate_floor=0.05,
        use_weight_norm=False,
        use_gate_norm=True,
        residual_gate_init_bias=4.0,
        # === Standard options ===
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
        # MagGated specific
        self.gate_rank = gate_rank
        self.use_mag_gate = use_mag_gate
        self.mag_gate_positions = mag_gate_positions
        self.use_residual_gate = use_residual_gate
        self.residual_gate_rank = residual_gate_rank
        self.gate_init_bias = gate_init_bias
        self.gate_floor = gate_floor
        self.use_weight_norm = use_weight_norm
        self.use_gate_norm = use_gate_norm
        self.residual_gate_init_bias = residual_gate_init_bias
        # Standard
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
