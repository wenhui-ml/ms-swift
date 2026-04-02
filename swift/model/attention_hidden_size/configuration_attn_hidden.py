# Copyright (c) 2024. All rights reserved.
# Attention Hidden-Size Transformer Configuration (V11)
"""Configuration for Attention Hidden-Size Transformer model.

Qwen3-compatible architecture with Self-Attention Residual Gate:
    h_new = α ⊙ h + β ⊙ o

where α and β are computed via full self-attention over the hidden-size dimension:
    Q = W_q · RMSNorm(h)         (content-dependent query)
    K_h = W_kh · RMSNorm(h)      (h's key)
    K_o = W_ko · RMSNorm(o)      (o's key)
    score_h = Q · K_h             (per-head scalar score)
    score_o = Q · K_o             (per-head scalar score)
    [α, β] = softmax([score_h, score_o])  (competitive selection per head)

Cross-layer information via gate_context propagated between layers.

All linear layers are standard nn.Linear.
"""

from transformers import PretrainedConfig


class AttnHiddenConfig(PretrainedConfig):
    """Configuration class for Attention Hidden-Size Transformer.

    This model is identical to Qwen3 except for the ResidualGate mechanism
    that replaces the standard additive residual connection (h + o) with
    a self-attention-based gated combination: α⊙h + β⊙o.

    The gate uses full self-attention principles:
    - Q = W_q · h (content-dependent, follows dynamic token content)
    - K = W_k · h / W_k · o (content-dependent keys)
    - Multi-head design for dimension-level granularity
    - Cross-layer gate_context for depth-wise information flow

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
        use_residual_gate: Whether to use ResidualGate (True) or standard add (False).
        residual_gate_num_heads: Number of heads for the hidden-size attention gate.
            Each head controls hidden_size // num_heads dimensions.
        residual_gate_context_dim: Dimension of cross-layer gate context.
            Set to 0 to disable cross-layer information.
        residual_gate_lr_scale: Learning rate multiplier for gate parameters.
        residual_gate_init_mode: Initialization mode for gate projections.
            - "sft": LoRA-style init (K=zeros). Initial h_new = h + o exactly.
              Use when transferring weights from a pretrained Qwen3 model.
            - "pretrain": All projections small random. Small initial perturbation.
              Use when training from scratch.
        residual_gate_init_remove_scale: Initial value of the learnable removal
            scale factor. Controls max removal per layer. Default 0.1 means
            each layer can remove at most ~10% of h or o initially.
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
        # === Attention Hidden-Size Gate specific ===
        use_residual_gate=True,
        residual_gate_num_heads=8,
        residual_gate_context_dim=16,
        residual_gate_lr_scale=5.0,
        residual_gate_init_mode="sft",
        residual_gate_init_remove_scale=0.1,
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
        # Attention Hidden-Size Gate
        self.use_residual_gate = use_residual_gate
        self.residual_gate_num_heads = residual_gate_num_heads
        self.residual_gate_context_dim = residual_gate_context_dim
        self.residual_gate_lr_scale = residual_gate_lr_scale
        self.residual_gate_init_mode = residual_gate_init_mode
        self.residual_gate_init_remove_scale = residual_gate_init_remove_scale
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
