# Copyright (c) 2024. All rights reserved.
# Attention Hidden-Size Transformer (V12 — Independent Synaptic Gating)
from .configuration_attn_hidden import AttnHiddenConfig
from .modeling_attn_hidden import (
    AttnHiddenForCausalLM,
    AttnHiddenModel,
    AttnHiddenDecoderLayer,
    SynapticGate,
    ResidualGate,  # Legacy alias for backward compatibility
)

__all__ = [
    "AttnHiddenConfig",
    "AttnHiddenForCausalLM",
    "AttnHiddenModel",
    "AttnHiddenDecoderLayer",
    "SynapticGate",
    "ResidualGate",
]
