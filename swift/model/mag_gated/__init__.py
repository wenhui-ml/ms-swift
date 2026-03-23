# Copyright (c) 2024. All rights reserved.
# Magnitude-Gated Transformer: Why Hidden Size Can Be 2-4x Smaller
from .configuration_mag_gated import MagGatedConfig
from .modeling_mag_gated import (
    MagGatedForCausalLM,
    MagGatedModel,
    MagGatedDecoderLayer,
    MagGatedLinear,
)

__all__ = [
    "MagGatedConfig",
    "MagGatedForCausalLM",
    "MagGatedModel",
    "MagGatedDecoderLayer",
    "MagGatedLinear",
]
