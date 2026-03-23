# Copyright (c) 2024. All rights reserved.
# Register MagGated Transformer model in ms-swift
"""Registration of MagGated Transformer for ms-swift training pipeline."""

from transformers import AutoConfig, AutoModelForCausalLM

from swift.model.mag_gated.configuration_mag_gated import MagGatedConfig
from swift.model.mag_gated.modeling_mag_gated import MagGatedForCausalLM


def register_mag_gated():
    """Register MagGated model with HuggingFace AutoClasses.
    
    This allows loading MagGated models with:
        AutoConfig.from_pretrained(path)
        AutoModelForCausalLM.from_pretrained(path)
    """
    AutoConfig.register("mag_gated", MagGatedConfig)
    AutoModelForCausalLM.register(MagGatedConfig, MagGatedForCausalLM)


# Auto-register on import
register_mag_gated()
