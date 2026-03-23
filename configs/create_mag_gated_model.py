#!/usr/bin/env python
"""Create a from-scratch MagGated Transformer model checkpoint for pretraining.

This script creates model configs and saves random-initialized checkpoints
so ms-swift can load them for pretraining.

Usage:
    # Create MagGated model (d=1024, gate_rank=16, all positions gated)
    python create_mag_gated_model.py --variant mag_gated_all --hidden_size 1024

    # Create MagGated model (bottleneck only: o_proj + down_proj)
    python create_mag_gated_model.py --variant mag_gated_bottleneck --hidden_size 1024

    # Create standard baseline (no gates, same architecture)
    python create_mag_gated_model.py --variant baseline --hidden_size 1024

    # Create MagGated with smaller d (the core experiment)
    python create_mag_gated_model.py --variant mag_gated_all --hidden_size 512
"""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer

from swift.model.mag_gated.configuration_mag_gated import MagGatedConfig
from swift.model.mag_gated.modeling_mag_gated import MagGatedForCausalLM


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_gate_parameters(model):
    """Count parameters specifically from gate components."""
    gate_params = 0
    mag_params = 0
    residual_gate_params = 0
    for name, p in model.named_parameters():
        if 'gate_A' in name or 'gate_B' in name:
            if 'residual_gate' in name:
                residual_gate_params += p.numel()
            else:
                gate_params += p.numel()
        elif name.endswith('.m'):
            mag_params += p.numel()
    return gate_params, mag_params, residual_gate_params


def create_model(args):
    """Create and save a MagGated model checkpoint."""

    # Determine gate settings based on variant
    if args.variant == 'baseline':
        use_mag_gate = False
        mag_gate_positions = 'none'
        use_residual_gate = False
    elif args.variant == 'mag_gated_all':
        use_mag_gate = True
        mag_gate_positions = 'all'
        use_residual_gate = True
    elif args.variant == 'mag_gated_bottleneck':
        use_mag_gate = True
        mag_gate_positions = 'bottleneck'
        use_residual_gate = True
    elif args.variant == 'residual_only':
        use_mag_gate = False
        mag_gate_positions = 'none'
        use_residual_gate = True
    else:
        raise ValueError(f"Unknown variant: {args.variant}")

    # Compute intermediate_size (roughly 3.5x hidden_size, rounded to multiple of 128)
    if args.intermediate_size is None:
        intermediate_size = int(args.hidden_size * 3.5)
        intermediate_size = ((intermediate_size + 127) // 128) * 128
    else:
        intermediate_size = args.intermediate_size

    # Compute num_attention_heads and num_key_value_heads
    if args.num_attention_heads is None:
        # Default: head_dim=128, adjust num_heads accordingly
        head_dim = args.head_dim
        num_attention_heads = max(1, args.hidden_size // head_dim)
    else:
        num_attention_heads = args.num_attention_heads
        head_dim = args.hidden_size // num_attention_heads

    if args.num_key_value_heads is None:
        # GQA: use 1/4 of attention heads as KV heads, minimum 1
        num_key_value_heads = max(1, num_attention_heads // 4)
    else:
        num_key_value_heads = args.num_key_value_heads

    # If tokenizer is provided, use its actual vocab size first
    actual_vocab_size = args.vocab_size
    tokenizer = None
    if args.tokenizer_from:
        print(f"Loading tokenizer from: {args.tokenizer_from}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_from, trust_remote_code=True)
        actual_vocab_size = len(tokenizer)
        print(f"  Tokenizer actual vocab size: {actual_vocab_size}")
        if actual_vocab_size != args.vocab_size:
            print(f"  (Overriding --vocab_size {args.vocab_size} with tokenizer's {actual_vocab_size})")

    config = MagGatedConfig(
        vocab_size=actual_vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        hidden_act='silu',
        max_position_embeddings=args.max_position_embeddings,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=args.tie_word_embeddings,
        rope_theta=1000000.0,
        gate_rank=args.gate_rank,
        use_mag_gate=use_mag_gate,
        mag_gate_positions=mag_gate_positions,
        use_residual_gate=use_residual_gate,
        residual_gate_rank=args.residual_gate_rank,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
    )

    print(f"\n{'='*60}")
    print(f"Creating MagGated Transformer: {args.variant}")
    print(f"{'='*60}")
    print(f"  hidden_size:       {config.hidden_size}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  num_layers:        {config.num_hidden_layers}")
    print(f"  num_attn_heads:    {config.num_attention_heads}")
    print(f"  num_kv_heads:      {config.num_key_value_heads}")
    print(f"  head_dim:          {config.head_dim}")
    print(f"  use_mag_gate:      {config.use_mag_gate}")
    print(f"  gate_positions:    {config.mag_gate_positions}")
    print(f"  use_residual_gate: {config.use_residual_gate}")
    print(f"  gate_rank:         {config.gate_rank}")
    print(f"  residual_gate_rank:{config.residual_gate_rank}")
    print(f"  tie_embeddings:    {config.tie_word_embeddings}")
    print(f"  vocab_size:        {config.vocab_size}")
    print(f"{'='*60}")

    # Create model
    print("\nInitializing model...")
    model = MagGatedForCausalLM(config)

    total_params, trainable_params = count_parameters(model)
    gate_params, mag_params, residual_gate_params = count_gate_parameters(model)

    print(f"\n  Total parameters:          {total_params:>12,}  ({total_params/1e6:.1f}M)")
    print(f"  Trainable parameters:      {trainable_params:>12,}  ({trainable_params/1e6:.1f}M)")
    print(f"  Gate parameters:           {gate_params:>12,}  ({gate_params/1e6:.1f}M)")
    print(f"  Magnitude parameters:      {mag_params:>12,}  ({mag_params/1e6:.1f}M)")
    print(f"  Residual gate parameters:  {residual_gate_params:>12,}  ({residual_gate_params/1e6:.1f}M)")
    print(f"  Gate overhead:             {(gate_params + mag_params + residual_gate_params) / total_params * 100:.2f}%")

    # Save
    output_dir = args.output_dir
    if output_dir is None:
        name = f"MagGated-{args.variant}-d{args.hidden_size}-L{args.num_hidden_layers}"
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'model_checkpoints', name)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving to: {output_dir}")
    model.save_pretrained(output_dir)

    # Save tokenizer if loaded
    if tokenizer is not None:
        print(f"Saving tokenizer to: {output_dir}")
        tokenizer.save_pretrained(output_dir)

    print(f"\n✓ Model saved successfully!")
    print(f"  Config: {os.path.join(output_dir, 'config.json')}")
    print(f"  Model:  {os.path.join(output_dir, 'model.safetensors')}")

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Create MagGated Transformer checkpoint")
    parser.add_argument('--variant', type=str, default='mag_gated_all',
                        choices=['baseline', 'mag_gated_all', 'mag_gated_bottleneck', 'residual_only'],
                        help='Model variant')
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--intermediate_size', type=int, default=None)
    parser.add_argument('--num_hidden_layers', type=int, default=24)
    parser.add_argument('--num_attention_heads', type=int, default=None)
    parser.add_argument('--num_key_value_heads', type=int, default=None)
    parser.add_argument('--head_dim', type=int, default=128)
    parser.add_argument('--gate_rank', type=int, default=16)
    parser.add_argument('--residual_gate_rank', type=int, default=16)
    parser.add_argument('--max_position_embeddings', type=int, default=8192)
    parser.add_argument('--vocab_size', type=int, default=151936,
                        help='Vocabulary size (default matches Qwen2)')
    parser.add_argument('--tie_word_embeddings', action='store_true', default=True)
    parser.add_argument('--no_tie_word_embeddings', dest='tie_word_embeddings', action='store_false')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--tokenizer_from', type=str, default=None,
                        help='Path to copy tokenizer from (e.g., Qwen/Qwen2.5-0.5B)')

    args = parser.parse_args()
    create_model(args)


if __name__ == '__main__':
    main()
