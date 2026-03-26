#!/usr/bin/env python
"""Create a ResidualGate Transformer model checkpoint for pretraining.

Usage:
    # Create ResidualGate model (d=1024, Qwen3-0.6B aligned)
    python create_mag_gated_model.py \
        --hidden_size 1024 --intermediate_size 3072 --num_hidden_layers 28 \
        --num_attention_heads 16 --num_key_value_heads 8 --head_dim 128 \
        --vocab_size 151936 --max_position_embeddings 40960 \
        --torch_dtype bfloat16 \
        --output_dir model_checkpoints/mag_gated-d1024-L28

    # Create baseline (no ResidualGate, standard transformer)
    python create_mag_gated_model.py --no_residual_gate \
        --hidden_size 1024 --output_dir model_checkpoints/baseline-d1024-L28
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer

from swift.model.mag_gated.configuration_mag_gated import MagGatedConfig
from swift.model.mag_gated.modeling_mag_gated import MagGatedForCausalLM


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_gate_parameters(model):
    gate_params = 0
    for name, p in model.named_parameters():
        if 'gate_A' in name or 'gate_B' in name:
            gate_params += p.numel()
    return gate_params


def create_model(args):
    """Create and save a ResidualGate model checkpoint."""

    # Compute intermediate_size if not specified
    if args.intermediate_size is None:
        intermediate_size = int(args.hidden_size * 3.0)
        intermediate_size = ((intermediate_size + 127) // 128) * 128
    else:
        intermediate_size = args.intermediate_size

    # Compute heads if not specified
    num_attention_heads = args.num_attention_heads
    head_dim = args.head_dim
    if num_attention_heads is None:
        num_attention_heads = max(1, args.hidden_size // head_dim)
    if head_dim is None:
        head_dim = args.hidden_size // num_attention_heads

    num_key_value_heads = args.num_key_value_heads
    if num_key_value_heads is None:
        num_key_value_heads = max(1, num_attention_heads // 2)

    config = MagGatedConfig(
        vocab_size=args.vocab_size,
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
        use_residual_gate=args.use_residual_gate,
        residual_gate_rank=args.residual_gate_rank,
        residual_gate_init_bias=args.residual_gate_init_bias,
        max_window_layers=args.max_window_layers,
        use_sliding_window=args.use_sliding_window,
        sliding_window=args.sliding_window,
        torch_dtype=args.torch_dtype,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
    )

    gate_status = "ON" if args.use_residual_gate else "OFF"
    print(f"\n{'='*60}")
    print(f"Creating ResidualGate Transformer (gate={gate_status})")
    print(f"{'='*60}")
    print(f"  hidden_size:           {config.hidden_size}")
    print(f"  intermediate_size:     {config.intermediate_size}")
    print(f"  num_layers:            {config.num_hidden_layers}")
    print(f"  num_attn_heads:        {config.num_attention_heads}")
    print(f"  num_kv_heads:          {config.num_key_value_heads}")
    print(f"  head_dim:              {config.head_dim}")
    print(f"  use_residual_gate:     {config.use_residual_gate}")
    print(f"  residual_gate_rank:    {config.residual_gate_rank}")
    print(f"  residual_gate_init_bias: {config.residual_gate_init_bias}")
    print(f"  tie_embeddings:        {config.tie_word_embeddings}")
    print(f"  vocab_size:            {config.vocab_size}")
    print(f"  torch_dtype:           {config.torch_dtype}")
    print(f"{'='*60}")

    print("\nInitializing model...")
    model = MagGatedForCausalLM(config)

    # Convert to target dtype
    if args.torch_dtype == "bfloat16":
        print("Converting model to bfloat16...")
        model = model.to(torch.bfloat16)
    elif args.torch_dtype == "float16":
        print("Converting model to float16...")
        model = model.to(torch.float16)

    total_params, trainable_params = count_parameters(model)
    gate_params = count_gate_parameters(model)

    print(f"\n  Total parameters:      {total_params:>12,}  ({total_params/1e6:.1f}M)")
    print(f"  Trainable parameters:  {trainable_params:>12,}  ({trainable_params/1e6:.1f}M)")
    print(f"  Gate parameters:       {gate_params:>12,}  ({gate_params/1e6:.1f}M)")
    if total_params > 0:
        print(f"  Gate overhead:         {gate_params / total_params * 100:.2f}%")

    # Save
    output_dir = args.output_dir
    if output_dir is None:
        name = f"ResGate-d{args.hidden_size}-L{args.num_hidden_layers}"
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'model_checkpoints', name)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving to: {output_dir}")
    model.save_pretrained(output_dir)

    # Copy tokenizer if specified
    if args.tokenizer_from:
        print(f"Copying tokenizer from: {args.tokenizer_from}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_from, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)

    print(f"\n✓ Model saved successfully!")
    print(f"  Config: {os.path.join(output_dir, 'config.json')}")
    print(f"  Model:  {os.path.join(output_dir, 'model.safetensors')}")

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Create ResidualGate Transformer checkpoint")
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--intermediate_size', type=int, default=None)
    parser.add_argument('--num_hidden_layers', type=int, default=28)
    parser.add_argument('--num_attention_heads', type=int, default=None)
    parser.add_argument('--num_key_value_heads', type=int, default=None)
    parser.add_argument('--head_dim', type=int, default=128)
    parser.add_argument('--residual_gate_rank', type=int, default=16)
    parser.add_argument('--residual_gate_init_bias', type=float, default=5.0)
    parser.add_argument('--max_position_embeddings', type=int, default=40960)
    parser.add_argument('--vocab_size', type=int, default=151936)
    parser.add_argument('--tie_word_embeddings', action='store_true', default=True)
    parser.add_argument('--no_tie_word_embeddings', dest='tie_word_embeddings', action='store_false')
    parser.add_argument('--use_residual_gate', action='store_true', default=True)
    parser.add_argument('--no_residual_gate', dest='use_residual_gate', action='store_false')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--tokenizer_from', type=str, default=None)
    parser.add_argument('--max_window_layers', type=int, default=28)
    parser.add_argument('--use_sliding_window', action='store_true', default=False)
    parser.add_argument('--sliding_window', type=int, default=None)
    parser.add_argument('--torch_dtype', type=str, default="bfloat16")
    parser.add_argument('--bos_token_id', type=int, default=151643)
    parser.add_argument('--eos_token_id', type=int, default=151645)

    args = parser.parse_args()
    create_model(args)


if __name__ == '__main__':
    main()
