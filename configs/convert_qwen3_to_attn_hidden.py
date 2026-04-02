#!/usr/bin/env python
"""Convert a trained Qwen3 checkpoint to Attention Hidden-Size Transformer.

Copies all shared weights (Attention, MLP, RMSNorm, Embedding) from Qwen3
and initializes ResidualGate parameters in LoRA-style (K=zeros → h_new = h+o).

Usage:
    python configs/convert_qwen3_to_attn_hidden.py \
        --qwen3_path /home/ubuntu/llm_weights/Qwen3-0.6B \
        --output_dir model_checkpoints/attn_hidden-d1024-L28-v11-sft \
        --residual_gate_num_heads 8 \
        --residual_gate_context_dim 16
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from swift.model.attention_hidden_size.configuration_attn_hidden import AttnHiddenConfig
from swift.model.attention_hidden_size.modeling_attn_hidden import AttnHiddenForCausalLM


def convert(args):
    print(f"{'='*60}")
    print(f"Converting Qwen3 → Attention Hidden-Size Transformer")
    print(f"  Source: {args.qwen3_path}")
    print(f"  Output: {args.output_dir}")
    print(f"  Init mode: sft (LoRA-style, K=zeros → h_new = h+o)")
    print(f"{'='*60}")

    # Load Qwen3 config
    qwen3_config = AutoConfig.from_pretrained(args.qwen3_path, trust_remote_code=True)
    print(f"\nQwen3 config:")
    print(f"  hidden_size: {qwen3_config.hidden_size}")
    print(f"  num_hidden_layers: {qwen3_config.num_hidden_layers}")
    print(f"  num_attention_heads: {qwen3_config.num_attention_heads}")

    # Create AttnHidden config (aligned with Qwen3)
    attn_config = AttnHiddenConfig(
        vocab_size=qwen3_config.vocab_size,
        hidden_size=qwen3_config.hidden_size,
        intermediate_size=qwen3_config.intermediate_size,
        num_hidden_layers=qwen3_config.num_hidden_layers,
        num_attention_heads=qwen3_config.num_attention_heads,
        num_key_value_heads=qwen3_config.num_key_value_heads,
        head_dim=qwen3_config.head_dim,
        hidden_act=qwen3_config.hidden_act,
        max_position_embeddings=qwen3_config.max_position_embeddings,
        initializer_range=qwen3_config.initializer_range,
        rms_norm_eps=qwen3_config.rms_norm_eps,
        use_cache=qwen3_config.use_cache,
        tie_word_embeddings=qwen3_config.tie_word_embeddings,
        rope_theta=qwen3_config.rope_theta,
        use_residual_gate=True,
        residual_gate_num_heads=args.residual_gate_num_heads,
        residual_gate_context_dim=args.residual_gate_context_dim,
        residual_gate_init_mode="sft",
        residual_gate_init_remove_scale=args.init_remove_scale,
        max_window_layers=getattr(qwen3_config, 'max_window_layers', qwen3_config.num_hidden_layers),
        bos_token_id=qwen3_config.bos_token_id,
        eos_token_id=qwen3_config.eos_token_id,
        attention_bias=getattr(qwen3_config, 'attention_bias', False),
        mlp_bias=getattr(qwen3_config, 'mlp_bias', False),
        attention_dropout=getattr(qwen3_config, 'attention_dropout', 0.0),
    )

    # Load Qwen3 model
    print(f"\nLoading Qwen3 model...")
    qwen3_model = AutoModelForCausalLM.from_pretrained(
        args.qwen3_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    qwen3_state = qwen3_model.state_dict()
    qwen3_params = sum(p.numel() for p in qwen3_model.parameters())
    print(f"  Qwen3 parameters: {qwen3_params:,}")

    # Create AttnHidden model
    print(f"\nCreating AttnHidden model (init_mode=sft)...")
    attn_model = AttnHiddenForCausalLM(attn_config)
    attn_model = attn_model.to(torch.bfloat16)
    attn_state = attn_model.state_dict()
    attn_params = sum(p.numel() for p in attn_model.parameters())
    print(f"  AttnHidden parameters: {attn_params:,}")

    # Copy weights
    print(f"\nCopying weights...")
    copied = 0
    skipped = 0
    gate_only = 0

    for name, param in attn_state.items():
        if 'residual_gate' in name:
            gate_only += 1
            continue  # Keep gate's LoRA-style initialization

        # Map AttnHidden name to Qwen3 name
        # AttnHidden: model.layers.0.self_attn.q_proj.weight
        # Qwen3:      model.layers.0.self_attn.q_proj.weight
        # They should be identical (same structure minus gate)
        if name in qwen3_state:
            if param.shape == qwen3_state[name].shape:
                attn_state[name] = qwen3_state[name]
                copied += 1
            else:
                print(f"  ⚠️  Shape mismatch: {name} "
                      f"(attn={param.shape}, qwen3={qwen3_state[name].shape})")
                skipped += 1
        else:
            print(f"  ⚠️  Not found in Qwen3: {name}")
            skipped += 1

    attn_model.load_state_dict(attn_state)

    print(f"\n  Copied: {copied} parameters")
    print(f"  Skipped: {skipped} parameters")
    print(f"  Gate (kept init): {gate_only} parameters")

    # Verify gate initialization
    gate = attn_model.model.layers[0].attn_residual_gate
    k_h_max = gate.k_h_proj.weight.data.abs().max().item()
    k_o_max = gate.k_o_proj.weight.data.abs().max().item()
    print(f"\n  Gate K_h max abs: {k_h_max:.6f} (should be 0.0)")
    print(f"  Gate K_o max abs: {k_o_max:.6f} (should be 0.0)")
    assert k_h_max < 1e-6, "K_h should be zeros for SFT init!"
    assert k_o_max < 1e-6, "K_o should be zeros for SFT init!"
    print(f"  ✓ Gate initialized in SFT mode (K=zeros → h_new = h+o)")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving to: {args.output_dir}")
    attn_model.save_pretrained(args.output_dir)

    # Make checkpoint self-contained
    src_dir = os.path.join(os.path.dirname(__file__), '..', 'swift', 'model', 'attention_hidden_size')
    for fname in ['configuration_attn_hidden.py', 'modeling_attn_hidden.py']:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(args.output_dir, fname)
        shutil.copy2(src_path, dst_path)

    # Patch relative import
    modeling_path = os.path.join(args.output_dir, 'modeling_attn_hidden.py')
    with open(modeling_path, 'r') as f:
        content = f.read()
    old_import = 'from .configuration_attn_hidden import AttnHiddenConfig'
    new_import = (
        "import importlib, pathlib as _pl\n"
        "_cfg_path = _pl.Path(__file__).parent / 'configuration_attn_hidden.py'\n"
        "_spec = importlib.util.spec_from_file_location('configuration_attn_hidden', _cfg_path)\n"
        "_mod = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_mod)\n"
        "AttnHiddenConfig = _mod.AttnHiddenConfig"
    )
    if old_import in content:
        content = content.replace(old_import, new_import)
        with open(modeling_path, 'w') as f:
            f.write(content)

    # Add auto_map
    config_json_path = os.path.join(args.output_dir, 'config.json')
    with open(config_json_path, 'r') as f:
        config_dict = json.load(f)
    config_dict['auto_map'] = {
        'AutoConfig': 'configuration_attn_hidden.AttnHiddenConfig',
        'AutoModelForCausalLM': 'modeling_attn_hidden.AttnHiddenForCausalLM',
    }
    with open(config_json_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Copy tokenizer
    print(f"Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.qwen3_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n{'='*60}")
    print(f"✓ Conversion complete!")
    print(f"  {copied} weights copied from Qwen3")
    print(f"  {gate_only} gate parameters initialized (SFT mode)")
    print(f"  Initial behavior: h_new = h + o (exact standard residual)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3 → AttnHidden")
    parser.add_argument('--qwen3_path', type=str, required=True,
                        help="Path to trained Qwen3 checkpoint")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Output directory for AttnHidden checkpoint")
    parser.add_argument('--residual_gate_num_heads', type=int, default=8)
    parser.add_argument('--residual_gate_context_dim', type=int, default=16)
    parser.add_argument('--init_remove_scale', type=float, default=0.1)
    args = parser.parse_args()
    convert(args)


if __name__ == '__main__':
    main()
