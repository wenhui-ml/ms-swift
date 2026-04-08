#!/usr/bin/env python
"""Convert a trained Qwen3 checkpoint to Attention Hidden-Size Transformer.

Copies all shared weights (Attention, MLP, RMSNorm, Embedding) from Qwen3
and initializes SynapticGate parameters for zero-loss fallback:
    w_forget = w_acquire = 0
    b_forget = b_acquire = +4.0
    → σ(0·RMSNorm(x) + 4.0) = σ(4.0) ≈ 0.982
    → h_new ≈ 0.982·h + 0.982·o ≈ h + o (exact standard residual)

Gate formula:
    gate_forget  = σ(w_forget ⊙ RMSNorm(h) + b_forget)
    gate_acquire = σ(w_acquire ⊙ RMSNorm(o) + b_acquire)
    h_new = gate_forget ⊙ h + gate_acquire ⊙ o

Usage:
    python configs/convert_qwen3_to_attn_hidden.py \
        --qwen3_path /home/ubuntu/llm_weights/Qwen3-0.6B \
        --output_dir model_checkpoints/attn_hidden-d1024-L28-v12-sft \
        --synaptic_gate_init_bias 4.0
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
    print(f"Converting Qwen3 → Attention Hidden-Size Transformer (V12)")
    print(f"  Source: {args.qwen3_path}")
    print(f"  Output: {args.output_dir}")
    print(f"  Init mode: sft (w=0, b=+{args.synaptic_gate_init_bias})")
    print(f"  → σ({args.synaptic_gate_init_bias}) ≈ {torch.sigmoid(torch.tensor(args.synaptic_gate_init_bias)).item():.4f}")
    print(f"  → h_new ≈ h + o (zero-loss fallback)")
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
        use_synaptic_gate=True,
        synaptic_gate_init_bias=args.synaptic_gate_init_bias,
        synaptic_gate_init_mode="sft",
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
    print(f"\nCreating AttnHidden model (init_mode=sft, init_bias={args.synaptic_gate_init_bias})...")
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
        if 'synaptic_gate' in name:
            gate_only += 1
            continue  # Keep gate's SFT initialization (w=0, b=+4.0)

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
    gate = attn_model.model.layers[0].attn_synaptic_gate
    w_forget_max = gate.w_forget.data.abs().max().item()
    w_acquire_max = gate.w_acquire.data.abs().max().item()
    b_forget_val = gate.b_forget.data.mean().item()
    b_acquire_val = gate.b_acquire.data.mean().item()
    print(f"\n  Gate w_forget max abs: {w_forget_max:.6f} (should be 0.0)")
    print(f"  Gate w_acquire max abs: {w_acquire_max:.6f} (should be 0.0)")
    print(f"  Gate b_forget mean: {b_forget_val:.4f} (should be {args.synaptic_gate_init_bias})")
    print(f"  Gate b_acquire mean: {b_acquire_val:.4f} (should be {args.synaptic_gate_init_bias})")
    assert w_forget_max < 1e-6, "w_forget should be zeros for SFT init!"
    assert w_acquire_max < 1e-6, "w_acquire should be zeros for SFT init!"
    sig_val = torch.sigmoid(torch.tensor(args.synaptic_gate_init_bias)).item()
    print(f"  ✓ Gate initialized in SFT mode (σ({args.synaptic_gate_init_bias})≈{sig_val:.4f} → h_new ≈ h+o)")

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

    # Parameter count summary
    gate_params = sum(p.numel() for n, p in attn_model.named_parameters() if 'synaptic_gate' in n)
    print(f"\n{'='*60}")
    print(f"✓ Conversion complete!")
    print(f"  {copied} weights copied from Qwen3")
    print(f"  {gate_only} gate parameters initialized (SFT mode)")
    print(f"  Gate overhead: {gate_params:,} params ({gate_params/attn_params*100:.3f}%)")
    print(f"  Per gate: 4 × {qwen3_config.hidden_size} = {4 * qwen3_config.hidden_size:,} scalars")
    print(f"  Initial behavior: h_new ≈ h + o (exact standard residual)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3 → AttnHidden (V12 Synaptic Gating)")
    parser.add_argument('--qwen3_path', type=str, required=True,
                        help="Path to trained Qwen3 checkpoint")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Output directory for AttnHidden checkpoint")
    parser.add_argument('--synaptic_gate_init_bias', type=float, default=4.0,
                        help="Initial bias for gate parameters (default: 4.0, σ(4.0)≈0.982)")
    args = parser.parse_args()
    convert(args)


if __name__ == '__main__':
    main()
