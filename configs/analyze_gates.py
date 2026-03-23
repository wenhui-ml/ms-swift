#!/usr/bin/env python
"""Analyze MagGated Transformer gate activations and dimension utilization.

This script loads a trained MagGated model and analyzes:
1. Per-layer gate activation statistics (sparsity, mean, distribution)
2. Dimension utilization rates (how many dims are actually used)
3. Effective rank of hidden states
4. Comparison with baseline model hidden state rank

Usage:
    python scripts/analyze_gates.py --model_dir output/mag_gated_all-d512-L12/checkpoint-5000
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from swift.model.mag_gated.register_mag_gated import register_mag_gated
from swift.model.mag_gated.modeling_mag_gated import MagGatedLinear, ResidualGate


def load_model_and_tokenizer(model_dir):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return model, tokenizer


def get_gate_params(model):
    """Extract gate parameters from model."""
    gate_info = defaultdict(dict)
    for name, param in model.named_parameters():
        if '.gate_A.' in name or '.gate_B.' in name:
            # Parse layer index and component
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    component = '.'.join(parts[i + 2:])
                    gate_info[layer_idx][component] = param.data.clone()
                    break
        elif name.endswith('.m') and 'layers.' in name:
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    component = '.'.join(parts[i + 2:])
                    gate_info[layer_idx][component] = param.data.clone()
                    break
    return dict(gate_info)


def analyze_gate_activations(model, tokenizer, texts, device='cuda'):
    """Run forward passes and collect gate activations."""
    model.eval()
    
    # Hook to capture gate activations
    gate_activations = defaultdict(list)
    residual_gate_activations = defaultdict(list)
    hooks = []
    
    for layer_idx, layer in enumerate(model.model.layers):
        # Hook MagGated linear layers
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            proj = getattr(layer.self_attn, proj_name)
            if isinstance(proj, MagGatedLinear):
                def make_hook(name, idx):
                    def hook_fn(module, input, output):
                        with torch.no_grad():
                            x = input[0]
                            gate = torch.sigmoid(module.gate_B(module.gate_A(x)))
                            gate_activations[f'L{idx}.{name}'].append(gate.cpu().float())
                    return hook_fn
                h = proj.register_forward_hook(make_hook(proj_name, layer_idx))
                hooks.append(h)
        
        for proj_name in ['up_proj', 'down_proj']:
            proj = getattr(layer.mlp, proj_name)
            if isinstance(proj, MagGatedLinear):
                def make_hook(name, idx):
                    def hook_fn(module, input, output):
                        with torch.no_grad():
                            x = input[0]
                            gate = torch.sigmoid(module.gate_B(module.gate_A(x)))
                            gate_activations[f'L{idx}.{name}'].append(gate.cpu().float())
                    return hook_fn
                h = proj.register_forward_hook(make_hook(proj_name, layer_idx))
                hooks.append(h)
        
        # Hook residual gates
        if hasattr(layer, 'attn_residual_gate'):
            def make_res_hook(name, idx):
                def hook_fn(module, input, output):
                    with torch.no_grad():
                        residual = input[0]
                        gate = torch.sigmoid(module.gate_B(module.gate_A(residual)))
                        residual_gate_activations[f'L{idx}.{name}'].append(gate.cpu().float())
                return hook_fn
            h = layer.attn_residual_gate.register_forward_hook(
                make_res_hook('attn_res_gate', layer_idx))
            hooks.append(h)
            h = layer.ffn_residual_gate.register_forward_hook(
                make_res_hook('ffn_res_gate', layer_idx))
            hooks.append(h)
    
    # Run forward passes
    print(f"Running {len(texts)} forward passes...")
    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(texts)} done")
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return gate_activations, residual_gate_activations


def compute_statistics(gate_activations):
    """Compute statistics from gate activations."""
    stats = {}
    for name, acts_list in gate_activations.items():
        # Concatenate all activations: (total_tokens, d_out)
        all_acts = torch.cat([a.reshape(-1, a.shape[-1]) for a in acts_list], dim=0)
        
        mean_act = all_acts.mean(dim=0)  # (d_out,) — average gate value per dim
        std_act = all_acts.std(dim=0)
        
        # Sparsity: fraction of dims with average gate < 0.1
        sparsity_01 = (mean_act < 0.1).float().mean().item()
        sparsity_03 = (mean_act < 0.3).float().mean().item()
        sparsity_05 = (mean_act < 0.5).float().mean().item()
        
        # Active dims: fraction of dims with average gate > 0.5
        active_ratio = (mean_act > 0.5).float().mean().item()
        
        # Per-token sparsity: for each token, how many dims are < 0.1
        per_token_sparsity = (all_acts < 0.1).float().mean(dim=-1).mean().item()
        
        stats[name] = {
            'global_mean': mean_act.mean().item(),
            'global_std': mean_act.std().item(),
            'sparsity_01': sparsity_01,
            'sparsity_03': sparsity_03,
            'sparsity_05': sparsity_05,
            'active_ratio': active_ratio,
            'per_token_sparsity': per_token_sparsity,
            'dim_mean_min': mean_act.min().item(),
            'dim_mean_max': mean_act.max().item(),
        }
    
    return stats


def print_report(stats, residual_stats=None):
    """Print a nice report of gate statistics."""
    
    print("\n" + "=" * 80)
    print("MagGated Gate Activation Analysis")
    print("=" * 80)
    
    if stats:
        print("\n--- MagGated Linear Gate Statistics ---")
        print(f"{'Layer.Proj':<25} {'Mean':>8} {'Sparse<0.1':>12} {'Sparse<0.3':>12} "
              f"{'Active>0.5':>12} {'TokenSparse':>12}")
        print("-" * 80)
        
        for name in sorted(stats.keys()):
            s = stats[name]
            print(f"{name:<25} {s['global_mean']:>8.4f} {s['sparsity_01']:>12.4f} "
                  f"{s['sparsity_03']:>12.4f} {s['active_ratio']:>12.4f} "
                  f"{s['per_token_sparsity']:>12.4f}")
    
    if residual_stats:
        print("\n--- Residual Gate Statistics ---")
        print(f"{'Layer.Gate':<25} {'Mean':>8} {'Sparse<0.1':>12} {'Sparse<0.3':>12} "
              f"{'Active>0.5':>12} {'TokenSparse':>12}")
        print("-" * 80)
        
        for name in sorted(residual_stats.keys()):
            s = residual_stats[name]
            print(f"{name:<25} {s['global_mean']:>8.4f} {s['sparsity_01']:>12.4f} "
                  f"{s['sparsity_03']:>12.4f} {s['active_ratio']:>12.4f} "
                  f"{s['per_token_sparsity']:>12.4f}")
    
    # Summary
    if stats:
        all_sparsity = [s['per_token_sparsity'] for s in stats.values()]
        all_mean = [s['global_mean'] for s in stats.values()]
        print(f"\n--- Summary ---")
        print(f"  Average gate activation:     {np.mean(all_mean):.4f}")
        print(f"  Average per-token sparsity:  {np.mean(all_sparsity):.4f}")
        print(f"  → {np.mean(all_sparsity)*100:.1f}% of dimensions are <0.1 per token")
        
        # Effective dimension usage
        total_dims = len(stats)
        print(f"  Number of gated projections: {total_dims}")


def main():
    parser = argparse.ArgumentParser(description="Analyze MagGated gate activations")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of text samples to analyze')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for statistics')
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_dir}")
    model, tokenizer = load_model_and_tokenizer(args.model_dir)
    
    # Generate sample texts (simple diverse prompts)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In mathematics, a prime number is a natural number greater than 1",
        "Machine learning is a subset of artificial intelligence",
        "The capital of France is Paris, which is known for",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1)",
        "Water is composed of hydrogen and oxygen atoms.",
        "The stock market experienced significant volatility today",
        "Climate change is one of the most pressing issues",
        "Shakespeare wrote many famous plays including Hamlet",
        "The theory of relativity was developed by Albert Einstein",
    ] * (args.num_samples // 10 + 1)
    sample_texts = sample_texts[:args.num_samples]
    
    # Analyze
    gate_acts, residual_acts = analyze_gate_activations(model, tokenizer, sample_texts)
    
    if not gate_acts and not residual_acts:
        print("\nNo gate activations found. This model may not use MagGated layers.")
        return
    
    stats = compute_statistics(gate_acts)
    residual_stats = compute_statistics(residual_acts)
    
    print_report(stats, residual_stats)
    
    if args.output:
        results = {'gate_stats': stats, 'residual_stats': residual_stats}
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nStatistics saved to: {args.output}")


if __name__ == '__main__':
    main()
