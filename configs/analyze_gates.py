#!/usr/bin/env python
"""Analyze ResidualGate Transformer gate activations and dimension utilization.

This script loads a trained MagGated model and analyzes:
1. Per-layer ResidualGate activation statistics (α retain, β accept)
2. Dimension utilization rates (how many dims are actually used)
3. Visualizations of gate activations (heatmaps and histograms)

The ResidualGate mechanism:
    h_new = α(h,o) ⊙ h + β(h,o) ⊙ o

where α and β are independently computed per-dimension gates informed by
magnitude/direction signals.

Usage:
    python configs/analyze_gates.py --model_dir output/mag_gated-d1024-L28/checkpoint-5000 --plot_dir plots/
"""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Ensure custom modules are registered
try:
    from swift.model.mag_gated.register_mag_gated import register_mag_gated
    from swift.model.mag_gated.modeling_mag_gated import ResidualGate
except ImportError:
    print("Warning: Could not import MagGated modules directly. Ensure swift is in PYTHONPATH.")
    class ResidualGate: pass


def load_model_and_tokenizer(model_dir, device='cuda'):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_dir} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return model, tokenizer


def find_residual_gates(model):
    """Recursively find all ResidualGate modules in the model."""
    residual_gates = {}

    for name, module in model.named_modules():
        if isinstance(module, ResidualGate) or 'ResidualGate' in str(type(module)):
            residual_gates[name] = module

    return residual_gates


def analyze_gate_activations(model, tokenizer, texts, device='cuda'):
    """Run forward passes and collect ResidualGate activations via hooks."""
    model.eval()

    res_alpha_activations = defaultdict(list)
    res_beta_activations = defaultdict(list)
    hooks = []

    residual_gates = find_residual_gates(model)

    if not residual_gates:
        print("No ResidualGate modules found in this model.")
        return {}, {}

    print(f"Found {len(residual_gates)} ResidualGate modules.")

    # Hook ResidualGate to capture α and β
    for name, module in residual_gates.items():
        def make_res_hook(n):
            def hook_fn(m, input, output):
                with torch.no_grad():
                    residual, new_output = input[0], input[1]
                    # Recompute alpha/beta exactly as in ResidualGate.forward()
                    eps = 1e-6
                    h_mag = residual.detach().abs()
                    o_mag = new_output.detach().abs()
                    mag_ratio = h_mag / (h_mag + o_mag + eps)
                    dir_agree = (residual.detach() * new_output.detach()) / (h_mag * o_mag + eps)
                    gate_input = torch.cat([residual, new_output, mag_ratio, dir_agree], dim=-1)
                    gate_hidden = m.gate_A(gate_input)
                    alpha = torch.sigmoid(m.gate_B_alpha(gate_hidden))
                    beta = torch.sigmoid(m.gate_B_beta(gate_hidden))

                    res_alpha_activations[n].append(alpha.cpu().float())
                    res_beta_activations[n].append(beta.cpu().float())
            return hook_fn
        hooks.append(module.register_forward_hook(make_res_hook(name)))

    # Run forward passes
    print(f"Running {len(texts)} forward passes on {device}...")
    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device if device != 'auto' else 'cuda') for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(texts)} done")

    # Remove hooks
    for h in hooks:
        h.remove()

    return res_alpha_activations, res_beta_activations


def compute_statistics(activations_dict):
    """Compute detailed statistics from collected activations."""
    stats = {}
    for name, acts_list in activations_dict.items():
        if not acts_list:
            continue
        # (total_tokens, d_out)
        all_acts = torch.cat([a.reshape(-1, a.shape[-1]) for a in acts_list], dim=0)

        mean_act = all_acts.mean(dim=0)  # (d_out,)

        # Metrics
        sparsity_01 = (all_acts < 0.1).float().mean().item()
        saturation_09 = (all_acts > 0.9).float().mean().item()
        collapsed_dead = (mean_act < 0.06).float().mean().item()
        collapsed_active = (mean_act > 0.9).float().mean().item()

        stats[name] = {
            'mean': all_acts.mean().item(),
            'std': all_acts.std().item(),
            'sparsity_01': sparsity_01,
            'saturation_09': saturation_09,
            'dead_ratio': collapsed_dead,
            'always_on_ratio': collapsed_active,
            'dim_means': mean_act.numpy().tolist(),
        }
    return stats


def plot_gates(stats, output_dir, prefix="gate"):
    """Generate plots for gate activations."""
    if not stats:
        return
    os.makedirs(output_dir, exist_ok=True)

    # 1. Summary plot: sparsity across layers
    names = sorted(stats.keys())
    layers = range(len(names))
    dead = [stats[n]['dead_ratio'] for n in names]
    always = [stats[n]['always_on_ratio'] for n in names]
    means = [stats[n]['mean'] for n in names]

    plt.figure(figsize=(12, 6))
    plt.plot(layers, dead, label='Dead Dims (<0.06)', marker='o')
    plt.plot(layers, always, label='Always-on Dims (>0.9)', marker='s')
    plt.plot(layers, means, label='Avg Activation', marker='^', linestyle='--')
    plt.title(f'{prefix.capitalize()} Utilization across Layers')
    plt.xlabel('Layer Index (sorted)')
    plt.ylabel('Ratio / Mean')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'{prefix}_summary.png'), dpi=150)
    plt.close()

    # 2. Heatmap of dimension means
    if len(names) > 0:
        plt.figure(figsize=(15, 8))
        plot_names = names[-40:] if len(names) > 40 else names
        dim_data = np.array([stats[n]['dim_means'] for n in plot_names])
        # Sort dimensions by average activation across all layers
        avg_dim = dim_data.mean(axis=0)
        sort_idx = np.argsort(avg_dim)[::-1]
        dim_data_sorted = dim_data[:, sort_idx]

        plt.imshow(dim_data_sorted, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Avg Activation')
        plt.title(f'{prefix.capitalize()} Dimension Specialization (Sorted by Activity)')
        plt.xlabel('Dimension Index (Sorted)')
        plt.ylabel('Layer/Gate')
        plt.yticks(range(len(plot_names)), plot_names, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}_heatmap.png'), dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze ResidualGate activations")
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--plot_dir', type=str, default='gate_analysis_plots')
    parser.add_argument('--output_json', type=str, default='gate_stats.json')
    parser.add_argument('--device', type=str, default='auto', help='auto, cuda, or cpu')
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_dir, device=args.device)

    texts = [
        "The standard Transformer architecture consists of encoder and decoder",
        "def quicksort(arr): if len(arr) <= 1: return arr",
        "Quantum mechanics is a fundamental theory in physics",
        "Today's weather is sunny with a clear blue sky",
        "Paris is the capital of France, known for the Eiffel Tower"
    ] * (args.num_samples // 5 + 1)
    texts = texts[:args.num_samples]

    res_alpha, res_beta = analyze_gate_activations(model, tokenizer, texts, device=args.device)

    stats_alpha = compute_statistics(res_alpha)
    stats_beta = compute_statistics(res_beta)

    # Print summary table
    print("\n" + "=" * 80)
    print("ResidualGate Analysis: α (retain gate) and β (accept gate)")
    print("  h_new = α ⊙ h + β ⊙ o")
    print("=" * 80)

    if stats_alpha:
        print(f"\n{'--- Retain Gate (α) ---':^80}")
        print(f"{'Gate Name':<45} {'Mean':>8} {'Dead':>10} {'Satur':>10}")
        print("-" * 80)
        for n in sorted(stats_alpha.keys()):
            s = stats_alpha[n]
            print(f"{n:<45} {s['mean']:>8.4f} {s['dead_ratio']:>10.4f} {s['saturation_09']:>10.4f}")

    if stats_beta:
        print(f"\n{'--- Accept Gate (β) ---':^80}")
        print(f"{'Gate Name':<45} {'Mean':>8} {'Dead':>10} {'Satur':>10}")
        print("-" * 80)
        for n in sorted(stats_beta.keys()):
            s = stats_beta[n]
            print(f"{n:<45} {s['mean']:>8.4f} {s['dead_ratio']:>10.4f} {s['saturation_09']:>10.4f}")

    # Plotting
    plot_gates(stats_alpha, args.plot_dir, "residual_alpha")
    plot_gates(stats_beta, args.plot_dir, "residual_beta")

    # Save JSON
    final_stats = {
        'residual_alpha': stats_alpha,
        'residual_beta': stats_beta,
    }
    with open(args.output_json, 'w') as f:
        json.dump(final_stats, f, indent=2)

    print(f"\nAnalysis complete. Plots saved to '{args.plot_dir}', stats to '{args.output_json}'.")


if __name__ == '__main__':
    main()
