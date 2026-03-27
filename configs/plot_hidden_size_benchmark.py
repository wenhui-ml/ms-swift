#!/usr/bin/env python
"""Plot loss curves for hidden-size benchmark comparison.

Compares training loss across different hidden-size configurations:
  1. Gated model (d=1024, with attention hidden-size gate)
  2. Baseline d=1024 (standard transformer, no gate)
  3. Baseline d=512  (standard transformer, no gate)
  4. Baseline d=256  (standard transformer, no gate)

Usage:
    python configs/plot_hidden_size_benchmark.py
    python configs/plot_hidden_size_benchmark.py --output_dir output --save_path plots/hidden_size_benchmark.png
"""

import argparse
import json
import os
import glob
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def find_trainer_state(output_dir):
    """Find the latest trainer_state.json in an output directory."""
    # Look for trainer_state.json in checkpoint dirs or root
    patterns = [
        os.path.join(output_dir, '**', 'trainer_state.json'),
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    
    if not files:
        return None
    
    # Return the one with the most recent modification time
    return max(files, key=os.path.getmtime)


def load_loss_from_trainer_state(state_path):
    """Load training loss from trainer_state.json."""
    with open(state_path, 'r') as f:
        state = json.load(f)
    
    steps = []
    losses = []
    
    for entry in state.get('log_history', []):
        if 'loss' in entry and 'step' in entry:
            steps.append(entry['step'])
            losses.append(entry['loss'])
    
    return np.array(steps), np.array(losses)


def smooth_curve(values, weight=0.9):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0] if len(values) > 0 else 0
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def main():
    parser = argparse.ArgumentParser(description="Plot hidden-size benchmark loss curves")
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Base output directory containing training runs')
    parser.add_argument('--save_path', type=str, default='plots/hidden_size_benchmark.png',
                        help='Path to save the plot')
    parser.add_argument('--smooth', type=float, default=0.9,
                        help='Smoothing factor (0=no smoothing, 1=max smoothing)')
    parser.add_argument('--max_step', type=int, default=None,
                        help='Maximum step to plot')
    args = parser.parse_args()

    # Define runs to compare
    runs = [
        {
            'name': 'Gated d=1024 (AttnResGate)',
            'dir': 'attn_res_gate-d1024-L28-v5.1',
            'color': '#e74c3c',  # red
            'linestyle': '-',
            'linewidth': 2.0,
        },
        {
            'name': 'Baseline d=1024 (no gate)',
            'dir': 'baseline-d1024-L28',
            'color': '#3498db',  # blue
            'linestyle': '-',
            'linewidth': 1.5,
        },
        {
            'name': 'Baseline d=512 (no gate)',
            'dir': 'baseline-d512-L28',
            'color': '#2ecc71',  # green
            'linestyle': '-',
            'linewidth': 1.5,
        },
        {
            'name': 'Baseline d=256 (no gate)',
            'dir': 'baseline-d256-L28',
            'color': '#f39c12',  # orange
            'linestyle': '-',
            'linewidth': 1.5,
        },
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    found_any = False
    
    for run in runs:
        run_dir = os.path.join(args.output_dir, run['dir'])
        
        if not os.path.exists(run_dir):
            print(f"  [SKIP] {run['name']}: {run_dir} not found")
            continue
        
        state_path = find_trainer_state(run_dir)
        if state_path is None:
            print(f"  [SKIP] {run['name']}: no trainer_state.json found in {run_dir}")
            continue
        
        steps, losses = load_loss_from_trainer_state(state_path)
        
        if len(steps) == 0:
            print(f"  [SKIP] {run['name']}: no loss data found")
            continue
        
        if args.max_step is not None:
            mask = steps <= args.max_step
            steps = steps[mask]
            losses = losses[mask]
        
        found_any = True
        smoothed = smooth_curve(losses, args.smooth)
        
        print(f"  [OK]   {run['name']}: {len(steps)} points, "
              f"final_loss={losses[-1]:.4f}, min_loss={losses.min():.4f}")
        
        # Raw loss (left plot)
        ax1.plot(steps, losses, 
                 color=run['color'], linestyle=run['linestyle'],
                 linewidth=0.5, alpha=0.3)
        
        # Smoothed loss (both plots)
        ax1.plot(steps, smoothed,
                 color=run['color'], linestyle=run['linestyle'],
                 linewidth=run['linewidth'], label=run['name'])
        
        ax2.plot(steps, smoothed,
                 color=run['color'], linestyle=run['linestyle'],
                 linewidth=run['linewidth'], label=run['name'])
    
    if not found_any:
        print("\nNo training data found! Make sure training has been run.")
        print("Expected directories:")
        for run in runs:
            print(f"  {os.path.join(args.output_dir, run['dir'])}")
        return
    
    # Format left plot (full range)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Hidden-Size Benchmark: Training Loss', fontsize=14)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # Format right plot (log scale)
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Loss (log scale)', fontsize=12)
    ax2.set_title('Hidden-Size Benchmark: Training Loss (Log Scale)', fontsize=14)
    ax2.set_yscale('log')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(left=0)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    plt.savefig(args.save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {args.save_path}")
    
    # Also save a summary table
    summary_path = args.save_path.replace('.png', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Hidden-Size Benchmark Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Model':<35} {'Final Loss':>12} {'Min Loss':>12} {'Steps':>8}\n")
        f.write("-" * 67 + "\n")
        
        for run in runs:
            run_dir = os.path.join(args.output_dir, run['dir'])
            if not os.path.exists(run_dir):
                continue
            state_path = find_trainer_state(run_dir)
            if state_path is None:
                continue
            steps, losses = load_loss_from_trainer_state(state_path)
            if len(steps) == 0:
                continue
            f.write(f"{run['name']:<35} {losses[-1]:>12.4f} {losses.min():>12.4f} {int(steps[-1]):>8}\n")
        
        f.write("\n")
        f.write("Hypothesis: If gated d=1024 achieves loss comparable to or better than\n")
        f.write("baseline d=1024, while baseline d=512 and d=256 show significantly\n")
        f.write("higher loss, it suggests the gate mechanism effectively utilizes\n")
        f.write("the full hidden-size capacity.\n")
    
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
