#!/usr/bin/env python3
"""
Analyze SynapticGate statistics from a trained AttnHidden checkpoint.

Runs a forward pass on sample text and reports per-layer gate statistics for:
  - attn_synaptic_gate_forget  : σ(w_forget ⊙ RMSNorm(h) + b_forget) — retention of residual h
  - attn_synaptic_gate_acquire : σ(w_acquire ⊙ RMSNorm(o) + b_acquire) — acceptance of attn output o
  - ffn_synaptic_gate_forget   : σ(w_forget ⊙ RMSNorm(h) + b_forget) — retention of residual h
  - ffn_synaptic_gate_acquire  : σ(w_acquire ⊙ RMSNorm(o) + b_acquire) — acceptance of FFN output o

RMSNorm inside the gate is parameter-free (signal conditioning only).

The gate formula is:
    h_new = gate_forget ⊙ h + gate_acquire ⊙ o

Gate values are in [0, 1] (sigmoid output):
  ≈ 1.0  → full pass-through (standard residual)
  < 0.9  → active gating (partial filtering)
  < 0.5  → strong filtering / blocking
  ≈ 0.0  → full blocking

Initial state (σ(4.0) ≈ 0.982): all gates ≈ 1 → h_new ≈ h + o

Columns:
  Mean   : mean gate value over all tokens and dimensions
  Std    : standard deviation
  Min    : minimum gate value
  Max    : maximum gate value
  Active : fraction of gate values with v < active_threshold (gate is doing something)
  Block  : fraction of gate values with v < block_threshold (gate is blocking)

Usage:
    python analyze_gates.py [checkpoint_path] [options]

Examples:
    python analyze_gates.py
    python analyze_gates.py /path/to/checkpoint-239
    python analyze_gates.py --layers 0,1,2,27
    python analyze_gates.py --max-length 512 --num-samples 4
"""

import sys
import os
import argparse
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = (
    "/home/ubuntu/wenhui/mag_gate/ms-swift/output/"
    "qwen3-0.6b-attn_hidden-d1024-L28-v12-sft/"
    "v1/checkpoint-latest"
)

SWIFT_ROOT = "/home/ubuntu/wenhui/mag_gate/ms-swift"

# ---------------------------------------------------------------------------
# Sample texts for the forward pass
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    (
        "The quick brown fox jumps over the lazy dog. "
        "Once upon a time in a land far away, there lived a wise king who ruled with justice and compassion. "
        "In mathematics, the Pythagorean theorem states that the square of the hypotenuse of a right triangle "
        "equals the sum of the squares of the other two sides. "
        "Scientists have discovered that the observable universe is approximately 13.8 billion years old "
        "and contains hundreds of billions of galaxies. "
        "Language models learn statistical patterns from large corpora of text data. "
        "The capital of France is Paris, famous for the Eiffel Tower and the Louvre museum. "
        "Neural networks consist of layers of interconnected nodes that transform input data. "
        "The Python programming language was created by Guido van Rossum in the late 1980s. "
        "Climate change poses significant challenges to ecosystems and human societies worldwide."
    ),
    (
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data. "
        "The transformer architecture, introduced in 2017, has revolutionized natural language processing. "
        "Attention mechanisms allow models to focus on relevant parts of the input sequence. "
        "Residual connections help mitigate the vanishing gradient problem in deep networks. "
        "Layer normalization stabilizes training by normalizing activations within each layer. "
        "The softmax function converts raw logits into a probability distribution over classes. "
        "Stochastic gradient descent updates model parameters in the direction of steepest loss descent. "
        "Regularization techniques such as dropout help prevent overfitting in neural networks."
    ),
    (
        "The history of computing begins with Charles Babbage's Analytical Engine in the 19th century. "
        "Alan Turing's theoretical work laid the foundation for modern computer science. "
        "The first electronic computer, ENIAC, was built in 1945 and weighed 30 tons. "
        "Transistors replaced vacuum tubes in the 1950s, enabling smaller and faster computers. "
        "The internet was originally developed as ARPANET in the late 1960s for military communication. "
        "Tim Berners-Lee invented the World Wide Web in 1989, transforming global communication. "
        "Moore's Law predicted that transistor density would double approximately every two years. "
        "Graphics processing units have become essential for training deep learning models."
    ),
    (
        "Proteins are large biomolecules consisting of chains of amino acid residues. "
        "DNA carries the genetic information that directs the development and functioning of living organisms. "
        "The cell is the basic structural and functional unit of all known living organisms. "
        "Photosynthesis converts light energy into chemical energy stored in glucose molecules. "
        "The human brain contains approximately 86 billion neurons connected by trillions of synapses. "
        "Evolution by natural selection is the primary mechanism of biological change over time. "
        "Viruses are submicroscopic agents that replicate inside living cells of organisms. "
        "Antibiotics are medicines that kill or inhibit the growth of bacteria."
    ),
]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_stats(tensor: torch.Tensor, active_th: float, block_th: float) -> dict:
    """Compute gate statistics over all elements of tensor."""
    t = tensor.float().flatten()
    return {
        "mean":   t.mean().item(),
        "std":    t.std().item(),
        "min":    t.min().item(),
        "max":    t.max().item(),
        "active": (t < active_th).float().mean().item(),
        "block":  (t < block_th).float().mean().item(),
    }


def merge_stats(stats_list: list) -> dict:
    """Merge statistics from multiple forward passes."""
    if len(stats_list) == 1:
        return stats_list[0]
    n = len(stats_list)
    merged = {}
    for key in stats_list[0]:
        vals = [s[key] for s in stats_list]
        if key == "mean":
            merged[key] = sum(vals) / n
        elif key == "std":
            merged[key] = (sum(v**2 for v in vals) / n) ** 0.5
        elif key == "min":
            merged[key] = min(vals)
        elif key == "max":
            merged[key] = max(vals)
        elif key in ("active", "block"):
            merged[key] = sum(vals) / n
    return merged


# ---------------------------------------------------------------------------
# Gate data collection
# ---------------------------------------------------------------------------

def collect_gate_data(model, inputs: dict) -> dict:
    """
    Run a forward pass and collect raw gate tensors from each layer.

    Returns:
        {layer_idx: {"attn": (gate_forget, gate_acquire),
                     "ffn":  (gate_forget, gate_acquire)}}
    """
    model.train()
    with torch.no_grad():
        _ = model(**inputs, use_cache=False)
    model.eval()

    gate_data = {}
    for layer_idx, layer in enumerate(model.model.layers):
        if not getattr(layer, "use_synaptic_gate", False):
            continue

        layer_data = {}
        for gate_key, gate_name in [("attn", "attn_synaptic_gate"),
                                     ("ffn",  "ffn_synaptic_gate")]:
            gate = getattr(layer, gate_name, None)
            if gate is None or gate._gate_raw is None:
                continue
            gate_forget, gate_acquire = gate._gate_raw
            layer_data[gate_key] = (gate_forget.cpu(), gate_acquire.cpu())

        if layer_data:
            gate_data[layer_idx] = layer_data

    return gate_data


def build_gate_entries(gate_data: dict, active_th: float, block_th: float) -> list:
    """
    Build list of (layer_idx, proj_label, stats_dict) entries.

    Each gate produces two rows: _forget (gate_forget) and _acquire (gate_acquire).
    """
    entries = []
    for layer_idx in sorted(gate_data.keys()):
        ld = gate_data[layer_idx]
        for gate_key, prefix in [("attn", "attn_synaptic_gate"),
                                  ("ffn",  "ffn_synaptic_gate")]:
            if gate_key not in ld:
                continue
            gate_forget, gate_acquire = ld[gate_key]
            entries.append((
                layer_idx,
                f"{prefix}_forget",
                compute_stats(gate_forget, active_th, block_th),
            ))
            entries.append((
                layer_idx,
                f"{prefix}_acquire",
                compute_stats(gate_acquire, active_th, block_th),
            ))
    return entries


def accumulate_entries(entries_list: list, active_th: float, block_th: float) -> list:
    """Merge entries across multiple forward passes."""
    if len(entries_list) == 1:
        return entries_list[0]

    from collections import defaultdict
    grouped = defaultdict(list)
    for entries in entries_list:
        for layer_idx, label, stats in entries:
            grouped[(layer_idx, label)].append(stats)

    merged = []
    seen = set()
    for layer_idx, label, _ in entries_list[0]:
        key = (layer_idx, label)
        if key not in seen:
            seen.add(key)
            merged.append((layer_idx, label, merge_stats(grouped[key])))
    return merged


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_table(entries: list):
    """Print gate statistics as a formatted table."""
    if not entries:
        print("No gate data collected.")
        return

    proj_width = max(len(label) for _, label, _ in entries)
    proj_width = max(proj_width, 12)

    val_w = 8

    hdr_layer = "Layer"
    hdr_proj  = "Gate"
    hdr_mean  = "Mean"
    hdr_std   = "Std"
    hdr_min   = "Min"
    hdr_max   = "Max"
    hdr_active= "Active"
    hdr_block = "Block"

    header = (
        f"  {hdr_layer:>7} | {hdr_proj:<{proj_width}} | "
        f"{hdr_mean:>{val_w}} | {hdr_std:>{val_w}} | "
        f"{hdr_min:>{val_w}} | {hdr_max:>{val_w}} | "
        f"{hdr_active:>{val_w}} | {hdr_block:>{val_w}}"
    )
    sep_layer  = "-" * 8
    sep_proj   = "-" * (proj_width + 2)
    sep_val    = "-" * (val_w + 1)
    divider = (
        f"  {sep_layer}+{sep_proj}+"
        + (f"{sep_val}+" * 5)
        + f"{sep_val[:-1]}"
    )

    print(header)
    print(divider)

    prev_layer = None
    for layer_idx, label, stats in entries:
        if prev_layer is not None and layer_idx != prev_layer:
            print()
        prev_layer = layer_idx

        row = (
            f"  {layer_idx:>7} | {label:<{proj_width}} | "
            f"{stats['mean']:>{val_w}.4f} | "
            f"{stats['std']:>{val_w}.4f} | "
            f"{stats['min']:>{val_w}.4f} | "
            f"{stats['max']:>{val_w}.4f} | "
            f"{stats['active']:>{val_w}.4f} | "
            f"{stats['block']:>{val_w}.4f}"
        )
        print(row)


def print_bias_table(model, layer_indices: list = None, gate_entries: list = None):
    """Print the learned bias values per layer gate, plus mean gate values."""
    entry_lookup = {}
    if gate_entries:
        for layer_idx, label, stats in gate_entries:
            gate_type = "attn_synaptic_gate" if label.startswith("attn") else "ffn_synaptic_gate"
            key = (layer_idx, gate_type)
            if key not in entry_lookup:
                entry_lookup[key] = {}
            if "_forget" in label:
                entry_lookup[key]["forget"] = stats["mean"]
            elif "_acquire" in label:
                entry_lookup[key]["acquire"] = stats["mean"]

    print()
    print("  Learned gate biases and mean gate values:")
    print(f"  {'Layer':>7} | {'Gate':<25} | {'b_f mean':>8} | {'b_a mean':>8} | {'w_f rms':>8} | {'w_a rms':>8} | {'g_f mean':>8} | {'g_a mean':>8}")
    print(f"  {'-'*8}+{'-'*27}+{'-'*10}+{'-'*10}+{'-'*10}+{'-'*10}+{'-'*10}+{'-'*10}")

    all_b_forget = []
    all_b_acquire = []
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_indices is not None and layer_idx not in layer_indices:
            continue
        if not getattr(layer, "use_synaptic_gate", False):
            continue
        for gate_name in ["attn_synaptic_gate", "ffn_synaptic_gate"]:
            gate = getattr(layer, gate_name, None)
            if gate is None:
                continue
            b_f = gate.b_forget.data.float().mean().item()
            b_a = gate.b_acquire.data.float().mean().item()
            w_f_rms = gate.w_forget.data.float().pow(2).mean().sqrt().item()
            w_a_rms = gate.w_acquire.data.float().pow(2).mean().sqrt().item()
            all_b_forget.append(b_f)
            all_b_acquire.append(b_a)

            key = (layer_idx, gate_name)
            entry = entry_lookup.get(key, {})
            mean_gf = entry.get("forget", 0.0)
            mean_ga = entry.get("acquire", 0.0)

            print(
                f"  {layer_idx:>7} | {gate_name:<25} | {b_f:>+8.4f} | {b_a:>+8.4f} | {w_f_rms:>8.5f} | {w_a_rms:>8.5f} | {mean_gf:>8.4f} | {mean_ga:>8.4f}"
            )

    if all_b_forget:
        print()
        init_bias = getattr(model.config, 'synaptic_gate_init_bias', 4.0)
        print(f"  Bias summary: b_forget min={min(all_b_forget):.4f} max={max(all_b_forget):.4f} mean={sum(all_b_forget)/len(all_b_forget):.4f}  init={init_bias:.1f}")
        print(f"                b_acquire min={min(all_b_acquire):.4f} max={max(all_b_acquire):.4f} mean={sum(all_b_acquire)/len(all_b_acquire):.4f}  init={init_bias:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze SynapticGate statistics from an AttnHidden V12 checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "checkpoint", nargs="?", default=DEFAULT_CHECKPOINT,
        help=f"Path to checkpoint directory (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated list of layer indices to show (default: all)",
    )
    parser.add_argument(
        "--max-length", type=int, default=256,
        help="Max token length per sample (default: 256)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=len(SAMPLE_TEXTS),
        help=f"Number of sample texts to use (default: {len(SAMPLE_TEXTS)})",
    )
    parser.add_argument(
        "--active-threshold", type=float, default=0.9,
        help="Gate activity threshold: gate < threshold means active (default: 0.9)",
    )
    parser.add_argument(
        "--block-threshold", type=float, default=0.5,
        help="Gate blocking threshold: gate < threshold means blocking (default: 0.5)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use: 'cpu', 'cuda', 'cuda:0', etc. (auto-detect by default)",
    )
    args = parser.parse_args()

    # --- Setup ---
    if SWIFT_ROOT not in sys.path:
        sys.path.insert(0, SWIFT_ROOT)

    checkpoint_path = str(args.checkpoint)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {device}")
    print(f"Max length : {args.max_length} tokens per sample")
    print(f"Samples    : {min(args.num_samples, len(SAMPLE_TEXTS))}")
    print(f"Active thr : gate < {args.active_threshold}")
    print(f"Block  thr : gate < {args.block_threshold}")

    # Parse layer filter
    layer_filter = None
    if args.layers:
        layer_filter = set(int(x.strip()) for x in args.layers.split(","))

    # --- Load tokenizer ---
    from transformers import AutoTokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    # --- Load model ---
    from transformers import AutoModelForCausalLM
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    # Verify gate presence
    sample_layer = model.model.layers[0]
    if not getattr(sample_layer, "use_synaptic_gate", False):
        print("\nERROR: This checkpoint does not have synaptic gates enabled.")
        sys.exit(1)

    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    init_bias = getattr(model.config, 'synaptic_gate_init_bias', 4.0)
    print(f"Layers      : {num_layers}")
    print(f"Hidden size : {hidden_size}")
    print(f"Init bias   : {init_bias} (σ≈{torch.sigmoid(torch.tensor(init_bias)).item():.4f})")
    print(f"Gate params : 4 × {hidden_size} = {4 * hidden_size:,} per gate")

    # --- Run forward passes ---
    texts = SAMPLE_TEXTS[: max(1, min(args.num_samples, len(SAMPLE_TEXTS)))]
    all_entries = []

    for sample_idx, text in enumerate(texts):
        print(f"\nForward pass {sample_idx + 1}/{len(texts)}...", end=" ", flush=True)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]
        print(f"({seq_len} tokens)")

        gate_data = collect_gate_data(model, inputs)
        entries = build_gate_entries(gate_data, args.active_threshold, args.block_threshold)
        all_entries.append(entries)

    # --- Merge and filter ---
    merged_entries = accumulate_entries(all_entries, args.active_threshold, args.block_threshold)

    if layer_filter is not None:
        merged_entries = [(li, lb, st) for li, lb, st in merged_entries if li in layer_filter]

    # --- Print table ---
    print()
    print(f"Gate Statistics (averaged over {len(texts)} sample(s)):")
    print(f"  Gate values in [0, 1]: ≈1.0 = pass-through, <0.9 = active, <0.5 = blocking")
    print_table(merged_entries)

    # --- Print bias table ---
    layer_indices_to_show = layer_filter
    print_bias_table(
        model,
        list(layer_indices_to_show) if layer_indices_to_show else None,
        gate_entries=merged_entries,
    )

    # --- Summary ---
    if merged_entries:
        all_means   = [st["mean"]   for _, _, st in merged_entries]
        all_stds    = [st["std"]    for _, _, st in merged_entries]
        all_actives = [st["active"] for _, _, st in merged_entries]
        all_blocks  = [st["block"]  for _, _, st in merged_entries]

        forget_means = [st["mean"] for _, lb, st in merged_entries if "_forget" in lb]
        acquire_means = [st["mean"] for _, lb, st in merged_entries if "_acquire" in lb]

        print()
        print("Summary:")
        print(f"  Global mean gate value       : {sum(all_means)/len(all_means):.4f}")
        print(f"  Mean forget gate (retain h)  : {sum(forget_means)/len(forget_means):.4f}")
        print(f"  Mean acquire gate (accept o) : {sum(acquire_means)/len(acquire_means):.4f}")
        print(f"  Global active fraction       : {sum(all_actives)/len(all_actives):.4f}")
        print(f"  Global blocking fraction     : {sum(all_blocks)/len(all_blocks):.4f}")
        print(f"  (Initial state: all gates ≈ {torch.sigmoid(torch.tensor(init_bias)).item():.4f})")


if __name__ == "__main__":
    main()
