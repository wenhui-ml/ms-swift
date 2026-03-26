# Copyright (c) 2024. All rights reserved.
"""
Attention Hidden-Size Residual Gate Monitor Callback: Logs gate statistics during training.

Monitors the grouped attention-based ResidualGate mechanism (V5):
    h_new = (1 - forget) ⊙ h + accept ⊙ o

where forget and accept are computed via grouped Q·K attention over the hidden-size dimension.

Key metrics to monitor:
    retain (1-forget): How much old residual info is preserved per group
    accept:            How much new sub-layer output is accepted per group
    forget_active:     Ratio of groups actively forgetting (forget > 0.9)
    forget_inactive:   Ratio of groups not forgetting (forget < 0.1) — should decrease during training
    tau (τ):           Temperature controlling gate sharpness

Detects degenerate gate behavior:
- All retain≈1, accept≈1: Gate not yet diverged (normal at start, should change after ~100-500 steps)
- All retain≈0: Residual info completely discarded — possible gradient issue
- All accept≈0: New info completely rejected — gate stuck, information bottleneck

Usage:
    from swift.model.mag_gated.gate_monitor_callback import MagGateMonitorCallback
    trainer = Trainer(model=model, callbacks=[MagGateMonitorCallback()], ...)
"""

import logging
from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


class MagGateMonitorCallback(TrainerCallback):
    """Callback that monitors ResidualGate values during training."""

    def __init__(self, log_every_n_steps: int = 1,
                 detail_every_n_steps: int = 10):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.detail_every_n_steps = detail_every_n_steps

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, model=None, **kwargs):
        if model is None:
            return
        unwrapped = model.module if hasattr(model, 'module') else model
        if hasattr(unwrapped, 'verify_gate_init'):
            result = unwrapped.verify_gate_init()
            if not result.get("all_ok", True):
                logger.error(
                    "[GateMonitor] ⚠️  CRITICAL: Gate initialization is WRONG at training start!\n"
                    "  Call model.reinit_gates() to fix."
                )

    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, model=None, logs=None, **kwargs):
        if model is None:
            return
        if state.global_step % self.log_every_n_steps != 0:
            return

        unwrapped = model.module if hasattr(model, 'module') else model
        if not hasattr(unwrapped, 'get_gate_stats'):
            return

        gate_stats = unwrapped.get_gate_stats()
        if not gate_stats:
            return

        # Log numeric stats to TensorBoard/WandB
        numeric_stats = {k: v for k, v in gate_stats.items() if isinstance(v, (int, float))}
        if logs is not None:
            logs.update(numeric_stats)

        # Console summary (V5: forget/accept gate semantics)
        retain_mean = gate_stats.get("gate/residual_alpha_global_mean", -1)  # (1-forget) mean
        accept_mean = gate_stats.get("gate/residual_beta_global_mean", -1)   # accept mean
        forget_active = gate_stats.get("gate/residual_global_sparsity", -1)  # groups with forget>0.9
        forget_inactive = gate_stats.get("gate/residual_global_saturation", -1)  # groups with forget<0.1
        forget_ratio = gate_stats.get("gate/residual_global_forget_ratio", -1)  # groups with forget>0.5

        msg = (
            f"[GateMonitor step={state.global_step}] "
            f"retain={retain_mean:.4f} | accept={accept_mean:.4f} | "
            f"forget_active={forget_active:.4f} | forget_inactive={forget_inactive:.4f} | "
            f"forget_ratio={forget_ratio:.4f}"
        )
        logger.info(msg)

        # Detailed breakdown at intervals
        if state.global_step % self.detail_every_n_steps == 0:
            self._log_detailed_breakdown(gate_stats, state.global_step)

    def _log_detailed_breakdown(self, gate_stats: dict, step: int):
        """Log detailed per-layer ResidualGate statistics (V5 grouped attention gate)."""
        lines = [f"\n{'='*80}"]
        lines.append(f"[GateMonitor step={step}] V5 Grouped Attention Gate Breakdown")
        lines.append(f"  h_new = (1-forget)⊙h + accept⊙o")
        lines.append(f"{'='*80}")

        # Parse per-layer stats into structured data
        layer_gates = {}  # {layer_idx: {gate_name: {stat_name: value}}}
        for key, value in gate_stats.items():
            if not isinstance(value, (int, float)):
                continue
            if not key.startswith("gate/layer"):
                continue
            if "global" in key:
                continue
            # key format: gate/layer{idx}_{gate_name}_{stat} or gate/layer{idx}_{gate_name}_beta_{stat}
            rest = key.replace("gate/layer", "")
            parts = rest.split("_", 1)
            if len(parts) < 2:
                continue
            try:
                layer_idx = int(parts[0])
            except ValueError:
                continue
            remainder = parts[1]
            if layer_idx not in layer_gates:
                layer_gates[layer_idx] = {}
            layer_gates[layer_idx][remainder] = value

        # Print compact table: one row per layer, showing attn and ffn gates
        lines.append(f"\n  {'Layer':>5} | {'Gate':>8} | {'Retain':>7} | {'Accept':>7} | {'Forget%':>8} | {'τ':>6}")
        lines.append(f"  {'-'*5}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}")

        for layer_idx in sorted(layer_gates.keys()):
            data = layer_gates[layer_idx]
            for gate_type in ["attn", "ffn"]:
                prefix = f"{gate_type}_residual_gate"
                retain = data.get(f"{prefix}_mean", -1)
                accept = data.get(f"{prefix}_beta_mean", -1)
                forget_r = data.get(f"{prefix}_forget_ratio", -1)
                # Get temperature from gate stats if available
                tau_key = f"gate/layer{layer_idx}_{prefix}_tau"
                tau = gate_stats.get(tau_key, -1)
                lines.append(
                    f"  {layer_idx:>5} | {gate_type:>8} | {retain:>7.4f} | {accept:>7.4f} | "
                    f"{forget_r*100:>7.2f}% | {tau:>6.3f}" if isinstance(tau, float) and tau > 0
                    else f"  {layer_idx:>5} | {gate_type:>8} | {retain:>7.4f} | {accept:>7.4f} | "
                    f"{forget_r*100:>7.2f}% |    -"
                )

        # Global summary
        lines.append(f"\n  Global Summary:")
        retain_g = gate_stats.get("gate/residual_alpha_global_mean", -1)
        accept_g = gate_stats.get("gate/residual_beta_global_mean", -1)
        forget_active_g = gate_stats.get("gate/residual_global_sparsity", -1)
        forget_inactive_g = gate_stats.get("gate/residual_global_saturation", -1)
        forget_ratio_g = gate_stats.get("gate/residual_global_forget_ratio", -1)
        lines.append(f"    retain (1-forget) mean:  {retain_g:.6f}")
        lines.append(f"    accept mean:             {accept_g:.6f}")
        lines.append(f"    groups actively forgetting (>0.9): {forget_active_g:.4f}")
        lines.append(f"    groups not forgetting (<0.1):      {forget_inactive_g:.4f}")
        lines.append(f"    groups with forget>0.5:            {forget_ratio_g:.4f}")

        lines.append(f"{'='*80}")
        logger.info("\n".join(lines))

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, model=None, **kwargs):
        if model is None:
            return
        unwrapped = model.module if hasattr(model, 'module') else model
        if not hasattr(unwrapped, 'get_gate_stats'):
            return

        gate_stats = unwrapped.get_gate_stats()
        if gate_stats:
            self._log_detailed_breakdown(gate_stats, state.global_step)
