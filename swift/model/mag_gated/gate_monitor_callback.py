# Copyright (c) 2024. All rights reserved.
"""
ResidualGate Monitor Callback: Logs gate statistics during training.

Monitors the dual-gate (α retain, β accept) ResidualGate mechanism:
    h_new = α(h,o) ⊙ h + β(h,o) ⊙ o

Detects degenerate gate behavior:
- α,β all → 1 (saturation): Model = standard Transformer, no gating effect
- α,β all → 0 (collapse): Model dying, information cannot flow
- Healthy: α,β show variation — selective retention/acceptance per dimension

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

        # Console summary
        alpha_mean = gate_stats.get("gate/residual_alpha_global_mean", -1)
        beta_mean = gate_stats.get("gate/residual_beta_global_mean", -1)
        sparsity = gate_stats.get("gate/residual_global_sparsity", -1)
        saturation = gate_stats.get("gate/residual_global_saturation", -1)
        forget_ratio = gate_stats.get("gate/residual_global_forget_ratio", -1)

        msg = (
            f"[GateMonitor step={state.global_step}] "
            f"α_mean={alpha_mean:.4f} | β_mean={beta_mean:.4f} | "
            f"sparsity={sparsity:.4f} | saturation={saturation:.4f} | "
            f"forget_ratio={forget_ratio:.4f}"
        )
        logger.info(msg)

        # Detailed breakdown at intervals
        if state.global_step % self.detail_every_n_steps == 0:
            self._log_detailed_breakdown(gate_stats, state.global_step)

    def _log_detailed_breakdown(self, gate_stats: dict, step: int):
        """Log detailed per-layer ResidualGate statistics."""
        lines = [f"\n{'='*80}"]
        lines.append(f"[GateMonitor step={step}] Detailed Per-Layer Gate Breakdown")
        lines.append(f"{'='*80}")

        # Parse per-layer stats
        layer_data = {}
        for key, value in gate_stats.items():
            if not isinstance(value, (int, float)):
                continue
            if key.startswith("gate/layer") and "global" not in key:
                parts = key.replace("gate/layer", "").split("_", 1)
                if len(parts) == 2:
                    try:
                        layer_idx = int(parts[0])
                    except ValueError:
                        continue
                    rest = parts[1]
                    if layer_idx not in layer_data:
                        layer_data[layer_idx] = {}
                    layer_data[layer_idx][rest] = value

        # Print table
        lines.append(f"\n   {'Layer':>5} | {'Proj':>12} | {'Mean':>7} | {'Std':>7} | {'Min':>7} | {'Max':>7} | {'Sparse':>7} | {'Satur':>7}")
        lines.append(f"  {'-'*6}-+-{'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

        for layer_idx in sorted(layer_data.keys()):
            data = layer_data[layer_idx]
            projs = {}
            for key, val in data.items():
                parts = key.rsplit("_", 1)
                if len(parts) == 2:
                    proj_name, stat_name = parts
                    if proj_name not in projs:
                        projs[proj_name] = {}
                    projs[proj_name][stat_name] = val

            for proj_name in sorted(projs.keys()):
                p = projs[proj_name]
                lines.append(
                    f"  {layer_idx:>6} | {proj_name:>12} | {p.get('mean', -1):>7.4f} | "
                    f"{p.get('std', -1):>7.4f} | {p.get('min', -1):>7.4f} | "
                    f"{p.get('max', -1):>7.4f} | {p.get('sparsity', -1):>7.4f} | "
                    f"{p.get('saturation', -1):>7.4f}"
                )

        # Summary
        lines.append(f"\n  Summary:")
        for key in ["gate/residual_alpha_global_mean", "gate/residual_beta_global_mean",
                     "gate/residual_global_sparsity", "gate/residual_global_saturation",
                     "gate/residual_global_forget_ratio"]:
            if key in gate_stats and isinstance(gate_stats[key], (int, float)):
                short_key = key.replace("gate/", "")
                lines.append(f"    {short_key:>35}: {gate_stats[key]:.6f}")

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
