# Copyright (c) 2024. All rights reserved.
"""
MagGate Monitor Callback: Logs gate statistics during training.

Detects degenerate gate behavior:
- Gates all → 1 (saturation): Model degrades to standard Transformer, gating has no effect
- Gates all → 0 (collapse): Model is dying, information cannot flow
- Gates undifferentiated: All gates ~0.5, no dimension specialization
- Healthy: Gates show variation (some on, some off, depending on input)

Usage with HuggingFace Trainer:
    from swift.model.mag_gated.gate_monitor_callback import MagGateMonitorCallback

    trainer = Trainer(
        model=model,
        callbacks=[MagGateMonitorCallback(log_every_n_steps=50)],
        ...
    )
"""

import logging
from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


class MagGateMonitorCallback(TrainerCallback):
    """Callback that monitors gate values during MagGated model training.

    Logs gate statistics to:
    1. TensorBoard / WandB (via trainer.log)
    2. Console logger (summary every N steps)
    3. Detailed per-layer breakdown every N_detail steps

    Warns if gates degenerate (all-on, all-off, or undifferentiated),
    indicating the model has collapsed or the gate mechanism is not learning.
    """

    def __init__(self, log_every_n_steps: int = 1,
                 detail_every_n_steps: int = 10,
                 warn_threshold_saturation: float = 0.95,
                 warn_threshold_sparsity: float = 0.95):
        """
        Args:
            log_every_n_steps: How often to log gate stats summary (default: every 1 steps).
            detail_every_n_steps: How often to log detailed per-layer breakdown (default: every 10 steps).
            warn_threshold_saturation: If global saturation > this, log a warning.
            warn_threshold_sparsity: If global sparsity > this, log a warning.
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.detail_every_n_steps = detail_every_n_steps
        self.warn_threshold_saturation = warn_threshold_saturation
        self.warn_threshold_sparsity = warn_threshold_sparsity
        # Track history for trend analysis
        self._history = []

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, model=None, **kwargs):
        """Verify gate initialization at the start of training."""
        if model is None:
            return

        unwrapped = model
        if hasattr(model, 'module'):
            unwrapped = model.module

        if hasattr(unwrapped, 'verify_gate_init'):
            result = unwrapped.verify_gate_init()
            if not result.get("all_ok", True):
                logger.error(
                    "[GateMonitor] ⚠️  CRITICAL: Gate initialization is WRONG at training start!\n"
                    "  This will cause gates to be stuck near 0.5 (no differentiation).\n"
                    "  The auto-fix in register_mag_gated should have caught this.\n"
                    "  If you see this, please report a bug."
                )

    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, model=None, logs=None, **kwargs):
        """Called when trainer logs metrics. We piggyback on this to log gate stats."""
        if model is None:
            return

        # Only log every N steps
        if state.global_step % self.log_every_n_steps != 0:
            return

        # Check if model has get_gate_stats method
        # Handle DeepSpeed / FSDP wrapped models
        unwrapped = model
        if hasattr(model, 'module'):
            unwrapped = model.module

        if not hasattr(unwrapped, 'get_gate_stats'):
            return

        gate_stats = unwrapped.get_gate_stats()
        if not gate_stats:
            return

        # Filter numeric values for logging (exclude string health status and tensor values)
        numeric_stats = {k: v for k, v in gate_stats.items()
                         if isinstance(v, (int, float))}

        # Log to TensorBoard/WandB via trainer
        if logs is not None:
            logs.update(numeric_stats)

        # === Console summary (every log_every_n_steps) ===
        health = gate_stats.get("gate/health", "unknown")
        global_mean = gate_stats.get("gate/global_mean", -1)
        global_std = gate_stats.get("gate/global_std", -1)
        global_sparsity = gate_stats.get("gate/global_sparsity", -1)
        global_saturation = gate_stats.get("gate/global_saturation", -1)
        dim_active_ratio = gate_stats.get("gate/dim_active_ratio", -1)
        dim_reuse_score = gate_stats.get("gate/dim_reuse_score", -1)

        # Residual gate stats
        res_mean = gate_stats.get("gate/residual_global_mean", -1)
        res_forget = gate_stats.get("gate/residual_global_forget_ratio", -1)

        # Magnitude (m) stats
        m_mean = gate_stats.get("mag/global_m_mean", -1)
        m_std = gate_stats.get("mag/global_m_std", -1)
        m_diff = gate_stats.get("mag/m_differentiation", -1)

        msg = (
            f"[GateMonitor step={state.global_step}] "
            f"health={health} | "
            f"mean={global_mean:.4f} | "
            f"std={global_std:.4f} | "
            f"sparsity={global_sparsity:.4f} | "
            f"saturation={global_saturation:.4f}"
        )

        # Add magnitude info
        if m_mean >= 0:
            msg += f" | m_mean={m_mean:.4f}"
        if m_diff >= 0:
            msg += f" | m_diff={m_diff:.4f}"

        # Add dimension analysis if available
        if dim_active_ratio >= 0:
            msg += f" | dim_active={dim_active_ratio:.3f}"
        if dim_reuse_score >= 0:
            msg += f" | dim_reuse={dim_reuse_score:.4f}"

        # Add residual gate info if available
        if res_mean >= 0:
            msg += f" | res_mean={res_mean:.4f}"
        if res_forget >= 0:
            msg += f" | res_forget={res_forget:.4f}"

        # Track history for trend analysis
        self._history.append({
            "step": state.global_step,
            "mean": global_mean,
            "std": global_std,
            "sparsity": global_sparsity,
            "saturation": global_saturation,
        })

        # Health-based logging
        if health == "warning_all_on":
            logger.warning(
                f"{msg}\n"
                f"  ⚠️  Gates are nearly ALL ON (saturation={global_saturation:.3f} > {self.warn_threshold_saturation})\n"
                f"  → Model has degenerated into a standard Transformer. Gating mechanism is not learning.\n"
                f"  → Root cause: gate_init_bias too high → sigmoid gradient too small → gates stuck.\n"
                f"  → Fix: lower gate_init_bias (recommended: 0.5), increase gate_A init std (0.1),\n"
                f"         use get_gate_param_groups() for higher gate learning rate."
            )
        elif health == "warning_all_off":
            logger.warning(
                f"{msg}\n"
                f"  ⚠️  Gates are nearly ALL OFF (sparsity={global_sparsity:.3f} > {self.warn_threshold_sparsity})\n"
                f"  → Model is dying! Information cannot flow through gated layers.\n"
                f"  → Consider: increase gate_floor, raise gate_init_bias, or reduce learning rate."
            )
        elif health == "warning_no_differentiation":
            logger.warning(
                f"{msg}\n"
                f"  ⚠️  Gates show NO DIFFERENTIATION (sparsity≈0, saturation≈0, all gates ≈ {global_mean:.3f})\n"
                f"  → Gate mechanism is not learning to distinguish dimensions!\n"
                f"  → All {int(dim_active_ratio * 100) if dim_active_ratio >= 0 else '?'}% dims have similar gate values.\n"
                f"  → This means: no dimension specialization, no time-multiplexing.\n"
                f"  → Consider: increase gate learning rate (use get_gate_param_groups()), "
                f"reduce gate_init_bias, or increase gate_rank."
            )
        else:
            logger.info(msg)

        # === Detailed per-layer breakdown (every detail_every_n_steps) ===
        if state.global_step % self.detail_every_n_steps == 0:
            self._log_detailed_breakdown(gate_stats, state.global_step)

        # === Trend analysis (every 5 log intervals) ===
        if len(self._history) >= 5 and len(self._history) % 5 == 0:
            self._log_trend_analysis(state.global_step)

    def _log_detailed_breakdown(self, gate_stats: dict, step: int):
        """Log detailed per-layer gate statistics."""
        lines = [f"\n{'='*80}"]
        lines.append(f"[GateMonitor step={step}] Detailed Per-Layer Gate Breakdown")
        lines.append(f"{'='*80}")

        # Collect per-layer data
        layer_data = {}
        for key, value in gate_stats.items():
            if not isinstance(value, (int, float)):
                continue
            # Parse keys like "gate/layer0_q_proj_mean"
            if key.startswith("gate/layer") and "_summary_" not in key:
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

        # Print MagGatedLinear stats per layer
        lines.append(f"\n  {'Layer':>6} | {'Proj':>12} | {'Mean':>7} | {'Std':>7} | {'Min':>7} | {'Max':>7} | {'Sparse':>7} | {'Satur':>7}")
        lines.append(f"  {'-'*6}-+-{'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

        for layer_idx in sorted(layer_data.keys()):
            data = layer_data[layer_idx]
            # Group by projection name
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
                mean = p.get("mean", -1)
                std = p.get("std", -1)
                mn = p.get("min", -1)
                mx = p.get("max", -1)
                sp = p.get("sparsity", -1)
                sa = p.get("saturation", -1)
                lines.append(
                    f"  {layer_idx:>6} | {proj_name:>12} | {mean:>7.4f} | {std:>7.4f} | "
                    f"{mn:>7.4f} | {mx:>7.4f} | {sp:>7.4f} | {sa:>7.4f}"
                )

        # Print summary stats
        lines.append(f"\n  Summary:")
        for key in ["gate/global_mean", "gate/global_std", "gate/global_sparsity",
                     "gate/global_saturation", "gate/dim_active_ratio", "gate/dim_reuse_score",
                     "gate/residual_global_mean", "gate/residual_global_forget_ratio",
                     "gate/residual_global_sparsity", "gate/residual_global_saturation"]:
            if key in gate_stats and isinstance(gate_stats[key], (int, float)):
                short_key = key.replace("gate/", "")
                lines.append(f"    {short_key:>35}: {gate_stats[key]:.6f}")

        lines.append(f"\n  Dimension Analysis:")
        lines.append(f"    Each MagGatedLinear has gate g(x) ∈ (0,1)^d_out — one gate per output dimension.")
        lines.append(f"    Each ResidualGate has forget f(h) ∈ (0,1)^d — one gate per hidden dimension.")

        dim_active = gate_stats.get("gate/dim_active_ratio", -1)
        dim_reuse = gate_stats.get("gate/dim_reuse_score", -1)
        if dim_active >= 0:
            lines.append(f"    Active dims (mean gate > 0.5): {dim_active*100:.1f}% of hidden_size")
        if dim_reuse >= 0:
            lines.append(f"    Reuse score (cross-layer std):  {dim_reuse:.4f}")
            if dim_reuse < 0.01:
                lines.append(f"    ⚠️  Low reuse score → dimensions NOT being used differently across layers")
                lines.append(f"       → No time-multiplexing happening yet")
            elif dim_reuse > 0.1:
                lines.append(f"    ✓  Good reuse score → dimensions being specialized per layer")

        lines.append(f"{'='*80}")
        logger.info("\n".join(lines))

    def _log_trend_analysis(self, step: int):
        """Analyze gate value trends over recent history."""
        if len(self._history) < 5:
            return

        recent = self._history[-5:]
        first = recent[0]
        last = recent[-1]

        mean_delta = last["mean"] - first["mean"]
        std_delta = last["std"] - first["std"]
        sparsity_delta = last["sparsity"] - first["sparsity"]

        steps_span = last["step"] - first["step"]

        lines = [
            f"[GateMonitor step={step}] Trend (last {steps_span} steps): "
            f"Δmean={mean_delta:+.4f} | Δstd={std_delta:+.4f} | Δsparsity={sparsity_delta:+.4f}"
        ]

        # Detect stagnation
        if abs(mean_delta) < 0.001 and abs(std_delta) < 0.001:
            lines.append(
                f"  ⚠️  Gate values STAGNANT — mean and std barely changed over {steps_span} steps."
                f" Gate mechanism may not be receiving sufficient gradient signal."
            )

        # Detect positive differentiation trend
        if sparsity_delta > 0.01 or std_delta > 0.005:
            lines.append(
                f"  ✓  Positive differentiation trend — gates are starting to specialize."
            )

        logger.info(" ".join(lines) if len(lines) == 1 else "\n".join(lines))

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, model=None, **kwargs):
        """Log final gate stats at end of training."""
        if model is None:
            return

        unwrapped = model
        if hasattr(model, 'module'):
            unwrapped = model.module

        if not hasattr(unwrapped, 'get_gate_stats'):
            return

        gate_stats = unwrapped.get_gate_stats()
        if not gate_stats:
            return

        health = gate_stats.get("gate/health", "unknown")
        global_mean = gate_stats.get("gate/global_mean", -1)
        global_std = gate_stats.get("gate/global_std", -1)
        global_sparsity = gate_stats.get("gate/global_sparsity", -1)
        global_saturation = gate_stats.get("gate/global_saturation", -1)
        dim_active = gate_stats.get("gate/dim_active_ratio", -1)
        dim_reuse = gate_stats.get("gate/dim_reuse_score", -1)
        res_mean = gate_stats.get("gate/residual_global_mean", -1)
        res_forget = gate_stats.get("gate/residual_global_forget_ratio", -1)

        logger.info(
            f"\n{'='*80}\n"
            f"[GateMonitor] Final gate statistics at end of training:\n"
            f"\n"
            f"  MagGatedLinear Gates (y = m ⊙ g(x) ⊙ V̂·x):\n"
            f"    Health:           {health}\n"
            f"    Global mean:      {global_mean:.4f}  (ideal: 0.3~0.7 with variation)\n"
            f"    Global std:       {global_std:.4f}  (ideal: > 0.05, higher = more differentiation)\n"
            f"    Global sparsity:  {global_sparsity:.4f}  (fraction of gates < 0.1 = 'off' dims)\n"
            f"    Global saturation:{global_saturation:.4f}  (fraction of gates > 0.9 = 'on' dims)\n"
            f"\n"
            f"  Dimension Analysis:\n"
            f"    Active dim ratio: {dim_active:.4f}  (fraction of dims with mean gate > 0.5)\n"
            f"    Reuse score:      {dim_reuse:.4f}  (cross-layer std, higher = more specialization)\n"
            f"\n"
            f"  ResidualGate (h_new = f(h) ⊙ h + output):\n"
            f"    Mean:             {res_mean:.4f}  (ideal: 0.7~0.95, selective retention)\n"
            f"    Forget ratio:     {res_forget:.4f}  (fraction of dims with f < 0.5)\n"
            f"\n"
            f"  Interpretation:\n"
            f"    ✓ Healthy: sparsity > 0.05 AND saturation < 0.90 AND std > 0.05\n"
            f"      → Some dims ON, some OFF, input-dependent → dimension multiplexing working\n"
            f"    ⚠ No differentiation: sparsity ≈ 0 AND saturation ≈ 0\n"
            f"      → All gates ≈ same value → no dimension specialization\n"
            f"      → Fix: increase gate lr (get_gate_param_groups), reduce gate_init_bias\n"
            f"    ⚠ All on: saturation > 0.95\n"
            f"      → Gates not learning → model = standard Transformer\n"
            f"    ⚠ All off: sparsity > 0.95\n"
            f"      → Model dying → check learning rate / gate_floor\n"
            f"\n"
            f"  Paper Thesis Verification:\n"
            f"    For d/2 MagGated ≥ d Standard, we need:\n"
            f"    1. dim_active_ratio < 0.7 (not all dims always on)\n"
            f"    2. dim_reuse_score > 0.05 (different layers use different dims)\n"
            f"    3. sparsity > 0.05 (some dims genuinely turned off)\n"
            f"{'='*80}"
        )

        # Log the detailed breakdown one final time
        self._log_detailed_breakdown(gate_stats, state.global_step)
