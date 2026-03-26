# Copyright (c) 2024. All rights reserved.
# Register Attention Hidden-Size Residual Gate Transformer in ms-swift
"""Registration of AttnResGate Transformer for ms-swift training pipeline.

This model uses grouped attention over the hidden-size dimension to compute
selective residual gates, replacing the standard h = h + o with:
    h_new = (1 - forget) ⊙ h + accept ⊙ o

Auto-registers:
1. AttnResGate (mag_gated) model with HuggingFace AutoClasses
2. MagGateMonitorCallback (AttnResGate monitor) into Trainer (via monkey-patch)
"""

import logging
from transformers import AutoConfig, AutoModelForCausalLM

from swift.model.mag_gated.configuration_mag_gated import MagGatedConfig
from swift.model.mag_gated.modeling_mag_gated import MagGatedForCausalLM
from swift.model.mag_gated.gate_monitor_callback import MagGateMonitorCallback

logger = logging.getLogger(__name__)

_CALLBACK_INJECTED = False


def register_mag_gated():
    """Register MagGated model with HuggingFace AutoClasses.

    This allows loading MagGated models with:
        AutoConfig.from_pretrained(path)
        AutoModelForCausalLM.from_pretrained(path)
    """
    AutoConfig.register("mag_gated", MagGatedConfig)
    AutoModelForCausalLM.register(MagGatedConfig, MagGatedForCausalLM)


def inject_gate_monitor_callback(trainer, log_every_n_steps: int = 10,
                                  detail_every_n_steps: int = 20):
    """Add MagGateMonitorCallback to an existing Trainer instance.

    Safe to call multiple times — only injects once.
    Only injects if the model is a MagGated model with gate stats support.

    Also verifies gate initialization and auto-fixes if corrupted (e.g., from
    checkpoints saved with old code that didn't preserve gate_init_bias).

    Usage:
        from swift.model.mag_gated.register_mag_gated import inject_gate_monitor_callback
        inject_gate_monitor_callback(trainer, log_every_n_steps=50)

    Or auto-inject by calling auto_inject_gate_monitor() before training.
    """
    global _CALLBACK_INJECTED

    if _CALLBACK_INJECTED:
        return

    # Check if the model is a MagGated model
    model = trainer.model
    if hasattr(model, 'module'):
        model = model.module

    if not hasattr(model, 'get_gate_stats'):
        return

    # === Verify and auto-fix gate initialization ===
    if hasattr(model, 'verify_gate_init'):
        result = model.verify_gate_init()
        if not result.get("all_ok", True):
            logger.warning(
                "[MagGated] ⚠️  Detected corrupted gate initialization in loaded checkpoint! "
                "Auto-fixing by re-initializing gate parameters..."
            )
            if hasattr(model, 'reinit_gates'):
                model.reinit_gates()
                # Verify again after fix
                result2 = model.verify_gate_init()
                if result2.get("all_ok", False):
                    logger.info("[MagGated] ✓ Gate initialization successfully repaired!")
                else:
                    logger.error("[MagGated] ✗ Gate initialization repair FAILED! Check model code.")

    # Check if callback already exists
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, MagGateMonitorCallback):
            _CALLBACK_INJECTED = True
            return

    callback = MagGateMonitorCallback(
        log_every_n_steps=log_every_n_steps,
        detail_every_n_steps=detail_every_n_steps,
    )
    trainer.add_callback(callback)
    _CALLBACK_INJECTED = True
    logger.info(
        f"[MagGated] ✓ GateMonitorCallback injected (logging every {log_every_n_steps} steps, "
        f"detail every {detail_every_n_steps} steps). Gate health will be monitored during training."
    )


def auto_inject_gate_monitor():
    """Monkey-patch HuggingFace Trainer.train() to auto-inject GateMonitorCallback.

    Call this once after register_mag_gated(). It patches Trainer so that
    whenever train() is called with a MagGated model, the callback is
    automatically added.

    This is the recommended way to integrate gate monitoring with swift pt.
    """
    try:
        from transformers import Trainer

        _original_train = Trainer.train

        def _patched_train(self, *args, **kwargs):
            inject_gate_monitor_callback(self, log_every_n_steps=50, detail_every_n_steps=50)
            return _original_train(self, *args, **kwargs)

        Trainer.train = _patched_train
        logger.info("[MagGated] ✓ Trainer.train() patched for auto gate monitoring.")
    except Exception as e:
        logger.warning(f"[MagGated] Could not patch Trainer for gate monitoring: {e}")


# Auto-register on import
register_mag_gated()
auto_inject_gate_monitor()
