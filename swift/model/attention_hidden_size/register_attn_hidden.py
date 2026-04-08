# Copyright (c) 2024. All rights reserved.
# Register Attention Hidden-Size Transformer in ms-swift
"""Registration of Attention Hidden-Size Transformer for ms-swift training pipeline.

Auto-registers:
1. AttnHidden (attn_hidden) model with HuggingFace AutoClasses
2. Gate monitor callback into Trainer (via monkey-patch)
"""

import logging
from transformers import AutoConfig, AutoModelForCausalLM

from swift.model.attention_hidden_size.configuration_attn_hidden import AttnHiddenConfig
from swift.model.attention_hidden_size.modeling_attn_hidden import AttnHiddenForCausalLM, SynapticGate

logger = logging.getLogger(__name__)

_CALLBACK_INJECTED = False


def register_attn_hidden():
    """Register AttnHidden model with HuggingFace AutoClasses."""
    AutoConfig.register("attn_hidden", AttnHiddenConfig)
    AutoModelForCausalLM.register(AttnHiddenConfig, AttnHiddenForCausalLM)


def inject_gate_monitor_callback(trainer, log_every_n_steps: int = 50,
                                  detail_every_n_steps: int = 50):
    """Add gate monitor callback to an existing Trainer instance."""
    global _CALLBACK_INJECTED

    if _CALLBACK_INJECTED:
        return

    model = trainer.model
    if hasattr(model, 'module'):
        model = model.module

    if not hasattr(model, 'get_gate_stats'):
        return

    # Check if callback already exists
    from transformers import TrainerCallback

    class AttnHiddenGateMonitorCallback(TrainerCallback):
        """Monitor SynapticGate statistics during training."""

        def __init__(self, log_every_n_steps=50, detail_every_n_steps=50):
            self.log_every_n_steps = log_every_n_steps
            self.detail_every_n_steps = detail_every_n_steps

        def on_log(self, args, state, control, logs=None, model=None, **kwargs):
            if model is None:
                return
            if hasattr(model, 'module'):
                model = model.module
            if not hasattr(model, 'get_gate_stats'):
                return

            step = state.global_step
            if step % self.log_every_n_steps != 0:
                return

            stats = model.get_gate_stats()
            if not stats:
                return

            # Log global stats
            for k, v in stats.items():
                if 'global' in k:
                    if logs is not None:
                        logs[k] = v

            # Log per-layer detail
            if step % self.detail_every_n_steps == 0:
                for k, v in stats.items():
                    if logs is not None:
                        logs[k] = v

            # Reset gate stats
            for layer in model.model.layers:
                if hasattr(layer, 'attn_synaptic_gate'):
                    layer.attn_synaptic_gate._gate_stats = None
                if hasattr(layer, 'ffn_synaptic_gate'):
                    layer.ffn_synaptic_gate._gate_stats = None

    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, AttnHiddenGateMonitorCallback):
            _CALLBACK_INJECTED = True
            return

    callback = AttnHiddenGateMonitorCallback(
        log_every_n_steps=log_every_n_steps,
        detail_every_n_steps=detail_every_n_steps,
    )
    trainer.add_callback(callback)
    _CALLBACK_INJECTED = True
    logger.info(
        f"[AttnHidden] ✓ GateMonitorCallback injected (logging every {log_every_n_steps} steps)"
    )


def auto_inject_gate_monitor():
    """Monkey-patch HuggingFace Trainer.train() to auto-inject GateMonitorCallback."""
    try:
        from transformers import Trainer

        _original_train = Trainer.train

        def _patched_train(self, *args, **kwargs):
            inject_gate_monitor_callback(self, log_every_n_steps=50, detail_every_n_steps=50)
            return _original_train(self, *args, **kwargs)

        Trainer.train = _patched_train
        logger.info("[AttnHidden] ✓ Trainer.train() patched for auto gate monitoring.")
    except Exception as e:
        logger.warning(f"[AttnHidden] Could not patch Trainer for gate monitoring: {e}")


# Auto-register on import
register_attn_hidden()
auto_inject_gate_monitor()
