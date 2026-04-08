# Copyright (c) 2024. All rights reserved.
# Optimizer callback for Attention Hidden-Size Transformer (V12 — Independent Synaptic Gating)
"""
Custom OptimizerCallback that creates separate parameter groups for
SynapticGate parameters with a scaled learning rate and zero weight decay.

Usage:
    swift pt --optimizer attn_hidden ...

Gate formula:
    gate_forget  = σ(w_forget ⊙ RMSNorm(h) + b_forget)
    gate_acquire = σ(w_acquire ⊙ RMSNorm(o) + b_acquire)
    h_new = gate_forget ⊙ h + gate_acquire ⊙ o

Gate parameters (identified by 'synaptic_gate' in name) get:
    - LR = base_lr * synaptic_gate_lr_scale (from model config, default 5.0)
    - weight_decay = 0.0

All other parameters use the default LR and weight decay from training args.

This is most useful for pretraining where both backbone and gate are trainable
but need different learning rates. For SFT, prefer using:
    --freeze_parameters_ratio 1.0
    --trainable_parameters_regex 'synaptic_gate'
which freezes the backbone entirely.
"""

import torch.nn as nn
from transformers import Trainer as HfTrainer
from typing import List, Tuple

from swift.utils import get_logger
from .base import OptimizerCallback

logger = get_logger()


class AttnHiddenOptimizerCallback(OptimizerCallback):
    """Optimizer callback with separate LR for SynapticGate parameters.

    Gate parameters (w_forget, b_forget, w_acquire, b_acquire) are grouped
    separately with:
        - Higher LR (base_lr × synaptic_gate_lr_scale)
        - Zero weight decay (gate params should NOT be regularized)

    This ensures the gate learns faster than the backbone while backbone
    weights are gently tuned.
    """

    def create_optimizer(self):
        args = self.args
        model = self.trainer.model

        # Unwrap DDP/DeepSpeed wrappers to access config
        raw_model = model
        if hasattr(raw_model, 'module'):
            raw_model = raw_model.module

        # Get gate LR scale from model config
        gate_lr_scale = 5.0  # default for V12
        if hasattr(raw_model, 'config'):
            gate_lr_scale = getattr(raw_model.config, 'synaptic_gate_lr_scale', 5.0)

        base_lr = args.learning_rate
        gate_lr = base_lr * gate_lr_scale

        # Get decay parameter names (typically weight matrices, not biases/norms)
        decay_parameters = set(HfTrainer.get_decay_parameter_names(None, model))

        # Split parameters into gate vs non-gate
        gate_params_no_decay = []
        other_params_decay = []
        other_params_no_decay = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            is_gate = 'synaptic_gate' in name

            if is_gate:
                # Gate params: always zero weight decay, use gate_lr
                # Gate params are element-wise scalars (w, b) — no decay needed
                gate_params_no_decay.append(param)
            else:
                if name in decay_parameters:
                    other_params_decay.append(param)
                else:
                    other_params_no_decay.append(param)

        optimizer_grouped_parameters = []

        if other_params_decay:
            optimizer_grouped_parameters.append({
                'params': other_params_decay,
                'weight_decay': args.weight_decay,
                'lr': base_lr,
            })

        if other_params_no_decay:
            optimizer_grouped_parameters.append({
                'params': other_params_no_decay,
                'weight_decay': 0.0,
                'lr': base_lr,
            })

        if gate_params_no_decay:
            optimizer_grouped_parameters.append({
                'params': gate_params_no_decay,
                'weight_decay': 0.0,
                'lr': gate_lr,
            })

        # Count parameters
        n_gate = sum(p.numel() for p in gate_params_no_decay)
        n_other = sum(p.numel() for p in other_params_decay) + sum(p.numel() for p in other_params_no_decay)

        logger.info(
            f'[AttnHidden V12 Optimizer] base_lr={base_lr}, gate_lr={gate_lr} '
            f'(scale={gate_lr_scale}x), gate_params={n_gate:,}, other_params={n_other:,}'
        )

        optimizer_cls, optimizer_kwargs = HfTrainer.get_optimizer_cls_and_kwargs(args, model)
        return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
