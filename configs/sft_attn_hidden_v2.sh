#!/bin/bash
# ============================================================================
# Attention Hidden-Size Transformer V12 SFT 训练脚本 (V2 — 改进版)
# (Independent Synaptic Gating — 独立突触门控)
#
# 非对称冻结训练（Asymmetric Frozen Tuning）：
#   绝对冻结 Backbone 所有权重（Attention、MLP、Norm、Embedding），
#   仅训练每层的 SynapticGate 参数（4d × 2 gates × N layers）。
#
# 相比 v1 的改进：
#   - gradient_accumulation 从 16 降为 8（步数翻倍，更细粒度控制）
#   - warmup_ratio 从 0.05 增加到 0.10
#   - 使用 flash_attention_3
#   - learning_rate 提高到 5e-4（因为仅训练 gate 极少参数，可用更大 LR）
#
# 前置步骤：
#   1. 先用 convert_qwen3_to_attn_hidden.py 从 Qwen3 迁移权重
#   2. 然后用本脚本进行 SFT 训练
#
# Usage:
#   # Step 1: 迁移权重
#   python configs/convert_qwen3_to_attn_hidden.py \
#       --qwen3_path /home/ubuntu/llm_weights/Qwen3-0.6B \
#       --output_dir model_checkpoints/qwen3-0.6b-attn_hidden-v12-v2
#
#   # Step 2: SFT 训练（仅训练 gate 参数，backbone 冻结）
#   bash configs/sft_attn_hidden_v2.sh model_checkpoints/qwen3-0.6b-attn_hidden-v12-v2 3 8
# ============================================================================

MODEL_DIR=${1:-model_checkpoints/qwen3-0.6b-attn_hidden-v12-v2}
MAX_EPOCHS=${2:-3}
nproc_per_node=${3:-8}

MODEL_NAME=$(basename $MODEL_DIR)

echo "============================================"
echo "Attention Hidden-Size V12 SFT Training (V2)"
echo "  (Independent Synaptic Gating)"
echo "  Mode: Asymmetric Frozen (gate-only)"
echo "  Model: $MODEL_DIR"
echo "  Max Epochs: $MAX_EPOCHS"
echo "  GPUs: $nproc_per_node"
echo "  Backbone: FROZEN"
echo "  Trainable: synaptic_gate params only"
echo "============================================"

NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model $MODEL_DIR \
    --model_type qwen3 \
    --template qwen3 \
    --tuner_type full \
    --freeze_parameters_ratio 1.0 \
    --trainable_parameters_regex 'synaptic_gate' \
    --dataset 'HuggingFaceH4/ultrachat_200k' \
    --torch_dtype bfloat16 \
    --num_train_epochs $MAX_EPOCHS \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 5e-4 \
    --gradient_accumulation_steps 8 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --deepspeed zero0 \
    --max_length 4096 \
    --warmup_ratio 0.10 \
    --weight_decay 0.0 \
    --dataloader_num_workers 64 \
    --dataset_num_proc 64 \
    --save_only_model true \
    --output_dir output/${MODEL_NAME}-v2 \
    --lr_scheduler_type cosine \
    --use_hf true \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --attn_impl flash_attention_3 \
    --packing true
