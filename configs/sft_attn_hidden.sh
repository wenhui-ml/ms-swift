#!/bin/bash
# ============================================================================
# Attention Hidden-Size Transformer V11 SFT 训练脚本
#
# 从 Qwen3 迁移权重后的 SFT 训练。
# Gate 使用 LoRA 风格初始化（K=zeros → 初始 h_new = h+o），
# 训练过程中 Gate 逐渐学到选择性过滤冗余/有害信息。
#
# 前置步骤：
#   1. 先用 convert_qwen3_to_attn_hidden.py 从 Qwen3 迁移权重
#   2. 然后用本脚本进行 SFT 训练
#
# Usage:
#   # Step 1: 迁移权重
#   python configs/convert_qwen3_to_attn_hidden.py \
#       --qwen3_path output/Qwen3-0.6B-standard/v1-20260326-020547/checkpoint-3500 \
#       --output_dir model_checkpoints/attn_hidden-d1024-L28-v11-sft
#
#   # Step 2: SFT 训练
#   bash configs/sft_attn_hidden.sh model_checkpoints/attn_hidden-d1024-L28-v11-sft 3500 8
# ============================================================================

MODEL_DIR=${1:-model_checkpoints/qwen3-0.6b-attn_hidden-d1024-L28-v11-sft}
MAX_EPOCHS=${2:-3}
nproc_per_node=${3:-8}

MODEL_NAME=$(basename $MODEL_DIR)

echo "============================================"
echo "Attention Hidden-Size V11 SFT Training"
echo "  Model: $MODEL_DIR"
echo "  Max Epochs: $MAX_EPOCHS"
echo "  GPUs: $nproc_per_node"
echo "============================================"

NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model $MODEL_DIR \
    --model_type qwen3 \
    --template qwen3 \
    --tuner_type full \
    --dataset 'HuggingFaceH4/ultrachat_200k#100000' \
    --torch_dtype bfloat16 \
    --num_train_epochs $MAX_EPOCHS \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --deepspeed zero0 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 64 \
    --dataset_num_proc 64 \
    --save_only_model true \
    --output_dir output/$MODEL_NAME \
    --lr_scheduler_type cosine \
    --use_hf true \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --attn_impl flash_attn \
    --packing true