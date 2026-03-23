#!/bin/bash
# ============================================================================
# Qwen3-0.6B 继续预训练 (对照基准)
# 
# Qwen3-0.6B specs:
#   hidden_size=1024, L=28, n_h=16, n_kv=8, intermediate=3072
#   ~600M params, vocab=151936, tie_embeddings=true
#
# Usage:
#   bash configs/pt_qwen3_0.6b.sh
# ============================================================================

nproc_per_node=16
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)

NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=7 \
swift pt \
    --model /home/ubuntu/llm_weights/Qwen3-0.6B \
    --tuner_type full \
    --dataset swift/chinese-c4 \
    --torch_dtype bfloat16 \
    --streaming true \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 64 / $nproc_per_node) \
    --packing true \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --max_steps 10000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --save_only_model true \
    --output_dir output/Qwen3-0.6B \
    --attn_impl flash_attn
