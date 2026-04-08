#!/bin/bash
# ============================================================================
# Attention Hidden-Size Transformer V12 预训练脚本
# (Independent Synaptic Gating — 独立突触门控)
#
# 使用纯逐元素独立突触门控，替代标准残差连接 h = h + o。
#
# V12 核心设计：
#   gate_forget  = σ(w_forget ⊙ RMSNorm(h) + b_forget)   — 每维度独立遗忘门
#   gate_acquire = σ(w_acquire ⊙ RMSNorm(o) + b_acquire)  — 每维度独立获取门
#   h_new = gate_forget ⊙ h + gate_acquire ⊙ o             — 纯逐元素操作
#
# RMSNorm 为无参数归一化（仅信号调理），防止深层 sigmoid 饱和。
#
# 仿生学：每个维度仿照独立突触缩放（Synaptic Scaling），
#   只根据自身信号决定保留/接受，无需全局协调。
#
# 关键：必须设置 NPROC_PER_NODE=GPU数量 来启动多卡 DDP 训练！
#
# Usage:
#   # 创建模型权重（首次运行前执行）
#   python configs/create_attn_hidden_model.py \
#       --tokenizer_from /home/ubuntu/llm_weights/Qwen3-0.6B \
#       --output_dir model_checkpoints/attn_hidden-d1024-L28-v12
#
#   # 训练
#   bash configs/pt_attn_hidden.sh model_checkpoints/attn_hidden-d1024-L28-v12 3500 8
# ============================================================================

MODEL_DIR=${1:-model_checkpoints/attn_hidden-d1024-L28-v12}
MAX_STEPS=${2:-3500}
nproc_per_node=${3:-8}

MODEL_NAME=$(basename $MODEL_DIR)

echo "============================================"
echo "Attention Hidden-Size V12 Pretraining"
echo "  (Independent Synaptic Gating)"
echo "  Model: $MODEL_DIR"
echo "  Max Steps: $MAX_STEPS"
echo "  GPUs: $nproc_per_node"
echo "============================================"

# ★ 关键：NPROC_PER_NODE 必须设置，否则 swift 不会用 torchrun 启动多进程
# ★ --optimizer attn_hidden: gate 参数使用 5x 学习率，其余用 base_lr
NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift pt \
    --model $MODEL_DIR \
    --model_type qwen3 \
    --template qwen3 \
    --tuner_type full \
    --optimizer attn_hidden \
    --dataset \
        /home/ubuntu/wenhui/mag_gate/local_datasets/cosmo_khan.jsonl \
        /home/ubuntu/wenhui/mag_gate/local_datasets/cosmo_math.jsonl \
        /home/ubuntu/wenhui/mag_gate/local_datasets/cosmo_stanford.jsonl \
        /home/ubuntu/wenhui/mag_gate/local_datasets/fineweb_100k.jsonl \
        /home/ubuntu/wenhui/mag_gate/local_datasets/magpie_10k.jsonl \
        /home/ubuntu/wenhui/mag_gate/local_datasets/skypile_60k.jsonl \
        /home/ubuntu/wenhui/mag_gate/local_datasets/starcoder_py.jsonl \
    --streaming false \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --deepspeed zero0 \
    --max_length 4096 \
    --max_steps $MAX_STEPS \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --dataloader_num_workers 64 \
    --dataset_num_proc 64 \
    --save_only_model true \
    --output_dir output/$MODEL_NAME \
    --lr_scheduler_type cosine \
    --use_hf true \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --attn_impl flash_attention_3 \
    --packing true
