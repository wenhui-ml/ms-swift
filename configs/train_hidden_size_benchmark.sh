#!/bin/bash
# ============================================================================
# Hidden-Size Benchmark 训练脚本
#
# 对比不同 hidden-size 下标准 Transformer 的训练效果：
#   1. baseline-d1024-L28 — 标准 Qwen3-0.6B 基模（已有训练结果可对比）
#   2. baseline-d512-L28  — 一半 hidden-size 标准基模
#   3. baseline-d256-L28  — 四分之一 hidden-size 标准基模
#
# 训练配置与 pt_mag_gated.sh 完全一致（数据集、lr、batch_size 等）
# 以确保公平对比。
#
# Usage:
#   # 训练所有 baseline 模型
#   bash configs/train_hidden_size_benchmark.sh
#
#   # 只训练 d=512
#   bash configs/train_hidden_size_benchmark.sh 512
#
#   # 只训练 d=256
#   bash configs/train_hidden_size_benchmark.sh 256
# ============================================================================

MAX_STEPS=3500
NPROC=${2:-2}
TARGET=${1:-all}  # all, 512, 256
MASTER_PORT=${3:-29501}
GPUS=${4:-6,7}  # 默认使用 GPU 6,7（避免与已有训练冲突）

train_model() {
    local MODEL_DIR=$1
    local MODEL_NAME=$(basename $MODEL_DIR)
    
    echo ""
    echo "============================================"
    echo "Training: $MODEL_NAME"
    echo "  Model: $MODEL_DIR"
    echo "  Max Steps: $MAX_STEPS"
    echo "  GPUs: $GPUS (nproc=$NPROC, port=$MASTER_PORT)"
    echo "============================================"
    
    NPROC_PER_NODE=$NPROC \
    MASTER_PORT=$MASTER_PORT \
    CUDA_VISIBLE_DEVICES=$GPUS \
    swift pt \
        --model $MODEL_DIR \
        --model_type qwen3 \
        --template qwen3 \
        --tuner_type full \
        --dataset \
            /home/ubuntu/wenhui/mag_gate/local_datasets_2.6m/cosmo_khan.jsonl \
            /home/ubuntu/wenhui/mag_gate/local_datasets_2.6m/cosmo_math.jsonl \
            /home/ubuntu/wenhui/mag_gate/local_datasets_2.6m/cosmo_stanford.jsonl \
            /home/ubuntu/wenhui/mag_gate/local_datasets_2.6m/fineweb_1000k.jsonl \
            /home/ubuntu/wenhui/mag_gate/local_datasets_2.6m/magpie_10k.jsonl \
            /home/ubuntu/wenhui/mag_gate/local_datasets_2.6m/skypile_600k.jsonl \
            /home/ubuntu/wenhui/mag_gate/local_datasets_2.6m/starcoder_py.jsonl \
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
    
    echo ""
    echo ">>> Finished training: $MODEL_NAME"
    echo ""
}

# ============================================================================
# 执行训练
# ============================================================================

if [ "$TARGET" = "all" ] || [ "$TARGET" = "512" ]; then
    if [ ! -d "model_checkpoints/baseline-d512-L28" ]; then
        echo "ERROR: model_checkpoints/baseline-d512-L28 not found!"
        echo "Run: bash configs/create_baseline_models.sh first"
        exit 1
    fi
    train_model model_checkpoints/baseline-d512-L28
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "256" ]; then
    if [ ! -d "model_checkpoints/baseline-d256-L28" ]; then
        echo "ERROR: model_checkpoints/baseline-d256-L28 not found!"
        echo "Run: bash configs/create_baseline_models.sh first"
        exit 1
    fi
    train_model model_checkpoints/baseline-d256-L28
fi

echo ""
echo "============================================"
echo "Hidden-Size Benchmark Training Complete!"
echo "============================================"
echo ""
echo "Training outputs:"
if [ "$TARGET" = "all" ] || [ "$TARGET" = "512" ]; then
    echo "  - output/baseline-d512-L28/"
fi
if [ "$TARGET" = "all" ] || [ "$TARGET" = "256" ]; then
    echo "  - output/baseline-d256-L28/"
fi
echo ""
echo "Compare loss curves with existing runs:"
echo "  - output/attn_res_gate-d1024-L28-v5.1/  (gated, d=1024)"
echo "  - output/baseline-d512-L28/              (no gate, d=512)"
echo "  - output/baseline-d256-L28/              (no gate, d=256)"
