#!/bin/bash
# ============================================================================
# MagGated 全部实验运行脚本
#
# 实验对比矩阵（全部使用相同数据、相同训练步数）：
#   1. Qwen3-0.6B (d=1024, L=28, ~600M)    ← 已有权重，继续预训练
#   2. baseline-d512-L28 (~150M)             ← 同架构无门控对照
#   3. mag_gated-d512-L28 (~155M)            ← ★核心：d减半+门控
#   4. mag_gated-d256-L28 (~80M)             ← d四分之一+门控
#
# Usage:
#   bash scripts/run_all_experiments.sh [max_steps] [nproc]
# ============================================================================

set -e

MAX_STEPS=${1:-3000}
NPROC=${2:-4}
DATASET="HuggingFaceTB/cosmopedia-100k"

echo "============================================================"
echo "  MagGated Transformer 实验套件"
echo "  核心问题: MagGated(d=512) 能否匹配 Qwen3-0.6B(d=1024)?"
echo "  Max Steps: $MAX_STEPS | GPUs: $NPROC"
echo "============================================================"

# --- 通用训练参数 ---
COMMON_ARGS="
    --tuner_type full
    --dataset $DATASET
    --torch_dtype bfloat16
    --streaming true
    --per_device_train_batch_size 4
    --per_device_eval_batch_size 4
    --deepspeed zero2
    --gradient_accumulation_steps 4
    --packing true
    --eval_steps 200
    --save_steps 500
    --save_total_limit 2
    --logging_steps 5
    --max_length 2048
    --max_steps $MAX_STEPS
    --warmup_ratio 0.05
    --weight_decay 0.1
    --adam_beta1 0.9
    --adam_beta2 0.95
    --dataloader_num_workers 4
    --dataset_num_proc 8
    --save_only_model true
    --lr_scheduler_type cosine
"

run_experiment() {
    local MODEL=$1
    local LR=$2
    local NAME=$3
    
    echo ""
    echo "============================================================"
    echo "  训练: $NAME"
    echo "  模型: $MODEL | LR: $LR"
    echo "============================================================"
    
    NPROC_PER_NODE=$NPROC \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    swift pt \
        --model $MODEL \
        --learning_rate $LR \
        --output_dir output/$NAME \
        $COMMON_ARGS
    
    echo "  ✓ 完成: $NAME"
}

# ============================================================
# 实验 1: Qwen3-0.6B baseline (继续预训练, lr=1e-5)
# ============================================================
run_experiment \
    "/home/ubuntu/llm_weights/Qwen3-0.6B" \
    "1e-5" \
    "qwen3-0.6b-baseline"

# ============================================================
# 实验 2: baseline-d512-L28 (从头训练, lr=3e-4, 无门控)
# ============================================================
run_experiment \
    "model_checkpoints/baseline-d512-L28" \
    "3e-4" \
    "baseline-d512-L28"

# ============================================================
# 实验 3: ★ mag_gated-d512-L28 (从头训练, lr=3e-4, MagGated)
# ============================================================
run_experiment \
    "model_checkpoints/mag_gated-d512-L28" \
    "3e-4" \
    "mag_gated-d512-L28"

# ============================================================
# 实验 4: mag_gated-d256-L28 (从头训练, lr=3e-4, MagGated激进)
# ============================================================
run_experiment \
    "model_checkpoints/mag_gated-d256-L28" \
    "3e-4" \
    "mag_gated-d256-L28"

echo ""
echo "============================================================"
echo "  全部实验完成！"
echo "============================================================"
echo ""
echo "对比 TensorBoard 或 logging.jsonl 中的训练 loss："
echo "  output/qwen3-0.6b-baseline/    (d=1024, 上限)"
echo "  output/baseline-d512-L28/      (d=512, 无门控下限)"
echo "  output/mag_gated-d512-L28/     (d=512, ★核心)"
echo "  output/mag_gated-d256-L28/     (d=256, 激进)"
