#!/bin/bash
# ============================================================================
# 创建 MagGated 实验所需的模型 checkpoint
#
# 实验核心问题：MagGated(d=512) 或 MagGated(d=256) 能否匹配 Qwen3-0.6B(d=1024)?
#
# Qwen3-0.6B 参考配置：
#   hidden_size=1024, L=28, intermediate=3072, n_h=16, n_kv=8, head_dim=128
#   vocab_size=151669, tie_embeddings=true, silu, rope_theta=1e6
#
# 实验模型（对齐 Qwen3-0.6B 的设计，只缩小 hidden_size）：
#   1. baseline-d512-L28     标准Transformer d=512（无门控对照）
#   2. mag_gated-d512-L28    MagGated d=512（核心实验：d减半）
#   3. mag_gated-d256-L28    MagGated d=256（激进实验：d四分之一）
#
# Qwen3-0.6B 本身作为 baseline（d=1024），直接从已有权重继续预训练。
# ============================================================================

PYTHON=/home/ubuntu/miniconda3/envs/wh-llm/bin/python
SCRIPT=/home/ubuntu/wenhui/mag_gate/ms-swift/configs/create_mag_gated_model.py
CKPT_DIR=/home/ubuntu/wenhui/mag_gate/ms-swift/model_checkpoints
TOKENIZER=/home/ubuntu/llm_weights/Qwen3-0.6B

echo "=========================================="
echo " 创建对齐 Qwen3-0.6B 的实验模型"
echo "=========================================="

# --------------------------------------------------------------------------
# 1. baseline-d512-L28: 标准Transformer, d=512, L=28（无门控对照）
#    用于证明：不加门控的小d模型性能不行
# --------------------------------------------------------------------------
echo ""
echo ">>> 1/3: baseline-d512-L28 (标准Transformer, 无门控)"
$PYTHON $SCRIPT \
    --variant baseline \
    --hidden_size 512 \
    --num_hidden_layers 28 \
    --num_attention_heads 4 \
    --num_key_value_heads 2 \
    --head_dim 128 \
    --vocab_size 151669 \
    --output_dir $CKPT_DIR/baseline-d512-L28 \
    --tokenizer_from $TOKENIZER

# --------------------------------------------------------------------------
# 2. mag_gated-d512-L28: MagGated, d=512, L=28（核心实验：d减半）
#    核心问题：MagGated(d=512) 能否匹配 Qwen3-0.6B(d=1024)?
# --------------------------------------------------------------------------
echo ""
echo ">>> 2/3: mag_gated-d512-L28 (MagGated, d=512, 核心实验)"
$PYTHON $SCRIPT \
    --variant mag_gated_all \
    --hidden_size 512 \
    --num_hidden_layers 28 \
    --num_attention_heads 4 \
    --num_key_value_heads 2 \
    --head_dim 128 \
    --gate_rank 16 \
    --vocab_size 151669 \
    --output_dir $CKPT_DIR/mag_gated-d512-L28 \
    --tokenizer_from $TOKENIZER

# --------------------------------------------------------------------------
# 3. mag_gated-d256-L28: MagGated, d=256, L=28（激进实验：d四分之一）
#    探索极限：MagGated(d=256) 能达到什么水平?
# --------------------------------------------------------------------------
echo ""
echo ">>> 3/3: mag_gated-d256-L28 (MagGated, d=256, 激进实验)"
$PYTHON $SCRIPT \
    --variant mag_gated_all \
    --hidden_size 256 \
    --num_hidden_layers 28 \
    --num_attention_heads 2 \
    --num_key_value_heads 1 \
    --head_dim 128 \
    --gate_rank 16 \
    --vocab_size 151669 \
    --output_dir $CKPT_DIR/mag_gated-d256-L28 \
    --tokenizer_from $TOKENIZER

echo ""
echo "=========================================="
echo " 所有模型创建完成！"
echo "=========================================="
echo ""
echo "实验对比矩阵："
echo "  Qwen3-0.6B (d=1024, L=28, ~600M)    ← 已有权重，继续预训练"
echo "  baseline-d512-L28 (~XXM)              ← 无门控对照"
echo "  mag_gated-d512-L28 (~XXM)             ← ★核心：d减半+门控"
echo "  mag_gated-d256-L28 (~XXM)             ← d四分之一+门控"
echo ""
echo "Checkpoints: $CKPT_DIR"
ls -la $CKPT_DIR/
