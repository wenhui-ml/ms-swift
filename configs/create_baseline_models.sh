#!/bin/bash
# ============================================================================
# 创建 hidden-size 对照 benchmark 的标准基模 checkpoint
#
# 目标：对比不同 hidden-size 下标准 Transformer 的训练效果
#   1. d=1024, L=28 — Qwen3-0.6B 标准基模（已有）
#   2. d=512,  L=28 — 一半 hidden-size 标准基模（无门控）
#   3. d=256,  L=28 — 四分之一 hidden-size 标准基模（无门控）
#
# 验证假设：attention hidden-size 门控机制能否在相同 hidden-size 下
#           达到更大 hidden-size 的效果（"释放 hidden-size 容量"）
#
# Usage:
#   bash configs/create_baseline_models.sh
# ============================================================================

TOKENIZER_FROM=model_checkpoints/mag_gated-d1024-L28_20260325_res_gate

echo "============================================"
echo "Creating baseline models for hidden-size benchmark"
echo "============================================"

# --- d=512, L=28 标准基模（无门控）---
# intermediate_size = 512 * 3 = 1536 (已对齐128)
# num_attention_heads = 512 / 128 = 4
# num_key_value_heads = 4 / 2 = 2
echo ""
echo ">>> Creating baseline-d512-L28 (no gate, standard transformer)"
python configs/create_mag_gated_model.py \
    --no_residual_gate \
    --hidden_size 512 \
    --intermediate_size 1536 \
    --num_hidden_layers 28 \
    --num_attention_heads 4 \
    --num_key_value_heads 2 \
    --head_dim 128 \
    --vocab_size 151936 \
    --max_position_embeddings 40960 \
    --torch_dtype bfloat16 \
    --tokenizer_from $TOKENIZER_FROM \
    --output_dir model_checkpoints/baseline-d512-L28

echo ""
echo ">>> Creating baseline-d256-L28 (no gate, standard transformer)"
# --- d=256, L=28 标准基模（无门控）---
# intermediate_size = 256 * 3 = 768 (已对齐128)
# num_attention_heads = 256 / 128 = 2
# num_key_value_heads = 2 / 2 = 1
python configs/create_mag_gated_model.py \
    --no_residual_gate \
    --hidden_size 256 \
    --intermediate_size 768 \
    --num_hidden_layers 28 \
    --num_attention_heads 2 \
    --num_key_value_heads 1 \
    --head_dim 128 \
    --vocab_size 151936 \
    --max_position_embeddings 40960 \
    --torch_dtype bfloat16 \
    --tokenizer_from $TOKENIZER_FROM \
    --output_dir model_checkpoints/baseline-d256-L28

echo ""
echo "============================================"
echo "All baseline models created!"
echo "============================================"
echo ""
echo "Model checkpoints:"
echo "  - model_checkpoints/baseline-d512-L28"
echo "  - model_checkpoints/baseline-d256-L28"
echo ""
echo "Next: run training with configs/train_hidden_size_benchmark.sh"
