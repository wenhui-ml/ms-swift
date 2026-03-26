#!/bin/bash
# 创建 ResidualGate 模型 (d=1024, Qwen3-0.6B 对标, init_bias=3.0)
python configs/create_mag_gated_model.py \
    --hidden_size 1024 \
    --intermediate_size 3072 \
    --num_hidden_layers 28 \
    --num_attention_heads 16 \
    --num_key_value_heads 8 \
    --head_dim 128 \
    --vocab_size 151936 \
    --max_position_embeddings 40960 \
    --residual_gate_init_bias 3.0 \
    --torch_dtype bfloat16 \
    --tokenizer_from model_checkpoints/mag_gated-d1024-L28_20260325_res_gate \
    --output_dir model_checkpoints/attn_res_gate-d1024-L28
