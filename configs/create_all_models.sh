#!/bin/bash
# 创建 ResidualGate 模型 (d=1024, Qwen3-0.6B 对标)
python configs/create_mag_gated_model.py \
    --hidden_size 1024 \
    --intermediate_size 3072 \
    --num_hidden_layers 28 \
    --num_attention_heads 16 \
    --num_key_value_heads 8 \
    --head_dim 128 \
    --vocab_size 151936 \
    --max_position_embeddings 40960 \
    --torch_dtype bfloat16 \
    --output_dir model_checkpoints/mag_gated-d1024-L28

python configs/create_mag_gated_model.py \
    --hidden_size 512 \
    --intermediate_size 1536 \
    --num_hidden_layers 28 \
    --num_attention_heads 16 \
    --num_key_value_heads 8 \
    --head_dim 128 \
    --vocab_size 151936 \
    --max_position_embeddings 40960 \
    --torch_dtype bfloat16 \
    --output_dir model_checkpoints/mag_gated-d512-L28

python configs/create_mag_gated_model.py \
    --hidden_size 256 \
    --intermediate_size 768 \
    --num_hidden_layers 28 \
    --num_attention_heads 16 \
    --num_key_value_heads 8 \
    --head_dim 128 \
    --vocab_size 151936 \
    --max_position_embeddings 40960 \
    --torch_dtype bfloat16 \
    --output_dir model_checkpoints/mag_gated-d256-L28