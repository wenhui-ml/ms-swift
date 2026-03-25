# 1. 重新生成全量对标版 (d=1024)
python configs/create_mag_gated_model.py \
    --variant mag_gated_all \
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