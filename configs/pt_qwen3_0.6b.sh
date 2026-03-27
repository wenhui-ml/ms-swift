MODEL_DIR=${1:-model_checkpoints/Qwen3-0.6B-standard}
MAX_STEPS=${2:-150}   
nproc_per_node=${3:-8}

MODEL_NAME=$(basename $MODEL_DIR)

echo "============================================"
echo "MagGated Pretraining: $MODEL_NAME"
echo "  Model: $MODEL_DIR"
echo "  Max Steps: $MAX_STEPS"
echo "  GPUs: $nproc_per_node"
echo "============================================"

# ★ 关键：NPROC_PER_NODE 必须设置，否则 swift 不会用 torchrun 启动多进程
NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift pt \
    --model $MODEL_DIR \
    --model_type qwen3 \
    --template qwen3 \
    --tuner_type full \
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
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    --save_only_model true \
    --output_dir output/$MODEL_NAME \
    --lr_scheduler_type cosine \
    --use_hf true \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --attn_impl flash_attention_3 \
    --packing true
