#!/bin/bash
# ============================================================================
# ResidualGate 模型对比评测脚本
#
# 评测目标：对比以下模型在标准 benchmark 上的表现
#   1. Qwen3-0.6B 标准基模（未训练）
#   2. Qwen3-0.6B 同数据预训练（baseline-d1024）
#   3. MagGated d=1024（与 Qwen3-0.6B 同 hidden_size）
#   4. MagGated d=512（hidden_size 缩小一半）
#   5. MagGated d=256（hidden_size 缩小四分之三）
#   6. Baseline d=512（无 gate，同 hidden_size 对照）
#   7. Baseline d=256（无 gate，同 hidden_size 对照）
#
# 注意：MagGated 是自定义架构，vLLM 不支持。
#   方案 A（默认）：使用 transformers 后端直接评测
#   方案 B：先 deploy 再用 eval_url 评测（适合大批量）
#
# Usage:
#   # 评测单个模型（transformers 后端）
#   bash configs/eval_all_models.sh <model_path> [eval_limit] [gpu_id]
#
#   # 评测所有模型
#   bash configs/eval_all_models.sh all [eval_limit] [gpu_id]
#
# Examples:
#   bash configs/eval_all_models.sh output_bm/mag_gated-d512-L28/v0-xxx/checkpoint-300 100 0
#   bash configs/eval_all_models.sh all 200 0
# ============================================================================

set -e

MODEL_PATH=${1:-"all"}
EVAL_LIMIT=${2:-200}       # 每个 benchmark 采样数，200 足够快速验证
GPU_ID=${3:-0}

# ---- 评测数据集配置 ----
# 对于预训练模型（非 instruct），推荐以下 benchmark：
#   - mmlu: 多领域知识（选择题，不需要指令遵循能力）
#   - arc: 科学推理（选择题）
#   - hellaswag: 常识推理（选择题）
#   - gsm8k: 数学推理（需要一定生成能力，预训练模型可能表现差）
#   - truthful_qa: 真实性（选择题）
#   - ceval: 中文知识评测（选择题）
#   - cmmlu: 中文多领域（选择题）
EVAL_DATASETS="mmlu arc hellaswag truthful_qa ceval cmmlu"

# ---- 评测函数 ----
eval_model() {
    local model_path=$1
    local model_name=$2
    local model_type=${3:-"qwen3"}  # mag_gated 模型也用 qwen3 template

    echo ""
    echo "============================================"
    echo "Evaluating: $model_name"
    echo "  Model path: $model_path"
    echo "  Model type: $model_type"
    echo "  Eval limit: $EVAL_LIMIT"
    echo "  GPU: $GPU_ID"
    echo "  Datasets: $EVAL_DATASETS"
    echo "============================================"

    CUDA_VISIBLE_DEVICES=$GPU_ID \
    swift eval \
        --model $model_path \
        --model_type $model_type \
        --template qwen3 \
        --eval_backend Native \
        --infer_backend transformers \
        --eval_dataset $EVAL_DATASETS \
        --eval_limit $EVAL_LIMIT \
        --eval_output_dir eval_output/$model_name \
        --use_hf true \
        --torch_dtype bfloat16

    echo "Done: $model_name → eval_output/$model_name"
    echo ""
}

# ---- 单模型评测 ----
if [ "$MODEL_PATH" != "all" ]; then
    MODEL_NAME=$(basename $(dirname $MODEL_PATH))-$(basename $MODEL_PATH)
    eval_model "$MODEL_PATH" "$MODEL_NAME"
    exit 0
fi

# ---- 批量评测所有模型 ----
echo "=========================================="
echo " Batch Evaluation: All Models"
echo " Eval limit per dataset: $EVAL_LIMIT"
echo "=========================================="

# 请根据实际训练输出修改以下路径！
# 格式：<模型路径> <标签名> <model_type>

# 1) Qwen3-0.6B 标准基模（未经额外训练）
eval_model \
    "model_checkpoints/Qwen3-0.6B-standard" \
    "qwen3-0.6b-original" \
    "qwen3"

# 2) 同数据预训练的模型（需要你先完成训练，然后填入正确的 checkpoint 路径）
# 取消注释并修改路径：

# eval_model \
#     "output_bm/baseline-d1024-L28/v0-xxx/checkpoint-300" \
#     "baseline-d1024-pretrained" \
#     "qwen3"

# eval_model \
#     "output_bm/baseline-d512-L28/v0-xxx/checkpoint-300" \
#     "baseline-d512-pretrained" \
#     "qwen3"

# eval_model \
#     "output_bm/baseline-d256-L28/v0-xxx/checkpoint-300" \
#     "baseline-d256-pretrained" \
#     "qwen3"

# 3) MagGated 模型（训练后）
# eval_model \
#     "output_bm/mag_gated-d1024-L28/v0-20260327-071047/checkpoint-300" \
#     "mag_gated-d1024-pretrained" \
#     "qwen3"

# eval_model \
#     "output_bm/mag_gated-d512-L28/v0-xxx/checkpoint-300" \
#     "mag_gated-d512-pretrained" \
#     "qwen3"

# eval_model \
#     "output_bm/mag_gated-d256-L28/v0-xxx/checkpoint-300" \
#     "mag_gated-d256-pretrained" \
#     "qwen3"

echo "=========================================="
echo " All evaluations complete!"
echo " Results saved in eval_output/"
echo "=========================================="
