#!/bin/bash
# ============================================================================
# ResidualGate 模型对比评测脚本 (lm-evaluation-harness)
#
# 使用 lm_eval (EleutherAI lm-evaluation-harness) 进行标准化评测。
# 所有模型使用 HuggingFace transformers 后端（自定义架构不支持 vLLM）。
#
# 前提条件：
#   1. 先运行 prepare_checkpoint_for_eval.py 将模型代码注入 checkpoint
#   2. conda activate wh-llm
#
# Usage:
#   # 评测单个模型
#   bash configs/eval_lm_harness.sh <model_path> [limit] [gpu_id]
#
#   # 评测所有模型
#   bash configs/eval_lm_harness.sh all [limit] [gpu_id]
#
# Examples:
#   bash configs/eval_lm_harness.sh output_bm/mag_gated-d1024-L28/v0-20260327-071047/checkpoint-300 200 0
#   bash configs/eval_lm_harness.sh all 200 1
# ============================================================================

set -e

MODEL_PATH=${1:-"all"}
LIMIT=${2:-200}    # 每个子任务取多少样本，200 足够对比，完整评测可设为 0（全量）
GPU_ID=${3:-0}

LM_EVAL_BIN="${LM_EVAL_BIN:-/home/ubuntu/miniconda3/envs/wh-llm/bin/lm_eval}"
OUTPUT_BASE="eval_results"

# ---- 评测 benchmark 选择 ----
# 对预训练基模（非 instruct），推荐这些不依赖指令遵循能力的选择题 benchmark：
#   mmlu:      多领域知识（57 个子任务，4选1）
#   arc_easy:  小学科学（简单，4选1）
#   arc_challenge: 科学推理（较难，4选1）
#   hellaswag: 常识推理（4选1）
#   winogrande: 共指消解（2选1）
#   truthfulqa_mc2: 真实性（多选）
#
# 注意：gsm8k 需要生成能力，预训练模型基本为 0 分，不建议包含。
TASKS="mmlu,arc_easy,arc_challenge,hellaswag,winogrande,truthfulqa_mc2"

# ---- 评测函数 ----
eval_model() {
    local model_path=$1
    local model_name=$2

    local output_dir="$OUTPUT_BASE/$model_name"
    mkdir -p "$output_dir"

    echo ""
    echo "============================================"
    echo "Evaluating: $model_name"
    echo "  Model path: $model_path"
    echo "  Tasks: $TASKS"
    echo "  Limit: $LIMIT (per subtask)"
    echo "  GPU: $GPU_ID"
    echo "  Output: $output_dir"
    echo "============================================"

    CUDA_VISIBLE_DEVICES=$GPU_ID \
    $LM_EVAL_BIN \
        --model hf \
        --model_args pretrained=$model_path,trust_remote_code=True,dtype=bfloat16 \
        --tasks $TASKS \
        --batch_size auto \
        --num_fewshot 0 \
        --limit $LIMIT \
        --output_path "$output_dir" \
        --trust_remote_code \
        2>&1 | tee "$output_dir/eval.log"

    echo ""
    echo "Done: $model_name → $output_dir"
    echo ""
}

# ---- 单模型评测 ----
if [ "$MODEL_PATH" != "all" ]; then
    # 从路径中提取有意义的名字
    MODEL_NAME=$(echo "$MODEL_PATH" | sed 's|/|_|g' | sed 's|^_||')
    eval_model "$MODEL_PATH" "$MODEL_NAME"
    exit 0
fi

# ---- 批量评测 ----
echo "=========================================="
echo " Batch Evaluation (lm-evaluation-harness)"
echo " Limit per subtask: $LIMIT"
echo " GPU: $GPU_ID"
echo "=========================================="

# 先确保所有 checkpoint 已 prepare
echo ""
echo "[Step 0] Preparing checkpoints..."
python3 configs/prepare_checkpoint_for_eval.py \
    model_checkpoints/mag_gated-d1024-L28 \
    model_checkpoints/mag_gated-d512-L28 \
    model_checkpoints/mag_gated-d256-L28 \
    model_checkpoints/baseline-d512-L28 \
    model_checkpoints/baseline-d256-L28 \
    2>/dev/null

# 同时处理所有已训练的 checkpoint
for ckpt in output_bm/*/v0-*/checkpoint-*; do
    [ -d "$ckpt" ] && python3 configs/prepare_checkpoint_for_eval.py "$ckpt" 2>/dev/null
done
echo ""

# ============================================
# 1. Qwen3-0.6B 标准基模（上界参考）
# ============================================
eval_model \
    "model_checkpoints/Qwen3-0.6B-standard" \
    "qwen3-0.6b-original"

# ============================================
# 2. 同数据预训练的 Qwen3-0.6B
#    取消注释并填入正确的 checkpoint 路径
# ============================================
# eval_model \
#     "output/Qwen3-0.6B-standard/v2-20260327-060706/checkpoint-150" \
#     "qwen3-0.6b-pretrained-150steps"

# ============================================
# 3. MagGated d=1024（与 Qwen3-0.6B 同参数量）
# ============================================
eval_model \
    "output_bm/mag_gated-d1024-L28/v0-20260327-071047/checkpoint-300" \
    "mag_gated-d1024-300steps"

# ============================================
# 4. MagGated d=512（hidden_size 缩小一半）
#    取消注释并填入正确的 checkpoint 路径
# ============================================
# eval_model \
#     "output_bm/mag_gated-d512-L28/v0-xxx/checkpoint-300" \
#     "mag_gated-d512-300steps"

# ============================================
# 5. MagGated d=256（hidden_size 缩小 3/4）
# ============================================
# eval_model \
#     "output_bm/mag_gated-d256-L28/v0-xxx/checkpoint-300" \
#     "mag_gated-d256-300steps"

# ============================================
# 6. Baseline d=512（无 gate，同 hidden_size 对照）
# ============================================
# eval_model \
#     "output_bm/baseline-d512-L28/v0-xxx/checkpoint-300" \
#     "baseline-d512-300steps"

# ============================================
# 7. Baseline d=256（无 gate，同 hidden_size 对照）
# ============================================
# eval_model \
#     "output_bm/baseline-d256-L28/v0-xxx/checkpoint-300" \
#     "baseline-d256-300steps"

echo ""
echo "=========================================="
echo " All evaluations complete!"
echo " Results saved in $OUTPUT_BASE/"
echo ""
echo " Quick comparison:"
echo "   python3 configs/compare_eval_results.py"
echo "=========================================="
