总结
问题根因
你遇到的 KeyError: 'mag_gated' 是因为 lm_eval 通过 AutoConfig.from_pretrained() 加载模型时，HuggingFace transformers 不认识 model_type="mag_gated" 这个自定义架构。训练时 ms-swift 内部通过 register_mag_gated.py 注册了该类型，但 lm_eval 是独立工具，没有这个注册逻辑。

解决方案
我创建了 configs/prepare_checkpoint_for_eval.py，它做三件事：

将 configuration_mag_gated.py 和 modeling_mag_gated.py 复制到 checkpoint 目录
修改 modeling_mag_gated.py 中的导入方式，绕过 HF 的静态导入检查
在 config.json 中添加 auto_map 字段，告诉 HF AutoClasses 从本地文件加载
已验证模型可以成功加载并通过 lm_eval 评测。

使用流程
# Step 1: 准备 checkpoint（只需运行一次）
python3 configs/prepare_checkpoint_for_eval.py \
    output_bm/mag_gated-d1024-L28/v0-20260327-071047/checkpoint-300 \
    output_bm/mag_gated-d512-L28/v0-xxx/checkpoint-300  # 其他训练好的模型

# Step 2: 单模型评测
conda activate wh-llm
CUDA_VISIBLE_DEVICES=0 lm_eval \
    --model hf \
    --model_args pretrained=output_bm/mag_gated-d1024-L28/v0-20260327-071047/checkpoint-300,trust_remote_code=True,dtype=bfloat16 \
    --tasks mmlu,arc_easy,arc_challenge,hellaswag,winogrande,truthfulqa_mc2 \
    --batch_size auto \
    --num_fewshot 0 \
    --limit 200 \
    --output_path eval_results/mag_gated-d1024 \
    --trust_remote_code

# Step 3: 批量评测（编辑脚本中的 checkpoint 路径后）
bash configs/eval_lm_harness.sh all 200 0

# Step 4: 查看对比结果
python3 configs/compare_eval_results.py
关于 vLLM
MagGated 架构无法使用 vLLM，因为 vLLM 需要自定义的 C++/CUDA kernel 注册。所有模型（包括 Qwen3-0.6B 原版）都应统一使用 --model hf（transformers 后端）以确保评测公平性。

关于评测可靠性
300 steps 预训练的 MMLU 分数（~26%）接近随机水平（25%），这在预期之内。关键点：

看相对差异而非绝对分数：mag_gated-d512 vs baseline-d512 的差值是核心指标
建议训练更多步：至少 1000+ steps 才能让模型学到足够知识，使 benchmark 差异有统计意义
增加 --limit：200 对于快速验证可以，正式对比建议用 500+ 或全量（不设 limit）
选择题类 benchmark 对基模友好：MMLU/ARC/HellaSwag 用 log-likelihood 评分，不依赖生成能力
新增文件
文件	用途
configs/prepare_checkpoint_for_eval.py	将自定义模型 checkpoint 准备为可独立加载的格式
configs/prepare_checkpoint_for_eval.sh	同上的 bash 版本（可删除，Python 版更可靠）
configs/eval_lm_harness.sh	使用 lm_eval 的批量评测脚本
configs/compare_eval_results.py	汇总对比评测结果


现在可以正常运行的命令
# 先准备 checkpoint（每次模型代码更新后只需运行一次）
python3 configs/prepare_checkpoint_for_eval.py \
    /home/ubuntu/wenhui/mag_gate/ms-swift/output/attn_res_gate-d1024-L28-v5.2/v0-20260327-053620/checkpoint-150

# 评测（mmlu + gsm8k 均正常）
conda activate wh-llm
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 \
  -m lm_eval \
  --model hf \
  --tasks "mmlu,gsm8k" \
  --batch_size auto \
  --trust_remote_code \
  --model_args pretrained=/home/ubuntu/wenhui/mag_gate/ms-swift/output/mag_gated-d1024-L28/v0-20260325-104446/checkpoint-3500 \
  --num_fewshot 0

注意：后续训练新模型后，需要对新的 checkpoint 重新运行 prepare_checkpoint_for_eval.py，因为它会自动使用最新的已修复代码。

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 \
  -m lm_eval \
  --model hf \
  --tasks "lambada_openai,piqa" \
  --batch_size auto \
  --trust_remote_code \
  --model_args /home/ubuntu/wenhui/mag_gate/ms-swift/output/attn_res_gate-d1024-L28-v5.2/v0-20260327-053620/checkpoint-150 \
  --num_fewshot 0