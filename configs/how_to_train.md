## V11 Attention Hidden-Size Transformer 完整实现

### 核心设计：减法修正式 Self-Attention Gate

```python
h_new = h + o - scale · (tanh(score_h) ⊙ h + tanh(score_o) ⊙ o)
```

- **score_h, score_o** 由完整 self-attention 计算（Q = W_q·h, K = W_k·h/o）
- **tanh** 限制移除量到 (-1, 1)
- **scale** 可学习，初始值 0.1（每层最多移除 ~10%）
- **初始行为 = h + o**（SFT 模式下精确成立）

### 两种训练模式

| 模式 | 初始化 | 初始行为 | 适用场景 |
|------|--------|---------|---------|
| **pretrain** | Q, K 小随机值 | h + o - ε（微小扰动） | 从头预训练 |
| **sft** | Q 随机, K=**零** | h + o（精确） | 从 Qwen3 迁移权重 |

### 文件清单

| 文件 | 说明 |
|------|------|
| [`swift/model/attention_hidden_size/modeling_attn_hidden.py`](ms-swift/swift/model/attention_hidden_size/modeling_attn_hidden.py) | 核心模型（减法修正 Gate） |
| [`swift/model/attention_hidden_size/configuration_attn_hidden.py`](ms-swift/swift/model/attention_hidden_size/configuration_attn_hidden.py) | 配置类 |
| [`configs/create_attn_hidden_model.py`](ms-swift/configs/create_attn_hidden_model.py) | 创建模型（从头） |
| [`configs/convert_qwen3_to_attn_hidden.py`](ms-swift/configs/convert_qwen3_to_attn_hidden.py) | 从 Qwen3 迁移权重 |
| [`configs/pt_attn_hidden.sh`](ms-swift/configs/pt_attn_hidden.sh) | 训练脚本 |

### 使用方法

**方式 A：从头预训练**
```bash
cd ms-swift
# 创建模型
python configs/create_attn_hidden_model.py \
    --residual_gate_init_mode pretrain \
    --tokenizer_from /home/ubuntu/llm_weights/Qwen3-0.6B/ \
    --output_dir model_checkpoints/attn_hidden-d1024-L28-v11

# 训练
bash configs/pt_attn_hidden.sh model_checkpoints/attn_hidden-d1024-L28-v11 3500 8
```

**方式 B：从 Qwen3 迁移权重（推荐，节省训练成本）**
```bash
cd ms-swift
# 迁移权重（Gate 用 LoRA 风格初始化，h_new = h+o）
python configs/convert_qwen3_to_attn_hidden.py \
    --qwen3_path /home/ubuntu/llm_weights/Qwen3-0.6B \
    --output_dir model_checkpoints/qwen3-0.6b-attn_hidden-d1024-L28-v11-sft

# 训练（全参数 fine-tuning，Gate 参数 lr×5）
bash configs/sft_attn_hidden.sh model_checkpoints/qwen3-0.6b-attn_hidden-d1024-L28-v11-sft 1 8
```