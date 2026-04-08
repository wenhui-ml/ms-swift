## V12 Attention Hidden-Size Transformer — 独立突触门控 (Independent Synaptic Gating)

### 核心设计：仿生学独立突触缩放

```python
gate_forget  = σ(w_forget ⊙ RMSNorm(h) + b_forget)    # 每个维度独立决定保留多少 h
gate_acquire = σ(w_acquire ⊙ RMSNorm(o) + b_acquire)   # 每个维度独立决定接受多少 o
h_new = gate_forget ⊙ h + gate_acquire ⊙ o              # 纯逐元素操作，无矩阵乘法
```

- **纯逐元素操作**（⊙）：没有任何矩阵乘法，每个维度完全独立
- **RMSNorm 无参数化**：门控内的 RMSNorm 不含可学习权重，仅做信号归一化，防止深层 sigmoid 饱和
- **σ (sigmoid)**：确保门控值在 [0, 1] 之间
- **最终组合用原始 h 和 o**：RMSNorm 只用于门控计算，不影响残差流的幅度
- **初始化**：w=0, b=+4.0 → σ(0·RMSNorm(x)+4.0) = σ(4.0)≈0.982 → h_new ≈ h+o（精确零损失退化）
- **每个 gate 仅 4d 参数**：如 d=1024，则每个 gate 仅 4096 个标量（RMSNorm 无额外参数）

### 为什么需要 RMSNorm？

论文指出 PreNorm 导致隐状态幅度不断无控增长，深层的 w⊙h 乘积会变得非常大，
导致 sigmoid 不可控饱和。无参数 RMSNorm 将 h 和 o 归一化到单位 RMS，
确保门控在任意深度都工作在 sigmoid 的有效区间内。

### 生物仿生学直觉

第 42 号维度代表的"逻辑特征"，它的遗忘与否，**只由流经第 42 号维度的信号大小**，
以及它自己学到的"突触敏感度"（w 和 b）决定。它**不需要知道第 99 号维度在干什么**。

这完全仿照生物神经元的独立突触缩放（Synaptic Scaling）：
每个突触根据自身的电位历史和当前刺激来独立决定信号强度。

### 两种训练模式

| 模式 | 初始化 | 初始行为 | 适用场景 |
|------|--------|---------|---------|
| **pretrain** | w=小随机值(std=0.01), b=+4.0 | h_new ≈ h+o + 微小噪声 | 从头预训练 |
| **sft** | w=**零**, b=+4.0 | h_new ≈ h+o（精确） | 从 Qwen3 迁移权重 |

### 参数开销极小

| 指标 | 值 |
|------|-----|
| 每个 gate 参数量 | 4 × d（如 4×1024=4096） |
| 每层参数量 | 8 × d（attn_gate + ffn_gate） |
| 28 层总 gate 参数 | 28×8×1024 = 229,376 |
| 相对 Qwen3-0.6B 开销 | ~0.04% |

### SFT 训练策略：非对称冻结训练

在 SFT 阶段，**绝对冻结 Backbone 的所有权重**（Attention、MLP、Norm、Embedding），
仅训练每层 SynapticGate 的 4d 个参数。这通过 ms-swift 的参数冻结机制实现：

```bash
--freeze_parameters_ratio 1.0         # 冻结所有参数
--trainable_parameters_regex 'synaptic_gate'  # 仅解冻 gate 参数
--weight_decay 0.0                    # gate 参数不用正则化
```

因为可训练参数极少（约 0.04%），可以使用较高学习率（如 1e-4 ~ 5e-4）。

### 文件清单

| 文件 | 说明 |
|------|------|
| `swift/model/attention_hidden_size/modeling_attn_hidden.py` | 核心模型（SynapticGate 独立突触门控） |
| `swift/model/attention_hidden_size/configuration_attn_hidden.py` | 配置类 |
| `swift/model/attention_hidden_size/__init__.py` | 模块导出 |
| `swift/model/attention_hidden_size/register_attn_hidden.py` | HuggingFace 注册 |
| `configs/create_attn_hidden_model.py` | 创建模型（从头） |
| `configs/convert_qwen3_to_attn_hidden.py` | 从 Qwen3 迁移权重 |
| `configs/analyze_gates.py` | 分析门控统计信息 |
| `configs/pt_attn_hidden.sh` | 预训练脚本 |
| `configs/sft_attn_hidden.sh` | SFT 训练脚本 |
| `configs/sft_attn_hidden_v2.sh` | SFT 训练脚本 V2（改进参数） |

### 使用方法

**方式 A：从头预训练**
```bash
cd ms-swift
# 创建模型
python configs/create_attn_hidden_model.py \
    --synaptic_gate_init_mode pretrain \
    --synaptic_gate_init_bias 4.0 \
    --tokenizer_from /home/ubuntu/llm_weights/Qwen3-0.6B/ \
    --output_dir model_checkpoints/attn_hidden-d1024-L28-v12

# 训练
bash configs/pt_attn_hidden.sh model_checkpoints/attn_hidden-d1024-L28-v12 3500 8
```

**方式 B：从 Qwen3 迁移权重（推荐，节省训练成本）**
```bash
cd ms-swift
# 迁移权重（Gate 用 SFT 初始化：w=0, b=+4.0 → h_new ≈ h+o）
python configs/convert_qwen3_to_attn_hidden.py \
    --qwen3_path /home/ubuntu/llm_weights/Qwen3-0.6B \
    --output_dir model_checkpoints/qwen3-0.6b-attn_hidden-d1024-L28-v12-sft

# 训练（非对称冻结：backbone 冻结，仅训练 gate 参数）
bash configs/sft_attn_hidden.sh model_checkpoints/qwen3-0.6b-attn_hidden-d1024-L28-v12-sft 3 8
```

**分析门控状态**
```bash
python configs/analyze_gates.py /path/to/checkpoint
```

### 关键设计优势

1. **绝对的仿生学**：每个维度独立决策，完全仿照神经元突触缩放
2. **不破坏特征流形**：无跨通道交叉组合，预训练特征在各自轨道运行
3. **极度轻量**：4d 参数/gate，SFT 数据轻松驱动，不会灾难性遗忘
4. **完美零损失退化**：初始 σ(4.0)≈0.982 → h_new ≈ h+o，梯度高速公路畅通
5. **RMSNorm 防深层饱和**：无参数 RMSNorm 确保 sigmoid 在任意深度有效工作
6. **非对称冻结训练**：SFT 阶段冻结 backbone，仅训练 gate，信任交叉熵信号
7. **无需反熵损失**：无需复杂的路由架构或辅助损失函数
