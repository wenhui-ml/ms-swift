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

---

## 20260408 更新：代码审查修复 & 完整训练流程

### 本次修复清单

| # | 问题 | 严重度 | 修复方式 |
|---|------|--------|---------|
| 1 | `SynapticGate` 缺少 RMSNorm | 🔴 Critical | 在 gate 内添加无参数 `_rms_norm_no_weight()` |
| 2 | SFT 脚本未冻结 backbone | 🔴 Critical | 添加 `--freeze_parameters_ratio 1.0 --trainable_parameters_regex 'synaptic_gate'` |
| 3 | 所有文档公式缺少 RMSNorm | 🟡 Medium | 全部更新为 `σ(w ⊙ RMSNorm(x) + b)` |
| 4 | `scripts/inspect_gate_values.py` 为 V11 遗留 | 🟡 Medium | 已删除（被 `configs/analyze_gates.py` 取代） |
| 5 | `swift/optimizers/attn_hidden.py` 为 V11 遗留 | 🟡 Medium | 更新为 V12 命名 (`synaptic_gate`)，注册到 `optimizers_map` |
| 6 | 预训练脚本未使用 gate 专用优化器 | 🟢 Low | `pt_attn_hidden.sh` 添加 `--optimizer attn_hidden` |

### 修复后的核心公式

```python
# 无参数 RMSNorm（仅归一化，无可学习 weight）
def _rms_norm_no_weight(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

# 门控计算（在归一化后的空间中）
normed_h = _rms_norm_no_weight(h)
normed_o = _rms_norm_no_weight(o)
gate_forget  = σ(w_forget ⊙ normed_h + b_forget)
gate_acquire = σ(w_acquire ⊙ normed_o + b_acquire)

# 最终组合（在原始空间中）
h_new = gate_forget ⊙ h + gate_acquire ⊙ o
```

关键：RMSNorm 只用于 gate 的 sigmoid 输入，最终的 hadamard 乘积用的是**原始** h 和 o。

### 修复后的完整文件清单

| 文件 | 说明 | 20260408 状态 |
|------|------|--------------|
| `swift/model/attention_hidden_size/modeling_attn_hidden.py` | 核心模型（SynapticGate + RMSNorm） | ✅ 修复 |
| `swift/model/attention_hidden_size/configuration_attn_hidden.py` | 配置类 | ✅ 修复 |
| `swift/model/attention_hidden_size/__init__.py` | 模块导出 | ✅ 无需修改 |
| `swift/model/attention_hidden_size/register_attn_hidden.py` | HuggingFace 注册 + gate monitor | ✅ 无需修改 |
| `swift/optimizers/attn_hidden.py` | Gate 专用优化器（gate 5x LR） | ✅ V11→V12 更新 |
| `swift/optimizers/mapping.py` | 优化器注册表 | ✅ 新增 attn_hidden |
| `configs/create_attn_hidden_model.py` | 创建模型（从头） | ✅ 文档修复 |
| `configs/convert_qwen3_to_attn_hidden.py` | 从 Qwen3 迁移权重 | ✅ 文档修复 |
| `configs/analyze_gates.py` | 分析门控统计信息 | ✅ 文档修复 |
| `configs/pt_attn_hidden.sh` | 预训练脚本 | ✅ 添加 `--optimizer attn_hidden` |
| `configs/sft_attn_hidden.sh` | SFT 训练脚本（非对称冻结） | ✅ 修复冻结策略 |
| `configs/sft_attn_hidden_v2.sh` | SFT 训练脚本 V2 | ✅ 修复冻结策略 |
| ~~`scripts/inspect_gate_values.py`~~ | ~~V11 遗留检查脚本~~ | ❌ 已删除 |

### 完整训练步骤

#### 方式 A：从头预训练（适合研究探索）

```bash
cd /home/ubuntu/wenhui/mag_gate/ms-swift

# Step 1: 创建随机初始化的模型
python configs/create_attn_hidden_model.py \
    --hidden_size 1024 \
    --intermediate_size 3072 \
    --num_hidden_layers 28 \
    --num_attention_heads 16 \
    --num_key_value_heads 8 \
    --head_dim 128 \
    --synaptic_gate_init_mode pretrain \
    --synaptic_gate_init_bias 4.0 \
    --tokenizer_from /home/ubuntu/llm_weights/Qwen3-0.6B/ \
    --output_dir model_checkpoints/attn_hidden-d1024-L28-v12

# Step 2: 预训练（backbone + gate 全部可训练，gate 用 5x LR）
#   --optimizer attn_hidden: gate 参数自动分组，使用 base_lr × 5 和 weight_decay=0
bash configs/pt_attn_hidden.sh model_checkpoints/attn_hidden-d1024-L28-v12 3500 8

# Step 3: 分析门控学到了什么
python configs/analyze_gates.py output/attn_hidden-d1024-L28-v12/checkpoint-latest
```

#### 方式 B：从 Qwen3 迁移权重 + SFT（推荐，节省训练成本）

```bash
cd /home/ubuntu/wenhui/mag_gate/ms-swift

# Step 1: 从 Qwen3 迁移权重
#   - 所有 Attention/MLP/Norm/Embedding 权重直接复制
#   - Gate 用 SFT 初始化：w=0, b=+4.0 → σ(0·RMSNorm(x)+4.0)=σ(4.0)≈0.982 → h_new ≈ h+o
python configs/convert_qwen3_to_attn_hidden.py \
    --qwen3_path /home/ubuntu/llm_weights/Qwen3-0.6B \
    --output_dir model_checkpoints/qwen3-0.6b-attn_hidden-d1024-L28-v12-sft \
    --synaptic_gate_init_bias 4.0

# Step 2: SFT 训练（非对称冻结训练）
#   - backbone 完全冻结（--freeze_parameters_ratio 1.0）
#   - 仅 synaptic_gate 解冻（--trainable_parameters_regex 'synaptic_gate'）
#   - weight_decay=0.0（gate 参数不正则化）
#   - 较高 LR（1e-4）因为仅训练 ~0.04% 的参数
bash configs/sft_attn_hidden.sh model_checkpoints/qwen3-0.6b-attn_hidden-d1024-L28-v12-sft 3 8

# Step 3: 分析门控学到了什么
python configs/analyze_gates.py output/qwen3-0.6b-attn_hidden-d1024-L28-v12-sft/checkpoint-latest
```

#### 方式 B-V2：改进版 SFT（更细粒度）

```bash
# 使用 V2 脚本：gradient_accumulation=8, warmup=0.10, lr=5e-4, flash_attention_3
bash configs/sft_attn_hidden_v2.sh model_checkpoints/qwen3-0.6b-attn_hidden-v12-v2 3 8
```

### 训练策略对比表

| 参数 | 预训练 (`pt_attn_hidden.sh`) | SFT v1 (`sft_attn_hidden.sh`) | SFT v2 (`sft_attn_hidden_v2.sh`) |
|------|-------|------|------|
| backbone | ✅ 可训练 | ❄️ 冻结 | ❄️ 冻结 |
| gate | ✅ 可训练 (5x LR) | ✅ 可训练 | ✅ 可训练 |
| optimizer | `attn_hidden` (分组 LR) | default | default |
| freeze_ratio | 0 | 1.0 | 1.0 |
| trainable_regex | — | `synaptic_gate` | `synaptic_gate` |
| learning_rate | 1e-4 | 1e-4 | 5e-4 |
| weight_decay | 0.1 (backbone only) | 0.0 | 0.0 |
| gradient_accum | 16 | 16 | 8 |
| warmup | 0.05 | 0.05 | 0.10 |
| attn_impl | flash_attention_3 | flash_attn | flash_attention_3 |
| gate_init_mode | pretrain | sft | sft |

### 关键运行验证

```bash
# 验证 gate 输出正确性（零损失退化测试）
python -c "
import torch
from swift.model.attention_hidden_size import SynapticGate
gate = SynapticGate(1024, init_bias=4.0, init_mode='sft')
h, o = torch.randn(2, 10, 1024), torch.randn(2, 10, 1024)
result = gate(h, o)
expected = torch.sigmoid(torch.tensor(4.0)) * h + torch.sigmoid(torch.tensor(4.0)) * o
print(f'Max diff: {(result - expected).abs().max():.2e}')  # Should be 0.00e+00
"

# 验证优化器注册
python -c "
from swift.optimizers.mapping import optimizers_map
assert 'attn_hidden' in optimizers_map
print('Registered:', list(optimizers_map.keys()))
"
```
