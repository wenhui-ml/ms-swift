# V5 设计方案：多源 Softmax Hidden-Size Gate

## 一、核心思想

将论文 AttnRes 的 **softmax 归一化** 思想应用到 hidden-size 维度的门控中。

### 论文 AttnRes 的关键成功因素

```
h_l = Σ α_{i→l} · v_i,  其中 Σ α = 1 (softmax 归一化)
```

1. **softmax 归一化**：权重和为 1，控制 hidden state 幅度
2. **竞争性选择**：softmax 迫使模型在多个源之间做尖锐选择
3. **RMSNorm on keys**：防止大幅度输出主导 attention 权重
4. **零初始化伪查询**：初始均匀分配，训练中逐步分化

### v5 的设计

在每个残差连接处，将 h 和 o 视为两个"源"，用 softmax 在它们之间做竞争性选择：

```
h_new = α_h ⊙ h + α_o ⊙ o
其中 α_h + α_o = 1 (softmax 归一化，per group)
```

## 二、详细架构

### 2.1 ResidualGate Forward

```python
def forward(self, h, o):
    # 1. Key 投影 + RMSNorm (防止幅度差异主导 softmax)
    key_h = RMSNorm(w_kh * h)   # (*, d)
    key_o = RMSNorm(w_ko * o)   # (*, d)
    
    # 2. 分组 + 点积 score
    key_h_g = key_h.view(*, n_groups, group_size)
    key_o_g = key_o.view(*, n_groups, group_size)
    
    score_h = (w_q * key_h_g).sum(-1) / (τ·√group_size)  # (*, n_groups)
    score_o = (w_q * key_o_g).sum(-1) / (τ·√group_size)  # (*, n_groups)
    
    # 3. Softmax 归一化 (在 h 和 o 之间竞争)
    logits = torch.stack([score_h, score_o], dim=-1)  # (*, n_groups, 2)
    weights = F.softmax(logits, dim=-1)               # (*, n_groups, 2)
    α_h = weights[..., 0]  # (*, n_groups)
    α_o = weights[..., 1]  # (*, n_groups)
    
    # 4. 加权混合
    h_g = h.view(*, n_groups, group_size)
    o_g = o.view(*, n_groups, group_size)
    result = α_h.unsqueeze(-1) * h_g + α_o.unsqueeze(-1) * o_g
    
    return result.view(*, d)
```

### 2.2 初始化

- `w_q = 0`：score_h = score_o = 0 → softmax([0, 0]) = [0.5, 0.5]
- 初始：`h_new = 0.5·h + 0.5·o = 0.5·(h + o)`
- 这不等于 `h + o`，但论文证明 softmax 归一化的等权平均从一开始就优于标准残差
- softmax'(0) 在均匀分布处梯度 = 0.25（良好的梯度流）

### 2.3 RMSNorm on Keys 的重要性

论文消融（Table 4）显示去掉 RMSNorm 会降低性能：
- Full AttnRes: 1.737 → 1.743 (w/o RMSNorm)
- Block AttnRes: 1.746 → 1.750 (w/o RMSNorm)

原因：h 的幅度随深度增长（PreNorm 稀释问题），如果不做 RMSNorm，
h 的 score 会系统性地大于 o 的 score，导致 α_h 主导，模型倾向于保留旧信息。

RMSNorm 将 h 和 o 归一化到相同尺度，让 softmax 的选择基于"方向"而非"幅度"。

## 三、Attention Head 与 n_groups 的关联性分析

### 3.1 标准 Transformer 中 head 的语义

hidden_size = num_heads × head_dim，每个 head 独立做 attention：
- 不同 head 学习不同的 attention pattern
- 某些 head 关注局部语法，某些关注全局语义
- head 的输出拼接后经过 o_proj 混合

### 3.2 n_groups 与 num_heads 的对齐

当 `n_groups = num_heads` 时，每个 group 恰好对应一个 head 的输出区域：

```
hidden_size = [head_0 | head_1 | ... | head_15]
n_groups    = [grp_0  | grp_1  | ... | grp_15 ]
```

**潜在优势**：
- 每个 head 可以独立决定"保留旧信息 vs 接受新信息"
- 如果某个 head 在当前层的 attention 输出不重要，可以选择保留旧信息
- 这实现了 **per-head 的深度信息选择**

**但论文发现**（Table 4）：multihead（H=16）劣于单头（1.752 vs 1.746）
- 论文解释："最优深度混合在通道间基本一致"
- 但这是在**深度维度**的发现，不一定适用于 hidden-size 维度

### 3.3 建议

v5 默认 `n_groups = 1`（全局单一 softmax），同时支持 `n_groups = num_heads` 作为实验选项。

理由：
1. 论文在深度维度发现单头更好
2. 但 hidden-size 维度可能不同——需要实验验证
3. n_groups=1 更简单，参数更少，先验证核心机制是否有效

## 四、与 v4 的关键差异

| 维度 | v4 (tanh 独立门控) | v5 (softmax 竞争选择) |
|------|-------------------|---------------------|
| 归一化 | 无（retain + accept 可以 > 2 或 < 0） | softmax（α_h + α_o = 1） |
| 幅度控制 | 无（信号可能衰减或放大） | 有（加权平均，幅度有界） |
| 初始行为 | h_new = h + o（精确恒等） | h_new = 0.5(h+o)（缩放但有界） |
| 梯度 | tanh'(0) = 1.0 | softmax'(uniform) = 0.25 |
| 竞争性 | 无（retain 和 accept 独立） | 有（α_h ↑ 则 α_o ↓） |
| RMSNorm | 无 | 有（on keys，防止幅度偏差） |

## 五、参数清单

每个 ResidualGate：
- `w_q`: (n_groups, group_size) — 伪查询，零初始化
- `w_kh`: (hidden_size,) — h 的 key 投影，ones 初始化
- `w_ko`: (hidden_size,) — o 的 key 投影，ones 初始化
- `key_norm`: RMSNorm(group_size) — key 归一化（无可学习权重，或共享）
- `log_tau`: (1,) — 温度参数

总参数：~2d + n_groups·group_size + 1 ≈ 3d（与 v4 相同）

## 六、实现注意事项

1. **RMSNorm 的实现**：对分组后的 key 做 RMSNorm，不需要可学习的 weight（论文中 RMSNorm inside φ 也没有可学习 weight）
2. **bfloat16 精度**：softmax 应在 float32 中计算，避免精度问题
3. **监控指标**：记录 α_h 和 α_o 的均值/方差/min/max，观察分化趋势
