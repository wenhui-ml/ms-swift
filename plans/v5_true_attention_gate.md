# V5：真正对标 Self-Attention 的 ResidualGate

## 一、三个问题的根因分析

### 问题 1：参数冗余（w_q · w_kh 退化为标量）

**根因**：在 per-dim 操作中，query 和 key 都是标量，它们的"点积"就是标量乘法。

**Self-attention 为什么不退化**：因为 Q 和 K 是 d_k 维向量，Q·K^T 是跨 d_k 个维度的求和。这个求和让 score 依赖于多个维度的联合信息。

**解决方案**：让 score 的计算涉及 **跨维度的信息聚合**。具体来说，将 h 和 o 投影到一个低维的 query/key 空间（rank 维），然后在这个空间中做真正的向量点积。

### 问题 2：无跨维度交互

**根因**：per-dim 独立操作，第 j 维的 gate 不知道其他维度的状态。

**Self-attention 为什么有跨维度交互**：Q = W_Q · x 是一个矩阵乘法，Q 的每个分量依赖于 x 的所有维度。

**解决方案**：使用低秩投影（矩阵乘法）来生成 query 和 key，让它们包含跨维度信息。

### 问题 3：softmax 初始化导致信号/梯度衰减

**根因**：softmax([0, 0]) = [0.5, 0.5]，所以 h_new = 0.5h + 0.5o，信号和梯度都衰减一半。

**Self-attention 为什么没有这个问题**：因为 attention 的 value 不是"要保留的原始信号"，而是"要提取的新信息"。attention 的输出会加到残差上：h_new = h + Attention(h)。

**解决方案**：将 gate 的输出定义为 **残差的修正量**，而不是替代残差本身。

## 二、重新理解目标

目标：**让有限的 hidden-size 最大化存储有价值的知识，剔除冗余和有害信息。**

在标准 Transformer 中：
```
h_new = h + o    (无条件接受所有新信息)
```

问题：o 中可能包含冗余或有害信息，无条件加入会污染残差流。

我们想要的是：
```
h_new = h + gate(h, o) ⊙ o    (选择性接受新信息)
```

或者更一般地：
```
h_new = retain(h, o) ⊙ h + accept(h, o) ⊙ o    (选择性保留+接受)
```

## 三、对标 Self-Attention 的正确方式

### 3.1 Self-Attention 的本质

Self-attention 做的事情是：对于每个 token position，从所有 token 中 **选择性地提取信息**。

```
Q = W_Q · x_i          (当前 token 的需求)
K = W_K · x_j          (候选 token 的标识)
V = W_V · x_j          (候选 token 的内容)
score = Q · K^T / √d_k (需求-标识匹配度)
α = softmax(score)     (归一化权重)
output = Σ α_j · V_j   (加权聚合内容)
```

关键：Q·K^T 是 **d_k 维向量的点积**，它聚合了跨维度的信息来计算一个标量 score。

### 3.2 映射到 ResidualGate

在 ResidualGate 中：
- 有 2 个"候选"：h（旧残差）和 o（新输出）
- 需要为每个 hidden-size 维度计算权重

**正确的映射**：

```
对于每个维度 j，我们需要一个 score 来决定 h_j 和 o_j 的权重。
这个 score 应该依赖于 h 和 o 的 **全局信息**（跨维度），而不仅仅是第 j 维。
```

具体来说：

```
# 将 h 和 o 投影到低维 "gate 空间"（跨维度聚合）
q = W_down · h    (d → r)  — h 的全局特征摘要
k = W_down · o    (d → r)  — o 的全局特征摘要

# 在 gate 空间中计算交互
interaction = q ⊙ k    (r 维) — 全局特征的匹配度

# 投影回 hidden-size，生成 per-dim 的 gate
gate_logits = W_up · interaction    (r → d)

# 归一化
gate = sigmoid(gate_logits + bias)
```

但这又回到了 V1 的低秩网络...

### 3.3 关键洞察：AttnRes 的做法

AttnRes 论文的做法是：
```
q_l = w_l           (per-layer learned query, d 维)
k_i = RMSNorm(v_i)  (历史层输出的方向, d 维)
score = q_l^T · k_i  (d 维向量的点积 → 标量)
```

AttnRes 的 score 是一个 **标量**（对所有维度求和），用于决定"整个层的权重"。

但我们需要的是 **per-dim 的权重**。

### 3.4 解决方案：多头 Attention over Depth

借鉴 multi-head attention 的思想：将 d 维分成 n_heads 组，每组独立计算 attention。

```
将 d 维分成 n_groups 组，每组 group_size = d / n_groups 维。
每组内部共享一个 score（跨组内维度聚合），但不同组有不同的 score。
```

这样：
- 组内有跨维度交互（通过组内求和）
- 不同组有不同的 gate 值（per-group 而非 per-dim）
- 参数量适中

```python
class ResidualGate(nn.Module):
    def __init__(self, hidden_size, n_groups=16, init_bias=5.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_groups = n_groups
        self.group_size = hidden_size // n_groups
        
        # Per-group query (n_groups 个 group_size 维向量)
        self.w_q = nn.Parameter(torch.zeros(n_groups, self.group_size))
        
        # Key projections (per-dim, 用于将 h/o 投影到 key 空间)
        self.w_kh = nn.Parameter(torch.ones(hidden_size))
        self.w_ko = nn.Parameter(torch.ones(hidden_size))
        
        # Per-group bias
        self.b_h = nn.Parameter(torch.full((n_groups,), init_bias / 2))
        self.b_o = nn.Parameter(torch.full((n_groups,), init_bias / 2))
        
        # Temperature
        self.log_tau = nn.Parameter(torch.zeros(1))
    
    def forward(self, residual, new_output):
        B, T, d = residual.shape
        tau = self.log_tau.exp()
        
        # Key projection (per-dim)
        key_h = self.w_kh * residual    # (B, T, d)
        key_o = self.w_ko * new_output  # (B, T, d)
        
        # Reshape to groups: (B, T, n_groups, group_size)
        key_h_g = key_h.view(B, T, self.n_groups, self.group_size)
        key_o_g = key_o.view(B, T, self.n_groups, self.group_size)
        
        # Query-Key dot product within each group (跨维度聚合！)
        # w_q: (n_groups, group_size)
        # key_h_g: (B, T, n_groups, group_size)
        # score: (B, T, n_groups) — 每组一个标量 score
        score_h = (self.w_q * key_h_g).sum(dim=-1) + self.b_h  # (B, T, n_groups)
        score_o = (self.w_q * key_o_g).sum(dim=-1) + self.b_o  # (B, T, n_groups)
        
        # Softmax over 2 candidates per group
        scores = torch.stack([score_h, score_o], dim=-1) / (tau + 1e-8)  # (B, T, n_groups, 2)
        weights = torch.softmax(scores, dim=-1)  # (B, T, n_groups, 2)
        
        # Expand back to per-dim: (B, T, n_groups, 1) → (B, T, d)
        alpha = weights[..., 0].unsqueeze(-1).expand_as(key_h_g).reshape(B, T, d)
        beta = weights[..., 1].unsqueeze(-1).expand_as(key_o_g).reshape(B, T, d)
        
        return alpha * residual + beta * new_output
```

### 3.5 初始化问题的解决

初始时 w_q=0：
```
score_h = 0 + b_h = init_bias/2
score_o = 0 + b_o = init_bias/2
softmax([init_bias/2, init_bias/2] / τ) = [0.5, 0.5]
h_new = 0.5h + 0.5o
```

仍然有衰减问题！

**解决方案**：不用 softmax 的 α+β=1 约束，而是用 **缩放的 softmax**：

```
h_new = 2 · (α ⊙ h + β ⊙ o)    其中 α + β = 1
```

初始时：h_new = 2 · (0.5h + 0.5o) = h + o ✅

但这引入了一个固定的 2x 缩放，不够优雅。

**更好的解决方案**：使用 **残差形式**：

```
h_new = h + β ⊙ (o - h)    其中 β ∈ [0, 1]
```

- β=0 → h_new = h（完全保留）
- β=1 → h_new = o（完全替换）
- β=0.5 → h_new = 0.5h + 0.5o（均匀混合）

初始化 β=0.5 → h_new = 0.5h + 0.5o，仍然有衰减。

**最终解决方案**：使用 **加法残差 + 选择性接受**：

```
h_new = h + α ⊙ o    其中 α ∈ [0, 1]
```

- α=1 → h_new = h + o（标准残差）
- α=0 → h_new = h（拒绝新信息）
- α=0.5 → h_new = h + 0.5o（部分接受）

初始化 α=1 → h_new = h + o ✅ 完美的近恒等启动！

**但这失去了"遗忘旧信息"的能力**。

### 3.6 最终方案：分离保留和接受

```
h_new = (1 - forget) ⊙ h + accept ⊙ o
```

其中 forget 和 accept 都是 [0, 1] 的 per-group gate：
- forget=0, accept=1 → h_new = h + o（标准残差）← 初始状态
- forget>0 → 遗忘部分旧信息
- accept<1 → 拒绝部分新信息

初始化：forget=0, accept=1 → h_new = h + o ✅

```python
class ResidualGate(nn.Module):
    def __init__(self, hidden_size, n_groups=16, init_accept_bias=5.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_groups = n_groups
        self.group_size = hidden_size // n_groups
        
        # Shared query for both forget and accept decisions
        self.w_q = nn.Parameter(torch.zeros(n_groups, self.group_size))
        
        # Key projections
        self.w_kh = nn.Parameter(torch.ones(hidden_size))
        self.w_ko = nn.Parameter(torch.ones(hidden_size))
        
        # Forget gate: sigmoid(score) → 0 initially (no forgetting)
        self.b_forget = nn.Parameter(torch.full((n_groups,), -init_accept_bias))  # sigmoid(-5)≈0.007
        
        # Accept gate: sigmoid(score) → 1 initially (full acceptance)
        self.b_accept = nn.Parameter(torch.full((n_groups,), init_accept_bias))   # sigmoid(5)≈0.993
        
        # Temperature
        self.log_tau = nn.Parameter(torch.zeros(1))
    
    def forward(self, residual, new_output):
        B, T, d = residual.shape
        tau = self.log_tau.exp()
        
        # Key projection
        key_h = (self.w_kh * residual).view(B, T, self.n_groups, self.group_size)
        key_o = (self.w_ko * new_output).view(B, T, self.n_groups, self.group_size)
        
        # Query-Key dot product (跨组内维度聚合)
        # score_h: "h 在这个组中的特征与 query 的匹配度"
        # score_o: "o 在这个组中的特征与 query 的匹配度"
        score_h = (self.w_q * key_h).sum(dim=-1) / (tau * math.sqrt(self.group_size) + 1e-8)
        score_o = (self.w_q * key_o).sum(dim=-1) / (tau * math.sqrt(self.group_size) + 1e-8)
        
        # Forget gate: 基于 h 的特征决定遗忘多少
        forget = torch.sigmoid(score_h + self.b_forget)  # (B, T, n_groups)
        
        # Accept gate: 基于 o 的特征决定接受多少
        accept = torch.sigmoid(score_o + self.b_accept)  # (B, T, n_groups)
        
        # Expand to per-dim
        forget = forget.unsqueeze(-1).expand(B, T, self.n_groups, self.group_size).reshape(B, T, d)
        accept = accept.unsqueeze(-1).expand(B, T, self.n_groups, self.group_size).reshape(B, T, d)
        
        # h_new = (1 - forget) ⊙ h + accept ⊙ o
        return (1 - forget) * residual + accept * new_output
```

## 四、最终方案的验证

### 初始化
- w_q=0 → score_h=0, score_o=0
- forget = sigmoid(0 + (-5)) = sigmoid(-5) ≈ 0.007 ≈ 0
- accept = sigmoid(0 + 5) = sigmoid(5) ≈ 0.993 ≈ 1
- h_new = (1-0.007)h + 0.993·o ≈ h + o ✅

### 梯度
```
∂h_new/∂h = (1-forget) + h · ∂(-forget)/∂h + o · ∂accept/∂h
```
初始时 forget≈0, ∂forget/∂h≈0（因为 w_q=0）：
```
∂h_new/∂h ≈ 1    ✅ 梯度不衰减！
```

### 跨维度交互
score_h = Σ_{i∈group} w_q_i · w_kh_i · h_i → 组内跨维度求和 ✅

### 参数量
- w_q: n_groups × group_size = d
- w_kh, w_ko: d + d = 2d
- b_forget, b_accept: n_groups + n_groups = 2·n_groups
- log_tau: 1
- 总计: 3d + 2·n_groups + 1 ≈ 3d（d=1024, n_groups=16 → ~3K）

## 五、对标 Self-Attention 的完整映射

| Self-Attention | ResidualGate V5 |
|---|---|
| Q = W_Q · x (d→d_k) | w_q (n_groups × group_size) — 分组 query |
| K = W_K · x (d→d_k) | key_h = w_kh ⊙ h, key_o = w_ko ⊙ o — per-dim key 投影 |
| Q·K^T / √d_k | (w_q ⊙ key).sum(group) / (τ·√group_size) — 组内点积 |
| softmax | sigmoid（独立的 forget 和 accept） |
| V | h 和 o 本身 |
| output = Σα·V | h_new = (1-forget)⊙h + accept⊙o |
| 残差连接 h + output | 内置在公式中（初始 forget≈0, accept≈1 → h+o） |
