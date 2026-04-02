# V11：完整 Self-Attention 残差门控 + 跨层深度 Attention

## 核心需求

1. **单层 hidden-size 的完整 self-attention**：QKV 权重全部是可学习的线性投影，跟随动态 token 内容变化，不依赖人为经验设计
2. **跨层 hidden-size attention**：建立深度方向的信息依赖，避免单层局部优化导致全局门控退化

## 一、需求一：单层 Hidden-Size Self-Attention Gate

### 1.1 当前 v5.2 的问题回顾

```
Q: 固定参数 w_q ∈ R^{G × G_sz}     ← 不跟随 token 内容
K: element-wise w_k ⊙ h             ← 不是线性投影
V: 直接用 h 和 o                     ← 没有 V 投影
```

### 1.2 完整 Self-Attention 的设计

将 h 和 o 视为 hidden-size 维度上的"两个 token"，在它们之间做标准的 self-attention：

```
输入序列：[h, o]  ← 2 个 "token"，每个 d 维

Q = W_q · [h, o]    ← (2, d) → (2, d_k)
K = W_k · [h, o]    ← (2, d) → (2, d_k)  
V = W_v · [h, o]    ← (2, d) → (2, d_v)

Attention = softmax(Q · K^T / √d_k)    ← (2, 2) 的 attention 矩阵
Output = Attention · V                   ← (2, d_v)

h_new = W_o · Output[0]    ← 取第一个位置的输出作为新的残差
```

**但这有问题**：这是在 2 个 "token" 之间做 attention，attention 矩阵只有 2×2，信息量太少。

### 1.3 更好的设计：将 hidden-size 分成多个 "token"

将 h ∈ R^d 和 o ∈ R^d 各自分成 G 个 group，每个 group 视为一个 "token"：

```
h = [h_1, h_2, ..., h_G]    ← G 个 "token"，每个 d/G 维
o = [o_1, o_2, ..., o_G]    ← G 个 "token"，每个 d/G 维

输入序列：[h_1, h_2, ..., h_G, o_1, o_2, ..., o_G]    ← 2G 个 "token"
```

在这 2G 个 "token" 之间做 self-attention：

```
Q = W_q · input    ← (2G, d/G) → (2G, d_k)
K = W_k · input    ← (2G, d/G) → (2G, d_k)
V = W_v · input    ← (2G, d/G) → (2G, d_v)

Attention = softmax(Q · K^T / √d_k)    ← (2G, 2G) 的 attention 矩阵
Output = Attention · V                   ← (2G, d_v)

h_new = W_o · Output[:G].reshape(d)    ← 取前 G 个位置的输出，拼接为 d 维
```

**这实现了**：
- **Q 跟随 token 内容**：Q = W_q · h_g，不同的 token 有不同的 Q
- **K 跟随 token 内容**：K = W_k · [h_g; o_g]，不同的内容有不同的 K
- **V 是可学习的投影**：V = W_v · [h_g; o_g]，不是直接用原始值
- **全局交互**：attention 矩阵是 (2G, 2G)，每个 group 可以看到所有其他 group
- **无人为经验设计**：纯粹的 self-attention，没有手工设计的物理信号

### 1.4 计算量分析

设 G = 16（与 v5.2 一致），d = 1024，d_k = d_v = d/G = 64：

- W_q, W_k, W_v: 3 × (d/G × d_k) = 3 × (64 × 64) = 12K 参数
- W_o: (G × d_v) × d = (16 × 64) × 1024 = 1M 参数 ← 太大！

**优化**：共享 W_q, W_k, W_v（所有 group 用同一组投影），W_o 用低秩分解。

```
W_o = W_o_down · W_o_up    ← (d, r) × (r, d)，r = 16
```

参数量：3 × 64² + d × r + r × d = 12K + 32K = 44K per gate

### 1.5 简化版：不分 group，直接在 h 和 o 上做 cross-attention

```python
class HiddenSizeSelfAttentionGate(nn.Module):
    def __init__(self, hidden_size, num_heads=4, head_dim=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        
        # QKV 投影（完整的线性投影，跟随 token 内容）
        # Q 来自 h（"我需要什么"）
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        # K 来自 o（"新信息提供什么"）
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        # V 来自 o（"新信息的内容"）
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        
        # 输出投影
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # Gate bias（控制初始行为）
        self.gate_bias = nn.Parameter(torch.zeros(hidden_size))
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
        
        # 初始化为小值，让初始行为接近标准残差
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(proj.weight, std=0.01)
    
    def forward(self, residual, new_output):
        # Q 来自 h：h 提出"我需要什么信息"
        q = self.q_proj(residual)    # (*, d) → (*, num_heads * head_dim)
        
        # K, V 来自 o：o 回答"我提供什么信息"
        k = self.k_proj(new_output)  # (*, d) → (*, num_heads * head_dim)
        v = self.v_proj(new_output)  # (*, d) → (*, num_heads * head_dim)
        
        # Reshape for multi-head
        # q: (batch, seq, num_heads, head_dim) → (batch, seq, num_heads, head_dim)
        orig_shape = residual.shape[:-1]
        q = q.view(*orig_shape, self.num_heads, self.head_dim)
        k = k.view(*orig_shape, self.num_heads, self.head_dim)
        v = v.view(*orig_shape, self.num_heads, self.head_dim)
        
        # Attention score：每个 head 产生一个标量 score
        # q · k 在 head_dim 维度上做点积
        score = (q * k).sum(dim=-1) * self.scale  # (*, num_heads)
        
        # Gate：sigmoid 将 score 转换为 [0, 1] 的门控值
        # 每个 head 控制 hidden_size / num_heads 个维度
        gate = torch.sigmoid(score)  # (*, num_heads)
        
        # V 投影 + gate 调制
        # 将 gate 扩展到每个维度
        gate_expanded = gate.unsqueeze(-1).expand_as(v)  # (*, num_heads, head_dim)
        gated_v = gate_expanded * v  # (*, num_heads, head_dim)
        
        # 输出投影
        gated_v = gated_v.reshape(*orig_shape, -1)  # (*, num_heads * head_dim)
        correction = self.o_proj(gated_v) + self.gate_bias  # (*, d)
        
        # 残差连接：h + correction
        # correction 编码了"从 o 中选择性提取的信息"
        return residual + correction
```

**等等，这个设计变成了 `h_new = h + correction(h, o)`，h 的系数恒为 1。**

这不满足你的需求——你要的是能够**过滤 h 中的冗余信息**。

### 1.6 完整版：同时门控 h 和 o

```python
class FullSelfAttentionGate(nn.Module):
    """
    完整的 Self-Attention 残差门控。
    
    将 [h, o] 视为 2 个 "token"，在它们之间做 multi-head cross-attention。
    
    每个 head 产生两个 score：
    - score_h：h 对自己的"自信度"（保留多少）
    - score_o：h 对 o 的"需求度"（接受多少）
    
    h_new = α ⊙ h + β ⊙ o
    
    其中 α, β 由 attention score 决定，是完全内容依赖的。
    """
    
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Q 投影：从 h 生成 query（"我需要什么"）
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # K 投影：从 h 和 o 分别生成 key
        self.k_h_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # V 投影：从 h 和 o 分别生成 value
        self.v_h_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 输出投影
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # RMSNorm
        self.norm_h = RMSNorm(hidden_size)
        self.norm_o = RMSNorm(hidden_size)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, residual, new_output):
        h_norm = self.norm_h(residual)
        o_norm = self.norm_o(new_output)
        
        orig_shape = residual.shape[:-1]
        head_shape = (*orig_shape, self.num_heads, self.head_dim)
        
        # Q from h
        q = self.q_proj(h_norm).view(head_shape)  # (*, H, D)
        
        # K from h and o
        k_h = self.k_h_proj(h_norm).view(head_shape)  # (*, H, D)
        k_o = self.k_o_proj(o_norm).view(head_shape)  # (*, H, D)
        
        # V from h and o
        v_h = self.v_h_proj(h_norm).view(head_shape)  # (*, H, D)
        v_o = self.v_o_proj(o_norm).view(head_shape)  # (*, H, D)
        
        # Attention scores
        score_h = (q * k_h).sum(-1) * self.scale  # (*, H)
        score_o = (q * k_o).sum(-1) * self.scale  # (*, H)
        
        # Softmax over [h, o] — 竞争选择
        scores = torch.stack([score_h, score_o], dim=-1)  # (*, H, 2)
        weights = F.softmax(scores, dim=-1)  # (*, H, 2)
        alpha = weights[..., 0].unsqueeze(-1)  # (*, H, 1)
        beta = weights[..., 1].unsqueeze(-1)   # (*, H, 1)
        
        # Weighted sum of values
        output = alpha * v_h + beta * v_o  # (*, H, D)
        output = output.reshape(*orig_shape, -1)  # (*, hidden_size)
        output = self.o_proj(output)
        
        return output
```

**但这个参数量太大了**：6 个 `Linear(d, d)` = 6d² ≈ 6M per gate。56 个 gate 共 336M，超过了模型本身的参数量。

### 1.7 低秩版本：保持 self-attention 语义，控制参数量

```python
class LowRankSelfAttentionGate(nn.Module):
    """
    低秩 Self-Attention 残差门控。
    
    用低秩投影替代全秩投影，保持 self-attention 的语义：
    - Q = W_q · h：h 提出"我需要什么"（内容依赖）
    - K_h = W_kh · h, K_o = W_ko · o：h 和 o 各自的"提供什么"
    - Score = Q · K / √r：语义匹配度
    - α, β = softmax/sigmoid(scores)：门控权重
    
    所有投影都是 d → r 的低秩投影（r << d），参数量可控。
    """
    
    def __init__(self, hidden_size, rank=32, init_bias=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        
        # Q 投影：h → R^r（h 提出"我需要什么"）
        self.q_proj = nn.Linear(hidden_size, rank, bias=False)
        
        # K 投影：h → R^r, o → R^r（各自"提供什么"）
        self.k_h_proj = nn.Linear(hidden_size, rank, bias=False)
        self.k_o_proj = nn.Linear(hidden_size, rank, bias=False)
        
        # 输出投影：r → d（将 attention 结果映射回 hidden_size）
        self.out_proj = nn.Linear(rank, hidden_size, bias=False)
        
        # RMSNorm
        self.norm = RMSNorm(hidden_size)
        
        # 缩放
        self.scale = rank ** -0.5
        
        # 初始化
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_h_proj.weight, std=0.02)
        nn.init.normal_(self.k_o_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.weight)  # 初始输出为 0
    
    def forward(self, residual, new_output):
        h_norm = self.norm(residual)
        o_norm = self.norm(new_output)
        
        # Q from h（内容依赖的 query）
        q = self.q_proj(h_norm)  # (*, r)
        
        # K from h and o
        k_h = self.k_h_proj(h_norm)  # (*, r)
        k_o = self.k_o_proj(o_norm)  # (*, r)
        
        # Attention scores（全局语义匹配）
        score_h = (q * k_h).sum(-1, keepdim=True) * self.scale  # (*, 1)
        score_o = (q * k_o).sum(-1, keepdim=True) * self.scale  # (*, 1)
        
        # Softmax over [h, o]
        scores = torch.cat([score_h, score_o], dim=-1)  # (*, 2)
        weights = F.softmax(scores, dim=-1)  # (*, 2)
        alpha = weights[..., 0:1]  # (*, 1)
        beta = weights[..., 1:2]   # (*, 1)
        
        # Weighted combination（标量门控，整个 vector 统一权重）
        h_new = alpha * residual + beta * new_output
        
        return h_new
```

**参数量**：3 × (d × r) + (r × d) + d = 4dr + d ≈ 4 × 1024 × 32 + 1024 ≈ 132K per gate

**但这是标量门控**（α 和 β 是标量，不是向量）。如果要维度级别的门控，需要更多参数。

## 二、需求二：跨层 Hidden-Size Attention

### 2.1 问题

单层门控只看到当前层的 h 和 o，不知道其他层发生了什么。这导致：
- 每层独立优化自己的门控策略
- 没有全局协调，可能导致所有层都选择类似的策略
- 无法建立"浅层保留 embedding，深层做语义变换"这样的全局分工

### 2.2 AttnRes 的跨层机制

AttnRes 天然具有跨层信息：每层的 query 可以看到所有历史层的 key/value。

### 2.3 在 hidden-size 维度上实现跨层信息

**方案 A：共享 gate 状态**

维护一个跨层的 "gate memory"，每层的 gate 可以读写这个 memory：

```
gate_memory = 0  # 初始化

for layer in layers:
    o = sublayer(RMSNorm(h))
    
    # Gate 读取 memory
    gate_input = [h, o, gate_memory]
    alpha, beta, gate_update = gate(gate_input)
    
    # 更新 h
    h = alpha * h + beta * o
    
    # 更新 memory
    gate_memory = gate_memory + gate_update
```

gate_memory 在深度方向上累积信息，让深层的 gate 知道浅层做了什么决策。

**方案 B：Block AttnRes 风格**

将 28 层分成 N 个 block（如 4 个 block，每个 7 层）。

Block 内：使用标准残差（保证信息不衰减）
Block 间：使用 hidden-size attention gate（选择性过滤）

```
Block 1 (layers 1-7):  标准残差 h = h + o
Block 2 (layers 8-14): 标准残差 h = h + o
Block 3 (layers 15-21): 标准残差 h = h + o
Block 4 (layers 22-28): 标准残差 h = h + o

Block 间 gate：
h_block2 = gate(h_block1_output, block2_input)
h_block3 = gate(h_block2_output, block3_input)
h_block4 = gate(h_block3_output, block4_input)
```

这样只有 3 个 gate（而不是 56 个），信息衰减大大减少。

**方案 C：深度方向的 Attention（最忠实于 AttnRes）**

在每一层，不仅看当前的 h 和 o，还看之前 K 层的 gate 输出：

```
# 维护最近 K 层的 gate 输出
history = [h_{l-K}, h_{l-K+1}, ..., h_{l-1}]

# 当前层的 gate 可以 attend 到历史
Q = W_q · h_l
K = W_k · stack(history)    # (K, d) → (K, r)
V = W_v · stack(history)    # (K, d) → (K, r)

# Attention over depth
attn = softmax(Q · K^T / √r)    # (1, K)
context = attn · V                # (1, r)

# 用 context 来指导当前层的门控
gate_logit = W_gate · [h_l, o_l, context]
alpha, beta = sigmoid(gate_logit)
```

这实现了跨层的信息传递，但需要额外的内存来存储历史。

## 三、综合设计：V11

结合两个需求，推荐以下设计：

### 3.1 架构概览

```
每一层：
  1. 标准子层处理：o = SubLayer(RMSNorm(h))
  2. Hidden-Size Self-Attention Gate：
     - Q = W_q · h（内容依赖的 query）
     - K_h = W_kh · h, K_o = W_ko · o（内容依赖的 key）
     - Score = Q · K / √r（全局语义匹配）
     - α, β = gate_function(scores)
     - h_new = α ⊙ h + β ⊙ o
  3. 跨层信息传递：
     - 将当前层的 gate 统计量（α, β 的均值）传递给下一层
     - 或者维护一个跨层的 gate memory
```

### 3.2 具体实现

```python
class V11ResidualGate(nn.Module):
    """
    V11: 完整 Self-Attention Gate + 跨层信息传递
    
    单层设计：
    - Q = W_q · h：内容依赖的 query（h 提出"我需要什么"）
    - K_h = W_kh · h：h 的 key（"h 提供什么"）
    - K_o = W_ko · o：o 的 key（"o 提供什么"）
    - Score = Q · K / √r：全局语义匹配
    - 多头设计：每个 head 独立计算 score，控制 d/H 个维度
    
    跨层设计：
    - 接收上一层的 gate_context（跨层信息）
    - 输出当前层的 gate_context 给下一层
    """
    
    def __init__(self, hidden_size, num_heads=8, context_dim=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.context_dim = context_dim
        
        # === 单层 Self-Attention ===
        # Q from h（内容依赖）
        self.q_proj = nn.Linear(hidden_size, num_heads, bias=False)
        # K from h and o（内容依赖）
        self.k_h_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.k_o_proj = nn.Linear(hidden_size, num_heads, bias=False)
        
        # === 跨层信息 ===
        # 接收上一层的 context
        self.context_proj = nn.Linear(context_dim, num_heads, bias=False)
        # 输出当前层的 context
        self.context_update = nn.Linear(num_heads * 2, context_dim, bias=False)
        
        # RMSNorm
        self.norm = RMSNorm(hidden_size)
        
        # 缩放
        self.scale = 1.0  # 因为 score 已经是标量
        
        # 初始化
        nn.init.zeros_(self.q_proj.weight)
        nn.init.zeros_(self.k_h_proj.weight)
        nn.init.zeros_(self.k_o_proj.weight)
        nn.init.zeros_(self.context_proj.weight)
        nn.init.zeros_(self.context_update.weight)
    
    def forward(self, residual, new_output, gate_context=None):
        h_norm = self.norm(residual)
        o_norm = self.norm(new_output)
        
        # Q from h
        q = self.q_proj(h_norm)  # (*, num_heads)
        
        # K from h and o
        k_h = self.k_h_proj(h_norm)  # (*, num_heads)
        k_o = self.k_o_proj(o_norm)  # (*, num_heads)
        
        # Scores
        score_h = q * k_h  # (*, num_heads) — element-wise，每个 head 一个标量
        score_o = q * k_o  # (*, num_heads)
        
        # 跨层调制
        if gate_context is not None:
            context_mod = self.context_proj(gate_context)  # (*, num_heads)
            score_h = score_h + context_mod
            score_o = score_o - context_mod  # 互补调制
        
        # Softmax over [h, o]
        scores = torch.stack([score_h, score_o], dim=-1)  # (*, num_heads, 2)
        weights = F.softmax(scores, dim=-1)
        alpha = weights[..., 0]  # (*, num_heads)
        beta = weights[..., 1]   # (*, num_heads)
        
        # 扩展到维度级别
        alpha_expanded = alpha.unsqueeze(-1).expand(
            *alpha.shape, self.head_dim
        ).reshape(*residual.shape)  # (*, d)
        beta_expanded = beta.unsqueeze(-1).expand(
            *beta.shape, self.head_dim
        ).reshape(*residual.shape)  # (*, d)
        
        h_new = alpha_expanded * residual + beta_expanded * new_output
        
        # 更新跨层 context
        gate_stats = torch.cat([alpha, beta], dim=-1)  # (*, num_heads * 2)
        new_context = self.context_update(gate_stats.detach())  # (*, context_dim)
        if gate_context is not None:
            new_context = gate_context + new_context  # 残差更新
        
        return h_new, new_context
```

### 3.3 参数量

- q_proj: d × H = 1024 × 8 = 8K
- k_h_proj: d × H = 8K
- k_o_proj: d × H = 8K
- context_proj: C × H = 16 × 8 = 128
- context_update: 2H × C = 16 × 16 = 256
- norm: d = 1K
- **总计：~25K per gate**

56 个 gate 共 ~1.4M 参数（占总参数 0.2%）

### 3.4 关键特性

1. **Q 是内容依赖的**：`Q = W_q · h`，不同 token 有不同的 query
2. **K 是内容依赖的**：`K_h = W_kh · h`, `K_o = W_ko · o`
3. **全局视野**：W_q 是 `Linear(d, H)`，看到整个 hidden vector
4. **跨层信息**：gate_context 在层间传递，让深层知道浅层的门控决策
5. **多头设计**：每个 head 控制 d/H 个维度，提供维度级别的门控粒度
6. **初始化为均匀**：所有权重初始化为 0，初始 score_h = score_o = 0，softmax([0,0]) = [0.5, 0.5]

## 四、与 v5.2 的对比

| 属性 | v5.2 | V11 |
|------|------|-----|
| Q 是否内容依赖 | ❌ 固定参数 | ✅ W_q · h |
| K 是否内容依赖 | ⚠️ element-wise | ✅ W_k · h, W_k · o |
| 全局视野 | ❌ 64 维 group | ✅ 整个 d 维 |
| 跨层信息 | ❌ 无 | ✅ gate_context |
| 多头 | ✅ 16 groups | ✅ 8 heads |
| 参数量/gate | ~4K | ~25K |
| 人为设计 | ⚠️ RMSNorm on keys, 分离 Q | ❌ 纯 self-attention |
