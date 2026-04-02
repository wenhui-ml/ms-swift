# V9：忠实于 AttnRes 原理的 Hidden-Size Attention 重新设计

## 一、回到 AttnRes 的第一性原理

### 1.1 AttnRes 解决的问题

标准残差 `h_l = h_0 + Σ o_i` 的问题：
- **PreNorm 稀释**：深层的 o_l 贡献被浅层的累积幅度稀释
- **信息冗余**：所有历史层的输出无条件叠加，冗余信息不断累积
- **梯度分布不均**：浅层梯度大，深层梯度小

### 1.2 AttnRes 的核心机制（从论文 Figure 1b）

```
h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i    (α 通过 softmax 归一化，Σα = 1)

其中：
  v_i = o_i                    (第 i 层的子层输出)
  q_l = w_l                    (第 l 层的 learned query)
  k_i = RMSNorm(v_i)           (第 i 层输出的归一化方向)
  score_{i→l} = q_l^T · k_i    (query-key 点积)
  α_{i→l} = softmax(score / τ) (在深度维度上 softmax)
```

**关键特性**：
1. **在深度维度上做 softmax**：L+1 个候选（embedding + L 层输出）竞争
2. **每层有独立的 query**：`w_l ∈ R^d`，编码"第 l 层需要什么信息"
3. **Key 是全维度的**：`k_i ∈ R^d`，编码"第 i 层提供什么信息"
4. **点积是在整个 d 维空间上做的**：`q_l^T · k_i` 是一个标量，捕捉全局语义匹配

### 1.3 AttnRes 为什么有效？

1. **全局视野**：q 和 k 都是 d 维向量，点积捕捉了整个 hidden vector 的语义匹配
2. **竞争选择**：softmax 在 L+1 个候选中选择，迫使模型做出有意义的选择
3. **内容依赖**：虽然 q 是 learned（不依赖输入），但 k 依赖于每层的实际输出
4. **幅度归一化**：RMSNorm 消除幅度差异，让选择只基于方向

## 二、你的 Hidden-Size 移植 vs AttnRes 原始设计

### 2.1 维度映射

| AttnRes 原始 | 你的 Hidden-Size 移植 |
|-------------|---------------------|
| 在**深度维度**上做 attention | 在**hidden-size 维度**上做 attention |
| L+1 个候选（各层输出） | 2 个候选（h 和 o） |
| 每层一个 query `w_l ∈ R^d` | 每组一个 query `w_q ∈ R^{group_size}` |
| Key 是 d 维向量 | Key 是 group_size 维向量 |
| 点积产生 1 个标量 score | 点积产生 n_groups 个标量 score |
| softmax 在 L+1 个候选上 | softmax 在 2 个候选上 |

### 2.2 移植中丢失了什么？

**丢失 1：候选数量从 L+1 降到 2**

AttnRes 在 L+1 个候选中选择，这提供了丰富的选择空间。你的设计只在 h 和 o 之间选择，选择空间极其有限。

更重要的是：AttnRes 的 softmax 在 L+1 个候选上归一化，每个候选的权重可以很小（1/(L+1)），但总和为 1。这意味着即使某个候选的权重很小，它的信息也不会完全丢失——它只是被稀释了。

而你的 softmax 在 2 个候选上归一化，α_h + α_o = 1。如果 α_h = 0.1，那 h 的信息就被衰减了 90%。经过多层后，这种衰减是灾难性的。

**丢失 2：Query 的维度从 d 降到 group_size**

AttnRes 的 query `w_l ∈ R^d` 看到整个 hidden vector，可以捕捉全局语义。

你的 query `w_q ∈ R^{group_size}` 只看到 64 维的 group，无法捕捉全局语义。

**丢失 3：点积的语义从"全局匹配"变成"局部匹配"**

AttnRes 的 `q_l^T · k_i` 是在整个 d 维空间上的点积，产生一个标量，表示"第 l 层的需求与第 i 层的输出在语义空间中的匹配度"。

你的 `(w_q · key_g).sum(-1)` 是在 group_size 维空间上的点积，产生一个标量，表示"这个 group 的 query 与 key 的局部匹配度"。

**丢失 4：从"选择哪一层"变成"选择 h 还是 o"**

AttnRes 的决策是"从哪些历史层获取信息"——这是一个有意义的语义决策。

你的决策是"在每个维度组上保留 h 还是接受 o"——这是一个低层次的信号处理决策。

## 三、根本问题的重新定义

### 3.1 你想要的是什么？

你想要在 **hidden-size 维度**上实现 AttnRes 的效果：让模型能够选择性地利用残差流中的信息，过滤冗余，保留有用的。

### 3.2 为什么在 hidden-size 维度上做这件事？

AttnRes 在深度维度上做 attention 需要保存所有历史层的输出，内存开销 O(L·d)。

如果能在 hidden-size 维度上实现类似的效果，就不需要额外的内存开销。

### 3.3 核心矛盾

AttnRes 的成功依赖于：
1. 在**多个候选**中做 softmax 选择
2. 用**全维度**的 query-key 点积做语义匹配

在 hidden-size 维度上，你只有 2 个候选（h 和 o），且每个 group 只有 64 维。这两个条件都不满足。

## 四、重新设计：忠实于 AttnRes 原理

### 4.1 核心思想：在 hidden-size 维度上创造"多个候选"

AttnRes 的力量来自于在多个候选中选择。在 hidden-size 维度上，我们可以通过**将 h 和 o 分解为多个子空间**来创造多个候选。

```
h = [h_1, h_2, ..., h_G]    (G 个子空间，每个 d/G 维)
o = [o_1, o_2, ..., o_G]    (G 个子空间)
```

对于每个子空间 g，有 2 个候选：h_g 和 o_g。

但这仍然只有 2 个候选。

**更好的方法**：将 h 和 o 通过不同的投影创造更多候选：

```
候选 1: h 本身
候选 2: o 本身
候选 3: W_1 · h  (h 的线性变换 1)
候选 4: W_2 · h  (h 的线性变换 2)
候选 5: W_3 · o  (o 的线性变换)
...
```

但这增加了太多参数。

### 4.2 换一个角度：不是"选择候选"，而是"选择维度"

AttnRes 在深度维度上选择"从哪一层获取信息"。

在 hidden-size 维度上，我们可以选择"保留哪些维度的信息"。

**这就是 attention 在 hidden-size 维度上的正确语义**：

```
对于 h 的每个维度 j：
  score_j = f(h, o, j)    (维度 j 的"重要性分数")
  
α = softmax(score_h) ∈ R^d    (在 d 个维度上 softmax)
β = softmax(score_o) ∈ R^d    (在 d 个维度上 softmax)

h_new = α ⊙ h + β ⊙ o
```

但这里 softmax 在 d 个维度上归一化，意味着 Σ α_j = 1。这会让每个维度的权重 ≈ 1/d，太小了。

### 4.3 真正的问题：softmax 的归一化维度

**AttnRes 的 softmax 在深度维度上归一化**（L+1 个候选），这是合理的——因为我们要在不同层之间分配注意力。

**你的 softmax 在 2 个候选上归一化**（h 和 o），这导致了零和博弈——增加 h 的权重必然减少 o 的权重。

**如果我们在 hidden-size 维度上归一化呢？**

```
score_j = q^T · k_j / √d    (q 是 learned query，k_j 是第 j 维的 key)
α_j = softmax(score)_j       (在 d 个维度上 softmax)

h_new_j = α_j · h_j + (1 - α_j) · o_j    ← 不对，α_j 太小
```

这不行，因为 softmax 在 d 维上归一化后每个 α_j ≈ 1/d。

### 4.4 关键洞察：需要的不是 softmax，而是 sigmoid + 全局信息

回到 AttnRes 的本质：它的 softmax 之所以有效，是因为它在**有意义的候选集**上做归一化。

在 hidden-size 维度上，没有一个自然的"候选集"可以做 softmax。每个维度不是一个"候选"——它们是一个向量的不同分量。

**正确的做法**：用 attention 机制来**提取全局信息**，然后用这个全局信息来指导每个维度的门控。

```
# Step 1: 用 attention 提取 h 和 o 的全局语义摘要
summary_h = Attention_pool(h)    # 从 h 中提取关键信息
summary_o = Attention_pool(o)    # 从 o 中提取关键信息

# Step 2: 用全局摘要来指导维度级别的门控
α = sigmoid(W_α · [summary_h; summary_o])    # d 维门控
β = sigmoid(W_β · [summary_h; summary_o])    # d 维门控

h_new = α ⊙ h + β ⊙ o
```

### 4.5 Attention Pooling 的设计

如何从 h ∈ R^d 中提取全局语义摘要？

**方法 A：Cross-Attention（h 查询 o）**

```
Q = W_q · h ∈ R^r        (h 提出"我需要什么")
K = W_k · o ∈ R^r        (o 回答"我提供什么")
V = W_v · o ∈ R^r        (o 的信息)

score = Q^T · K / √r     (标量匹配度)
summary = score · V       (加权信息)
```

这捕捉了"h 需要什么"和"o 提供什么"之间的匹配度。

**方法 B：Self-Attention over dimensions（维度间 attention）**

将 h ∈ R^d 视为 d 个 1 维 token，在维度间做 self-attention：

```
h_reshaped = h.view(n_heads, head_dim)    # (n_heads, head_dim)
Q = W_q · h_reshaped                      # (n_heads, r)
K = W_k · h_reshaped                      # (n_heads, r)
V = W_v · h_reshaped                      # (n_heads, r)

attn = softmax(Q · K^T / √r)             # (n_heads, n_heads)
summary = attn · V                         # (n_heads, r)
```

这让不同的维度组之间可以交换信息，实现真正的"全局视野"。

但这个计算量太大了（O(n_heads²)）。

**方法 C：低秩全局交互（最实用）**

```
# 将 d 维压缩到 r 维（全局摘要）
global_h = W_compress · h ∈ R^r    # (d → r)
global_o = W_compress · o ∈ R^r    # (d → r)

# 在低秩空间中做交互
interaction = global_h * global_o   # element-wise，捕捉 h-o 交互

# 从低秩空间展开回 d 维
gate_logit = W_expand · interaction ∈ R^d    # (r → d)

α = sigmoid(gate_logit + bias_α)
β = sigmoid(-gate_logit + bias_β)    # 互补关系
```

这个设计的关键：
- `W_compress` 将 d 维压缩到 r 维，实现**全局信息聚合**
- `global_h * global_o` 在低秩空间中捕捉 h 和 o 的**交互模式**
- `W_expand` 将交互信息展开回 d 维，指导每个维度的门控
- α 和 β 通过共享 gate_logit 的正负号实现**互补关系**

## 五、V9 最终设计

```python
class ResidualGateV9(nn.Module):
    """
    基于全局交互的 Hidden-Size Attention Gate。
    
    核心思想：
    1. 将 h 和 o 压缩到低秩空间（全局信息聚合）
    2. 在低秩空间中计算 h-o 交互（语义匹配）
    3. 将交互信息展开回 d 维（维度级门控）
    
    这忠实于 AttnRes 的原理：
    - AttnRes 用 q·k 做全局语义匹配 → 我们用低秩压缩 + 交互
    - AttnRes 用 softmax 做选择 → 我们用 sigmoid 做独立门控
    - AttnRes 在深度维度选择 → 我们在 hidden-size 维度选择
    
    h_new = α ⊙ h + β ⊙ o
    
    其中 α, β ∈ (0, 1)^d，由 h 和 o 的全局交互决定。
    """
    
    def __init__(self, hidden_size, rank=16, init_bias=3.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        
        # 全局压缩：d → r（共享，h 和 o 用同一个投影）
        self.compress = nn.Linear(hidden_size, rank, bias=False)
        
        # 交互后展开：r → d
        self.expand = nn.Linear(rank, hidden_size, bias=False)
        
        # 可学习的温度
        self.log_tau = nn.Parameter(torch.zeros(1))
        
        # 偏置（控制初始行为）
        self.bias_alpha = nn.Parameter(torch.full((hidden_size,), init_bias))
        self.bias_beta = nn.Parameter(torch.full((hidden_size,), init_bias))
        
        # RMSNorm for inputs
        self.norm = RMSNorm(hidden_size)
        
        # 初始化：compress 和 expand 为小随机值
        # 这样初始时 gate_logit ≈ 0，α ≈ sigmoid(bias)，β ≈ sigmoid(bias)
        nn.init.normal_(self.compress.weight, std=0.01)
        nn.init.normal_(self.expand.weight, std=0.01)
    
    def forward(self, residual, new_output):
        # 归一化（消除幅度差异，只看内容/方向）
        h_norm = self.norm(residual)
        o_norm = self.norm(new_output)
        
        # 全局压缩：d → r
        global_h = self.compress(h_norm)    # (*, r)
        global_o = self.compress(o_norm)    # (*, r)
        
        # 温度缩放
        tau = self.log_tau.exp() + 1e-8
        
        # 全局交互：element-wise 乘积捕捉 h-o 的语义匹配
        # 这类似于 AttnRes 的 q^T · k，但在低秩空间中
        interaction = global_h * global_o / tau    # (*, r)
        
        # 展开回 d 维：r → d
        gate_logit = self.expand(interaction)    # (*, d)
        
        # 门控
        alpha = torch.sigmoid(gate_logit + self.bias_alpha)     # 保留门
        beta = torch.sigmoid(-gate_logit + self.bias_beta)      # 接受门
        
        return alpha * residual + beta * new_output
```

### 5.1 为什么这个设计忠实于 AttnRes？

| AttnRes 原理 | V9 的实现 |
|-------------|----------|
| 全局语义匹配（q^T · k 在 d 维上） | 低秩压缩后的交互（compress(h) * compress(o) 在 r 维上） |
| RMSNorm 消除幅度 | RMSNorm 消除幅度 |
| 温度缩放 τ | 可学习温度 τ |
| 内容依赖的权重 | α, β 依赖于 h 和 o 的内容 |
| 选择性聚合 | α ⊙ h + β ⊙ o |

### 5.2 为什么这个设计解决了 v5.2 的问题？

| v5.2 的问题 | V9 的解决 |
|------------|----------|
| 分组点积视野太窄（64 维） | 低秩压缩看到整个 d 维 |
| Q 是固定参数 | gate 依赖于 h 和 o 的实际内容 |
| K 是 element-wise 乘法 | compress 是矩阵乘法（线性变换） |
| softmax 约束 α+β=1 | 独立 sigmoid，α+β 可以 > 1 或 < 1 |

### 5.3 初始化行为

- compress 和 expand 初始化为小随机值（std=0.01）
- 初始时 gate_logit ≈ 0（因为 compress 输出小，interaction 更小，expand 后仍然小）
- α ≈ sigmoid(bias_alpha) ≈ sigmoid(3) ≈ 0.953
- β ≈ sigmoid(bias_beta) ≈ sigmoid(3) ≈ 0.953
- h_new ≈ 0.953·h + 0.953·o ≈ h + o（近似标准残差）

### 5.4 梯度分析

```
∂h_new/∂h = α · I + ∂α/∂h · h + ∂β/∂h · o
```

- 直通项：α · I ≈ 0.953 · I（初始时）
- 间接项：通过 compress → interaction → expand → sigmoid 的链式求导

**关键**：α 是 sigmoid 输出，范围 (0, 1)。初始时 α ≈ 0.953，梯度直通系数接近 1。

**但 α 可以学到 < 1 的值**，这会导致信息衰减。这是设计意图——当 h 中的信息冗余时，α 应该降低。

**与 v5.2 的区别**：v5.2 的 softmax 约束 α + β = 1，所以 α 降低必然导致 β 升高。V9 的独立 sigmoid 允许 α 和 β 同时降低（抑制）或同时升高（协同）。

### 5.5 互补关系的设计

```python
alpha = sigmoid(gate_logit + bias_alpha)     # gate_logit 正 → α 高
beta = sigmoid(-gate_logit + bias_beta)      # gate_logit 正 → β 低
```

gate_logit 的正负号决定了 α 和 β 的互补关系：
- gate_logit > 0：α 高，β 低 → 保留 h，减少 o
- gate_logit < 0：α 低，β 高 → 减少 h，接受 o
- gate_logit ≈ 0：α ≈ β ≈ sigmoid(bias) → 均衡混合

**但 bias_alpha 和 bias_beta 是独立的**，所以 α 和 β 的基准值可以不同。

## 六、参数量分析

- compress: d × r = 1024 × 16 = 16K
- expand: r × d = 16 × 1024 = 16K
- log_tau: 1
- bias_alpha: d = 1024
- bias_beta: d = 1024
- norm: d = 1024 (weight)
- **总计：~35K per gate**

56 个 gate 共 ~2M 参数，占总参数量 596M 的 0.3%。

## 七、与 v5.2 的关键区别总结

1. **全局视野**：compress(h) 看到整个 1024 维，而不是 64 维的 group
2. **内容依赖**：gate 依赖于 h 和 o 的实际内容（通过 compress 提取语义摘要）
3. **交互机制**：global_h * global_o 捕捉 h-o 的语义匹配，而不是固定 Q 与 K 的点积
4. **独立 sigmoid**：α + β 不受约束，可以协同或抑制
5. **初始化接近标准残差**：训练初期不破坏信息流

## 八、潜在风险

1. **α 可能学到恒为 0**：如果 gate_logit 学到很大的负值，α → 0，信息衰减
   - 缓解：bias_alpha = 3.0 提供了强正偏置，gate_logit 需要 < -3 才能让 α < 0.5
   - 监控：训练过程中监控 alpha.mean()

2. **compress 可能退化**：如果 compress 学到接近零的权重，gate 退化为常数
   - 这实际上是好的退化：gate 退化为 sigmoid(bias)，接近标准残差

3. **交互项可能太弱**：global_h * global_o 的幅度可能很小
   - 缓解：温度 τ 可以学习，放大交互信号
   - 或者用 `global_h^T · global_o`（标量点积）+ expand 来替代 element-wise 乘积

## 九、实验建议

### 9.1 最小可行实验

1. 在 Qwen3-0.6B 配置上训练 V9，对比标准残差和 v5.2
2. 训练 3500 步（与之前实验一致）
3. 评估 lambada_openai 和 piqa

### 9.2 消融实验

1. **rank**：4, 8, 16, 32
2. **init_bias**：1.0, 3.0, 5.0
3. **交互方式**：element-wise 乘积 vs 标量点积 vs 拼接后 MLP
4. **是否共享 compress**：h 和 o 用同一个 compress vs 不同的 compress
5. **互补关系**：α 和 β 共享 gate_logit vs 独立 gate_logit

### 9.3 关键监控指标

- alpha.mean() 和 beta.mean()：是否在合理范围
- alpha.std()：是否有足够的变化（说明内容依赖在起作用）
- gate_logit 的分布：是否有正有负（说明在做选择）
- 不同层的 alpha 分布：浅层 vs 深层是否不同
