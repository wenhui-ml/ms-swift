# V6 深度分析：为什么 Hidden-Size Attention Gate 失败了，以及如何重新设计

## 一、实验事实总结

| 指标 | Qwen3-0.6B | attn_res_gate v5.2 | 差异 |
|------|-----------|-------------------|------|
| train/loss | 3.54 | ~3.5 | 基本一致 |
| train/token_acc | 0.389 | ~0.38 | 基本一致 |
| lambada_openai acc | 0.0621 | **0.0000** | 完全失败 |
| lambada_openai ppl | 3,498 | **4,441,009** | 差 1270 倍 |
| piqa acc | 0.5805 | 0.5250 | 接近随机 |

**核心矛盾**：训练 loss 几乎一样，但评估指标差距巨大。

## 二、这个矛盾说明了什么？

### 2.1 训练 loss 相似 ≠ 模型能力相同

训练 loss 是在 teacher-forcing 模式下计算的：给定正确的前 t-1 个 token，预测第 t 个 token。这只需要**局部的 n-gram 统计能力**。

lambada_openai 测试的是：给定一个完整的句子（去掉最后一个词），预测最后一个词。这需要**理解整个句子的语义**，是一个长距离依赖任务。

**结论**：attn_res_gate 模型学到了局部统计模式（足以降低 cross-entropy），但**完全没有学到长距离语义理解**。

### 2.2 为什么长距离信息传递失败了？

标准 Transformer 的残差流 `h_L = h_0 + Σ f_l(h_l)` 保证了 embedding 层的信息 h_0 可以无损地传到最后一层。这是 Transformer 能做长距离依赖的基础。

attn_res_gate 的残差流 `h_{l+1} = α_h · h_l + α_o · o_l`（α_h + α_o = 1）在每一层都对 h 做了衰减。经过 28 层 × 2 个 gate = 56 次衰减后，h_0 的信息几乎完全丢失。

## 三、当前 Hidden-Size Attention Gate 的根本性缺陷分析

### 3.1 缺陷一：分组点积的"视野"太窄

当前设计将 hidden_size=1024 分成 16 个 group，每组 64 维。每个 group 独立计算一个标量 score：

```
score_g = (w_q_g · RMSNorm(w_k_g ⊙ h_g)).sum() / √64
```

**问题**：每个 group 只能看到 64 个维度的信息。它不知道其他 group 发生了什么。

在 Transformer 中，语义信息是分布在**整个 hidden_size** 上的（distributed representation）。一个 64 维的窗口无法捕捉到"这个 token 是否需要保留底层的 embedding 信息"这种全局性的决策。

**类比**：这就像让 16 个人各自只看一幅画的 1/16，然后独立决定是否保留这幅画。每个人看到的局部信息不足以做出正确的全局决策。

### 3.2 缺陷二：Q 是静态的（不依赖输入内容）

```python
self.w_qh = nn.Parameter(torch.zeros(n_groups, group_size))  # 固定参数
self.w_qo = nn.Parameter(torch.randn(n_groups, group_size) * 0.1)  # 固定参数
```

Q 是可学习的固定参数，不依赖于当前 token 的内容。这意味着：

- 对于所有 token、所有位置、所有上下文，同一个 group 使用**完全相同的 Q** 来评估 h 和 o
- Gate 无法做出"这个特定 token 在这个特定上下文中需要什么"的决策
- 它只能学到一个**全局平均最优**的策略

**对比标准 Attention**：Q = W_q · h，Q 是输入依赖的。不同的 token 有不同的 Q，可以"提出不同的问题"。

### 3.3 缺陷三：K 的投影能力太弱

```python
key_h = self.w_kh * residual  # element-wise 乘法，不是矩阵乘法
```

K 的投影是逐元素乘法（Hadamard product），不是线性变换（矩阵乘法）。这意味着：

- K 只能对每个维度做独立的缩放，不能做维度间的混合
- 无法学到"维度 i 和维度 j 的组合模式"
- 信息提取能力极其有限

**对比标准 Attention**：K = W_k · h，W_k 是一个矩阵，可以做任意的线性变换。

### 3.4 缺陷四：RMSNorm 消除了关键的幅度信息

```python
key_h_g = key_h_g * rsqrt(mean(key_h_g²) + ε)  # 归一化到单位 RMS
```

RMSNorm 消除了 h 和 o 的幅度差异。虽然这解决了"h 随深度增长主导 softmax"的问题，但也丢失了一个重要信号：**幅度本身携带了信息重要性的线索**。

在 Transformer 中，幅度大的维度通常对应更重要的特征。完全消除幅度意味着 gate 无法区分"重要的大信号"和"不重要的小信号"。

### 3.5 缺陷五：softmax 约束 α_h + α_o = 1 导致信息不可逆丢失

这是最根本的问题。softmax 保证了 `α_h + α_o = 1`，这意味着：

- `h_new = α_h · h + (1-α_h) · o`
- 这是 h 和 o 的**凸组合**
- `‖h_new‖ ≤ max(‖h‖, ‖o‖)`

**信息论视角**：凸组合是一个**有损压缩**操作。h 和 o 的信息被混合后，无法再分离。经过 56 次有损压缩后，原始的 embedding 信息被不可逆地丢失了。

**对比标准残差**：`h + o` 是**无损叠加**。h 的信息完整保留，o 的信息被加入。虽然幅度会增长，但信息不会丢失。

## 四、为什么 sigmoid 独立门控也没有帮助？

用户反馈 sigmoid 版本（`h_new = (1-f)·h + a·o`）的训练 loss 和 token_acc 也与标准 Qwen3 一样。

这说明问题不仅仅是 softmax vs sigmoid，而是**整个 hidden-size attention 机制的设计范式**有问题。

### 4.1 sigmoid 版本的问题

即使 `(1-f) + a` 可以 > 1，sigmoid 版本仍然有缺陷三（K 投影太弱）和缺陷二（Q 是静态的）。gate 的决策质量太低，无法做出有意义的选择。

### 4.2 更深层的问题：gate 的决策粒度

当前设计在**每个维度组**上做独立的 retain/accept 决策。但 Transformer 的信息是以**整个 hidden vector** 为单位编码的。

一个 token 的语义不是"维度 1-64 编码了名词，维度 65-128 编码了动词"这样的。语义是分布在所有维度上的。在维度组级别做 retain/accept 决策，就像在像素级别决定是否保留一张图片——你需要看到整张图片才能做出有意义的决策。

## 五、发散思考：什么样的残差门控才能工作？

### 5.1 约束条件

从 Transformer 的信息流范式出发，一个有效的残差门控必须满足：

1. **梯度直通**：`∂h_new/∂h` 中必须有一个接近 1 的分量，保证梯度可以无衰减地从顶层传回底层
2. **信息保持**：h 中的信息不能被不可逆地丢失
3. **内容依赖**：gate 的决策应该依赖于当前 token 的内容，而不是固定的
4. **全局视野**：gate 需要看到整个 hidden vector 才能做出有意义的决策

### 5.2 方向一：门控子层输出而非残差流

**核心思想**：不要动残差流 h，只门控子层输出 o。

```
h_new = h + g(h, o) ⊙ o
```

其中 `g(h, o) ∈ [0, 1]^d` 是一个门控函数。

**优势**：
- 梯度直通：`∂h_new/∂h = I + ...`，恒有直通项 I
- 信息保持：h 完整保留
- 可以退化为标准残差：g = 1 时 h_new = h + o

**g 的设计**：
```
g = sigmoid(W_down · ReLU(W_up · [h; o]) + b)
```
- `[h; o]` 拼接后有 2d 维，提供全局视野
- `W_up ∈ R^{r × 2d}`, `W_down ∈ R^{d × r}`，低秩瓶颈控制参数量
- 这本质上是一个小型 MLP，有足够的表达能力

**参数量**：r=16 时，2d·r + r·d = 3dr = 3×1024×16 = 49K per gate

### 5.3 方向二：标量门控（最简单）

**核心思想**：用一个标量 gate 控制整个子层输出的贡献。

```
h_new = h + g · o
```

其中 `g ∈ R` 是一个标量。

**g 的设计**：
```
# 基于 h 和 o 的全局统计量
h_norm = ‖h‖
o_norm = ‖o‖
cos_sim = (h · o) / (h_norm · o_norm)
g = sigmoid(w1·log(h_norm/o_norm) + w2·cos_sim + b)
```

**优势**：
- 极简：每个 gate 只有 3 个参数
- 全局视野：基于整个 vector 的统计量
- 梯度直通：h 完整保留

**劣势**：
- 粒度太粗：整个 o 要么全接受要么全拒绝
- 无法做维度级别的选择

### 5.4 方向三：Cross-Attention 风格的门控

**核心思想**：用 h 作为 Query 去"查询" o，决定 o 的哪些信息应该被接受。

```
Q = W_q · h          # h 提出"我需要什么"
K = W_k · o          # o 回答"我提供什么"
V = o                # o 的原始信息

# 在 hidden-size 维度上做 attention
# 但不是 group-wise 的，而是用低秩投影做全局交互
score = Q^T · K / √d  # 标量 score
g = sigmoid(score + b)
h_new = h + g · o
```

**或者更精细的版本**：
```
Q = W_q · h          # (d,) → (r,)
K = W_k · o          # (d,) → (r,)
score = (Q * K).sum() / √r  # 标量
g = sigmoid(score + b)

# 维度级别的门控
V = W_v · o          # (d,) → (d,)
h_new = h + g · V
```

### 5.5 方向四：AttnRes 的正确移植——在深度维度上做 attention

**核心思想**：AttnRes 的成功是因为它在**深度维度**上做 softmax，而不是在 hidden-size 维度上。

当前的设计错误地将 AttnRes 的机制从深度维度移植到了 hidden-size 维度。正确的移植应该是：

```
# 保存每一层的子层输出
outputs = [o_1, o_2, ..., o_L]  # 每层的 attention/FFN 输出

# 在第 l 层，用 learned query 查询所有历史层的输出
q_l = w_l                       # per-layer query
k_i = RMSNorm(o_i)              # 历史层输出的方向
α_{i→l} = softmax(q_l · k_i)   # 权重

# 加权聚合
h_l = Σ α_{i→l} · o_i
```

但这需要保存所有历史层的输出，内存开销大。

### 5.6 方向五：层级自适应残差缩放（最实用）

**核心思想**：不做维度级别的门控，而是让每一层学习一个**标量缩放因子**来控制子层输出的贡献。

```
h_new = h + λ_l · o
```

其中 `λ_l` 是第 l 层的可学习标量，初始化为 1.0。

**这就是 DeepNet 的 α 缩放**，已经被证明对深层 Transformer 有效。

**优势**：
- 极简：每层 1 个参数
- 梯度直通：完美保留
- 已有理论和实验支持

**劣势**：
- 不是内容依赖的
- 粒度最粗

## 六、推荐方案：方向一的变体——轻量级 MLP 门控子层输出

综合考虑所有约束条件，推荐以下设计：

```python
class ResidualGateV6(nn.Module):
    def __init__(self, hidden_size, bottleneck_dim=16):
        super().__init__()
        # 低秩 MLP：全局视野 + 内容依赖 + 维度级别输出
        self.gate_up = nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.gate_down = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(hidden_size))
        
        # 初始化：gate ≈ 1（近似标准残差）
        nn.init.zeros_(self.gate_up.weight)
        nn.init.zeros_(self.gate_down.weight)
        nn.init.ones_(self.gate_bias)  # sigmoid(1) ≈ 0.73
        # 或者用更大的 bias 让初始 gate 更接近 1
    
    def forward(self, residual, new_output):
        # 用子层输出 o 的内容来决定门控
        # "o 自己决定自己的哪些维度应该被接受"
        gate_input = new_output  # 或者用 residual，或者两者
        gate = torch.sigmoid(
            self.gate_down(F.silu(self.gate_up(gate_input))) + self.gate_bias
        )
        return residual + gate * new_output
```

### 为什么这个设计能工作？

1. **梯度直通**：`h_new = h + gate·o`，`∂h_new/∂h = I + ...`，恒有直通项
2. **全局视野**：`gate_up` 是 `Linear(d, r)`，看到整个 hidden vector
3. **内容依赖**：gate 依赖于 o 的内容（或 h 的内容）
4. **维度级别**：`gate_down` 输出 d 维，每个维度独立门控
5. **可退化**：gate_up/down 权重为 0 时，gate = sigmoid(bias) ≈ 常数，退化为标量缩放
6. **参数量小**：r=16 时，d·r + r·d + d = 2dr + d ≈ 33K per gate

### 初始化策略

- `gate_up.weight = 0, gate_down.weight = 0` → 初始时 gate = sigmoid(bias)
- `gate_bias = 3.0` → sigmoid(3) ≈ 0.953 → 初始 `h_new ≈ h + 0.95·o ≈ h + o`
- 这保证了初始行为接近标准残差，模型可以从标准 Transformer 的行为开始，逐步学习门控

### 与当前 v5.2 的对比

| 属性 | v5.2 Hidden-Size Attention | V6 MLP Gate |
|------|--------------------------|-------------|
| 梯度直通 | ❌ α_h < 1 衰减 | ✅ 恒有 I |
| 全局视野 | ❌ 64 维 group | ✅ 整个 d 维 |
| 内容依赖 | ❌ Q 是固定参数 | ✅ 依赖 o 的内容 |
| K 投影能力 | ❌ element-wise | ✅ 矩阵乘法 |
| 信息保持 | ❌ 凸组合有损 | ✅ h 完整保留 |
| 参数量/gate | ~4K | ~33K |
| 可退化为标准残差 | ❌ | ✅ |

## 七、实验建议

1. **V6 vs Qwen3 baseline**：在相同的 3.47B token 上训练，对比 lambada 和 piqa
2. **消融实验**：
   - gate_input = o vs h vs [h;o]
   - bottleneck_dim = 4, 8, 16, 32
   - gate_bias 初始值 = 1.0, 3.0, 5.0
3. **监控 gate 统计**：训练过程中 gate 的均值、方差、极端值比例
