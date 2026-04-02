# V8 分析：深度方向的 SSM 视角——从 Mamba 的经验中学习

## 一、类比的建立

### 1.1 标准 SSM/Mamba 的维度

Mamba 在**序列维度（时间）**上运行：

```
h_t = A · h_{t-1} + B · x_t     (状态更新)
y_t = C · h_t                    (输出)
```

- 时间轴：token 序列 t = 1, 2, ..., T
- 状态 h_t：在时间维度上递推
- A：遗忘矩阵（控制历史信息的衰减）
- B：输入矩阵（控制新信息的注入）

### 1.2 ResidualGate 的维度

ResidualGate 在**深度维度（层）**上运行：

```
h_{l+1} = α_l · h_l + β_l · o_l   (残差更新)
```

- 深度轴：层 l = 1, 2, ..., L
- 状态 h_l：在深度维度上递推
- α_l：遗忘因子（控制历史残差的保留）
- β_l：输入因子（控制子层输出的接受）

### 1.3 对应关系

| SSM/Mamba（序列维度） | ResidualGate（深度维度） |
|---------------------|----------------------|
| h_t（隐状态） | h_l（残差流） |
| x_t（输入 token） | o_l（子层输出） |
| A（遗忘矩阵） | α_l（保留门） |
| B（输入矩阵） | β_l（接受门） |
| 时间步 t | 层 l |
| 序列长度 T | 网络深度 L |

**这个类比是精确的**：ResidualGate 就是在深度维度上运行的 SSM。

## 二、Mamba/SSM 遇到过的问题

### 2.1 问题一：线性递推的信息衰减

经典 SSM 的核心问题：`h_t = A · h_{t-1} + B · x_t`

如果 A 的特征值 |λ| < 1，信息会指数衰减：

```
h_T 中 x_1 的贡献 ∝ A^{T-1} · B · x_1
```

当 T 很大时，A^{T-1} → 0，早期信息丢失。

**对应到 ResidualGate**：如果 α_l < 1（softmax 约束下必然如此），经过 L 层后：

```
h_L 中 h_0 的贡献 ∝ ∏_{l=1}^{L} α_l · h_0
```

这正是我们观察到的问题：lambada perplexity 爆炸，因为 embedding 信息 h_0 在深层被衰减到几乎为零。

### 2.2 Mamba 的解决方案：选择性状态空间（Selective SSM）

Mamba 的核心创新是让 A, B, C **依赖于输入内容**：

```
A_t = f_A(x_t)    (输入依赖的遗忘)
B_t = f_B(x_t)    (输入依赖的注入)
C_t = f_C(x_t)    (输入依赖的输出)
```

**关键**：Mamba 的 A_t 是通过 `A_t = -exp(Linear(x_t))` 参数化的，保证 A_t < 0（连续时间下对应衰减），但衰减速率是输入依赖的。

### 2.3 但 Mamba 也有长距离依赖的问题

Mamba 在极长序列上的表现不如 Transformer attention，正是因为 SSM 的递推本质导致信息衰减。Mamba-2 和后续工作通过以下方式缓解：

1. **结构化状态空间**：使用对角化的 A 矩阵，让不同维度有不同的衰减速率
2. **选择性机制**：让衰减速率依赖于输入内容
3. **混合架构**：在 SSM 层之间穿插 attention 层（Jamba, Zamba 等）

### 2.4 关键洞察：SSM 的信息衰减是**特性**而非 bug

在序列维度上，信息衰减是合理的——远处的 token 通常不如近处的重要。SSM 通过选择性衰减来实现"记住重要的，遗忘不重要的"。

**但在深度维度上，情况完全不同**：

- 序列维度：远处的 token 可能不重要（局部性假设）
- 深度维度：底层的 embedding 信息**始终重要**（因为最终的 LM head 需要它来预测 token）

这是 ResidualGate 与 Mamba 的**根本区别**：在深度维度上，我们不能像序列维度那样让信息自然衰减。

## 三、从 SSM 视角重新理解失败原因

### 3.1 所有失败方案的共同问题

所有方案（V1 MLP gate, V5.x sigmoid, V5.2 softmax attention）都是在深度维度上运行的 SSM，且都有 α < 1 的衰减。

即使 α 是内容依赖的（V1 的 MLP gate），只要 α 的**期望值** < 1，经过 56 次递推后信息就会衰减到接近零。

**数学证明**：设 α_l 是独立同分布的随机变量，E[α_l] = μ < 1。则：

```
E[∏_{l=1}^{56} α_l] = μ^{56}
```

即使 μ = 0.95（sigmoid(3) ≈ 0.953），μ^{56} ≈ 0.056。
即使 μ = 0.99，μ^{56} ≈ 0.57。

**要让信息不衰减，需要 μ ≥ 1**，但 sigmoid 的输出范围是 (0, 1)，不可能达到 1。

### 3.2 为什么训练 loss 正常？

训练 loss 是在 teacher-forcing 下计算的。每个 token 的预测只需要**局部信息**（最近几层的输出），不需要底层的 embedding 信息。

这就像 SSM 在短序列上表现正常——因为信息还没来得及衰减。

### 3.3 为什么 lm_eval 发散？

lm_eval 的 lambada 任务需要理解整个句子的语义来预测最后一个词。这需要**底层 embedding 信息**通过所有 28 层传到顶层。

在 ResidualGate 中，这个信息经过 56 次 α < 1 的衰减后几乎为零。模型无法利用底层信息，只能依赖最近几层的局部模式，导致 perplexity 爆炸。

## 四、SSM 的解决方案能否移植到深度维度？

### 4.1 方案 A：让 α 可以 ≥ 1

在 SSM 中，A 的特征值可以接近 1（慢衰减）。如果我们让 α 可以 ≥ 1：

```
α = softplus(logit) + 0.5    # 范围 (0.5, +∞)
```

**问题**：α > 1 意味着残差流的幅度会指数增长，导致数值不稳定。

### 4.2 方案 B：对角化 + 不同衰减速率

Mamba 使用对角化的 A 矩阵，让不同维度有不同的衰减速率。

映射到 ResidualGate：让不同维度组有不同的 α：

```
α_group_1 ≈ 0.99  (慢衰减，保留长距离信息)
α_group_2 ≈ 0.50  (快衰减，只保留局部信息)
```

**这就是 v5.2 的分组设计**，但它没有工作。原因：即使某些组的 α 接近 1，sigmoid 的输出永远 < 1，56 次递推后仍然衰减。

### 4.3 方案 C：混合架构（SSM + 直通残差）

Jamba 等混合架构在 SSM 层之间穿插 attention 层。

映射到 ResidualGate：在某些层使用 gate，在其他层使用标准残差。

```
if layer_idx % 4 == 0:
    h_new = α · h + β · o    # 每 4 层做一次门控
else:
    h_new = h + o             # 其他层标准残差
```

**这可能有效**，因为标准残差层保证了信息的直通，而门控层提供了选择性过滤。

### 4.4 方案 D：残差旁路 + 门控（最关键的洞察）

**核心思想**：将残差流分成两部分——一部分始终直通（保证信息不丢失），另一部分经过门控（提供选择性过滤）。

```
h_new = h + gate · (α · h + β · o - h)
      = h + gate · ((α-1) · h + β · o)
      = (1 - gate + gate·α) · h + gate·β · o
```

当 gate = 0 时：h_new = h（完全直通，标准残差的退化）
当 gate = 1 时：h_new = α · h + β · o（完全门控）

**gate 控制"门控的强度"**，而不是直接控制 h 的系数。

这保证了即使 gate 学到了错误的值，h 的信息也不会完全丢失（因为 gate 可以退化为 0）。

### 4.5 方案 E：Mamba-2 的启示——将 SSM 与 Attention 统一

Mamba-2 证明了 SSM 可以被视为一种特殊的线性 attention。

**关键公式**：

```
SSM: y_t = Σ_{s=1}^{t} (∏_{τ=s+1}^{t} A_τ) · B_s · x_s
Attention: y_t = Σ_{s=1}^{t} softmax(q_t · k_s) · v_s
```

SSM 的权重 `∏ A_τ · B_s` 是**乘法累积**的，而 attention 的权重 `softmax(q·k)` 是**加法归一化**的。

**映射到深度维度**：

如果我们在深度维度上做 attention（而不是 SSM），就不会有乘法累积导致的信息衰减：

```
h_L = Σ_{l=0}^{L-1} attention_weight(l→L) · o_l
```

这正是 **AttnRes 的原始设计**——在深度维度上做 attention，而不是 SSM。

## 五、核心结论

### 5.1 为什么所有方案都失败了

**所有方案都是深度方向的 SSM**，而 SSM 的递推本质导致信息衰减。无论 gate 的设计多么精巧（MLP、sigmoid、softmax、attention），只要是 `h_{l+1} = α · h_l + β · o_l` 的递推形式，且 α < 1，信息就会指数衰减。

### 5.2 正确的方向

要解决"残差信息冗余和有害"的问题，同时避免信息衰减，有两个根本不同的方向：

**方向 A：保留递推形式，但保证 α 的期望值 = 1**

这需要一种新的参数化方式，让 α 的期望值恒为 1，但方差可以学习。例如：

```
α = 1 + tanh(logit) · scale    # 范围 (1-scale, 1+scale)
```

当 scale 很小时，α ≈ 1 ± ε，信息几乎不衰减，但 gate 仍然可以做微小的调节。

**方向 B：放弃递推形式，改用 attention 形式**

这就是 AttnRes 的原始设计：在深度维度上做 attention，让每一层可以直接访问所有历史层的输出，而不是通过递推间接访问。

```
h_L = Σ_{l=0}^{L-1} softmax(q_L · k_l / √d) · v_l
```

这避免了递推导致的信息衰减，但需要保存所有历史层的输出（内存开销 O(L·d)）。

**方向 C：混合方案——标准残差 + 选择性门控子层输出**

```
h_new = h + gate(h, o) · o
```

其中 gate 的范围是 [0, 2] 而不是 [0, 1]：

```
gate = 1 + tanh(MLP(h, o))    # 范围 (0, 2)
```

- gate = 1：标准残差 h + o
- gate = 0：跳过子层 h + 0 = h
- gate = 2：强化子层 h + 2o

**这保证了 h 始终直通（系数恒为 1），同时允许 o 的贡献在 [0, 2] 范围内调节。**

但这不能解决"h 中的信息冗余"问题——h 始终以系数 1 直通。

**方向 D：在 h 上做投影而非缩放**

不是缩放 h 的幅度，而是**旋转/投影** h 的方向：

```
h_new = W_rotate(h) · h + o
```

其中 W_rotate 是一个接近单位矩阵的可学习旋转矩阵。这保持了 h 的幅度（不衰减），但改变了 h 的方向（过滤冗余信息）。

**参数化**：`W_rotate = I + ΔW`，其中 ΔW 是低秩的。

```
W_rotate = I + U · V^T    # U ∈ R^{d×r}, V ∈ R^{d×r}
```

这保证了：
- 初始时 W_rotate = I（标准残差）
- 训练后 W_rotate 可以旋转 h 的方向
- h 的幅度不衰减（因为 W_rotate 接近正交矩阵）
- 梯度直通：∂h_new/∂h = W_rotate ≈ I

**但这不是内容依赖的**——所有 token 使用相同的旋转。

**方向 E：内容依赖的投影（最有前景）**

```
h_new = (I + gate(h, o) · ΔW) · h + o
```

其中 gate(h, o) 是一个标量门控，ΔW 是低秩的方向调整矩阵。

- gate = 0：h_new = h + o（标准残差）
- gate > 0：h 在被加入 o 之前先做方向调整

这同时实现了：
1. 梯度直通（系数接近 I）
2. 信息不衰减（幅度不变，只旋转方向）
3. 内容依赖（gate 依赖于 h 和 o）
4. 过滤冗余（通过旋转将冗余维度的信息转移到有用维度）

## 六、最终推荐：方向 C 的简化版

考虑到实验的可行性和之前方案的教训，推荐最简单的有效方案：

```python
class ResidualGateV8(nn.Module):
    """
    深度方向 SSM 视角的残差门控。
    
    核心洞察：所有 α < 1 的递推都会导致信息衰减。
    解决方案：保持 h 的系数恒为 1，只门控 o 的贡献强度。
    
    h_new = h + gate(h, o) · o
    
    gate 的范围是 (-1, 2)，通过 1 + tanh 实现：
    - gate ≈ 1：标准残差（默认行为）
    - gate ≈ 0：跳过子层（当 o 有害时）
    - gate ≈ 2：强化子层（当 o 特别有价值时）
    - gate < 0：反转子层（当 o 方向错误时）
    
    关键：h 的系数恒为 1，信息永远不会衰减。
    """
    
    def __init__(self, hidden_size, bottleneck_dim=8):
        super().__init__()
        # 用 h 和 o 的内容来决定 o 的贡献强度
        self.gate_up = nn.Linear(hidden_size * 2, bottleneck_dim, bias=False)
        self.gate_down = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        
        # 初始化为 0 → gate = 1 + tanh(0) = 1 → 标准残差
        nn.init.zeros_(self.gate_up.weight)
        nn.init.zeros_(self.gate_down.weight)
    
    def forward(self, residual, new_output):
        # 拼接 h 和 o 作为 gate 输入（全局视野）
        gate_input = torch.cat([residual, new_output], dim=-1)
        gate_logit = self.gate_down(F.silu(self.gate_up(gate_input)))
        gate = 1.0 + torch.tanh(gate_logit)  # 范围 (0, 2)，中心在 1
        
        return residual + gate * new_output
```

**但你说这不能解决"h 中的信息冗余"问题。**

你是对的。这个方案只能控制 o 的贡献，不能过滤 h 中的冗余信息。

## 七、真正解决"h 冗余"的方案：方向 D/E

如果必须同时解决"h 冗余"和"信息不衰减"，需要一种**不改变幅度但改变方向**的操作。

### 7.1 内容依赖的低秩残差投影

```python
class ResidualGateV8_Full(nn.Module):
    """
    内容依赖的残差投影 + 子层门控。
    
    h_new = (I + gate_h · ΔW) · h + gate_o · o
    
    ΔW 是低秩矩阵，gate_h 和 gate_o 是标量门控。
    
    - gate_h = 0 时：h_new = h + gate_o · o（标准残差 + o 门控）
    - gate_h > 0 时：h 在加入 o 之前先做方向调整
    - ΔW 学习"哪些方向的信息是冗余的，应该被旋转到其他方向"
    """
    
    def __init__(self, hidden_size, rank=4, bottleneck_dim=8):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 低秩方向调整矩阵 ΔW = U · V^T
        self.delta_U = nn.Linear(hidden_size, rank, bias=False)  # d → r
        self.delta_V = nn.Linear(rank, hidden_size, bias=False)  # r → d
        
        # 内容依赖的门控
        self.gate_proj = nn.Linear(hidden_size * 2, bottleneck_dim, bias=False)
        self.gate_h_out = nn.Linear(bottleneck_dim, 1, bias=False)  # 标量 gate for h 投影
        self.gate_o_out = nn.Linear(bottleneck_dim, hidden_size, bias=False)  # 维度级 gate for o
        
        # 初始化为 0 → 标准残差
        nn.init.zeros_(self.delta_U.weight)
        nn.init.zeros_(self.delta_V.weight)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_h_out.weight)
        nn.init.zeros_(self.gate_o_out.weight)
    
    def forward(self, residual, new_output):
        # 内容依赖的门控
        gate_input = torch.cat([residual, new_output], dim=-1)
        gate_hidden = F.silu(self.gate_proj(gate_input))
        
        gate_h = torch.tanh(self.gate_h_out(gate_hidden))  # 标量，范围 (-1, 1)
        gate_o = 1.0 + torch.tanh(self.gate_o_out(gate_hidden))  # 维度级，范围 (0, 2)
        
        # 低秩方向调整：h_adjusted = h + gate_h · ΔW · h
        delta_h = self.delta_V(self.delta_U(residual))  # ΔW · h
        h_adjusted = residual + gate_h * delta_h
        
        # 最终输出
        return h_adjusted + gate_o * new_output
```

**这个方案的关键性质**：

1. **h 的幅度不衰减**：`h_adjusted = h + gate_h · ΔW · h`，h 的主分量始终保留
2. **方向可调整**：ΔW 学习将冗余方向的信息旋转到有用方向
3. **内容依赖**：gate_h 和 gate_o 都依赖于 h 和 o 的内容
4. **初始化为标准残差**：所有权重初始化为 0 时，h_new = h + o
5. **o 的贡献可调节**：gate_o 范围 (0, 2)，可以跳过、标准、或强化

**参数量**：rank=4, bottleneck_dim=8, d=1024
- delta_U: d×r = 4K
- delta_V: r×d = 4K
- gate_proj: 2d×b = 16K
- gate_h_out: b×1 = 8
- gate_o_out: b×d = 8K
- 总计：~32K per gate，56 个 gate 共 ~1.8M（占总参数 0.3%）
