# ResidualGate V4：维度级自适应学习

## 一、思维转变：从规则设计到自适应学习

### 之前的思路（规则性设计）
```
"幅度大 → 保留"
"方向一致 → 接受"
"幅度比 + 方向一致性 → 线性组合 → sigmoid → gate"
```
这是在用人类的先验知识设计规则。但 self-attention 的成功恰恰在于 **不预设规则**。

### Self-attention 的启示
```
x → W_Q(x) = query    "我在找什么"
x → W_K(x) = key      "我能提供什么"
attention = softmax(query · key^T)   "匹配度"
```

Self-attention 不预设"什么 token 应该被关注"。它通过 W_Q 和 W_K 的训练，自动学习出最优的匹配规则。

### 核心问题
**如何在 hidden-size 维度上实现类似 self-attention 的自适应学习？**

## 二、维度级 Self-Attention Gate

### 2.1 类比

| Self-Attention（token 维度） | Residual Gate（hidden-size 维度） |
|---|---|
| 多个 token 竞争注意力 | h 和 o 在每个维度上竞争 |
| Q, K 从输入 x 投影 | Q, K 从 h 和 o 投影 |
| attention weight 是数据依赖的 | gate α, β 是数据依赖的 |
| W_Q, W_K 是可学习的 | gate 投影矩阵是可学习的 |

### 2.2 设计

在每个残差连接处，h 和 o 是两个"候选"。我们需要为每个维度 j 计算"保留 h_j 多少"和"接受 o_j 多少"。

**关键洞察**：不要预设 gate 应该看什么信号（幅度、方向等），而是让 gate 从 h 和 o 的原始内容中自适应地学习。

但直接对 d 维做投影（如 V1 的 gate_A: 4d→16）参数太多。我们需要一个 **参数高效** 的自适应机制。

### 2.3 方案：Per-dim 的 Query-Key Attention

```python
class ResidualGate(nn.Module):
    def __init__(self, hidden_size, init_bias=5.0):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 可学习的投影向量（不是矩阵！）
        # 将 h 和 o 投影到 "gate 空间" 的标量
        # 每个维度 j: q_j = w_q_j · h_j,  k_j = w_k_j · o_j
        self.w_q = nn.Parameter(torch.zeros(hidden_size))  # query 投影
        self.w_k = nn.Parameter(torch.zeros(hidden_size))  # key 投影
        
        # per-dim bias
        self.b_alpha = nn.Parameter(torch.full((hidden_size,), init_bias))
        self.b_beta = nn.Parameter(torch.full((hidden_size,), init_bias))
    
    def forward(self, residual, new_output):
        # 每个维度独立的 "query-key" 匹配
        # q_j = w_q_j · h_j  (h 在第 j 维的 "query")
        # k_j = w_k_j · o_j  (o 在第 j 维的 "key")
        q = self.w_q * residual      # (B, T, d) — 带梯度！
        k = self.w_k * new_output    # (B, T, d) — 带梯度！
        
        # gate logit = q·k 的某种组合
        # ...但这只是逐元素乘法，表达能力有限
```

**问题**：per-dim 的标量投影太简单了。`w_q_j · h_j` 只是一个标量乘法，没有跨维度的信息交互。

### 2.4 更好的方案：低秩投影到 gate 空间

Self-attention 的 W_Q, W_K 是 d×d_k 的矩阵，将 d 维输入投影到 d_k 维的 query/key 空间。

类比地，我们可以将 h 和 o 投影到一个低维的 "gate 空间"，然后在 gate 空间中计算匹配度：

```python
class ResidualGate(nn.Module):
    def __init__(self, hidden_size, rank=16, init_bias=5.0):
        super().__init__()
        
        # 将 h 和 o 投影到低维 gate 空间
        self.proj_h = nn.Linear(hidden_size, rank, bias=False)  # h → gate space
        self.proj_o = nn.Linear(hidden_size, rank, bias=False)  # o → gate space
        
        # 从 gate 空间投影回 hidden_size，生成 α 和 β
        self.out_alpha = nn.Linear(rank, hidden_size, bias=True)
        self.out_beta = nn.Linear(rank, hidden_size, bias=True)
```

**等等——这不就是 V1 吗？** V1 的 gate_A 把 [h, o, mag, dir] 投影到 rank=16，然后 gate_B 投影回 d。

**区别在于**：V1 把 h, o, mag, dir 拼接后一起投影，而这里是 h 和 o 分别投影后交互。

## 三、真正的自适应学习方案

让我重新思考。Self-attention 的核心不是投影矩阵的大小，而是 **数据依赖的权重计算**。

在 token 维度上，self-attention 的权重是 `softmax(Q·K^T)`——query 和 key 的点积。

在 hidden-size 维度上，我们需要的是：**h 和 o 的某种数据依赖的交互，产生 per-dim 的权重。**

### 3.1 最简洁的自适应方案

```python
class ResidualGate(nn.Module):
    """
    维度级自适应 gate。
    
    核心思想：不预设 gate 应该看什么信号，
    而是让 h 和 o 通过可学习的投影在低维空间中交互，
    自适应地学习每个维度的保留/接受权重。
    
    类比 self-attention：
      - h 和 o 是两个 "token"
      - 通过投影到 gate 空间后交互
      - 产生数据依赖的权重
    """
    def __init__(self, hidden_size, rank=16, init_bias=5.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        
        # 共享的下投影：将 h 和 o 投影到低维 gate 空间
        # 这是自适应学习的核心：网络自己学习"看什么"
        self.down = nn.Linear(hidden_size, rank, bias=False)
        
        # 独立的上投影：从 gate 空间生成 α 和 β
        self.up_alpha = nn.Linear(rank, hidden_size, bias=True)
        self.up_beta = nn.Linear(rank, hidden_size, bias=True)
        
        # 初始化：up 权重小 + bias=init_bias → 初始 gate ≈ sigmoid(init_bias)
        nn.init.normal_(self.up_alpha.weight, std=0.01)
        nn.init.constant_(self.up_alpha.bias, init_bias)
        nn.init.normal_(self.up_beta.weight, std=0.01)
        nn.init.constant_(self.up_beta.bias, init_bias)
        nn.init.normal_(self.down.weight, std=0.02)
    
    def forward(self, residual, new_output):
        # h 和 o 在 gate 空间中的表示
        h_gate = self.down(residual)      # (B, T, rank) — 带梯度
        o_gate = self.down(new_output)    # (B, T, rank) — 带梯度
        
        # 在 gate 空间中交互：逐元素乘积捕捉 h-o 的关系
        # 这是自适应的：网络学习在 gate 空间中 h 和 o 的哪些特征重要
        interaction = h_gate * o_gate     # (B, T, rank) — 数据依赖的交互
        
        # 投影回 hidden_size，生成 per-dim gate
        alpha = torch.sigmoid(self.up_alpha(interaction))  # (B, T, d)
        beta = torch.sigmoid(self.up_beta(interaction))    # (B, T, d)
        
        return alpha * residual + beta * new_output
```

### 3.2 分析

**自适应性**：
- `self.down` 学习"从 h 和 o 中提取什么特征"——不预设是幅度还是方向
- `h_gate * o_gate` 是数据依赖的交互——类似 attention 中 Q·K 的点积
- `self.up_alpha/beta` 学习"如何从交互特征生成 per-dim gate"

**梯度路径**：
- h → down → h_gate → interaction → up_alpha → α → α⊙h → h_new → loss
- o → down → o_gate → interaction → up_beta → β → β⊙o → h_new → loss
- **h 和 o 都有完整的梯度路径**，Attention/FFN 可以感知 gate 的存在

**参数量**：
- down: d × rank (无 bias)
- up_alpha: rank × d + d (有 bias)
- up_beta: rank × d + d (有 bias)
- 总计: d×rank + 2×(rank×d + d) = 3×d×rank + 2d
- d=1024, rank=16 → 3×1024×16 + 2×1024 = 49152 + 2048 ≈ **51K per gate**

这比 V1 的 ~100K 少一半，但比纯物理信号方案多很多。

### 3.3 问题：h_gate * o_gate 的交互是否足够？

逐元素乘积 `h_gate * o_gate` 只捕捉了 h 和 o 在 gate 空间中的 **逐维度** 关系。

更丰富的交互方式：
- `h_gate * o_gate`：逐元素乘积（当前方案）
- `h_gate + o_gate`：逐元素加法（线性组合）
- `h_gate - o_gate`：差异信号
- `torch.cat([h_gate, o_gate, h_gate * o_gate])`：拼接多种交互

但更多的交互 = 更多的参数。让我们保持简洁。

### 3.4 改进：加入加法和乘法两种交互

```python
def forward(self, residual, new_output):
    h_gate = self.down(residual)      # (B, T, rank)
    o_gate = self.down(new_output)    # (B, T, rank)
    
    # 两种交互模式：
    # 加法：捕捉 h 和 o 的共同特征（"两者都有什么"）
    # 乘法：捕捉 h 和 o 的关系（"两者如何交互"）
    gate_signal = h_gate * o_gate + h_gate + o_gate  # (B, T, rank)
    
    alpha = torch.sigmoid(self.up_alpha(gate_signal))
    beta = torch.sigmoid(self.up_beta(gate_signal))
    
    return alpha * residual + beta * new_output
```

**但这又回到了 V1 的思路**——用一个网络从 h 和 o 中提取特征来计算 gate。

## 四、真正的突破：重新定义问题

让我退一步，重新定义问题。

### 4.1 Self-attention 为什么有效？

Self-attention 有效不是因为 Q·K 的点积，而是因为：
1. **数据依赖**：权重完全由输入决定
2. **端到端学习**：W_Q, W_K 通过梯度优化
3. **表达能力**：d_k 维的投影空间足够丰富

### 4.2 ResidualGate 的最小自适应设计

要实现维度级的自适应学习，最小需要什么？

**答案**：一个从 h 和 o 到 per-dim gate 的 **可微映射**，且这个映射是 **数据依赖** 的。

最简洁的数据依赖映射：

```python
# h 和 o 的逐元素交互，通过可学习的 per-dim 权重组合
gate_signal = w_h * h + w_o * o + w_ho * (h * o)
#             ^^^^     ^^^^       ^^^^^^^^^^^^^
#             h的贡献   o的贡献    h-o的交互（数据依赖）
```

其中 w_h, w_o, w_ho 都是 per-dim 的可学习向量。

这个设计：
- **数据依赖**：gate_signal 取决于 h 和 o 的实际值
- **端到端学习**：w_h, w_o, w_ho 通过梯度优化
- **h 和 o 有梯度路径**：gate 对 h 和 o 有梯度
- **per-dim**：每个维度独立的 gate
- **参数极少**：3d per gate（d=1024 → 3K）

```python
class ResidualGate(nn.Module):
    def __init__(self, hidden_size, init_bias=5.0):
        super().__init__()
        
        # Per-dim 权重：控制 h, o, h*o 对 gate 的贡献
        self.w_h = nn.Parameter(torch.zeros(hidden_size))      # h 的直接贡献
        self.w_o = nn.Parameter(torch.zeros(hidden_size))      # o 的直接贡献
        self.w_ho = nn.Parameter(torch.zeros(hidden_size))     # h-o 交互项
        
        # Per-dim bias
        self.b_alpha = nn.Parameter(torch.full((hidden_size,), init_bias))
        self.b_beta = nn.Parameter(torch.full((hidden_size,), init_bias))
    
    def forward(self, residual, new_output):
        # 数据依赖的 gate 信号
        # h*o 项是关键：它捕捉了 h 和 o 在每个维度上的关系
        # - h_j > 0, o_j > 0 (同向正) → h*o > 0
        # - h_j > 0, o_j < 0 (反向)   → h*o < 0
        # - h_j 大, o_j 大 (都强)      → |h*o| 大
        # - h_j 小 或 o_j 小           → |h*o| 小
        gate_signal = self.w_h * residual + self.w_o * new_output + self.w_ho * (residual * new_output)
        
        alpha = torch.sigmoid(gate_signal + self.b_alpha)
        beta = torch.sigmoid(-gate_signal + self.b_beta)  # 注意负号：α 和 β 互补
        # 或者用独立的权重：
        # alpha = sigmoid(w_h_a * h + w_o_a * o + w_ho_a * h*o + b_alpha)
        # beta  = sigmoid(w_h_b * h + w_o_b * o + w_ho_b * h*o + b_beta)
        
        return alpha * residual + beta * new_output
```

### 4.3 为什么 h*o 项是自适应学习的关键？

`h_j * o_j` 是一个 **二阶交互项**，它天然地编码了：
- **方向关系**：同号 → 正，异号 → 负（= dir_agree 的未归一化版本）
- **幅度关系**：两者都大 → 绝对值大，一方小 → 绝对值小
- **数据依赖**：完全由 h 和 o 的实际值决定

而且 **h*o 对 h 和 o 都有梯度**：
```
∂(h*o)/∂h = o
∂(h*o)/∂o = h
```

这意味着 Attention/FFN 可以通过 h*o 项感知 gate 的存在，并调整自己的输出。

### 4.4 与之前方案的对比

| | V1 低秩网络 | V2/V3 物理信号 | V4 自适应 |
|---|---|---|---|
| gate 信号 | [h,o,mag,dir] → 网络 | detached mag, dir | h, o, h*o（带梯度） |
| 数据依赖 | ✅ 通过网络 | ❌ 只看 detached 物理量 | ✅ 通过 h*o 交互 |
| h/o 有梯度 | ✅ 但经过低秩瓶颈 | ❌ 全部 detached | ✅ 直接梯度 |
| 参数量 | ~100K | ~2K | ~5K |
| 可解释性 | 低 | 高 | 中（w_h, w_o, w_ho 可解释） |
| 预设规则 | 无（但结构复杂） | 有（幅度→保留等） | 无（自适应学习） |

## 五、最终方案：独立 α/β 的自适应 gate

为了给 α 和 β 更多的自由度（不强制互补），使用独立的权重：

```python
class ResidualGate(nn.Module):
    def __init__(self, hidden_size, init_bias=5.0):
        super().__init__()
        
        # α (retain gate) 的自适应权重
        self.w_alpha_h  = nn.Parameter(torch.zeros(hidden_size))
        self.w_alpha_o  = nn.Parameter(torch.zeros(hidden_size))
        self.w_alpha_ho = nn.Parameter(torch.zeros(hidden_size))
        self.b_alpha    = nn.Parameter(torch.full((hidden_size,), init_bias))
        
        # β (accept gate) 的自适应权重
        self.w_beta_h   = nn.Parameter(torch.zeros(hidden_size))
        self.w_beta_o   = nn.Parameter(torch.zeros(hidden_size))
        self.w_beta_ho  = nn.Parameter(torch.zeros(hidden_size))
        self.b_beta     = nn.Parameter(torch.full((hidden_size,), init_bias))
    
    def forward(self, residual, new_output):
        # α: 保留门 — 自适应学习何时保留旧信息
        alpha = torch.sigmoid(
            self.w_alpha_h * residual 
            + self.w_alpha_o * new_output 
            + self.w_alpha_ho * (residual * new_output) 
            + self.b_alpha
        )
        
        # β: 接受门 — 自适应学习何时接受新信息
        beta = torch.sigmoid(
            self.w_beta_h * residual 
            + self.w_beta_o * new_output 
            + self.w_beta_ho * (residual * new_output) 
            + self.b_beta
        )
        
        return alpha * residual + beta * new_output
```

### 参数量
- 6d + 2d = 8d per gate
- d=1024 → **8K per gate**
- 28 层 × 2 gate/层 = 56 gates → 总计 **~450K gate 参数**
- 对比 0.6B 模型的总参数量，gate 开销 < 0.1%

### 初始化
- w = 0, b = init_bias = 5.0
- 初始时：gate_signal = 0, α = β = sigmoid(5.0) = 0.993
- h_new ≈ 0.993h + 0.993o ≈ h + o ✅ 近恒等

### 梯度分析
```
∂α/∂h = σ'(·) · (w_alpha_h + w_alpha_ho · o)
∂α/∂o = σ'(·) · (w_alpha_o + w_alpha_ho · h)
```
- h 和 o 都有直接的梯度路径
- 梯度强度由 w 的大小控制（初始为 0，逐渐增长）
- 不经过低秩瓶颈，梯度信号强

## 六、为什么这是最符合自适应学习原则的设计

1. **不预设规则**：不假设"幅度大就保留"或"方向一致就接受"
2. **数据依赖**：gate 完全由 h 和 o 的实际值决定
3. **端到端学习**：h 和 o 有完整的梯度路径
4. **h*o 交互项**：自然地捕捉了方向和幅度的联合信息，但不强制分离
5. **per-dim 独立**：每个维度有自己的学习参数
6. **参数高效**：8d per gate，远小于低秩网络
7. **计算高效**：纯逐元素运算，无矩阵乘法
8. **初始化安全**：w=0 → 近恒等启动
