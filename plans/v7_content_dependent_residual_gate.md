# V7 设计：内容依赖的残差门控——从第一性原理重新出发

## 一、重新定义问题

### 1.1 核心需求

**不同的 token/上下文需要不同程度地依赖历史残差信息。**

这是一个比"防止梯度消失"更深层的需求。它要求：

- 对于某些 token（如功能词、标点），当前层的子层输出 o 可能已经足够，不需要太多历史残差 h
- 对于某些 token（如需要长距离依赖的代词、指代），历史残差 h 中的信息至关重要
- 这个决策必须是**内容依赖的**（content-dependent），不能是固定的

### 1.2 为什么标准残差不够？

标准残差 `h_new = h + o` 是**无条件的**：无论 token 是什么，无论上下文是什么，h 的贡献系数恒为 1，o 的贡献系数恒为 1。

这意味着：
- 模型无法选择性地"重置"残差流（当历史信息有害时）
- 模型无法选择性地"强化"残差流（当历史信息特别重要时）
- 所有 token 都以相同的方式混合历史和当前信息

### 1.3 为什么 `h_new = h + gate(o) ⊙ o` 不够？

这个设计只门控了 o，h 始终以系数 1 直通。它解决了"o 的贡献可以被调节"，但没有解决"h 的贡献可以被调节"。

当历史残差 h 中包含有害信息时（比如早期层的错误中间表示），这个设计无法减少 h 的影响。

## 二、约束条件的重新审视

### 2.1 梯度直通的必要性

梯度直通（`∂h_new/∂h` 中有接近 1 的分量）是**必要的**，但不是说 h 的系数必须恒为 1。

关键是：**梯度路径必须存在，但可以是内容依赖的**。

如果 `h_new = f(h) · h + g(h, o) · o`，其中 `f(h)` 是一个可学习的函数，那么：
- `∂h_new/∂h = f(h) · I + ∂f/∂h · h + ∂g/∂h · o`
- 只要 `f(h)` 不恒为 0，梯度就能流动

**关键洞察**：梯度直通不要求 f=1，只要求 f≠0。

### 2.2 信息保持的重新理解

"信息保持"不是说 h 的信息必须 100% 保留，而是说**有用的信息应该被保留，有害的信息可以被过滤**。

这正是门控机制的价值所在。

### 2.3 内容依赖的实现方式

门控函数 f(h) 和 g(h, o) 必须依赖于当前 token 的内容。这意味着：
- 不能是固定参数（如 v5.2 的 w_qh, w_qo）
- 必须是 h 和/或 o 的函数

## 三、从 Transformer 信息流范式出发的新设计

### 3.1 残差流的语义

在 Pre-Norm Transformer 中，残差流 h 在每一层都经过 LayerNorm 归一化后输入子层。这意味着：

- **子层看到的是 h 的方向信息**（LayerNorm 消除了幅度）
- **子层的输出 o 是对 h 的"修正建议"**
- **残差连接 h + o 是"接受修正建议"**

从这个视角看，门控机制的语义是：
- **α(h, o) · h**：保留多少"原始状态"
- **β(h, o) · o**：接受多少"修正建议"

### 3.2 什么信号应该驱动门控？

**信号 1：子层输出的"置信度"**

如果 o 的幅度很大，说明子层对这个 token 有强烈的"修正意见"，应该更多地接受 o。
如果 o 的幅度很小，说明子层认为 h 已经很好了，不需要太多修正。

```
confidence_o = ‖o‖ / (‖h‖ + ‖o‖ + ε)  ∈ [0, 1]
```

**信号 2：h 和 o 的方向一致性**

如果 h 和 o 方向一致（cos_sim > 0），说明子层在"强化"已有信息，可以同时保留 h 和接受 o。
如果 h 和 o 方向相反（cos_sim < 0），说明子层在"纠正"已有信息，应该更多地接受 o，减少 h 的影响。

```
cos_sim = (h · o) / (‖h‖ · ‖o‖ + ε)  ∈ [-1, 1]
```

**信号 3：当前 token 的"上下文需求"（内容依赖）**

这是最难捕捉的信号。不同的 token 对历史信息的需求不同。

一个轻量级的方法：用 h 本身（经过 LayerNorm）来预测"我需要多少历史信息"。

```
context_need = MLP_small(LayerNorm(h))  ∈ R^d
```

### 3.3 V7 设计方案

```python
class ResidualGateV7(nn.Module):
    """
    内容依赖的残差门控。
    
    核心思想：
    1. 用 h 的内容预测"需要多少历史信息"（α gate）
    2. 用 o 的内容预测"接受多少新信息"（β gate）
    3. 用物理信号（幅度比、方向一致性）作为辅助调制
    4. 保证梯度直通：α 不恒为 0
    
    h_new = α(h) ⊙ h + β(o) ⊙ o
    
    其中 α 和 β 是独立的 sigmoid 门，可以同时高（协同）或同时低（抑制）。
    """
    
    def __init__(self, hidden_size, bottleneck_dim=8, init_alpha_bias=3.0, init_beta_bias=3.0):
        super().__init__()
        self.hidden_size = hidden_size
        
        # α gate：基于 h 的内容，决定保留多少历史信息
        # 用低秩 MLP 捕捉 h 的全局语义
        self.alpha_up = nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.alpha_down = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.alpha_bias = nn.Parameter(torch.full((hidden_size,), init_alpha_bias))
        
        # β gate：基于 o 的内容，决定接受多少新信息
        # 用低秩 MLP 捕捉 o 的全局语义
        self.beta_up = nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.beta_down = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.beta_bias = nn.Parameter(torch.full((hidden_size,), init_beta_bias))
        
        # 物理信号调制（可选，轻量级）
        self.w_cos = nn.Parameter(torch.zeros(1))  # 方向一致性权重
        
        # 初始化：权重为 0，gate 由 bias 决定
        nn.init.zeros_(self.alpha_up.weight)
        nn.init.zeros_(self.alpha_down.weight)
        nn.init.zeros_(self.beta_up.weight)
        nn.init.zeros_(self.beta_down.weight)
        
        # RMSNorm for gate inputs
        self.norm_h = nn.RMSNorm(hidden_size)
        self.norm_o = nn.RMSNorm(hidden_size)
    
    def forward(self, residual: torch.Tensor, new_output: torch.Tensor) -> torch.Tensor:
        # 归一化输入（消除幅度差异，让 gate 只看方向/内容）
        h_norm = self.norm_h(residual)
        o_norm = self.norm_o(new_output)
        
        # α gate：h 的内容决定保留多少历史信息
        alpha_logit = (
            self.alpha_down(F.silu(self.alpha_up(h_norm)))  # 内容依赖项
            + self.alpha_bias                                 # 基准偏置
        )
        alpha = torch.sigmoid(alpha_logit)  # ∈ (0, 1)^d
        
        # β gate：o 的内容决定接受多少新信息
        beta_logit = (
            self.beta_down(F.silu(self.beta_up(o_norm)))   # 内容依赖项
            + self.beta_bias                                 # 基准偏置
        )
        beta = torch.sigmoid(beta_logit)  # ∈ (0, 1)^d
        
        # 可选：用方向一致性调制（detached，不影响主梯度流）
        # cos_sim = F.cosine_similarity(residual.detach(), new_output.detach(), dim=-1, eps=1e-8)
        # cos_mod = self.w_cos * cos_sim.unsqueeze(-1)  # (*, 1)
        # alpha = torch.sigmoid(alpha_logit + cos_mod)
        # beta = torch.sigmoid(beta_logit + cos_mod)
        
        return alpha * residual + beta * new_output
```

### 3.4 为什么这个设计能解决"直通残差冗余"问题？

1. **α 是内容依赖的**：`alpha = sigmoid(MLP(h_norm) + bias)`
   - 当 h 中的信息对当前 token 有用时，MLP 学到输出正值，α 接近 1
   - 当 h 中的信息冗余或有害时，MLP 学到输出负值，α 接近 0
   - 这是真正的内容依赖门控，不是固定的

2. **β 是内容依赖的**：`beta = sigmoid(MLP(o_norm) + bias)`
   - 当 o 提供了有价值的新信息时，β 接近 1
   - 当 o 的修正建议不重要时，β 接近 0

3. **梯度直通**：
   - `∂h_new/∂h = alpha · I + ∂alpha/∂h · h`
   - 初始时 alpha ≈ sigmoid(3) ≈ 0.95，梯度直通系数接近 1
   - 训练后 alpha 可以在 (0, 1) 范围内变化，但不会恒为 0

4. **α + β 可以 > 1**：当 h 和 o 都有价值时，两者都被保留（协同）
   - 这是 sigmoid 相对于 softmax 的优势

### 3.5 初始化分析

- `alpha_up.weight = 0, alpha_down.weight = 0` → 初始时 MLP 输出 0
- `alpha_bias = 3.0` → `alpha = sigmoid(3) ≈ 0.953`
- `beta_bias = 3.0` → `beta = sigmoid(3) ≈ 0.953`
- 初始 `h_new ≈ 0.953·h + 0.953·o ≈ h + o`（近似标准残差）

这保证了训练初期的行为接近标准 Transformer，模型可以从一个好的起点开始学习门控。

## 四、与其他方案的对比

### 4.1 vs v5.2 Hidden-Size Attention Gate

| 属性 | v5.2 | V7 |
|------|------|-----|
| α 是否内容依赖 | ❌ w_qh 是固定参数 | ✅ MLP(h_norm) |
| β 是否内容依赖 | ❌ w_qo 是固定参数 | ✅ MLP(o_norm) |
| 全局视野 | ❌ 64 维 group | ✅ 整个 d 维 |
| K 投影能力 | ❌ element-wise | ✅ 矩阵乘法 |
| 梯度直通 | ❌ softmax 衰减 | ✅ alpha ≈ 0.95 |
| α+β 可以 > 1 | ❌ softmax 约束 | ✅ 独立 sigmoid |
| 参数量/gate | ~4K | ~2×(d·r + r·d + d) ≈ 66K |

### 4.2 vs sigmoid 独立门控（用户之前实验过的）

用户说 sigmoid 版本也没有帮助。关键区别：

- **之前的 sigmoid 版本**：gate 基于 h 和 o 的物理特性（幅度、方向），Q 是固定参数
- **V7**：gate 基于 h 和 o 的**内容**（通过 MLP 提取），是真正的内容依赖

这是本质区别。物理特性（幅度、方向）是 token 无关的统计量，而内容（MLP 提取的语义特征）是 token 相关的。

### 4.3 vs `h_new = h + gate(o) ⊙ o`

- **方向三**：只门控 o，h 始终直通
- **V7**：同时门控 h 和 o，都是内容依赖的

V7 能够解决"直通残差冗余"问题，方向三不能。

## 五、关键设计决策

### 5.1 为什么用 RMSNorm 归一化 gate 输入？

- h 的幅度随深度增长（PreNorm 累积效应）
- 如果不归一化，深层的 h 幅度大，MLP 的输出会被幅度主导
- RMSNorm 确保 gate 只看内容（方向），不看幅度
- 这与 Transformer 中 LayerNorm 的作用类似

### 5.2 为什么用低秩 MLP 而不是全秩线性层？

- 全秩线性层 `Linear(d, d)` 参数量 d² = 1M，太大
- 低秩 MLP `Linear(d, r) → Linear(r, d)` 参数量 2dr，r=8 时约 16K
- 低秩瓶颈强制 gate 学习 h 的**低维语义摘要**，而不是逐维度的独立决策
- 这与 LoRA 的设计哲学一致

### 5.3 为什么 α 基于 h，β 基于 o？

- α 决定"保留多少历史信息"，这个决策应该基于"历史信息是什么"（h 的内容）
- β 决定"接受多少新信息"，这个决策应该基于"新信息是什么"（o 的内容）
- 这是最自然的信息论分工

**替代方案**：α 和 β 都基于 [h; o] 的拼接。这提供了更多信息，但参数量翻倍，且可能导致 α 和 β 之间的耦合。

### 5.4 为什么不用 softmax？

- softmax 约束 α + β = 1，这是一个人为的限制
- 当 h 和 o 都有价值时（协同场景），应该允许 α + β > 1
- 当 h 和 o 都不重要时（抑制场景），应该允许 α + β < 1
- 独立 sigmoid 提供了更大的灵活性

## 六、潜在问题和缓解措施

### 6.1 问题：α 可能学到恒为 0（完全遗忘历史）

**缓解**：
- 初始化 alpha_bias = 3.0，sigmoid(3) ≈ 0.95，从高值开始
- 对 alpha_bias 施加 L2 正则化，防止其偏离初始值太远
- 监控训练过程中 alpha 的均值，如果持续下降则调整

### 6.2 问题：MLP 可能学到恒为 0（gate 退化为常数）

**缓解**：
- 这实际上是一个好的退化：gate 退化为常数 sigmoid(bias)，接近标准残差
- 如果 MLP 权重保持为 0，说明内容依赖的门控没有帮助，模型选择了标准残差行为

### 6.3 问题：参数量增加

- 每个 gate 约 66K 参数（r=8, d=1024）
- 每层 2 个 gate，28 层共 56 个 gate
- 总增加：56 × 66K ≈ 3.7M 参数
- 相对于 596M 的总参数量，增加约 0.6%，可以接受

## 七、实验建议

### 7.1 消融实验

1. **α 的输入**：h_norm vs h vs LayerNorm(h)
2. **β 的输入**：o_norm vs o vs LayerNorm(o)
3. **bottleneck_dim**：4, 8, 16, 32
4. **init_bias**：1.0, 3.0, 5.0
5. **是否加物理信号调制**（cos_sim）

### 7.2 监控指标

- `alpha.mean()` 和 `beta.mean()`：是否在合理范围内（0.3-0.9）
- `alpha.std()` 和 `beta.std()`：是否有足够的变化（说明内容依赖在起作用）
- 不同 token 类型的 alpha 分布：功能词 vs 内容词 vs 标点

### 7.3 对比基线

- Qwen3-0.6B 标准残差（已有）
- V7 MLP Gate（新方案）
- V7 消融：只门控 o（`h_new = h + beta·o`）
- V7 消融：只门控 h（`h_new = alpha·h + o`）

## 八、一句话总结

**V7 用两个独立的低秩 MLP 分别从 h 和 o 的内容中提取语义特征，生成内容依赖的 α 和 β 门控，实现"不同 token/上下文对历史残差信息的不同程度依赖"，同时通过初始化保证训练初期接近标准残差行为。**
