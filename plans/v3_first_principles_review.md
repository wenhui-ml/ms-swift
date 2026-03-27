# V3 方案的第一性原理审查：信息传播与信息提取

## 一、LLM 中残差连接的第一性原理

### 1.1 残差流是什么？

在 Pre-Norm Transformer 中，残差流 h 是所有层输出的线性叠加：

```
h_l = h_0 + f_1(h_0) + f_2(h_1) + ... + f_{l-1}(h_{l-2})
```

**第一性原理**：残差流是一个 **共享的通信总线**。每一层从总线上读取信息（通过 LayerNorm + Attention/FFN），处理后将结果写回总线（通过加法）。

### 1.2 信息传播的本质

在这个总线模型中：
- **读取**：LayerNorm(h) → Attention/FFN → output o
- **写入**：h_new = h + o

标准残差的写入是 **无条件的**：无论 o 的质量如何，都直接加到总线上。

### 1.3 ResidualGate 要解决什么问题？

ResidualGate 要让写入变成 **有条件的**：

```
h_new = α⊙h + β⊙o
```

**核心问题**：α 和 β 应该基于什么信息来决定？

## 二、V3 方案的第一性原理审查

### 2.1 V3 的 gate 信号来源

V3 用以下信号计算 α 和 β：
1. `RMSNorm(h.detach())` — h 的方向
2. `RMSNorm(o.detach())` — o 的方向
3. `log|h| - log|o|` — 幅度比
4. `preference` — 可学习的偏好向量

### 🔴 根本问题：V3 的 gate 信号全部是 detached 的

**所有信号都来自 h 和 o 的 detached 版本**。这意味着：

**gate 的决策完全基于"当前时刻 h 和 o 的物理特性"，而不是基于"这个决策对下游 loss 的影响"。**

这违反了一个重要的第一性原理：

> **信息提取的第一性原理**：在端到端学习中，每个决策应该能够通过梯度反馈来优化，使其对最终目标（loss）的贡献最大化。

V3 的 gate 决策是基于 **局部物理特性**（幅度、方向）的启发式规则，而不是通过梯度学习的端到端优化。

### 2.2 这个问题有多严重？

让我们分析 V3 中梯度的流动：

```
L → h_new → α⊙h + β⊙o
         ↓
∂L/∂h = α · ∂L/∂h_new + [∂α/∂h · h + ∂β/∂h · o] · ∂L/∂h_new
                          ^^^^^^^^^^^^^^^^^^^^^^^^
                          这一项在 V3 中为 0！因为 α, β 不依赖于 h, o 的梯度
```

**V3 中 gate 对 h 和 o 没有梯度路径**。这意味着：
- h 和 o 的更新不会考虑"gate 会如何处理我的输出"
- gate 的参数（preference, temperature, bias）只能通过 `α·h` 和 `β·o` 的间接路径学习
- 具体来说：`∂L/∂preference = ∂L/∂h_new · (∂α/∂preference · h + ∂β/∂preference · o)`

这个间接路径是存在的，所以 gate 参数可以学习。但 **h 和 o 的生成过程无法感知 gate 的存在**。

### 2.3 对比：V1 的梯度路径

V1 中 gate_input 包含了带梯度的 h 和 o：

```
gate_input = [h, o, mag.detach(), dir.detach()]
gate_hidden = gate_A(gate_input)  ← h, o 的梯度通过这里流动
α = sigmoid(gate_B_alpha(gate_hidden))
```

V1 中：
```
∂L/∂h 包含 ∂α/∂h · h · ∂L/∂h_new  ← gate 对 h 有梯度
```

这意味着 V1 中，**h 的生成过程知道 gate 的存在**，可以调整自己的输出以更好地通过 gate。

### 🟡 但是：V1 的梯度路径也有问题

V1 中 h 和 o 通过 gate 网络有梯度，但这个梯度路径经过了 gate_A（4d→16）的低秩瓶颈，信号非常弱。而且这个梯度可能导致 **梯度干扰**：gate 的梯度会反向影响 Attention/FFN 的参数更新，可能不是好事。

## 三、重新思考：gate 到底需不需要对 h/o 有梯度？

### 3.1 类比：LSTM 的 gate

LSTM 的 forget gate：
```
f = sigmoid(W_f · [h_{t-1}, x_t] + b_f)
c_t = f ⊙ c_{t-1} + i ⊙ c̃_t
```

LSTM 的 gate **对输入有梯度**。这是因为 LSTM 的 gate 需要学习"什么时候遗忘"这个复杂的时序决策，需要端到端优化。

### 3.2 类比：Attention 的权重

Attention 的权重：
```
α = softmax(Q·K^T / √d)
output = α · V
```

Attention 的权重 **对 Q, K 有梯度**。这是因为 attention 需要学习"关注什么"这个复杂的内容依赖决策。

### 3.3 ResidualGate 的情况

ResidualGate 的决策是"在每个维度上保留多少旧信息、接受多少新信息"。

**关键问题**：这个决策是否需要端到端优化？

**论点 A（需要梯度）**：
- gate 的最优决策取决于下游任务，不能仅靠局部物理特性决定
- 例如：某个维度的 h 和 o 幅度相当、方向一致，但下游任务可能需要更多 o 的信息

**论点 B（不需要梯度）**：
- gate 的决策本质上是一个"信号质量评估"问题，可以用物理特性近似
- 类比：信号处理中的自动增益控制（AGC）不需要知道下游任务是什么
- detached gate 的优势：不会干扰主网络的梯度流，训练更稳定

### 3.4 我的判断

**两种方案都有道理，但 V3 的纯 detached 方案可能过于保守。**

最优方案应该是：**gate 的核心信号来自 detached 的物理特性（稳定），但保留一条轻量级的梯度路径让 gate 能够微调（灵活）。**

## 四、V3 的另一个第一性原理问题

### 4.1 preference 向量的语义问题

V3 的 `preference` 是一个 per-dim 的可学习向量，与 `dir_h` 和 `dir_o` 做点积。

**问题**：`preference` 在训练中是固定的（对所有 token 相同），但 `dir_h` 和 `dir_o` 是 token-dependent 的。

这意味着 `preference × dir_h` 的含义是："第 j 维的 h 方向是否与 preference 的固定方向一致"。

**但这个"固定方向"是什么意思？** 在 Transformer 中，残差流的每个维度没有固定的语义方向。同一个维度在不同 token 上可能表示完全不同的信息。

**更根本的问题**：RMSNorm 是全局归一化（除以所有维度的 RMS），它改变了每个维度的相对大小。`dir_h_j` 不是第 j 维的方向（符号），而是第 j 维在 RMSNorm 后的值。这个值的符号取决于 h_j 的符号和 h 的整体幅度。

所以 `preference × dir_h` 实际上是在学习"第 j 维在 RMSNorm 后的值与 preference 的相关性"，这不是一个清晰的物理信号。

### 4.2 对比：AttnRes 的 query 为什么有效

AttnRes 的 query w_l 与 RMSNorm(v_i) 做点积，但这是在 **整个 d 维空间** 上做的：

```
α_{i→l} = softmax(w_l^T · RMSNorm(v_i))  ← 这是一个标量
```

这个标量表示"第 l 层的需求与第 i 层的输出方向在 d 维空间中的匹配度"。这是有意义的，因为 d 维空间中的方向确实编码了语义信息。

但 V3 的 `preference_j × dir_h_j` 是 **逐元素** 的乘法，每个维度独立。单个维度的 RMSNorm 后的值没有清晰的语义含义。

## 五、修正方案

### 5.1 回归本质：gate 应该基于什么信息？

从信息传播的第一性原理出发：

1. **幅度信息**（detached，物理信号）：
   - `|h_j|` 和 `|o_j|` 的相对大小 → 谁在这个维度上更强
   - 这是一个可靠的物理信号，不需要梯度

2. **方向信息**（detached，物理信号）：
   - h_j 和 o_j 的符号关系 → 是否冲突
   - 这也是一个可靠的物理信号

3. **上下文信息**（需要梯度）：
   - "当前 token 在当前层需要什么" → 这需要从 h 或 o 的内容中提取
   - 这是 V1 的 gate_A([h, o, ...]) 试图捕捉的

### 5.2 最终方案：物理信号 + 轻量级上下文

```python
class ResidualGate(nn.Module):
    def __init__(self, hidden_size, init_bias=5.0):
        super().__init__()
        
        # ---- 物理信号参数（detached 路径）----
        self.w_mag = nn.Parameter(torch.zeros(1))      # 幅度信号权重
        self.w_dir = nn.Parameter(torch.zeros(1))      # 方向信号权重
        
        # ---- per-dim 基准 ----
        self.b_alpha = nn.Parameter(torch.full((hidden_size,), init_bias))
        self.b_beta = nn.Parameter(torch.full((hidden_size,), init_bias))
    
    def forward(self, residual, new_output):
        eps = 1e-5
        
        # ---- 物理信号（detached）----
        h_det = residual.detach()
        o_det = new_output.detach()
        
        h_mag = h_det.abs()
        o_mag = o_det.abs()
        
        # 对数幅度比（中间地带分辨率高）
        log_mag_ratio = torch.log(h_mag + eps) - torch.log(o_mag + eps)
        
        # 逐元素方向一致性（简单的符号乘积，归一化到 [-1, 1]）
        dir_agree = (h_det * o_det) / (h_mag * o_mag + eps)
        
        # ---- Gate 计算 ----
        # α: 保留门
        #   幅度大 → 保留（log_mag > 0 → α 增加）
        #   方向一致 → 保留（dir_agree > 0 → α 增加）
        alpha = torch.sigmoid(
            self.w_mag * log_mag_ratio
            + self.w_dir * dir_agree
            + self.b_alpha
        )
        
        # β: 接受门
        #   o 幅度大 → 接受（-log_mag > 0 → β 增加）
        #   方向一致 → 接受（dir_agree > 0 → β 增加）
        beta = torch.sigmoid(
            -self.w_mag * log_mag_ratio
            + self.w_dir * dir_agree
            + self.b_beta
        )
        
        return alpha * residual + beta * new_output
```

### 5.3 为什么去掉了 preference？

1. **语义不清晰**：per-dim preference 与 RMSNorm 后的值做点积，物理含义模糊
2. **不必要**：per-dim bias (b_alpha, b_beta) 已经提供了"每个维度的基准偏好"
3. **简洁**：去掉 preference 后，gate 的行为完全由物理信号（幅度、方向）和 per-dim bias 决定，更加可解释

### 5.4 为什么 α 和 β 共享 w_mag 和 w_dir？

注意 α 和 β 的公式中：
- `w_mag` 对 α 是正号，对 β 是负号 → 幅度信号自然地让 α 和 β 互补
- `w_dir` 对 α 和 β 都是正号 → 方向一致时两者都增加（协同保留）

这种对称设计减少了参数量（2 个标量 + 2d 偏置 = 2d+2），同时保持了物理合理性。

### 5.5 参数量

- w_mag: 1
- w_dir: 1
- b_alpha: d
- b_beta: d
- **总计: 2d + 2 ≈ 2K per gate (d=1024)**

这是所有方案中参数最少的，但物理含义最清晰。

## 六、最终结论

从信息传播和信息提取的第一性原理出发：

1. **gate 的核心信号应该是 detached 的物理特性**（幅度比、方向一致性），因为这些是可靠的、不需要端到端优化的信号质量指标

2. **per-dim bias 提供了足够的可学习自由度**，让每个维度可以有不同的基准保留/接受策略

3. **不需要 preference 向量**，因为它的物理含义不清晰，且 per-dim bias 已经覆盖了它的功能

4. **对数幅度比优于线性幅度比**，在中间地带有更好的分辨率

5. **α 和 β 共享物理信号权重但用相反符号**，自然地实现了互补关系

这个最终方案是最简洁、最符合第一性原理的设计。
