# ResidualGate V3：融合 AttnRes 思想的设计

## 一、AttnRes 论文的核心洞察

### 1.1 标准残差的根本问题

论文指出标准残差 `h_l = h_{l-1} + f_{l-1}(h_{l-1})` 的三个根本限制：

1. **无选择性访问**：每层只能访问前一层的压缩状态 h_{l-1}，无法选择性地访问更早层的输出
2. **不可逆信息丢失**：一旦信息在聚合中丢失，深层无法恢复
3. **输出增长问题**：深层需要产生越来越大的输出才能在累积的残差中产生影响（PreNorm dilution）

### 1.2 AttnRes 的解决方案

用 softmax attention over depth 替代固定的加法残差：

```
h_l = Σ α_{i→l} · v_i
```

其中 α 是通过 learned query w_l 和 key=RMSNorm(v_i) 计算的 softmax attention 权重。

**关键设计**：
- Query: `q_l = w_l`（per-layer 可学习向量，d 维）
- Key: `k_i = RMSNorm(v_i)`（对历史输出做 RMSNorm 防止幅度主导）
- Value: `v_i = f_i(h_i)`（各层的原始输出）
- Attention: `α_{i→l} = softmax(q_l^T · RMSNorm(v_i))`

### 1.3 Block AttnRes（实用版本）

将 L 层分成 N 个 block，block 内用标准残差累积，block 间用 attention：
- 内存从 O(Ld) 降到 O(Nd)
- N≈8 就能恢复大部分收益

## 二、AttnRes 与我们的 ResidualGate 的对比

| 维度 | AttnRes | 我们的 ResidualGate |
|------|---------|-------------------|
| 作用范围 | 跨层（depth attention） | 单层（layer-local gate） |
| 信息来源 | 所有历史层的输出 | 只有当前残差 h 和当前层输出 o |
| 权重计算 | softmax attention（归一化） | 独立 sigmoid（非归一化） |
| 参数 | per-layer query w_l（d 维） | gate 网络参数 |
| 核心思想 | 选择性聚合历史信息 | 选择性保留/接受当前信息 |
| 解决的问题 | PreNorm dilution, 跨层信息选择 | 维度级别的信息流控制 |

**关键区别**：AttnRes 是在 **层间** 做选择性聚合，我们的 ResidualGate 是在 **维度级别** 做保留/接受决策。两者是正交的，可以互补。

## 三、从 AttnRes 借鉴的关键思想

### 3.1 RMSNorm 防止幅度主导

AttnRes 的一个精妙设计：对 key 做 RMSNorm，防止幅度大的层输出主导 attention 权重。

**启发**：我们的 mag_ratio 直接用原始幅度，大幅度的维度会主导 gate 决策。应该对幅度做归一化处理。

### 3.2 Learned Query 作为 "需求信号"

AttnRes 用 per-layer learned query 表示"这一层需要什么样的信息"。

**启发**：我们的 ResidualGate 缺少"需求信号"——当前的 gate 只看 h 和 o 的物理特性（幅度、方向），但不知道"模型在这个位置需要什么"。

### 3.3 Softmax 归一化 vs 独立 Sigmoid

AttnRes 用 softmax 确保权重和为 1（零和博弈：给一个层更多权重意味着给其他层更少）。

我们的 ResidualGate 用独立 sigmoid，α 和 β 可以同时为 1 或同时为 0。

**问题**：独立 sigmoid 的自由度太高，可能导致 α≈1, β≈1（退化为标准残差）。

## 四、V3 设计：AttnRes-Inspired Magnitude-Direction Gate

### 核心思想

借鉴 AttnRes 的 "query-key attention" 思想，但在维度级别操作：

1. **用 learned per-dim query 表示"每个维度的需求偏好"**
2. **用 RMSNorm 归一化的 h 和 o 作为 key，消除幅度偏差**
3. **用 cosine similarity（方向一致性）作为核心 gate 信号**
4. **用幅度信息作为辅助调制**

### V3 公式

```python
class ResidualGate(nn.Module):
    def __init__(self, hidden_size, init_bias=5.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = 1e-5
        
        # Learned per-dim "preference" vector (类似 AttnRes 的 query)
        # 表示每个维度倾向于保留旧信息还是接受新信息
        # 初始化为 0 → 初始时无偏好
        self.preference = nn.Parameter(torch.zeros(hidden_size))
        
        # Per-dim bias for α and β (控制基准 gate 值)
        self.b_alpha = nn.Parameter(torch.full((hidden_size,), init_bias))
        self.b_beta = nn.Parameter(torch.full((hidden_size,), init_bias))
        
        # Learnable temperature for direction signal
        # 控制方向一致性对 gate 的影响强度
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, residual, new_output):
        # ---- 1. 方向信号（借鉴 AttnRes 的 RMSNorm + dot product）----
        # RMSNorm 消除幅度差异，只保留方向信息
        h_norm = residual.detach() * torch.rsqrt(
            residual.detach().pow(2).mean(-1, keepdim=True) + self.eps
        )  # (B, T, d), 方向归一化
        o_norm = new_output.detach() * torch.rsqrt(
            new_output.detach().pow(2).mean(-1, keepdim=True) + self.eps
        )  # (B, T, d), 方向归一化
        
        # Per-dim cosine-like similarity（在 RMSNorm 空间中）
        # 这比原始的 dir_agree 更稳定，因为 RMSNorm 消除了全局幅度差异
        dir_h = h_norm  # h 的方向
        dir_o = o_norm  # o 的方向
        
        # ---- 2. 幅度信号 ----
        h_mag = residual.detach().abs()
        o_mag = new_output.detach().abs()
        # 对数幅度比（比线性比更均匀，中间地带分辨率更高）
        log_mag_ratio = torch.log(h_mag + self.eps) - torch.log(o_mag + self.eps)
        # log_mag_ratio > 0: h 更强; < 0: o 更强; ≈ 0: 势均力敌
        
        # ---- 3. Gate 计算（借鉴 AttnRes 的 query-key 思想）----
        # preference 作为 "query"，dir_h 和 dir_o 作为 "key"
        # preference > 0 的维度倾向于与 h 方向一致时保留
        # preference < 0 的维度倾向于与 o 方向一致时接受
        
        # α: 保留门
        # - 当 h 的方向与 preference 一致时，α 高
        # - 当 h 幅度大时（log_mag_ratio > 0），α 高
        alpha_logit = (
            self.preference * dir_h          # preference-direction alignment
            + self.temperature * log_mag_ratio  # magnitude modulation
            + self.b_alpha                    # per-dim baseline
        )
        
        # β: 接受门
        # - 当 o 的方向与 preference 一致时，β 高
        # - 当 o 幅度大时（log_mag_ratio < 0），β 高
        beta_logit = (
            self.preference * dir_o          # preference-direction alignment
            - self.temperature * log_mag_ratio  # magnitude modulation (反号)
            + self.b_beta                    # per-dim baseline
        )
        
        alpha = torch.sigmoid(alpha_logit)
        beta = torch.sigmoid(beta_logit)
        
        return alpha * residual + beta * new_output
```

### 参数量
- preference: d
- b_alpha: d
- b_beta: d
- temperature: 1
- **总计: 3d + 1 ≈ 3K per gate (d=1024)**

## 五、V3 的物理解释

### 5.1 preference 向量的含义

`preference` 是一个 per-dim 的可学习向量，类似 AttnRes 中的 query w_l。

- `preference_j > 0`：第 j 维"偏好"与 h 方向一致的信息
  - 当 h_j 和 preference_j 同号时，`preference_j * dir_h_j > 0` → α 高
  - 当 o_j 和 preference_j 同号时，`preference_j * dir_o_j > 0` → β 高
  
- `preference_j < 0`：第 j 维"偏好"与 h 方向相反的信息
  - 这允许模型学习"某些维度应该被反转"

- `preference_j ≈ 0`：第 j 维无方向偏好，gate 主要由幅度和 bias 决定

### 5.2 log_mag_ratio 的优势

相比线性 mag_ratio = |h|/(|h|+|o|+ε)：

```
线性 mag_ratio:
  |h|=100, |o|=1   → 0.990  ┐
  |h|=10,  |o|=1   → 0.909  ├ 差异 0.081
  |h|=1,   |o|=1   → 0.500  ┘ 差异 0.409

对数 log_mag_ratio:
  |h|=100, |o|=1   → 4.605  ┐
  |h|=10,  |o|=1   → 2.303  ├ 差异 2.302
  |h|=1,   |o|=1   → 0.000  ┘ 差异 2.303
```

对数空间中，幅度比的分布更均匀，中间地带的分辨率更高。

### 5.3 temperature 的作用

`temperature` 控制幅度信号对 gate 的影响强度：
- temperature 大 → 幅度差异对 gate 影响大（幅度主导）
- temperature 小 → 幅度差异对 gate 影响小（方向和 bias 主导）
- 模型可以自动学习最优的幅度敏感度

### 5.4 RMSNorm 方向归一化的作用

借鉴 AttnRes 的关键设计：对 h 和 o 做 RMSNorm 后再计算方向信号。

这解决了原始 dir_agree 的问题：
- 原始：`dir_agree = h*o / (|h|*|o| + ε)` — 逐元素除法，数值不稳定
- V3：`dir_h = RMSNorm(h)` — 全局归一化，数值稳定，且保留了维度间的相对方向信息

## 六、6 个区域在 V3 中的行为

假设训练后 preference 学到了合理的值：

| 区域 | h方向 | o方向 | log_mag | preference*dir_h | preference*dir_o | α | β |
|------|-------|-------|---------|-----------------|-----------------|---|---|
| A: h强+同向 | + | + | 大正 | + | + | 高 | 中 |
| B: h强+反向 | + | - | 大正 | + | - | 高 | 低 |
| C: o强+同向 | + | + | 大负 | + | + | 中 | 高 |
| D: o强+反向 | + | - | 大负 | + | - | 低 | 高 |
| E: 均衡+同向 | + | + | ≈0 | + | + | 高 | 高 |
| F: 均衡+反向 | + | - | ≈0 | + | - | 中 | 中 |

✅ 所有区域的行为都符合预期。

## 七、与 V1、V2 的对比

| 维度 | V1 低秩网络 | V2 直接映射 | V3 AttnRes-Inspired |
|------|------------|------------|-------------------|
| 参数量 | ~100K/gate | ~4K/gate | ~3K/gate |
| 核心信号 | [h,o,mag,dir] | mag, dir | RMSNorm方向, log幅度 |
| 方向处理 | 逐元素 h*o/(|h||o|) | 同左 | RMSNorm 全局归一化 |
| 幅度处理 | 线性 |h|/(|h|+|o|) | 同左 | 对数 log(|h|/|o|) |
| 需求信号 | 无（纯物理） | 无 | preference 向量 |
| 可解释性 | 黑盒 | 高 | 高 |
| 数值稳定性 | 中 | 中 | 高（RMSNorm + log） |
| 中间地带分辨率 | 取决于rank | 需要 per-dim scale | 对数空间天然均匀 |

## 八、进一步思考：是否需要 AttnRes 的跨层机制？

AttnRes 的核心价值是 **跨层选择性聚合**。我们的 ResidualGate 是 **层内** 的 gate。

两者可以结合：
1. **层内**：用 V3 的 ResidualGate 做维度级别的保留/接受
2. **层间**：用 Block AttnRes 做跨层的选择性聚合

但 Block AttnRes 需要存储历史 block 的表示，增加了内存开销。对于从头预训练的小模型（0.6B），层内 gate 可能已经足够。

**建议**：先实施 V3 的层内 gate，验证效果后再考虑是否加入跨层机制。

## 九、实施计划

1. 在 modeling_mag_gated.py 中实现 V3 版 ResidualGate
2. 更新 config 支持 V3 参数
3. 更新 gate_monitor_callback 适配 V3 的新统计量
4. 更新 create_mag_gated_model.py
5. 运行测试验证
6. 对比 V1 和 V3 的训练效果
