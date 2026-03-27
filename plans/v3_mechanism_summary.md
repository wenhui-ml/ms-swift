# V3 方案总结：基于 Magnitude-Direction 的仿 Attention Residual 机制

## 一、AttnRes 的核心机制回顾

AttnRes 在 **层间** 做 attention：

```
h_l = Σ α_{i→l} · v_i     (对所有历史层输出做加权聚合)

其中：
  q_l = w_l                 (per-layer learned query: "我需要什么")
  k_i = RMSNorm(v_i)        (历史层输出的方向: "我提供什么")
  α_{i→l} = softmax(q_l^T · k_i)  (query-key 相似度 → 权重)
```

**本质**：每一层通过 learned query 去"查询"历史层的输出，根据方向匹配度分配权重。RMSNorm 确保只看方向不看幅度。

## 二、V3 如何将 AttnRes 机制映射到维度级别

V3 在 **维度级别** 做类似 attention 的操作：

### 2.1 类比关系

| AttnRes（层间） | V3 ResidualGate（维度级别） |
|----------------|--------------------------|
| 多个历史层输出 v_0, v_1, ..., v_{l-1} | 两个候选：residual h 和 new_output o |
| per-layer query w_l ∈ R^d | per-dim preference ∈ R^d |
| RMSNorm(v_i) 消除幅度 | RMSNorm(h), RMSNorm(o) 消除幅度 |
| softmax attention 权重 | sigmoid gate α, β |
| 加权聚合 Σ α_i · v_i | α⊙h + β⊙o |

### 2.2 V3 的"仿 attention"过程

对于每个维度 j，V3 执行以下"attention-like"操作：

```
Step 1: 提取方向信号（类似 AttnRes 的 Key）
  dir_h = RMSNorm(h.detach())    → h 的方向（消除幅度）
  dir_o = RMSNorm(o.detach())    → o 的方向（消除幅度）

Step 2: 计算 query-key 匹配度（类似 AttnRes 的 q^T · k）
  match_h_j = preference_j × dir_h_j    → "preference 与 h 方向的匹配度"
  match_o_j = preference_j × dir_o_j    → "preference 与 o 方向的匹配度"

Step 3: 幅度调制（AttnRes 没有的额外信号）
  log_mag_j = log(|h_j|) - log(|o_j|)  → 对数幅度比

Step 4: 生成 gate（类似 AttnRes 的 attention weight）
  α_j = sigmoid(match_h_j + τ·log_mag_j + b_α_j)
  β_j = sigmoid(match_o_j - τ·log_mag_j + b_β_j)

Step 5: 加权聚合（类似 AttnRes 的 Σ α_i · v_i）
  h_new_j = α_j · h_j + β_j · o_j
```

## 三、Magnitude 和 Direction 各自的角色

### 3.1 Direction（方向）的角色 — 类似 AttnRes 的 Query-Key 匹配

```
preference_j × dir_h_j
```

这是 V3 的核心 attention 机制：

- `preference` 是 learned query，编码了"第 j 维偏好什么方向的信息"
- `dir_h` 和 `dir_o` 是 key，编码了"h 和 o 在第 j 维提供什么方向的信息"
- 乘积 `preference × dir` 就是 query-key 的点积匹配度

**工作原理**：
- 如果 preference_j = +2.0（偏好正方向）
  - h_j 为正 → match_h 为正 → α 高（保留）
  - o_j 为正 → match_o 为正 → β 高（接受）
  - h_j 为负 → match_h 为负 → α 低（遗忘）
  - o_j 为负 → match_o 为负 → β 低（拒绝）

- 如果 preference_j = -1.5（偏好负方向）
  - 行为反转：负方向的信息被保留/接受，正方向的被遗忘/拒绝

- 如果 preference_j ≈ 0（无偏好）
  - 方向信号不影响 gate，完全由幅度和 bias 决定

**关键**：RMSNorm 确保 dir_h 和 dir_o 只反映方向，不受幅度影响。这直接借鉴了 AttnRes 对 key 做 RMSNorm 的设计。

### 3.2 Magnitude（幅度）的角色 — AttnRes 没有的额外维度

AttnRes 通过 RMSNorm 完全消除了幅度信息。但在维度级别的 gate 中，幅度信息是有价值的：

```
log_mag_ratio = log(|h_j|) - log(|o_j|)
```

幅度信号提供了"谁更强"的物理信息：

- `log_mag > 0`（h 更强）→ α 增加，β 减少
- `log_mag < 0`（o 更强）→ α 减少，β 增加
- `log_mag ≈ 0`（势均力敌）→ 幅度不影响，由方向和 bias 决定

**temperature τ** 控制幅度信号的影响强度：
- τ 大 → 幅度主导 gate 决策（"谁强听谁的"）
- τ 小 → 方向主导 gate 决策（"方向对的才接受"）
- τ 可学习 → 模型自动找到最优的幅度-方向平衡

### 3.3 Direction 和 Magnitude 的协同

两个信号在 sigmoid 内部做加法，形成互补的决策依据：

```
α_logit = [方向匹配度] + [幅度优势] + [基准偏置]
        = preference·dir_h + τ·log_mag + b_α
```

| 场景 | 方向匹配 | 幅度优势 | α 结果 |
|------|---------|---------|--------|
| h 方向对 + h 强 | + | + | 很高（双重支持） |
| h 方向对 + h 弱 | + | - | 中等（方向支持但幅度不支持） |
| h 方向错 + h 强 | - | + | 中等（幅度支持但方向不支持） |
| h 方向错 + h 弱 | - | - | 很低（双重反对） |

这种加法组合让 gate 能够在方向和幅度之间做 **软权衡**，而不是硬性的 if-else 分支。

## 四、与 AttnRes 的关键差异

### 4.1 为什么用 sigmoid 而不是 softmax？

AttnRes 用 softmax 是因为它在多个历史层之间分配权重（零和博弈）。

V3 只有两个候选（h 和 o），用 sigmoid 而非 softmax 的原因：
- **允许 α+β > 1**：当 h 和 o 协同时（同向），两者都应该被保留
- **允许 α+β < 1**：当两者都不可靠时，可以同时衰减
- **更灵活**：softmax(α, β) 会强制 α+β=1，限制了表达能力

### 4.2 为什么保留幅度信号？

AttnRes 完全消除幅度是因为它在层间操作，不同层的输出幅度差异很大（PreNorm dilution 问题）。

V3 在维度级别操作，h 和 o 的幅度差异包含有价值的信息：
- 幅度大的维度通常携带更重要的信息
- 幅度比反映了"谁在这个维度上更有发言权"

所以 V3 **同时使用方向（RMSNorm 后）和幅度（对数比）**，而 AttnRes 只使用方向。

### 4.3 为什么用 per-dim preference 而不是 per-layer query？

AttnRes 的 query w_l 是 per-layer 的（每层一个 d 维向量），用于在多个历史层之间选择。

V3 的 preference 是 per-dim 的（每个 gate 一个 d 维向量），用于在 h 和 o 之间选择。

两者的粒度不同：
- AttnRes query：选择"从哪一层获取信息"（层级选择）
- V3 preference：选择"在每个维度上保留还是接受"（维度级选择）

## 五、数据流图

```
residual (h)          new_output (o)
    │                      │
    ├──── detach ──────────┤
    │         │            │         │
    │    RMSNorm(h)   RMSNorm(o)    │
    │      dir_h        dir_o       │
    │         │            │        │
    │    ┌────┴────┐  ┌───┴────┐   │
    │    │preference│  │preference│  │
    │    │ × dir_h │  │ × dir_o │  │
    │    └────┬────┘  └───┬────┘   │
    │         │            │        │
    │    ┌────┴────────────┴────┐   │
    │    │  log(|h|) - log(|o|) │   │
    │    │    × temperature     │   │
    │    └────┬────────────┬────┘   │
    │         │            │        │
    │    ┌────┴────┐  ┌───┴────┐   │
    │    │ + b_α   │  │ + b_β  │   │
    │    │ sigmoid │  │ sigmoid │   │
    │    └────┬────┘  └───┬────┘   │
    │         α            β        │
    │         │            │        │
    │    ┌────┴────┐  ┌───┴────┐   │
    ├───→│  α ⊙ h  │  │ β ⊙ o │←──┤
    │    └────┬────┘  └───┬────┘
    │         │            │
    │         └─────┬──────┘
    │               │
    │           h_new = α⊙h + β⊙o
```

## 六、一句话总结

**V3 将 AttnRes 的 "learned query 查询 RMSNorm 归一化的 key" 机制从层间降维到维度级别，用 per-dim preference 向量作为 query，用 RMSNorm(h) 和 RMSNorm(o) 作为 key 计算方向匹配度，再用对数幅度比作为辅助调制信号，通过 sigmoid 生成独立的保留门 α 和接受门 β。**
