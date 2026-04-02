# 物理信号分类与旧/新信息汇总策略

## 一、两个物理信号的完整组合空间

mag_ratio ∈ [0, 1] 和 dir_agree ∈ [-1, 1] 构成一个 2D 空间。
我们将其划分为 **6 个语义区域**，每个区域对应不同的物理含义和最优策略。

### 完整分类表

| 区域 | mag_ratio | dir_agree | 物理含义 | α 策略 | β 策略 | 直觉 |
|------|-----------|-----------|----------|--------|--------|------|
| A | 高 ~1 | 高 +1 | h强，o弱，同向 | 高 | 中 | h主导，o是微弱的同向补充 |
| B | 高 ~1 | 低 -1 | h强，o弱，反向 | 高 | 低 | h主导，o是噪声/干扰 |
| C | 低 ~0 | 高 +1 | h弱，o强，同向 | 中 | 高 | o带来新信息，与h方向一致 |
| D | 低 ~0 | 低 -1 | h弱，o强，反向 | 低 | 高 | o覆写h，新信息替代旧信息 |
| E | 中 ~0.5 | 高 +1 | 势均力敌，同向 | 高 | 高 | 两者协同，都保留（≈标准残差） |
| F | 中 ~0.5 | 低 -1 | 势均力敌，反向 | 竞争 | 竞争 | 最需要学习的区域 |

### 详细分析

#### 区域 A：h 强 + 同向（mag_ratio↑, dir_agree↑）
```
h_j = 5.0,  o_j = 0.3  →  mag_ratio = 0.94,  dir_agree = +1.0
```
- h 在该维度已经建立了强信号
- o 是同方向的微弱补充
- **最优策略**：α 高（保留 h），β 中等（接受 o 的补充但不过度放大）
- **等效于**：h_new ≈ h + small_o，接近标准残差

#### 区域 B：h 强 + 反向（mag_ratio↑, dir_agree↓）
```
h_j = 5.0,  o_j = -0.3  →  mag_ratio = 0.94,  dir_agree = -1.0
```
- h 在该维度有强信号
- o 试图反向修正，但很弱
- **最优策略**：α 高（保留 h），β 低（拒绝反向干扰）
- **这是 gate 最有价值的场景**：标准残差会让 h_new = 5.0 + (-0.3) = 4.7，gate 可以保护 h

#### 区域 C：o 强 + 同向（mag_ratio↓, dir_agree↑）
```
h_j = 0.3,  o_j = 5.0  →  mag_ratio = 0.06,  dir_agree = +1.0
```
- h 在该维度信号弱
- o 带来强的同向新信息
- **最优策略**：α 中等（保留 h 的方向信息），β 高（大力接受 o）
- **等效于**：h_new ≈ h + strong_o，接近标准残差但可以微调比例

#### 区域 D：o 强 + 反向（mag_ratio↓, dir_agree↓）
```
h_j = 0.3,  o_j = -5.0  →  mag_ratio = 0.06,  dir_agree = -1.0
```
- h 在该维度信号弱
- o 带来强的反向新信息（覆写）
- **最优策略**：α 低（放弃旧信息），β 高（接受新信息）
- **这是"维度分时复用"的核心场景**：旧维度被回收，写入新信息

#### 区域 E：势均力敌 + 同向（mag_ratio≈0.5, dir_agree↑）
```
h_j = 3.0,  o_j = 3.0  →  mag_ratio = 0.50,  dir_agree = +1.0
```
- 两者强度相当，方向一致
- **最优策略**：α 高，β 高（两者协同，都保留）
- **等效于标准残差**：h_new = h + o

#### 区域 F：势均力敌 + 反向（mag_ratio≈0.5, dir_agree↓）
```
h_j = 3.0,  o_j = -3.0  →  mag_ratio = 0.50,  dir_agree = -1.0
```
- 两者强度相当，方向冲突
- **最需要学习的区域**：标准残差会让 h_new ≈ 0（相互抵消）
- **最优策略取决于上下文**：可能需要保留 h（α 高 β 低），也可能需要接受 o（α 低 β 高）
- 这个区域是 per-dim bias 发挥作用的地方

## 二、从分类表推导 α 和 β 的映射函数

### 观察规律

从上面的分类表中，我们可以提取出以下规律：

**α（保留门）的规律**：
1. mag_ratio 高 → α 高（h 强时保留）
2. dir_agree 高 → α 高（同向时保留）
3. 两个信号对 α 的影响是**正相关**的

**β（接受门）的规律**：
1. mag_ratio 低（即 1-mag_ratio 高）→ β 高（o 强时接受）
2. dir_agree 高 → β 高（同向时接受）
3. dir_agree 低 + mag_ratio 低 → β 仍然高（区域 D：覆写场景）

### 关键洞察：β 对 dir_agree 的响应是非对称的

- 当 mag_ratio 低时（o 主导），无论 dir_agree 正负，β 都应该高
- 当 mag_ratio 高时（h 主导），dir_agree 负 → β 应该低

这意味着 **β 对 dir_agree 的敏感度应该随 mag_ratio 变化**。

### 改进的 V2 公式

基于上述分析，我提出一个更精确的映射：

```python
# α: 保留门
# - mag_ratio 高 → α 高
# - dir_agree 高 → α 高
α_logit = w_α_mag * mag_ratio + w_α_dir * dir_agree + b_α

# β: 接受门  
# - (1-mag_ratio) 高 → β 高（o 强时接受）
# - dir_agree 的影响被 mag_ratio 调制：
#   当 mag_ratio 低时（o 主导），dir_agree 对 β 影响小（无论同反向都接受）
#   当 mag_ratio 高时（h 主导），dir_agree 负 → β 低（拒绝反向干扰）
β_logit = w_β_mag * (1 - mag_ratio) + w_β_dir * dir_agree * mag_ratio + b_β
#                                                          ^^^^^^^^^^^
#                                                    mag_ratio 调制 dir 的影响
```

**为什么要用 `dir_agree * mag_ratio` 调制 β？**

| 场景 | mag_ratio | dir_agree | dir_agree * mag_ratio | β 行为 |
|------|-----------|-----------|----------------------|--------|
| 区域 B: h强+反向 | 0.94 | -1.0 | -0.94 | β 低 ✅ 拒绝反向干扰 |
| 区域 D: o强+反向 | 0.06 | -1.0 | -0.06 | β 不受影响 ✅ 接受覆写 |
| 区域 E: 均衡+同向 | 0.50 | +1.0 | +0.50 | β 中高 ✅ 接受协同 |
| 区域 F: 均衡+反向 | 0.50 | -1.0 | -0.50 | β 中低 ✅ 谨慎 |

这个调制项完美地捕捉了"只有当 h 足够强时，方向冲突才应该降低 β"的物理直觉。

## 三、最终 V2 公式

```python
class ResidualGate(nn.Module):
    def __init__(self, hidden_size, init_bias=5.0):
        super().__init__()
        # 5 个标量权重 + 2 个 per-dim 偏置
        self.w_alpha_mag = nn.Parameter(torch.zeros(1))    # mag → α
        self.w_alpha_dir = nn.Parameter(torch.zeros(1))    # dir → α
        self.w_beta_mag  = nn.Parameter(torch.zeros(1))    # (1-mag) → β
        self.w_beta_dir  = nn.Parameter(torch.zeros(1))    # dir*mag → β (调制项)
        self.w_beta_dir_raw = nn.Parameter(torch.zeros(1)) # dir → β (直接项，可选)
        
        self.b_alpha = nn.Parameter(torch.full((hidden_size,), init_bias))
        self.b_beta  = nn.Parameter(torch.full((hidden_size,), init_bias))
    
    def forward(self, residual, new_output):
        eps = 1e-5
        h_mag = residual.detach().abs()
        o_mag = new_output.detach().abs()
        
        mag_ratio = h_mag / (h_mag + o_mag + eps)          # ∈ [0, 1]
        dir_agree = (residual.detach() * new_output.detach()) / (h_mag * o_mag + eps)  # ∈ [-1, 1]
        
        # α: 保留门
        alpha = torch.sigmoid(
            self.w_alpha_mag * mag_ratio 
            + self.w_alpha_dir * dir_agree 
            + self.b_alpha
        )
        
        # β: 接受门（dir 的影响被 mag_ratio 调制）
        beta = torch.sigmoid(
            self.w_beta_mag * (1.0 - mag_ratio) 
            + self.w_beta_dir * dir_agree * mag_ratio  # 调制项
            + self.b_beta
        )
        
        return alpha * residual + beta * new_output
```

## 四、6 个区域的 α/β 验证

假设训练后学到合理的权重：
- w_α_mag = 3.0, w_α_dir = 2.0
- w_β_mag = 3.0, w_β_dir = 2.0
- b_α = b_β = 3.0（从 5.0 下降，允许更多动态范围）

| 区域 | mag | dir | α_logit | α | β_logit | β | h_new |
|------|-----|-----|---------|---|---------|---|-------|
| A: h强+同向 | 0.94 | +1.0 | 3+2.82+2.0=7.82 | 1.00 | 3+0.18+1.88=5.06 | 0.99 | ≈h+o |
| B: h强+反向 | 0.94 | -1.0 | 3+2.82-2.0=3.82 | 0.98 | 3+0.18-1.88=1.30 | 0.79 | ≈h+0.79o |
| C: o强+同向 | 0.06 | +1.0 | 3+0.18+2.0=5.18 | 0.99 | 3+2.82+0.12=5.94 | 1.00 | ≈h+o |
| D: o强+反向 | 0.06 | -1.0 | 3+0.18-2.0=1.18 | 0.76 | 3+2.82-0.12=5.70 | 1.00 | ≈0.76h+o |
| E: 均衡+同向 | 0.50 | +1.0 | 3+1.50+2.0=6.50 | 1.00 | 3+1.50+1.00=5.50 | 1.00 | ≈h+o |
| F: 均衡+反向 | 0.50 | -1.0 | 3+1.50-2.0=2.50 | 0.92 | 3+1.50-1.00=3.50 | 0.97 | ≈0.92h+0.97o |

**关键验证**：
- 区域 B（h强+反向）：β=0.79，成功降低了反向干扰的接受度 ✅
- 区域 D（o强+反向）：β=1.00，α=0.76，成功实现了覆写 ✅
- 区域 E（协同）：α≈1, β≈1，等效于标准残差 ✅
- 区域 F（冲突）：α=0.92, β=0.97，轻微偏向接受新信息 ✅

## 五、与 V1 的表达能力对比

V1（低秩网络）可以学习 **任意** 从 [h, o, mag, dir] 到 [α, β] 的映射（受限于 rank=16 的瓶颈）。

V2（直接映射）只能学习 mag 和 dir 的 **线性组合** 经过 sigmoid 的映射。

**V2 缺失的能力**：
1. 不能学习 h 和 o 的原始值对 gate 的影响（只能通过 mag/dir 间接感知）
2. 不能学习 mag 和 dir 之间的非线性交互（除了我们手动设计的 `dir*mag` 调制项）
3. 不能学习跨维度的 gate 依赖（每个维度完全独立）

**V2 的优势**：
1. 物理可解释：每个参数都有明确的物理含义
2. 不会过拟合：参数极少，不会学到虚假的 gate 模式
3. 训练稳定：没有低秩瓶颈的梯度消失问题
4. 计算高效：纯逐元素运算，无矩阵乘法

**结论**：对于 residual gate 这个特定任务，V2 的表达能力是 **足够的**。因为 gate 的核心功能就是根据"谁更强"和"是否同向"来决定保留/接受比例，这正是 mag_ratio 和 dir_agree 直接编码的信息。
