# ResidualGate V2 改进：连续值中间地带的处理

## 问题

实际训练中 mag_ratio 和 dir_agree 是连续值，大部分落在中间地带：
- mag_ratio ≈ 0.6~0.9（h 通常比 o 强，因为残差累积）
- dir_agree ≈ -0.3~0.3（h 和 o 的方向关系接近随机）

纯线性映射 `w * mag + w * dir + b` 在这个窄范围内的分辨率不足。

## 方案对比

### 方案 1：非线性预处理（推荐）

对 mag_ratio 和 dir_agree 做非线性变换，拉伸中间区域的分辨率：

```python
# 将 mag_ratio 从 [0,1] 映射到更有区分度的空间
# 使用 centered sigmoid-like 变换：2*(mag-0.5) 然后 tanh 拉伸
mag_centered = 2.0 * (mag_ratio - 0.5)  # [-1, 1]，中心在 0
# dir_agree 已经在 [-1, 1]

# 或者更简单：直接用 mag_ratio 和 (1-mag_ratio) 的对数比
# log_mag_ratio = log(|h| + eps) - log(|o| + eps)  # 无界，对数空间更均匀
```

**优势**：保持物理可解释性，增加中间区域分辨率
**劣势**：需要手动选择非线性函数

### 方案 2：极简低秩网络（只用 mag/dir 输入）

保留低秩结构，但输入只用 [mag_ratio, dir_agree]（2d）而非 [h, o, mag, dir]（4d）：

```python
# 输入：[mag_ratio, dir_agree] → 2d per dimension
# 但这是逐元素的，所以实际上是 (B, T, d, 2) → 需要跨维度的网络

# 方案 2a：per-dim 独立的小网络（不现实，参数太多）
# 方案 2b：共享的小网络，输入是 (mag, dir) 的 2 维标量
```

这个方向的问题是：mag_ratio 和 dir_agree 是 per-dim 的标量，如果要用网络处理，要么是 per-dim 独立的（参数爆炸），要么是跨维度共享的（但每个维度的 mag/dir 是独立的，跨维度没有意义）。

### 方案 3：混合方案 — 物理粗调 + 可学习细调

```python
# 粗调：物理信号直接映射（V2 的核心）
coarse_alpha = w_alpha_mag * mag_ratio + w_alpha_dir * dir_agree

# 细调：per-dim 可学习的非线性响应曲线
# 用 per-dim 的 scale 和 shift 参数调节 sigmoid 的形状
alpha = sigmoid(s_alpha * coarse_alpha + b_alpha)
#               ^^^^^^^
#               per-dim scale，控制该维度对物理信号的敏感度
```

**这个方案的关键洞察**：不同维度对 mag/dir 信号的敏感度应该不同。
- 某些维度可能对幅度比很敏感（s_alpha 大）
- 某些维度可能对方向一致性很敏感
- 某些维度可能几乎不需要 gate（s_alpha ≈ 0，纯靠 bias）

## 推荐方案：方案 3 的精炼版

```python
class ResidualGate(nn.Module):
    def __init__(self, hidden_size, init_bias=5.0):
        super().__init__()
        self.hidden_size = hidden_size
        
        # ---- α (retain gate) ----
        # 标量权重：物理信号的全局影响方向
        self.w_alpha_mag = nn.Parameter(torch.zeros(1))
        self.w_alpha_dir = nn.Parameter(torch.zeros(1))
        # per-dim 参数：每个维度的敏感度和基准
        self.s_alpha = nn.Parameter(torch.ones(hidden_size))   # 敏感度 scale
        self.b_alpha = nn.Parameter(torch.full((hidden_size,), init_bias))  # 基准偏置
        
        # ---- β (accept gate) ----
        self.w_beta_mag = nn.Parameter(torch.zeros(1))
        self.w_beta_dir = nn.Parameter(torch.zeros(1))
        self.s_beta = nn.Parameter(torch.ones(hidden_size))
        self.b_beta = nn.Parameter(torch.full((hidden_size,), init_bias))
    
    def forward(self, residual, new_output):
        eps = 1e-5
        h_mag = residual.detach().abs()
        o_mag = new_output.detach().abs()
        
        mag_ratio = h_mag / (h_mag + o_mag + eps)
        dir_agree = (residual.detach() * new_output.detach()) / (h_mag * o_mag + eps)
        
        # 物理信号的线性组合（标量权重，全局方向）
        alpha_signal = self.w_alpha_mag * mag_ratio + self.w_alpha_dir * dir_agree
        beta_signal = self.w_beta_mag * (1.0 - mag_ratio) + self.w_beta_dir * dir_agree
        
        # per-dim scale + bias → sigmoid
        # s 控制该维度对物理信号的敏感度
        # b 控制该维度的基准 gate 值
        alpha = torch.sigmoid(self.s_alpha * alpha_signal + self.b_alpha)
        beta = torch.sigmoid(self.s_beta * beta_signal + self.b_beta)
        
        return alpha * residual + beta * new_output
```

### 参数量
- 4 个标量权重 + 4×d 个 per-dim 参数 = 4 + 4d
- d=1024 → 4100 per gate，仍然远小于 V1 的 ~100K

### 初始化分析
- w=0, s=1, b=5.0
- 初始时：alpha_signal = 0, beta_signal = 0
- alpha = sigmoid(1.0 * 0 + 5.0) = sigmoid(5.0) = 0.993 ✅
- 训练后 s 分化：
  - s_j 大 → 第 j 维对物理信号敏感，gate 动态范围大
  - s_j 小 → 第 j 维对物理信号不敏感，gate 接近常数（由 b_j 决定）
  - s_j 负 → 第 j 维的 gate 响应反转（物理上可能有意义）

### 为什么 per-dim scale 解决了中间地带问题

当 mag_ratio 在 0.6~0.9 的窄范围内变化时：
- alpha_signal = w * mag_ratio 的变化范围也很窄（比如 w=3 时，变化 0.9）
- 但 s_alpha 可以放大这个变化：s=5 时，sigmoid 输入变化 4.5
- sigmoid 在 0 附近的梯度最大，所以 s 和 b 的配合可以让 sigmoid 的"敏感区"对准实际的 mag_ratio 分布

具体来说：
- 如果某维度的 mag_ratio 通常在 0.7 附近
- 模型可以学到 b_j ≈ -s_j * w * 0.7（让 sigmoid 的中心对准 0.7）
- 然后 s_j 控制在 0.7 附近的分辨率

## 与 V1 的最终对比

| 维度 | V1 低秩网络 | V2 精炼版 |
|------|------------|-----------|
| 参数量 | ~100K/gate | ~4K/gate |
| 输入 | [h, o, mag, dir] 4d | mag, dir 2个标量 |
| 网络结构 | Linear(4d→16) → Linear(16→d) | 标量权重 → per-dim scale+bias → sigmoid |
| 非线性 | 两层线性 + sigmoid | per-dim scale 提供自适应非线性 |
| 可解释性 | 黑盒 | 完全可解释 |
| 中间地带分辨率 | 取决于 rank | per-dim scale 自适应调节 |
| 梯度路径 | h,o 通过 gate 网络有梯度 | 纯 detached 信号，无梯度干扰 |
