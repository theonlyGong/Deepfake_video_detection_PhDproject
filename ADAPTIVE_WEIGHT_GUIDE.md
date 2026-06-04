# 自适应损失权重使用指南

## 功能总结

你的项目已实现以下功能：

### ✅ 已实现的功能

1. **视频分帧 + 人脸截取 + 数据增强**
   - 使用 `decord` 平均分帧（8帧）
   - SCRFD 人脸检测和对齐
   - 随机数据增强（Random Erasing、Random Crop、Albumentations 等）

2. **Swin Transformer 特征提取**
   - 返回局部特征（用于 consistency loss）
   - 返回全局特征（用于分类 CE loss）

3. **Consistency Loss + CE Loss**
   - Consistency Loss：计算视频帧间特征的一致性
   - CE Loss：分类交叉熵损失

4. **自适应损失权重（新增）** 🆕
   - 自动学习两个损失的权重
   - 不确定性高的损失权重自动降低
   - 权重参数通过 `backward()` 自动更新

---

## 使用方法

### 1. 使用自适应权重（推荐）

默认启用自适应权重：

```bash
python trainer3.py --adaptive-weight True --epochs 30 --batch 4
```

输出示例：
```
使用自适应损失权重，初始 log_var: [0.0, 0.0]
...
[Training epoch 0] loss: 0.823, acc: 0.612, w_con: 0.523, w_ce: 0.477
```

### 2. 使用固定权重（传统方式）

如果仍想使用固定权重：

```bash
python trainer3.py --adaptive-weight False --consistency-rate 0.5 --epochs 30
```

---

## 原理说明

### 自适应权重公式

参考论文 **"Multi-Task Learning Using Uncertainty to Weigh Losses"**

```
L_total = Σ (1 / 2*σ_i²) * L_i + log(σ_i)

其中:
- σ_i 是任务 i 的不确定性（可学习参数）
- 不确定性越高 → 权重越低
- log(σ_i) 是正则化项，防止权重趋于0
```

### 与传统固定权重对比

| 方式 | 优点 | 缺点 |
|------|------|------|
| **固定权重** | 简单、可控 | 需要手动调参、可能不是最优 |
| **自适应权重** | 自动学习最优权重、鲁棒性强 | 训练初期可能不稳定 |

---

## 关键代码说明

### 自适应权重模块 (`adaptive_loss.py`)

```python
class AdaptiveLossWeight(nn.Module):
    def __init__(self, num_losses=2):
        super().__init__()
        # 可学习的对数方差
        self.log_var = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_var[i])  # 1/σ²
            weighted_loss = precision * loss + self.log_var[i]
            total_loss += weighted_loss
        return total_loss, weights
```

### 训练中使用

```python
# 初始化
loss_weight = AdaptiveLossWeight(num_losses=2).to(device)
optimizer = torch.optim.AdamW([...model_params..., *loss_weight.parameters()], lr=lr)

# 训练循环
L_con = Cosine_similarity(...)
ce_loss = CrossEntropyLoss(...)
total_loss, weights = loss_weight([L_con, ce_loss])
total_loss.backward()  # 自动更新权重
```

---

## 训练技巧

### 1. 学习率调整

自适应权重参数通常需要较小的学习率：

```python
# 为模型和权重设置不同学习率
optimizer = torch.optim.AdamW([
    {'params': model.parameters(), 'lr': 0.0001},
    {'params': loss_weight.parameters(), 'lr': 0.00001}  # 权重参数学习率更小
])
```

### 2. 权重初始化

默认 `log_var = 0` 对应权重约 0.5。如果想偏向某个任务：

```python
# 初始偏向 CE loss
loss_weight.log_var.data[0] = 0.5   # Consistency 不确定性高
loss_weight.log_var.data[1] = 0.0   # CE 不确定性低
```

### 3. 监控权重变化

训练时观察权重变化，确保它们在学习：

```python
print(f"Weights: w_con={weights[0]:.3f}, w_ce={weights[1]:.3f}")
print(f"Log_vars: {loss_weight.log_var.data}")
```

---

## 故障排除

### Q: 权重不变化？
A: 确保权重参数被添加到优化器：
```python
pg.extend(loss_weight.parameters())
```

### Q: 权重都变成0？
A: 检查学习率是否过大，或添加梯度裁剪：
```python
torch.nn.utils.clip_grad_norm_(loss_weight.parameters(), max_norm=1.0)
```

### Q: 训练不稳定？
A: 尝试固定权重跑几轮预热，再切换到自适应权重。

---

## 参考文献

1. **"Multi-Task Learning Using Uncertainty to Weigh Losses"** (Kendall et al., 2018)
   - 提出基于不确定性的多任务损失权重学习方法

2. **"GradNorm: Gradient Normalization for Adaptive Loss Balancing"** (Chen et al., 2018)
   - 基于梯度大小的动态权重调整

3. **"Dynamic Weight Averaging"** (Liu et al., 2019)
   - 任务相关的动态权重平均方法