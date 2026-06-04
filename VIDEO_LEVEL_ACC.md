# 视频级准确率统计说明

## 目标
**视频级准确率统计**。

---

### 视频级统计
- **Batch=4** 时：**4个视频样本**
- 每个视频8帧预测结果**聚合**（概率平均）→ 得到1个视频预测
- 准确率的含义：4个视频中有多少个视频整体判定正确

---

## 核心修改逻辑

### 1. 预测聚合方式
```python
# 视频级聚合 [batch, 8, 2] → [batch, 2]
pred_probs = torch.softmax(pred, dim=2)        # [batch, 8, 2]
video_pred_probs = pred_probs.mean(dim=1)     # [batch, 2]
video_pred_classes = torch.argmax(video_pred_probs, dim=1)  # [batch]
```

### 2. 标签使用方式
```python
# 使用视频标签（每视频取第一帧标签，因为8帧标签相同）
video_labels = labels[:, 0].long().to(device)  # [batch]
```

### 3. 准确率计算
```python
# 视频级比较
accu_num += torch.eq(video_pred_classes, video_labels).sum()
sample_num += batch_size  # 4
```

### 4. Loss计算（保持不变）
```python
# Loss用帧级计算，以获得更好的梯度
pred_flat = torch.flatten(pred, 0, 1)  # [batch*8, 2]
labels_flat = torch.flatten(labels, 0, 1).long().to(device)
ce_loss = loss_function(pred_flat, labels_flat)
```

---

## AUC计算修改

### 修改后
- 收集每个视频的预测概率（8帧平均）→ 4个概率值
- 与4个视频标签计算AUC
- **符合场景**：判断整个视频的真假

---

### 优势
1. **符合业务需求**：评估的是视频级检测能力
2. **更合理的评估**：每个视频只算1个样本
3. **一致性聚合**：8帧结果投票/平均，提高鲁棒性

---

## 使用建议

### 训练时
- **Loss**：仍按帧级计算（为了更好的梯度传播）
- **准确率**：按视频级统计（为了真实评估）

### 推理时
- 对测试视频的8帧分别预测
- 将8帧概率取平均
- 平均概率 > 0.5 判定为 Fake

---

## 可能的影响

| 指标 | 预期变化 |
|------|----------|
| 训练准确率 | 可能下降（从帧级→视频级更严格） |
| 验证准确率 | 可能下降（同上） |
| AUC | 可能变化（样本数减少，但更符合实际） |
| 训练稳定性 | 可能提升（避免帧级过拟合） |

---

## 运行示例

```bash
python trainer.py --batch 4 --epochs 30
```

预期输出：
```
[INFO] Batch size: 4, Epochs: 30
...
[Epoch 1 Step 10/100] Loss: 0.6543, Video Acc: 0.6500, L_con: 0.1234, L_ce: 0.5312
...
[Epoch 1 Training Summary]
  Loss: 0.5432
  Video Accuracy: 0.6800    ← 视频级准确率
  Video AUC: 0.7234       ← 视频级AUC
  
[Epoch 1 Validation Summary]
  Loss: 0.5123
  Video Accuracy: 0.7100
  Video AUC: 0.7543
```