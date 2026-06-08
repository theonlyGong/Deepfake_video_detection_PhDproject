# Deepfake Video Detection

基于 Swin Transformer 和帧间一致性的深度伪造视频检测系统

## 📋 项目概述

本项目是一个针对 Deepfake 视频的检测系统，通过分析视频帧之间的一致性特征，结合 Swin Transformer 强大的特征提取能力，实现对伪造视频的有效识别。

### 核心特点

- 🎯 **视频级判定**：8帧聚合预测（求预测的平均logits值），输出视频整体真假判定
- 🔗 **帧间一致性**：引入 Consistency Loss，捕捉帧间特征关联
- ⚖️ **自适应权重**：自动学习 Consistency Loss 和 CE Loss 的最佳权重

## 新增功能：

- 🚀 **智能 GPU 选择**：自动选择空闲显存最多的 GPU
- 📊 **视频级准确率**：评估指标符合业务场景（视频级而非帧级）

---

## 🏗️ 系统架构

```
视频输入 → SCRFD人脸检测 → 8帧提取对齐 → Swin Transformer → 双任务输出
                    ↓
        Consistency Loss ← 帧间特征一致性
        CE Loss ← 交叉熵损失
                    ↓
        自适应权重融合 → 输出判定结果
```

---

## 🔬 技术细节

### 1. 视频预处理流程

```markdown
视频路径 → 平均提取8帧 → SCRFD人脸检测 → 人脸对齐 → 裁剪 → 数据增强
```

- **帧提取**：均匀采样8帧，覆盖整个视频时长
- **人脸检测**：SCRFD_10g 模型，检测人脸框和5个关键点
- **人脸对齐**：根据双眼关键点旋转对齐
- **人脸裁剪**：外扩0.2倍宽高，保留上下文
- **数据增强**：逐帧独立随机增强（每帧8帧各自独立增强）

### 2. Swin Transformer 特征提取

- **输入**：8张 224×224 RGB 图像
- **PatchEmbed**：4×4 Patch，输出 56×56 特征图
- **4个Stage**：逐级下采样（56×56 → 28×28 → 14×14 → 7×7）
- **输出**：
  - 全局特征：`[8, 768]`（8帧，每帧768维）
  - 分类预测：`[8, 2]`（8帧，每帧Real/Fake概率）

### 3. 双损失函数设计

#### Consistency Loss（一致性损失）

```python
# 计算8帧特征向量的余弦相似度
# 目标：真实视频帧间相似度高（特征一致）
#       伪造视频帧间相似度高（特征一致）

for feat_i, feat_j in combinations(features, 2):
    cos_sim = cosine_similarity(feat_i, feat_j)
    loss += 1 - cos_sim  

L_con = mean(all_pairs)
```

#### CE Loss（交叉熵损失）

```python
# 标准的分类损失
# 每帧独立计算，然后取平均

L_ce = CrossEntropyLoss(pred, label)
```

### 4. 自适应损失权重

```python
# 自动学习 w_con 和 w_ce
# 原理：不确定性高的任务权重自动降低

# 可学习参数
log_var = nn.Parameter(torch.zeros(2))  # [log_var_con, log_var_ce]

# 权重计算
w_con = exp(-log_var[0])
w_ce = exp(-log_var[1])

# 总损失
Total Loss = w_con × L_con + w_ce × L_ce + log_var[0] + log_var[1]
```

### 5. 视频级预测聚合

```python
# 8帧预测 → 视频预测

# Step 1: Softmax转概率
frame_probs = softmax(pred, dim=2)  # [batch, 8, 2]

# Step 2: 平均聚合
video_probs = frame_probs.mean(dim=1)  # [batch, 2]

# Step 3: 取最大概率类别
video_pred = argmax(video_probs, dim=1)  # [batch]

# Step 4: 与视频标签比较
accuracy = (video_pred == video_labels).sum() / batch_size
```

---

## 📁 项目结构

```
gly_deepfake_video_detection/
├── backbones/
│   ├── __init__.py
│   └── model.py              # Swin Transformer 模型class
├── scrfd_opencv_gpu/
│   ├── scrfd_face_detect.py  # SCRFD 人脸检测器
│   └── face_allign_crop.py   # 人脸对齐与裁剪
├── data_aug.py               # 数据增强方法，随机数据增强模式
├── dataset.py                # 视频数据集加载
├── loss.py                   # Consistency Loss
├── adaptive_loss.py          # 自适应损失权重
├── trainer.py                # 训练主程序
├── create_txt.py             # 生成数据列表
├── requirements.txt          # 依赖列表
├── install.bat               # Windows安装脚本
├── install.sh                # Linux/Mac安装脚本
└── README.md                 # 本文件
```

---

## 🚀 快速开始

### 1. 环境安装

```bash
# 方法1：使用自动安装脚本
# Windows:
install.bat

# Linux/Mac:
chmod +x install.sh
./install.sh

# 方法2：手动安装
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 数据准备

项目期望的数据结构：

```
data/
├── Real/
│   ├── Train/
│   ├── Validate/
│   └── Test/
└── Fake/
    ├── Train/
    ├── Validate/
    └── Test/
```

生成数据列表：

```bash
python create_txt.py
```

### 3. 开始训练

```bash
# 基础训练
python trainer.py --batch 4 --epochs 30

# 使用自适应权重（推荐）
python trainer.py --adaptive-weight True --batch 4 --epochs 30

# 使用固定权重
python trainer.py --adaptive-weight False --consistency-rate 0.5 --batch 4

# 指定GPU
python trainer.py --gpu-id 7 --batch 4

# 自动选择空闲GPU
python trainer.py --device auto --batch 4
```

---

## ⚙️ 可调整参数说明

### 训练参数

| 参数名 | 类型 | 默认值 | 说明 | 示例 |
|--------|------|--------|------|------|
| `--num_classes` | int | 2 | 类别数量（Real/Fake=2） | `--num_classes 2` |
| `--weights` | str | `./checkpoints/...` | 预训练权重路径 | `--weights ./model.pth` |
| `--freeze-layers` | bool | False | 是否冻结backbone | `--freeze-layers True` |
| `--epochs` | int | 30 | 训练轮数 | `--epochs 50` |
| `--lr` | float | 0.0001 | 学习率 | `--lr 0.0005` |
| `--batch` | int | 4 | Batch大小（视频数） | `--batch 8` |

### 损失权重参数

| 参数名 | 类型 | 默认值 | 说明 | 示例 |
|--------|------|--------|------|------|
| `--adaptive-weight` | bool | True | 是否使用自适应权重 | `--adaptive-weight True` |
| `--consistency_rate` | float | 0.5 | 固定权重时Consistency占比 | `--consistency_rate 0.6` |

**说明**：
- `adaptive-weight=True`：自动学习权重，consistency_rate 无效
- `adaptive-weight=False`：使用固定权重，total_loss = 0.5×L_con + 0.5×L_ce

### 设备参数

| 参数名 | 类型 | 默认值 | 说明 | 示例 |
|--------|------|--------|------|------|
| `--device` | str | `auto` | 设备类型 | `auto`/`cuda`/`cpu` |
| `--gpu-id` | int | None | 指定GPU编号 | `--gpu-id 7` |

**说明**：
- `--device auto`：自动选择空闲显存最多的GPU
- `--gpu-id 7`：强制使用7号GPU
- 不指定`--gpu-id`：自动选择

### 完整参数示例

```bash
python trainer.py \
    --batch 4 \
    --epochs 30 \
    --lr 0.0001 \
    --adaptive-weight True \
    --device auto \
    --weights ./checkpoints/swin_tiny_patch4_window7_224_22k.pth
```

---

## 🎨 数据增强策略

在 `data_aug.py` 中定义了5种增强策略，训练时**随机选择一种**应用于整个batch：

### 1. base（基础）
- Resize + ToTensor + Normalize

### 2. RE（Random Erasing）
- RandomHorizontalFlip
- RandomErasing（p=0.8）

### 3. RandCrop（随机裁剪）
- RandomResizedCrop
- RandomHorizontalFlip

### 4. DFDC_Selium（Albumentations组合）
- ImageCompression
- GaussNoise / GaussianBlur
- HorizontalFlip
- RandomBrightnessContrast / FancyPCA / HueSaturationValue
- ShiftScaleRotate

### 5. RA（Random Augment）
- 从3种增强中随机选一种

**重要**：每个视频的8帧**各自独立**进行随机增强

---

## 📊 训练日志解读

```
[INFO] Starting Deepfake Detection Training
[INFO] Arguments: Namespace(batch=4, epochs=30, ...)
[INFO] Using device: cuda:7

============================================================
Epoch 1/30 - Training
============================================================
[Epoch 1 Step 10/250] Loss: 0.6543, Video Acc: 0.6500, L_con: 0.1234, L_ce: 0.5312
...

[Epoch 1 Training Summary]
  Loss: 0.5432
  Video Accuracy: 0.6800    ← 视频级准确率
  Video AUC: 0.7234        ← 视频级AUC
  Weights - Consistency: 0.4876, CE: 0.5124

[Epoch 1 Validation Summary]
  Loss: 0.5123
  Video Accuracy: 0.7100
  Video AUC: 0.7543
  Best threshold: 0.5234
  TPR: 0.8234, FPR: 0.2134
```

### 关键指标说明

| 指标 | 含义 |
|------|------|
| `Video Acc` | 视频级准确率（batch个视频中正确判定的比例） |
| `Video AUC` | 视频级AUC（基于视频级平均概率计算） |
| `L_con` | Consistency Loss（帧间一致性损失） |
| `L_ce` | CE Loss（分类交叉熵损失） |
| `Weights` | 自适应学习的损失权重 |

---

## 🔧 常见问题

### Q1: 如何选择 consistency_rate？

**A**: 
- 如果 `adaptive-weight=True`，不需要设置，模型自动学习
- 如果 `adaptive-weight=False`，建议范围 0.3~0.7
  - 值越大，越重视帧间一致性
  - 值越小，越重视单帧分类

### Q2: Batch大小如何选择？

**A**:
- 显存充足（≥24GB）：`--batch 8`
- 显存有限（12GB）：`--batch 4`
- 显存很小（8GB）：`--batch 2`

注意：每个batch实际处理 `batch × 8` 帧

### Q3: 为什么使用视频级准确率而不是帧级？

**A**:
- **业务逻辑**：最终目标是判定视频真假，不是单帧
- **避免数据泄露**：同一视频的8帧高度相似，帧级统计会导致指标虚高
- **一致性聚合**：8帧投票比单帧更鲁棒

### Q4: 如何查看当前使用的GPU？

**A**: 训练日志会自动输出所有GPU的显存使用情况，并标注选择的GPU

---

## 📄 Published Paper

论文发布：[Swin-Fake: A Consistency Learning Transformer-Based Deepfake Video Detector](https://www.mdpi.com/2079-9292/13/15/3045)

