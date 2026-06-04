# 项目依赖安装指南

## 快速安装（推荐）

### Windows 用户
双击运行 `install.bat` 文件，自动完成所有安装：
```cmd
install.bat
```

### Linux / Mac 用户
在终端中执行：
```bash
chmod +x install.sh
./install.sh
```

---

## 手动安装

如果你更喜欢手动安装，可以使用以下方法：

### 方法一：直接使用清华源安装
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 方法二：配置清华源为默认源（永久生效）
```bash
# 配置清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 之后可以直接使用
pip install -r requirements.txt
```

### 方法三：升级 pip 并安装
```bash
# 1. 升级 pip（使用清华源）
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## GPU 支持（可选）

如果你的电脑有 NVIDIA 显卡，建议安装 GPU 版本的 PyTorch：

### 查看 CUDA 版本
```bash
nvidia-smi
```

### 安装对应版本的 PyTorch

**CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**仅 CPU 版本:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 验证安装

安装完成后，可以运行以下命令验证关键依赖：

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import albumentations; print(f'Albumentations: {albumentations.__version__}')"
python -c "from decord import VideoReader; print('Decord: OK')"
```

---

## 常见问题

### 1. decord 安装失败
如果 decord 安装失败，可以尝试：
```bash
pip install decord -i https://pypi.tuna.tsinghua.edu.cn/simple
```
或者在 Windows 上使用：
```bash
pip install decord-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. opencv 安装失败
如果 OpenCV 安装出现问题，可以尝试分开安装：
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. albumentations 版本冲突
如果与其他包冲突，可以：
```bash
pip install albumentations --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install imgaug qudida -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 依赖列表

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | >=1.12.0 | 深度学习框架 |
| torchvision | >=0.13.0 | 视觉工具库 |
| opencv-python | >=4.6.0 | 计算机视觉 |
| opencv-contrib-python | >=4.6.0 | OpenCV扩展 |
| albumentations | >=1.3.0 | 图像增强 |
| decord | >=0.6.0 | 视频解码 |
| numpy | >=1.21.0 | 数值计算 |
| Pillow | >=9.0.0 | 图像处理 |
| scikit-learn | >=1.0.0 | 机器学习工具 |
| loguru | >=0.6.0 | 日志记录 |
| matplotlib | >=3.5.0 | 可视化 |
| onnxruntime | >=1.12.0 | ONNX推理 |
| onnxruntime-gpu | >=1.12.0 | GPU推理支持 |
| tqdm | >=4.64.0 | 进度条 |