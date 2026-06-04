#!/bin/bash

echo "============================================"
echo " Deepfake Detection 项目依赖安装脚本"
echo " 使用清华源加速下载"
echo "============================================"
echo ""

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 Python3，请先安装 Python 3.8+"
    exit 1
fi

# 显示 Python 版本
echo "[INFO] 检测到 Python 版本："
python3 --version
echo ""

# 升级 pip
echo "[INFO] 正在升级 pip..."
python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
echo ""

# 配置清华源（可选，永久生效）
echo "[INFO] 配置清华源为默认源..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
echo ""

# 安装依赖
echo "[INFO] 开始安装项目依赖（使用清华源）..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

if [ $? -ne 0 ]; then
    echo ""
    echo "[错误] 安装过程中出现错误，请检查网络连接"
    exit 1
fi

echo ""
echo "============================================"
echo "[成功] 所有依赖安装完成！"
echo "============================================"
echo ""
echo "提示：如果需要使用GPU加速，请确保已安装CUDA"
echo "可用的CUDA版本可在 https://pytorch.org 查询"