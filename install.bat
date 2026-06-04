@echo off
chcp 65001
cls
echo ============================================
echo  Deepfake Detection 项目依赖安装脚本
echo  使用清华源加速下载
echo ============================================
echo.

:: 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

:: 显示 Python 版本
echo [INFO] 检测到 Python 版本：
python --version
echo.

:: 升级 pip
echo [INFO] 正在升级 pip...
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
echo.

:: 配置清华源（可选，永久生效）
echo [INFO] 配置清华源为默认源...
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
echo.

:: 安装依赖
echo [INFO] 开始安装项目依赖（使用清华源）...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

if errorlevel 1 (
    echo.
    echo [错误] 安装过程中出现错误，请检查网络连接
    pause
    exit /b 1
)

echo.
echo ============================================
echo [成功] 所有依赖安装完成！
echo ============================================
echo.
echo 提示：如果需要使用GPU加速，请确保已安装CUDA
echo 可用的CUDA版本可在 https://pytorch.org 查询
echo.
pause