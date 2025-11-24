@echo off
REM PVFP快速运行脚本 - Windows版本

echo ========================================
echo PVFP - 联邦深度强化学习VNF并行部署
echo ========================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.7
    pause
    exit /b 1
)

echo [1/3] 检查环境...
python -c "import tensorflow" >nul 2>&1
if errorlevel 1 (
    echo [警告] TensorFlow未安装，正在安装依赖...
    pip install -r requirements.txt
)

echo [2/3] 创建日志目录...
if not exist "logs\models" mkdir logs\models
if not exist "logs\results" mkdir logs\results
if not exist "logs\plots" mkdir logs\plots

echo [3/3] 运行主程序...
echo.
python main.py

echo.
echo ========================================
echo 实验完成！
echo 结果保存在 logs\results\ 目录
echo ========================================
pause
