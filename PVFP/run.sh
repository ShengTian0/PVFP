#!/bin/bash
# PVFP快速运行脚本 - Linux/Mac版本

echo "========================================"
echo "PVFP - 联邦深度强化学习VNF并行部署"
echo "========================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到Python，请先安装Python 3.7"
    exit 1
fi

echo "[1/3] 检查环境..."
python3 -c "import tensorflow" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[警告] TensorFlow未安装，正在安装依赖..."
    pip3 install -r requirements.txt
fi

echo "[2/3] 创建日志目录..."
mkdir -p logs/models
mkdir -p logs/results
mkdir -p logs/plots

echo "[3/3] 运行主程序..."
echo ""
python3 main.py

echo ""
echo "========================================"
echo "实验完成！"
echo "结果保存在 logs/results/ 目录"
echo "========================================"
