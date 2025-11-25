# PVFP: 基于联邦深度强化学习的VNF并行部署

## 项目概述

本项目实现了论文 "Parallel Placement of Virtualized Network Functions via Federated Deep Reinforcement Learning" 中提出的PVFP框架。该框架通过联邦深度强化学习实现跨多个网络域的虚拟网络功能(VNF)并行部署优化。

## 核心特性

### 1. VNF并行化规则 (Rule 1, 2, 3)
- **Rule 1**: 执行顺序依赖检查
- **Rule 2**: 位置约束验证（first/mid/final）
- **Rule 3**: 优先级冲突解决

### 2. SFC分解算法 (Algorithm 1)
- 基于域资源比例的智能SFC分解
- 公式: `|Fi| = |F| × (Σ rcpu_v for v in Vi) / (Σ rcpu_v for all v)`

### 3. 域级DQN训练
- 3层全连接网络，每层600个神经元
- ReLU激活函数
- 经验回放机制
- 自适应ε-greedy探索策略: `ε_i = R_i_t / R_i_{t-1}`
- 软目标网络更新: `θ_target ← τ·θ_predict + (1-τ)·θ_target`

### 4. 联邦聚合 (Algorithm 2)
- 时滞加权聚合: `ς(ti - t1) = 1 / (ti - t1 + 1)^λ`
- 聚合公式: `Θ(t+1) = (1 - Σδi)·Θ(t) + Σ(δi·θi(t))`

## 项目结构

```
PVFP/
├── config.py                # 配置文件（超参数）
├── main.py                  # 主程序入口
├── requirements.txt         # 依赖包
├── README.md               # 本文件
├── pvfp/                   # 核心代码
│   ├── __init__.py
│   ├── cloud/              # 云端组件
│   │   ├── __init__.py
│   │   ├── decomposer.py   # SFC分解器
│   │   └── aggregator.py   # 联邦聚合器
│   ├── domain/             # 域级组件
│   │   ├── __init__.py
│   │   ├── vnf_parallel.py # 并行规则
│   │   └── dqn_agent.py    # DQN代理
│   ├── env/                # 环境
│   │   ├── __init__.py
│   │   └── network_env.py  # 网络环境
│   └── utils/              # 工具
│       ├── __init__.py
│       └── topo_loader.py  # 拓扑加载器
└── logs/                   # 日志和结果
    ├── models/             # 模型保存
    └── results/            # 结果保存
```

## 环境要求

### 硬件要求
- GPU: NVIDIA GPU (支持CUDA)
- 内存: 至少 8GB RAM
- 存储: 至少 2GB 可用空间

### 软件要求
- Python 3.7 或 3.8
- CUDA 9.0 + cuDNN 7.0 (匹配TensorFlow 1.10)
- TensorFlow GPU 1.10.0

## 安装步骤

### 1. 创建虚拟环境（推荐）

```bash
# 使用conda
conda create -n pvfp python=3.7
conda activate pvfp

# 或使用venv
python -m venv pvfp_env
# Windows
pvfp_env\Scripts\activate
# Linux/Mac
source pvfp_env/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证GPU环境

```python
import tensorflow as tf
print(tf.__version__)  # 应该显示 1.10.0
print(tf.test.is_gpu_available())  # 应该返回 True
```

## 使用方法

### 快速开始

```bash
python main.py
```

### 自定义配置

修改 `config.py` 中的参数：

```python
# 网络规模
SCALE = 'small'  # 或 'large'

# 域数量
NUM_DOMAINS = 4

# DQN超参数
LEARNING_RATE = 5e-4
BATCH_SIZE = 128
GAMMA = 0.99

# 联邦学习参数
LAMBDA_STALENESS = 5
DELTA_BASE = 0.9
AGGREGATION_EPOCHS = 100
LOCAL_EPOCHS = 10
```

### 运行完整实验

```python
from main import PVFPFramework

# 创建框架实例
pvfp = PVFPFramework(scale='small', num_domains=4)

# 执行联邦训练
training_results = pvfp.run_federated_training(
    num_sfcs=10,           # SFC请求数
    aggregation_rounds=50  # 聚合轮数
)

# 评估模型
evaluation_results = pvfp.evaluate(num_sfcs=20)

# 打印聚合统计
pvfp.aggregator.print_summary()
```

## 核心算法详解

### Algorithm 1: SFC分解

```
输入: 拓扑 G(V,E), 域划分 {G1, G2, ..., Gk}, SFC请求集合 C
输出: 每个域的SFC段

1. 计算每个域的可用CPU资源 rcpu_v
2. 对于每个SFC F:
   a. 计算总可用资源: R_total = Σ rcpu_v (所有域)
   b. 对于每个域 Gi:
      - 计算域资源比例: ratio_i = Σ rcpu_v (域i) / R_total
      - 分配VNF数量: |Fi| = ⌊|F| × ratio_i⌋
   c. 分配剩余VNF到资源最多的域
3. 返回分解后的SFC段
```

### Algorithm 2: 联邦聚合

```
输入: 各域模型权重 {θ1, θ2, ..., θk}, 时间戳 {t1, t2, ..., tk}
输出: 全局模型权重 Θ

1. 找到最早上传时间 t1 = min(t1, t2, ..., tk)
2. 对于每个域 i:
   a. 计算时滞因子: ς_i = 1 / ((ti - t1 + 1)^λ)
   b. 计算聚合权重: δ_i = ς_i × δ_base
3. 归一化权重: 确保 Σδ_i ≤ 1
4. 聚合: Θ(t+1) = (1 - Σδ_i) × Θ(t) + Σ(δ_i × θ_i(t))
5. 返回 Θ(t+1)
```

## 性能指标

### 训练指标
- **Loss**: DQN训练损失
- **Reward**: 平均奖励值
- **Epsilon**: 探索率变化

### 评估指标
- **端到端延迟 (End-to-End Latency)**: SFC部署的总延迟
- **资源开销 (Resource Cost)**: CPU和带宽使用量
- **部署成功率 (Success Rate)**: 成功部署的SFC比例

### 联邦学习指标
- **时滞因子 (Staleness Factor)**: 各域的时滞程度
- **聚合权重 (Aggregation Weight)**: 各域在聚合中的权重

## 实验配置

### 小规模网络
- 节点数: 12
- 链路数: 15
- CPU容量: 30 核/节点
- 带宽: 4 Mbps/链路

### 大规模网络
- 节点数: 35
- 链路数: 79
- CPU容量: 20 核/节点
- 带宽: 2 Mbps/链路

### VNF类型
7类VNF: NAT, LB, NIDS, Gateway, VPN, FW, Caching

### 超参数（论文配置）
- 学习率 α: 5×10^-4
- 批次大小: 128
- 折扣因子 γ: 0.99
- 软更新参数 τ: 0.01
- 时滞指数 λ: 5
- 基础权重 δ: 0.9

## 结果可视化

结果保存在 `logs/results/` 目录下，包含：
- 训练损失曲线
- 奖励变化曲线
- 延迟对比图
- 资源开销统计

可以使用以下代码绘制结果：

```python
import json
import matplotlib.pyplot as plt

# 加载结果
with open('logs/results/pvfp_results_XXXXXX.json', 'r') as f:
    results = json.load(f)

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(results['training_history']['losses'])
plt.xlabel('Aggregation Round')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('loss_curve.png')
plt.show()
```

## 故障排除

### 1. GPU不可用
```
错误: Could not find CUDA
解决: 安装匹配TensorFlow 1.10的CUDA 9.0和cuDNN 7.0
```

### 2. 内存不足
```
错误: ResourceExhaustedError
解决: 减小BATCH_SIZE或REPLAY_BUFFER_SIZE
```

### 3. 导入错误
```
错误: ModuleNotFoundError
解决: 确保在项目根目录运行，并正确安装所有依赖
```

## 论文引用

如果使用本代码，请引用原论文：

```
@article{pvfp2024,
  title={Parallel Placement of Virtualized Network Functions via Federated Deep Reinforcement Learning},
  journal={IEEE Transactions on Network and Service Management},
  year={2024},
  doi={10.1109/TNSM.2024.XXXXX}
}
```

## 许可证

本项目仅用于学术研究和教育目的。

## 联系方式

如有问题或建议，欢迎提出Issue。

---

**最后更新**: 2025年11月
