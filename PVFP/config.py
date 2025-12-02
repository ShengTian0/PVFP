# -*- coding: utf-8 -*-
"""
PVFP配置文件
包含所有超参数和网络配置
"""

# ==================== 网络拓扑配置 ====================
# 小规模网络配置
SMALL_SCALE = {
    'nodes': 12,
    'links': 15,
    'cpu_capacity': 30,  # 每个节点的CPU核心数
    'bandwidth': 4,      # 每条链路带宽 (Mbps)
}

# 大规模网络配置
# 注意：对于 large 规模，我们在 TopologyLoader 中使用 Erdős–Rényi 随机图，
# 节点数固定为 50，边连接概率在生成函数中单独指定（参考 Algorithm2 实验配置）。
LARGE_SCALE = {
    'nodes': 50,
    'links': 79,
    'cpu_capacity': 20,
    'bandwidth': 2,
}

# ==================== VNF配置 ====================
# 7类VNF类型
VNF_TYPES = ['NAT', 'LB', 'NIDS', 'Gateway', 'VPN', 'FW', 'Caching']

# VNF执行时间 (ms)
VNF_EXEC_TIME_MIN = 5
VNF_EXEC_TIME_MAX = 10

# VNF资源需求
VNF_CPU_REQUIREMENT = 1  # 每个VNF需要的CPU核心数
VNF_MEMORY_REQUIREMENT = 512  # MB

# ==================== SFC配置 ====================
MAX_SFC_LENGTH = 10  # SFC最大长度
MIN_SFC_LENGTH = 3   # SFC最小长度
LARGE_SFC_LENGTH = 4  # large规模下SFC固定长度
LARGE_PARALLEL_NUM = 3  # large规模下并行功能数
LARGE_PARALLEL_POSITION = 1  # large规模下并行功能起始位置(与算法2配置对应, 1表示第一个功能)

# ==================== DQN超参数 ====================
# 网络结构
DQN_HIDDEN_LAYERS = 3
DQN_NEURONS_PER_LAYER = 600
DQN_ACTIVATION = 'relu'

# 训练超参数
LEARNING_RATE = 5e-4  # 论文使用的学习率
BATCH_SIZE = 128
GAMMA = 0.99  # 折扣因子
TAU = 0.01    # 软更新参数 φ

# 经验回放
REPLAY_BUFFER_SIZE = 128  # 可根据域资源调整
MIN_REPLAY_SIZE = 64

# 探索策略
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
# 关闭基于奖励的自适应ε，使用标准指数衰减ε-greedy
USE_ADAPTIVE_EPSILON = False

# ==================== 联邦学习配置 ====================
# 论文中默认将网络划分为3个联邦域
NUM_DOMAINS = 3  # 域的数量
AGGREGATION_EPOCHS = 100  # 全局聚合轮数
LOCAL_EPOCHS = 10  # 每个域的本地训练轮数（已根据loss收敛速度下调）

# 时滞参数
LAMBDA_STALENESS = 5  # 时滞衰减指数 λ
DELTA_BASE = 0.9      # 基础聚合权重 δ

# ==================== 延迟计算配置 ====================
BASE_TRANSMISSION_DELAY = 20  # 基准传输延迟 (ms)
PACKET_SIZE = 500             # 数据包大小 (bytes)
PACKET_HEADER = 60            # 包头大小 (bytes)
READ_WRITE_PER_BIT = 0.08     # 每比特读写时间 (ms)

# ==================== 实验配置 ====================
NUM_EXPERIMENTS = 30  # 独立实验次数
RANDOM_SEED = 42

# 日志和保存
LOG_INTERVAL = 10
SAVE_INTERVAL = 50
MODEL_SAVE_PATH = './logs/models/'
RESULT_SAVE_PATH = './logs/results/'

# ==================== 网络模型配置 ====================
FUNCTION_NODE_RATIO = 0.4
MAX_DEPLOYMENT_COST = 10
MAX_LINK_COST = 10
MIN_DEPLOY_BUDGET = 20
MAX_DEPLOY_BUDGET = 20

# 大规模网络下的部署成本与链路成本配置（参考 Algorithm2 实验）
MAX_LARGE_DEPLOYMENT_COST = 10
LARGE_USE_FIXED_DEPLOY_COST = False
LARGE_FIXED_DEPLOY_COST = 5
LARGE_USE_FIXED_LINK_COST = False
LARGE_FIXED_LINK_COST = 3

# ==================== GPU配置 ====================
USE_GPU = True
GPU_MEMORY_FRACTION = 0.8
