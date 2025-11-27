# -*- coding: utf-8 -*-
"""
环境检查脚本
验证所有依赖是否正确安装
"""

import sys

def check_python_version():
    """检查Python版本"""
    print("\n" + "="*60)
    print("检查Python版本...")
    print("="*60)
    
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor in [7, 8]:
        print("✓ Python版本符合要求 (3.7 或 3.8)")
        return True
    else:
        print("✗ 警告: 建议使用Python 3.7或3.8 (TensorFlow 1.10兼容性)")
        return False

def check_tensorflow():
    """检查TensorFlow/DirectML"""
    print("\n" + "="*60)
    print("检查TensorFlow/DirectML...")
    print("="*60)
    
    try:
        try:
            import tensorflow as tf
        except ImportError:
            import tensorflow_directml as tf
        print(f"✓ TensorFlow版本: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU可用: {gpus}")
        else:
            print("⚠ GPU不可用 (将使用CPU训练，速度较慢)")
        return True
    except ImportError:
        print("✗ TensorFlow/DirectML未安装")
        print("  Windows可安装: pip install tensorflow-directml")
        return False
    except Exception as e:
        print(f"✗ TensorFlow/DirectML检查失败: {e}")
        return False

def check_numpy():
    """检查NumPy"""
    print("\n" + "="*60)
    print("检查NumPy...")
    print("="*60)
    
    try:
        import numpy as np
        print(f"✓ NumPy版本: {np.__version__}")
        return True
    except ImportError:
        print("✗ NumPy未安装")
        return False

def check_networkx():
    """检查NetworkX"""
    print("\n" + "="*60)
    print("检查NetworkX...")
    print("="*60)
    
    try:
        import networkx as nx
        print(f"✓ NetworkX版本: {nx.__version__}")
        return True
    except ImportError:
        print("✗ NetworkX未安装")
        return False

def check_matplotlib():
    """检查Matplotlib"""
    print("\n" + "="*60)
    print("检查Matplotlib...")
    print("="*60)
    
    try:
        import matplotlib
        print(f"✓ Matplotlib版本: {matplotlib.__version__}")
        return True
    except ImportError:
        print("✗ Matplotlib未安装")
        return False

def check_project_structure():
    """检查项目结构"""
    print("\n" + "="*60)
    print("检查项目结构...")
    print("="*60)
    
    import os
    
    required_dirs = [
        'pvfp',
        'pvfp/cloud',
        'pvfp/domain',
        'pvfp/env',
        'pvfp/utils',
        'experiments',
        'tests',
        'visualization'
    ]
    
    required_files = [
        'config.py',
        'main.py',
        'requirements.txt',
        'README.md',
        'pvfp/cloud/decomposer.py',
        'pvfp/cloud/aggregator.py',
        'pvfp/domain/vnf_parallel.py',
        'pvfp/domain/dqn_agent.py',
        'pvfp/env/network_env.py',
        'pvfp/utils/topo_loader.py'
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ 目录存在: {dir_path}")
        else:
            print(f"✗ 目录缺失: {dir_path}")
            all_exist = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ 文件存在: {file_path}")
        else:
            print(f"✗ 文件缺失: {file_path}")
            all_exist = False
    
    return all_exist

def check_imports():
    """检查核心模块导入"""
    print("\n" + "="*60)
    print("检查核心模块导入...")
    print("="*60)
    
    try:
        from pvfp.domain.vnf_parallel import VNFParallelRules
        print("✓ vnf_parallel模块导入成功")
        
        from pvfp.cloud.decomposer import SFCDecomposer
        print("✓ decomposer模块导入成功")
        
        from pvfp.cloud.aggregator import FederatedAggregator
        print("✓ aggregator模块导入成功")
        
        from pvfp.domain.dqn_agent import DQNAgent
        print("✓ dqn_agent模块导入成功")
        
        from pvfp.env.network_env import VNFPlacementEnv
        print("✓ network_env模块导入成功")
        
        from pvfp.utils.topo_loader import TopologyLoader
        print("✓ topo_loader模块导入成功")
        
        return True
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 导入检查失败: {e}")
        return False

def create_log_directories():
    """创建日志目录"""
    print("\n" + "="*60)
    print("创建日志目录...")
    print("="*60)
    
    import os
    
    dirs = [
        'logs',
        'logs/models',
        'logs/results',
        'logs/plots'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ 创建/验证目录: {dir_path}")
    
    return True

def main():
    """主函数"""
    print("\n" + "#"*60)
    print("#" + " "*18 + "PVFP环境检查" + " "*18 + "#")
    print("#"*60)
    
    checks = [
        ("Python版本", check_python_version),
        ("TensorFlow", check_tensorflow),
        ("NumPy", check_numpy),
        ("NetworkX", check_networkx),
        ("Matplotlib", check_matplotlib),
        ("项目结构", check_project_structure),
        ("模块导入", check_imports),
        ("日志目录", create_log_directories)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name}检查失败: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "="*60)
    print("检查总结")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    print("\n" + "="*60)
    print(f"总计: {passed}/{total} 项检查通过")
    print("="*60)
    
    if passed == total:
        print("\n🎉 所有检查通过！环境配置正确，可以开始运行实验。")
        print("\n快速开始:")
        print("  python main.py")
        print("\n或运行测试:")
        print("  python tests/test_parallel_rules.py")
        print("  python tests/test_decomposer.py")
    else:
        print("\n⚠ 部分检查未通过，请根据上述提示安装缺失的依赖。")
        print("\n安装所有依赖:")
        print("  pip install -r requirements.txt")
    
    print("\n")

if __name__ == "__main__":
    main()
