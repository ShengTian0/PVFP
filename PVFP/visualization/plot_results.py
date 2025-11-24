# -*- coding: utf-8 -*-
"""
结果可视化脚本
绘制训练损失、奖励、延迟等曲线
"""

import sys
sys.path.append('..')

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob


def load_latest_results(results_dir='../logs/results/'):
    """加载最新的结果文件"""
    
    result_files = glob(os.path.join(results_dir, 'pvfp_results_*.json'))
    
    if not result_files:
        print(f"未找到结果文件在 {results_dir}")
        return None
    
    # 获取最新的文件
    latest_file = max(result_files, key=os.path.getctime)
    
    print(f"加载结果文件: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results


def plot_loss_curve(losses, save_path='loss_curve.png'):
    """绘制损失曲线"""
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2, color='blue', marker='o', markersize=4)
    plt.xlabel('聚合轮次', fontsize=12)
    plt.ylabel('平均损失', fontsize=12)
    plt.title('训练损失曲线', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"损失曲线已保存: {save_path}")
    plt.close()


def plot_reward_curve(rewards, save_path='reward_curve.png'):
    """绘制奖励曲线"""
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, linewidth=2, color='green', marker='s', markersize=4)
    plt.xlabel('聚合轮次', fontsize=12)
    plt.ylabel('平均奖励', fontsize=12)
    plt.title('平均奖励曲线', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"奖励曲线已保存: {save_path}")
    plt.close()


def plot_combined_metrics(losses, rewards, save_path='combined_metrics.png'):
    """绘制组合指标图"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 损失曲线
    ax1.plot(losses, linewidth=2, color='blue', marker='o', markersize=4)
    ax1.set_xlabel('聚合轮次', fontsize=12)
    ax1.set_ylabel('平均损失', fontsize=12)
    ax1.set_title('训练损失', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 奖励曲线
    ax2.plot(rewards, linewidth=2, color='green', marker='s', markersize=4)
    ax2.set_xlabel('聚合轮次', fontsize=12)
    ax2.set_ylabel('平均奖励', fontsize=12)
    ax2.set_title('平均奖励', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"组合指标图已保存: {save_path}")
    plt.close()


def plot_training_summary(results, save_dir='../logs/plots/'):
    """绘制训练总结图表"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    training_history = results.get('training_history', {})
    losses = training_history.get('losses', [])
    rewards = training_history.get('rewards', [])
    
    if not losses or not rewards:
        print("警告: 结果数据为空，无法绘图")
        return
    
    print("\n" + "="*70)
    print("生成可视化图表")
    print("="*70 + "\n")
    
    # 绘制损失曲线
    plot_loss_curve(losses, os.path.join(save_dir, 'loss_curve.png'))
    
    # 绘制奖励曲线
    plot_reward_curve(rewards, os.path.join(save_dir, 'reward_curve.png'))
    
    # 绘制组合图
    plot_combined_metrics(losses, rewards, os.path.join(save_dir, 'combined_metrics.png'))
    
    # 打印统计信息
    print("\n" + "="*70)
    print("训练统计摘要")
    print("="*70)
    print(f"总训练轮次: {len(losses)}")
    print(f"最终损失: {losses[-1]:.4f}")
    print(f"最小损失: {min(losses):.4f} (轮次 {losses.index(min(losses))+1})")
    print(f"最终奖励: {rewards[-1]:.2f}")
    print(f"最大奖励: {max(rewards):.2f} (轮次 {rewards.index(max(rewards))+1})")
    print(f"平均损失: {np.mean(losses):.4f}")
    print(f"平均奖励: {np.mean(rewards):.2f}")
    print("="*70 + "\n")


def main():
    """主函数"""
    
    # 加载结果
    results = load_latest_results()
    
    if results is None:
        print("无法加载结果，请先运行训练")
        return
    
    # 生成图表
    plot_training_summary(results)
    
    print("可视化完成!")


if __name__ == "__main__":
    # 设置中文字体支持（可选）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    main()
