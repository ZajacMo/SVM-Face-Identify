import pickle
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings

# 忽略关于类别数量的警告
warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50% of the number of samples.")

def load_training_history(history_path='training_history.pkl'):
    """加载训练历史数据"""
    try:
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        return history
    except FileNotFoundError:
        print(f"错误: 未找到训练历史文件 {history_path}，请先运行train.py进行训练")
        return None


def plot_test_results(test_results_path='plots/test_results.json'):
    """绘制测试结果可视化图表"""
    if not os.path.exists(test_results_path):
        print(f"错误: 未找到测试结果文件 {test_results_path}")
        return

    # 加载测试结果
    with open(test_results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False

    # 1. 准确率展示
    plt.figure(figsize=(8, 6))
    plt.bar(['准确率'], [results['accuracy']], color='skyblue')
    plt.ylim(0, 1.0)
    plt.title('模型准确率')
    plt.ylabel('准确率')
    plt.text(0, results['accuracy'] + 0.02, f'{results["accuracy"]:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig('results/plots/accuracy.png', dpi=300)
    print("准确率图表已保存为 results/plots/accuracy.png")

    # 2. 预测结果分布
    actual = [case['actual_person'] for case in results['test_cases']]
    predicted = [case['predicted_person'] for case in results['test_cases']]
    confidence = [case['confidence'] for case in results['test_cases']]

    # 3. 置信度分布
    plt.figure(figsize=(10, 6))
    sns.histplot(confidence, bins=10, kde=True, color='green')
    plt.title('预测置信度分布')
    plt.xlabel('置信度')
    plt.ylabel('样本数量')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/plots/confidence_distribution.png', dpi=300)
    print("置信度分布图已保存为 results/plots/confidence_distribution.png")

    # 4. 混淆矩阵（仅显示前20个类别以保持清晰）
    unique_classes = list(set(actual + predicted))
    if len(unique_classes) > 20:
        unique_classes = unique_classes[:20]

    cm = confusion_matrix(actual, predicted, labels=unique_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title('混淆矩阵（前20个类别）')
    plt.xlabel('预测类别')
    plt.ylabel('实际类别')
    plt.tight_layout()
    plt.savefig('results/plots/confusion_matrix.png', dpi=300)
    print("混淆矩阵图已保存为 results/plots/confusion_matrix.png")

def plot_loss_curve(history):
    """绘制损失率随epoch变化的曲线"""
    if not history or 'loss_history' not in history:
        print("错误: 训练历史数据不完整，无法绘制损失曲线")
        return

    # 提取损失数据
    epochs, losses = zip(*history['loss_history'])
    learning_rate = history.get('learning_rate', '未知')
    total_epochs = history.get('epochs', len(epochs))
    batch_size = history.get('batch_size', '未知')

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', linewidth=2)
    plt.title(f'训练损失曲线 (学习率: {learning_rate}, 总epochs: {total_epochs}, 批大小: {batch_size})')
    plt.xlabel('Epoch')
    plt.ylabel('损失值 (交叉熵)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 保存图像
    plt.savefig('results/plots/loss_curve.png', dpi=300)
    print(f"损失曲线已保存为 results/plots/loss_curve.png")

    # 显示图像
    plt.close()


def plot_all():
    # 加载训练历史
    history = load_training_history()
    if history:
        # 绘制损失曲线
        plot_loss_curve(history)
        plot_test_results()
        print("所有可视化图表已保存到 plots 文件夹")
    else:
        print("无法绘制训练历史，程序退出")

if __name__ == '__main__':
    plot_all()