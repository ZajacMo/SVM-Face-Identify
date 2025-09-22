import os
import json
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.utils import BPNeuralNetwork, preprocess_image
import pickle

def load_casia_dataset(label_file, image_size=(32, 32)):
    X = []
    y = []
    
    # 读取标签文件
    with open(label_file, 'r') as f:
        data = json.load(f)
    
    # 提取所有唯一标签并排序
    labels = [entry['label'] for entry in data]
    unique_labels = sorted(list(set(labels)), key=lambda x: int(x))
    class_to_idx = {label: i for i, label in enumerate(unique_labels)}
    
    # 加载图像数据
    for entry in data:
        img_path = entry['path']
        img_array = preprocess_image(img_path, image_size)
        X.append(img_array)
        y.append(class_to_idx[entry['label']])
    
    # 转换为numpy数组
    X = np.array(X)
    y = np.array(y, dtype=np.int32)  # 确保y是整数类型
    
    # 对标签进行one-hot编码
    num_classes = len(unique_labels)
    y_onehot = np.zeros((y.shape[0], num_classes))
    y_onehot[np.arange(y.shape[0]), y] = 1
    
    return X, y_onehot, class_to_idx

def train():
    # 配置参数
    DATA_DIR = 'labels/train_labels.json'
    IMAGE_SIZE = (32, 32)
    HIDDEN_SIZE = 128
    EPOCHS = 1000
    LEARNING_RATE = 0.01
    BATCH_SIZE = 32
    MODEL_SAVE_PATH = 'results/face_model.pkl'
    LABELS_SAVE_PATH = 'results/class_labels.pkl'
    
    # 加载数据集
    print(f"Loading CASIA-FaceV5 dataset from {DATA_DIR}...")
    X, y, class_to_idx = load_casia_dataset(DATA_DIR, IMAGE_SIZE)
    print(f"Dataset loaded: {X.shape[0]} images, {len(class_to_idx)} classes")
    
    # 初始化神经网络
    input_size = IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3  # RGB图像
    output_size = len(class_to_idx)
    model = BPNeuralNetwork(input_size, HIDDEN_SIZE, output_size)
    
    # 训练模型
    print("Starting model training...")
    model.train(X, y, EPOCHS, LEARNING_RATE, BATCH_SIZE)
    
    # 保存模型和标签
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    with open(LABELS_SAVE_PATH, 'wb') as f:
        pickle.dump(class_to_idx, f)
    
    # 保存训练历史数据
    training_history = {
        'loss_history': model.loss_history,
        'learning_rate': model.learning_rate,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE
    }
    os.makedirs('results', exist_ok=True)
    with open('results/training_history.pkl', 'wb') as f:
        pickle.dump(training_history, f)
    
    print(f"模型已保存至 {MODEL_SAVE_PATH}")
    print(f"类别标签已保存至 {LABELS_SAVE_PATH}")
    print(f"训练历史数据已保存至 training_history.pkl")

if __name__ == '__main__':
    train()