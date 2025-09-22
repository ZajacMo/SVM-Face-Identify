import os
import json
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.utils import preprocess_image

class SVMClassifier:
    def __init__(self, C=1.0, kernel='rbf', gamma='scale'):
        self.model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
        self.class_to_idx = None
        self.idx_to_class = None

    def load_dataset(self, label_file, image_size=(32, 32)):
        X = []
        y = []
        
        with open(label_file, 'r') as f:
            data = json.load(f)
        
        labels = [entry['label'] for entry in data]
        unique_labels = sorted(list(set(labels)), key=lambda x: int(x))
        self.class_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        for entry in data:
            img_path = entry['path']
            img_array = preprocess_image(img_path, image_size)
            X.append(img_array)
            y.append(self.class_to_idx[entry['label']])
        
        return np.array(X), np.array(y)

    def train(self, X, y):
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return accuracy

    def save_model(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'class_to_idx': self.class_to_idx,
                'idx_to_class': self.idx_to_class
            }, f)

    @classmethod
    def load_model(cls, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        classifier = cls()
        classifier.model = data['model']
        classifier.class_to_idx = data['class_to_idx']
        classifier.idx_to_class = data['idx_to_class']
        return classifier

# 训练脚本
def svm_train():
    TRAIN_LABELS = 'labels/train_labels.json'
    TEST_LABELS = 'labels/test_labels.json'
    MODEL_SAVE_PATH = 'results/svm_face_model.pkl'
    IMAGE_SIZE = (32, 32)
    
    # 初始化分类器
    classifier = SVMClassifier(C=10.0, kernel='rbf', gamma=0.001)
    
    # 加载数据
    print("Loading dataset...")
    X_train, y_train = classifier.load_dataset(TRAIN_LABELS, IMAGE_SIZE)
    X_test, y_test = classifier.load_dataset(TEST_LABELS, IMAGE_SIZE)
    
    # 训练模型
    print("Training SVM model...")
    classifier.train(X_train, y_train)
    
    # 评估模型
    train_accuracy = classifier.evaluate(X_train, y_train)
    test_accuracy = classifier.evaluate(X_test, y_test)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # 保存模型
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    classifier.save_model(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    svm_train()