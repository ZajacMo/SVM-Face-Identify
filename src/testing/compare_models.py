import os
import json
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
import sys
import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.utils import preprocess_image
from src.training.svm_classifier import SVMClassifier

class ModelComparer:
    def __init__(self):
        self.bp_model = None
        self.svm_model = None
        self.test_data = None
        self.test_labels = None
        self.class_to_idx = None

    def load_bp_model(self, model_path, labels_path):
        class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == 'utils' and name == 'BPNeuralNetwork':
                        from src.utils.utils import BPNeuralNetwork
                        return BPNeuralNetwork
                    return super().find_class(module, name)

        with open(model_path, 'rb') as f:
            self.bp_model = CustomUnpickler(f).load()
        with open(labels_path, 'rb') as f:
            self.class_to_idx = pickle.load(f)
        return self

    def load_svm_model(self, model_path):
        self.svm_model = SVMClassifier.load_model(model_path)
        return self

    def load_test_data(self, test_labels_path, image_size=(32, 32)):
        with open(test_labels_path, 'r') as f:
            test_samples = json.load(f)

        X = []
        y = []
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        for sample in test_samples:
            img_path = os.path.join(project_root, sample['path'])
            if os.path.exists(img_path):
                img_array = preprocess_image(img_path, image_size)
                X.append(img_array)
                y.append(self.class_to_idx[sample['label']])

        self.test_data = np.array(X)
        self.test_labels = np.array(y)
        return self

    def evaluate_bp(self):
        if self.bp_model is None or self.test_data is None:
            raise ValueError("BP模型或测试数据未加载")

        bp_predictions = []
        for img in self.test_data:
            output = self.bp_model.forward(img.reshape(1, -1))
            bp_predictions.append(np.argmax(output))

        bp_accuracy = np.mean(np.array(bp_predictions) == self.test_labels)
        return {
            'accuracy': bp_accuracy,
            'predictions': bp_predictions
        }

    def evaluate_svm(self):
        if self.svm_model is None or self.test_data is None:
            raise ValueError("SVM模型或测试数据未加载")

        svm_predictions = self.svm_model.predict(self.test_data)
        svm_accuracy = accuracy_score(self.test_labels, svm_predictions)
        return {
            'accuracy': svm_accuracy,
            'predictions': svm_predictions.tolist()
        }

    def compare(self, output_dir='results/comparison'):
        os.makedirs(output_dir, exist_ok=True)

        # 评估两个模型
        bp_results = self.evaluate_bp()
        svm_results = self.evaluate_svm()

        # 保存数值结果
        comparison_results = {
            'bp_accuracy': bp_results['accuracy'],
            'svm_accuracy': svm_results['accuracy'],
            'test_samples': len(self.test_labels),
            'comparison_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
            json.dump(comparison_results, f, indent=2)

        # 绘制准确率对比图
        # 设置中文字体支持，移除找不到的字体
        plt.rcParams["font.family"] = ["SimHei"]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        plt.figure(figsize=(10, 6))
        models = ['BP神经网络', 'SVM']
        accuracies = [bp_results['accuracy'], svm_results['accuracy']]

        plt.bar(models, accuracies, color=['blue', 'orange'])
        plt.ylim(0, 1.0)
        plt.ylabel('准确率')
        plt.title('BP神经网络与SVM分类准确率对比')
        plt.xticks(rotation=45)

        # 在柱状图上显示准确率数值
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.02, f'{v:.4f}', ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
        plt.close()

        print("模型比较完成！")
        print(f"BP神经网络准确率: {bp_results['accuracy']:.4f}")
        print(f"SVM准确率: {svm_results['accuracy']:.4f}")
        print(f"比较结果已保存至 {output_dir}")

def compare():
    comparer = ModelComparer()
    comparer.load_bp_model('results/face_model.pkl', 'results/class_labels.pkl')
    comparer.load_svm_model('results/svm_face_model.pkl')
    comparer.load_test_data('labels/test_labels.json')
    comparer.compare()


if __name__ == '__main__':
    compare()
