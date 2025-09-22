from src.data.data_split import split
from src.training.train import train
from src.training.svm_classifier import svm_train
from src.testing.test import test
from src.testing.compare_models import compare
from src.plotting.plot_training_history import plot_all

if __name__ == '__main__':
    print("=====================数据集划分=====================")
    split()
    print("=====================BP神经网络训练=====================")
    train()
    print("=====================BP神经网络测试=====================")
    test()
    print("=====================SVM模型=====================")
    svm_train()
    print("=====================模型比较=====================")
    compare()
    print("=====================图像绘制=====================")
    plot_all()
    
