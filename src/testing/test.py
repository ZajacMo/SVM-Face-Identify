import os
import numpy as np
import pickle
import random
import json
import datetime
from PIL import Image
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.utils import preprocess_image

class FaceRecognitionTester:
    def __init__(self, model_path, labels_path):
        # 加载训练好的模型和标签
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(labels_path, 'rb') as f:
            self.class_to_idx = pickle.load(f)
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
    
    def predict(self, image_path, image_size=(32, 32)):
        # 预处理图像并进行预测
        img_array = preprocess_image(image_path, image_size)
        img_array = img_array.reshape(1, -1)
        output = self.model.forward(img_array)
        predicted_idx = np.argmax(output)
        confidence = output[0][predicted_idx]
        return self.idx_to_class[predicted_idx], confidence
    
    def automatic_test(self, test_labels_path, image_size=(32, 32)):
        # 自动测试模式：根据测试标签文件进行测试
        log_dir = 'plots'
        log_filename = 'test_log.txt'
        log_path = os.path.join(log_dir, log_filename)
        
        # 加载测试标签
        with open(test_labels_path, 'r', encoding='utf-8') as f:
            test_samples = json.load(f)
        
        correct = 0
        total = len(test_samples)
        results = []
        
        # 获取项目根目录
        project_root =  ""

        
        # 创建日志文件并写入测试信息
        os.makedirs(log_dir, exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"=== 自动测试模式 ===\n")
            log_file.write(f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"测试标签文件: {test_labels_path}\n")
            log_file.write(f"总测试样本数: {total}\n\n")
            
            for sample in test_samples:
                img_rel_path = sample['path']
                actual_person = sample['label']
                
                # 构造绝对路径
                img_path = os.path.join(project_root, img_rel_path)
                
                if not os.path.exists(img_path):
                    warning_msg = f"警告: 图片路径不存在 - {img_path}\n"
                    print(warning_msg.strip())  # 在控制台显示警告
                    log_file.write(warning_msg)
                    continue
                
                # 进行预测
                predicted_person, confidence = self.predict(img_path, image_size)
                
                # 显示结果
                result = "正确" if predicted_person == actual_person else "错误"
                log_file.write(f"图片: {img_rel_path}\n")
                log_file.write(f"实际人物: {actual_person}\n")
                log_file.write(f"预测人物: {predicted_person} (置信度: {confidence:.2f})\n")
                log_file.write(f"结果: {result}\n\n")
                
                # 收集结果数据
                results.append({
                    'image_path': img_rel_path,
                    'actual_person': actual_person,
                    'predicted_person': predicted_person,
                    'confidence': float(confidence),
                    'result': result
                })
                
                if result == "正确":
                    correct += 1
        
            # 写入测试总结
            log_file.write(f"\n=== 测试总结 ===\n")
            log_file.write(f"测试完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"总测试样本数: {total}\n")
            log_file.write(f"正确识别数: {correct}\n")
            if total > 0:
                accuracy = correct / total
                log_file.write(f"准确率: {accuracy:.2f} ({correct}/{total})\n")
            else:
                log_file.write("没有测试样本可供测试。\n")
                accuracy = 0
            log_file.write(f"测试结果已保存至 results\\plots\\test_results.json\n")
        
        # 控制台显示测试完成信息
        print(f"自动测试完成，日志文件已保存至: {log_path}")
        
        # 保存测试结果到JSON文件
        os.makedirs('results\\plots', exist_ok=True)
        with open('results\\plots\\test_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'accuracy': float(accuracy),
                'total_tests': total,
                'correct': correct,
                'test_cases': results,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, ensure_ascii=False, indent=2)
        print(f"测试结果已保存至 results\\plots\\test_results.json")
    
    def manual_test(self, image_size=(32, 32)):
        # 手动测试模式：用户输入图片路径
        print(f"=== 手动测试模式 ===")
        while True:
            img_path = input("请输入要测试的图片路径(输入'q'退出): ")
            if img_path.lower() == 'q':
                break
            
            if not os.path.exists(img_path):
                print("图片路径不存在，请重试!")
                continue
            
            try:
                predicted_person, confidence = self.predict(img_path, image_size)
                print(f"预测人物: {predicted_person}")
                print(f"置信度: {confidence:.2f}\n")
            except Exception as e:
                print(f"处理图片时出错: {str(e)}")

def test():
    # 配置参数
    MODEL_PATH = 'results/face_model.pkl'
    LABELS_PATH = 'results/class_labels.pkl'
    TEST_LABELS_PATH = 'labels/test_labels.json'
    IMAGE_SIZE = (32, 32)
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print("错误: 未找到模型文件，请先运行train.py训练模型!")
        exit(1)
    
    # 创建测试器实例
    tester = FaceRecognitionTester(MODEL_PATH, LABELS_PATH)
    
    # 选择测试模式
    print("欢迎使用人脸识别测试系统")
    print("1. 自动测试模式")
    print("2. 手动测试模式")
    
    while True:
        choice = input("请选择测试模式(1/2): ")
        if choice == '1':
            tester.automatic_test(TEST_LABELS_PATH, IMAGE_SIZE)
            break
        elif choice == '2':
            tester.manual_test(IMAGE_SIZE)
            break
        else:
            print("无效选择，请输入1或2!")


if __name__ == '__main__':
    test()