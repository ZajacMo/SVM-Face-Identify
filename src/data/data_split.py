import os
import random
import json


def split_dataset(source_dir, label_dir, test_size_per_class=1):
    # 创建标签目录
    os.makedirs(label_dir, exist_ok=True)
    
    # 初始化标签列表
    train_labels = []
    test_labels = []
    class_names = []
    
    # 遍历每个人员目录
    for person in os.listdir(source_dir):
        person_path = os.path.join(source_dir, person)
        if not os.path.isdir(person_path):
            continue
        
        # 获取该人员的所有图片
        images = [f for f in os.listdir(person_path) if f.endswith('.bmp')]
        if len(images) <= test_size_per_class:
            # 如果图片数量不足，全部放入训练集
            test_images = []
            train_images = images
        else:
            # 随机选择test_size_per_class张图片作为测试集
            test_images = random.sample(images, test_size_per_class)
            train_images = [img for img in images if img not in test_images]
        
        class_names.append(person)
        
        # 收集训练集标签
        for img in train_images:
            img_path = os.path.join(person_path, img)
            train_labels.append({"path": img_path, "label": person})
        
        # 收集测试集标签
        for img in test_images:
            img_path = os.path.join(person_path, img)
            test_labels.append({"path": img_path, "label": person})


    # 保存类别名称
    with open(os.path.join(label_dir, 'classes.json'), 'w', encoding='utf-8') as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)
    
    # 保存训练集标签
    with open(os.path.join(label_dir, 'train_labels.json'), 'w', encoding='utf-8') as f:
        json.dump(train_labels, f, indent=2, ensure_ascii=False)
    
    # 保存测试集标签
    with open(os.path.join(label_dir, 'test_labels.json'), 'w', encoding='utf-8') as f:
        json.dump(test_labels, f, indent=2, ensure_ascii=False)


# if __name__ == '__main__':
def split():
    source_dir = '64_CASIA-FaceV5'
    label_dir = 'labels'
    split_dataset(source_dir, label_dir)
    print(f"类别标签已保存到{label_dir}目录，包含classes.json、train_labels.json和test_labels.json")