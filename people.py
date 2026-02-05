import cv2
import numpy as np
import os
from pathlib import Path


def detect_person_opencv_dnn(image_path):
    """
    使用 OpenCV 预训练的 SSD 模型检测人体
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法加载图片 {image_path}")
        return False

    # 加载预训练的 MobileNet-SSD 模型
    # 请确保这些模型文件存在于当前目录
    model_path = "MobileNetSSD_deploy.caffemodel"
    config_path = "MobileNetSSD_deploy.prototxt"

    try:
        net = cv2.dnn.readNetFromCaffe(config_path, model_path)
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保以下文件存在:")
        print(f"1. {model_path}")
        print(f"2. {config_path}")
        return False

    # 准备输入图像
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)

    # 进行检测
    net.setInput(blob)
    detections = net.forward()

    # 检查是否有"人"被检测到（在 MobileNet-SSD 中，'person' 是第15类）
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        # 只检查"人"类别（通常 class_id = 15）
        if confidence > 0.5 and class_id == 15:  # 15 是 person 类别
            return True

    return False


import cv2



def process_test_folder(folder_path="./test"):
    """
    遍历test文件夹中的所有图片文件
    """
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return

    # 获取文件夹中的所有文件
    results = {}

    # 方法1: 使用os.walk遍历所有子文件夹
    print(f"开始检测文件夹: {folder_path}")
    print("=" * 50)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件扩展名
            if any(file.lower().endswith(ext) for ext in image_extensions):
                file_path = os.path.join(root, file)
                print(f"处理文件: {file_path}")

                # 检测图片中是否有人
                has_person = detect_person_opencv_dnn(file_path)
                results[file_path] = has_person

                status = "检测到人体" if has_person else "未检测到人体"
                print(f"  结果: {status}")
                print("-" * 30)

process_test_folder()