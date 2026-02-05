import os
import cv2
import numpy as np
from ultralytics import YOLO


def detect_and_draw_yolo(image_path, output_folder="./output"):
    """
    使用 YOLOv8 检测并画出边界框
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return False, None

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    try:
        # 加载模型
        model = YOLO('yolov8n.pt')

        # 推理
        results = model(img, verbose=False)

        person_detected = False

        for result in results:
            # 获取原始图像副本用于绘图
            img_with_boxes = img.copy()

            # 检查是否有"人"Q
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # 只处理人的检测结果
                    if cls == 0:  # 0 代表人
                        person_detected = True

                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # 绘制边界框
                        color = (0, 255, 0)  # 绿色框
                        thickness = 2
                        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)

                        # 添加标签（类别和置信度）
                        label = f"Person: {conf:.2f}"
                        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                        # 绘制标签背景
                        cv2.rectangle(img_with_boxes,
                                      (x1, y1 - label_size[1] - 5),
                                      (x1 + label_size[0], y1),
                                      color, -1)

                        # 绘制标签文字
                        cv2.putText(img_with_boxes, label,
                                    (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 0), 1)

                        # 在每个角点添加小圆点（可选，更美观）
                        cv2.circle(img_with_boxes, (x1, y1), 3, color, -1)
                        cv2.circle(img_with_boxes, (x2, y1), 3, color, -1)
                        cv2.circle(img_with_boxes, (x1, y2), 3, color, -1)
                        cv2.circle(img_with_boxes, (x2, y2), 3, color, -1)

                # 如果没有检测到人，显示提示信息
                if not person_detected:
                    text = "No Person Detected"
                    font_scale = 0.7
                    thickness = 2

                    # 计算文本位置（居中）
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    text_x = (img.shape[1] - text_size[0]) // 2
                    text_y = (img.shape[0] + text_size[1]) // 2

                    cv2.putText(img_with_boxes, text,
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (0, 0, 255), thickness)  # 红色文字

            # 保存带检测框的图片
            filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{name_without_ext}_detected.jpg")
            cv2.imwrite(output_path, img_with_boxes)

            print(f"检测结果已保存到: {output_path}")

            # 也可以显示图片（可选）
            # cv2.imshow('Detection Result', img_with_boxes)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            return person_detected, img_with_boxes

    except Exception as e:
        print(f"YOLO检测出错: {e}")
        return False, None


def process_test_folder_with_drawing(folder_path="./test", output_base="./output"):
    """
    遍历test文件夹中的所有图片文件并绘制检测结果
    """
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return

    print(f"开始检测文件夹: {folder_path}")
    print("=" * 60)

    stats = {
        'total_images': 0,
        'with_person': 0,
        'without_person': 0
    }

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件扩展名
            if any(file.lower().endswith(ext) for ext in image_extensions):
                file_path = os.path.join(root, file)
                stats['total_images'] += 1

                print(f"\n处理文件: {file_path}")

                # 获取相对路径以创建输出子文件夹
                rel_path = os.path.relpath(root, folder_path)
                if rel_path == ".":
                    rel_path = ""
                output_folder = os.path.join(output_base, rel_path)

                # 检测并绘制
                has_person, result_img = detect_and_draw_yolo(file_path, output_folder)

                if has_person:
                    stats['with_person'] += 1
                    status = "✅ 检测到人体"
                else:
                    stats['without_person'] += 1
                    status = "❌ 未检测到人体"

                print(f"  结果: {status}")

    # 打印统计信息
    print("\n" + "=" * 60)
    print("检测统计:")
    print(f"总图片数: {stats['total_images']}")
    print(f"有人图片: {stats['with_person']}")
    print(f"无人图片: {stats['without_person']}")
    print(f"检测率: {stats['with_person'] / stats['total_images'] * 100:.1f}%" if stats[
                                                                                      'total_images'] > 0 else "检测率: 0%")
process_test_folder_with_drawing()
