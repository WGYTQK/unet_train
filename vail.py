import torch
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, jaccard_score
import glob
import cv2
import itertools
from typing import List, Tuple
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
from model_unetskip import AnomalySegmentationNetwork as CombinedNetwork, DepthwiseSeparableConv

def create_transparent_overlay(original: np.ndarray, pred_prob: np.ndarray) -> np.ndarray:
    """
    创建透明叠加效果，高亮显示预测目标区域
    Args:
        original: 原始BGR图像 (H, W, 3)
        pred_prob: 预测概率图 (H, W)
    Returns:
        np.ndarray: 透明叠加效果图像
    """
    # 创建高亮区域（红色）
    highlight = np.zeros_like(original)
    pred_mask = (pred_prob > 0.5).astype(np.uint8)
    highlight[pred_mask > 0] = [0, 0, 255]  # 红色高亮

    # 创建透明叠加
    overlay = original.copy()
    alpha = 0.2  # 透明度
    overlay = cv2.addWeighted(overlay, 1 - alpha, highlight, alpha, 0)

    # 添加轮廓
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)  # 黄色轮廓

    return overlay


# 将多个张量拼接成一个图像 (保持不变)
def reshape_tensors(tensors: List[np.ndarray], rows: int, cols: int) -> np.ndarray:
    assert len(tensors) == rows * cols, 'Number of tensors does not fit the shape'
    lists = [[tensors[i * cols + j] for j in range(cols)] for i in range(rows)]
    row_tensors = [np.concatenate(lists[i], axis=-1) for i in range(rows)]
    image_tensor = np.concatenate(row_tensors, axis=-2)
    return image_tensor


# 修改后的二分类指标计算函数
def calculate_metrics(target: np.ndarray, pred_prob: np.ndarray,img_path) -> Tuple[float, float, float, float, float]:
    """
    单类别分割评估指标计算
    Args:
        target: 真实标签 (H,W)，值为0(背景)或1(目标)
        pred_prob: 预测概率 (H,W)，值为[0,1]
    Returns:
        tuple: (precision, recall, f1, auroc, iou)
    """
    # 获取预测类别 (阈值为0.5)
    pred_class = (pred_prob > 0.5).astype(np.uint8)

    # 展平数组
    target_flat = target.flatten()
    pred_class_flat = pred_class.flatten()
    pred_prob_flat = pred_prob.flatten()

    # 计算Precision、Recall、F1
    precision = precision_score(target_flat, pred_class_flat, zero_division=1)
    recall = recall_score(target_flat, pred_class_flat, zero_division=1)
    f1 = f1_score(target_flat, pred_class_flat, zero_division=1)

    # 计算AUROC (需要正负样本都存在)
    if np.unique(target_flat).size > 1:
        try:
            auroc = roc_auc_score(target_flat, pred_prob_flat)
        except:
            auroc = 0.5
            print(img_path)
    else:
        auroc = 0.5  # 只有单一类别时设为0.5

    # 计算IoU (Jaccard指数)
    iou = jaccard_score(target_flat, pred_class_flat, zero_division=1)

    return precision, recall, f1, auroc, iou


# 修改后的验证函数
def validate_model(model: torch.nn.Module, args) -> Tuple[float, float, float, float, float]:
    model.eval()
    device = next(model.parameters()).device

    # 获取所有测试图像路径
    val_paths = sorted(glob.glob(f"{args.val_path}/*.png") +
                         glob.glob(f"{args.val_path}/*.jpg"))

    val_paths = val_paths[:int(len(val_paths)*0.4)]
    metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'auroc': [],
        'iou': []
    }
    num_name = 0
    with torch.no_grad():
        for img_path in tqdm(val_paths, desc="Validating"):
            # 1. 加载并预处理图像
            image = cv2.imread(img_path)
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            # 分离通道
            y, cr, cb = cv2.split(img_yuv)
            # 只对亮度通道进行均衡化
            y_eq = cv2.equalizeHist(y)
            # 合并通道
            img_yuv_eq = cv2.merge((y_eq, cr, cb))
            # 转换回BGR颜色空间
            image = cv2.cvtColor(img_yuv_eq, cv2.COLOR_YCrCb2BGR)
            image = cv2.resize(image, (args.height, args.height))
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0

            # 2. 生成灰度图像 (如果模型需要)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0

            # 3. 加载真实标签
            fname = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = f"{args.anomaly_source_path}/{fname}.png"
            if os.path.exists(gt_path):
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)


                # gt = np.where(gt > 0, 0, 255).astype(np.uint8)  # 确保输出是 uint8

                gt = cv2.resize(gt, (args.height, args.height))
                gt = (gt > 0).astype(np.uint8)  # 将标签二值化为0或1
            else:
                gt = np.zeros((args.height, args.height), dtype=np.uint8)  # 全背景


            _, out_mask = model(
                image_tensor.to(device),
                gray_tensor.to(device)
            )

            # 5. 处理模型输出 (单通道)
            if out_mask.shape[1] > 1:  # 如果输出多通道，取第一个通道
                pred_prob = out_mask[:, 1, :, :].squeeze().cpu().numpy()
            else:
                pred_prob = out_mask.squeeze().cpu().numpy()  # (H,W)

            # 6. 计算指标
            precision, recall, f1, auroc, iou = calculate_metrics(gt, pred_prob,img_path)

            # 存储结果
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['auroc'].append(auroc)
            metrics['iou'].append(iou)
            num_name = num_name+1
            # 7. 可视化保存 (可选)
            if args.save_visualization:
                # 创建可视化目录
                vis_dir = os.path.join(args.output_dir, "visualization")
                os.makedirs(vis_dir, exist_ok=True)

                # # 1. 创建热力图叠加效果
                # heatmap = create_heatmap_overlay(image, pred_prob)
                # cv2.imwrite(os.path.join(vis_dir, f"heatmap_{fname}.jpg"), heatmap)
                #
                # # 2. 创建并排对比图（原图 + 分割结果）
                # comparison = create_comparison_image(image, gt, pred_prob)
                # cv2.imwrite(os.path.join(vis_dir, f"comparison_{fname}.jpg"), comparison)

                # 3. 创建透明叠加效果
                overlay = create_transparent_overlay(image, pred_prob)
                cv2.imwrite(os.path.join(vis_dir, f"overlay_{num_name}.jpg"), overlay)

                # # 4. 保存原始预测概率图（用于进一步分析）
                # pred_prob_scaled = (pred_prob * 255).astype(np.uint8)
                # cv2.imwrite(os.path.join(vis_dir, f"prob_{num_name}.png"), pred_prob_scaled)

    # 计算平均指标
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print(f"Validation Results - "
          f"Precision: {avg_metrics['precision']:.4f}, "
          f"Recall: {avg_metrics['recall']:.4f}, "
          f"F1: {avg_metrics['f1']:.4f}, "
          f"AUROC: {avg_metrics['auroc']:.4f}, "
          f"IoU: {avg_metrics['iou']:.4f}")

    return avg_metrics['precision'],avg_metrics['recall'],avg_metrics['f1'],avg_metrics['auroc'],avg_metrics['iou']
