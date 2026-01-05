import torch
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, jaccard_score
import glob
import cv2
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
from model_unetskip import AnomalySegmentationNetwork as CombinedNetwork, DepthwiseSeparableConv
import matplotlib.pyplot as plt


class EnhancedMetricsCalculator:
    """增强的指标计算器"""

    @staticmethod
    def calculate_detailed_metrics(target, pred_prob, img_path=""):
        """计算详细指标"""
        # 获取预测类别
        pred_class = (pred_prob > 0.5).astype(np.uint8)

        # 展平数组
        target_flat = target.flatten()
        pred_class_flat = pred_class.flatten()
        pred_prob_flat = pred_prob.flatten()

        metrics = {}

        try:
            # 基础指标
            metrics['precision'] = precision_score(target_flat, pred_class_flat, zero_division=1)
            metrics['recall'] = recall_score(target_flat, pred_class_flat, zero_division=1)
            metrics['f1'] = f1_score(target_flat, pred_class_flat, zero_division=1)
            metrics['iou'] = jaccard_score(target_flat, pred_class_flat, zero_division=1)
        except:
            metrics['precision'] = metrics['recall'] = metrics['f1'] = metrics['iou'] = 0.0

        # AUROC
        try:
            if np.unique(target_flat).size > 1:
                metrics['auroc'] = roc_auc_score(target_flat, pred_prob_flat)
            else:
                metrics['auroc'] = 0.5
        except:
            print(f"AUROC calculation failed for {img_path}")
            metrics['auroc'] = 0.5

        # 计算混淆矩阵相关指标
        try:
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(target_flat, pred_class_flat, labels=[0, 1]).ravel()

            metrics['specificity'] = tn / (tn + fp + 1e-8)  # 真阴性率
            metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn + 1e-8)
            metrics['fpr'] = fp / (fp + tn + 1e-8)  # 假阳性率
            metrics['fnr'] = fn / (fn + tp + 1e-8)  # 假阴性率
        except:
            metrics['specificity'] = metrics['accuracy'] = 0.0
            metrics['fpr'] = metrics['fnr'] = 0.0

        return metrics

    @staticmethod
    def calculate_small_target_metrics(target, pred_prob, small_area_threshold=100):
        """计算小目标相关指标"""
        import cv2

        pred_class = (pred_prob > 0.5).astype(np.uint8)

        # 找到真实小目标
        contours, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        small_targets = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < small_area_threshold:
                small_targets.append(contour)

        num_small_targets = len(small_targets)

        if num_small_targets == 0:
            return {
                'small_target_detection_rate': 0.0,
                'small_target_precision': 0.0,
                'small_target_recall': 0.0,
                'small_target_f1': 0.0
            }

        # 计算小目标检测率
        detected = 0
        for contour in small_targets:
            mask = np.zeros_like(target)
            cv2.drawContours(mask, [contour], -1, 1, -1)

            if np.any(pred_class & mask):
                detected += 1

        detection_rate = detected / num_small_targets

        # 计算小目标精度和召回率
        pred_contours, _ = cv2.findContours(pred_class, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        small_preds = []
        for contour in pred_contours:
            area = cv2.contourArea(contour)
            if area < small_area_threshold:
                small_preds.append(contour)

        true_positives = 0
        for pred_contour in small_preds:
            pred_mask = np.zeros_like(target)
            cv2.drawContours(pred_mask, [pred_contour], -1, 1, -1)

            if np.any(target & pred_mask):
                true_positives += 1

        if len(small_preds) > 0:
            precision = true_positives / len(small_preds)
        else:
            precision = 0.0

        recall = detected / num_small_targets

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return {
            'small_target_detection_rate': detection_rate,
            'small_target_precision': precision,
            'small_target_recall': recall,
            'small_target_f1': f1
        }

    @staticmethod
    def calculate_boundary_metrics(target, pred_prob, boundary_width=2):
        """计算边界相关指标"""
        # 二值化
        pred_class = (pred_prob > 0.5).astype(np.uint8)

        # 计算边界
        def get_boundary(mask):
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(mask, kernel, iterations=1)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            boundary = dilated - eroded
            return boundary

        pred_boundary = get_boundary(pred_class)
        target_boundary = get_boundary(target.astype(np.uint8))

        # 计算边界指标
        intersection = np.sum(pred_boundary * target_boundary)
        union = np.sum(pred_boundary) + np.sum(target_boundary)

        return {
            'boundary_iou': intersection / (union + 1e-8),
            'boundary_precision': intersection / (np.sum(pred_boundary) + 1e-8),
            'boundary_recall': intersection / (np.sum(target_boundary) + 1e-8)
        }


def create_enhanced_overlay(original, pred_prob, gt=None, alpha=0.3):
    """创建增强的透明叠加效果"""
    # 确保图像格式正确
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    # 创建热力图
    pred_normalized = cv2.normalize(pred_prob, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(pred_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    # 创建叠加
    overlay = original.copy()
    overlay = cv2.addWeighted(overlay, 1 - alpha, heatmap, alpha, 0)

    # 添加预测边界
    pred_mask = (pred_prob > 0.5).astype(np.uint8)
    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, pred_contours, -1, (0, 255, 255), 1)  # 黄色边界

    # 添加真实边界（如果有）
    if gt is not None:
        gt_contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 1)  # 绿色边界

    # 添加置信度文本
    confidence = np.max(pred_prob)
    cv2.putText(overlay, f"Conf: {confidence:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return overlay


def create_comparison_figure(original, pred_prob, gt, fname, save_dir):
    """创建对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 原始图像
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 真实掩码
    axes[0, 1].imshow(gt, cmap='gray')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')

    # 预测概率图
    im = axes[0, 2].imshow(pred_prob, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Prediction Probability')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])

    # 预测二值图
    pred_binary = (pred_prob > 0.5).astype(np.uint8)
    axes[1, 0].imshow(pred_binary, cmap='gray')
    axes[1, 0].set_title('Binary Prediction')
    axes[1, 0].axis('off')

    # 叠加效果
    overlay = create_enhanced_overlay(original, pred_prob, gt)
    axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Overlay')
    axes[1, 1].axis('off')

    # 误差图
    error = np.abs(pred_binary - gt)
    axes[1, 2].imshow(error, cmap='Reds', vmin=0, vmax=1)
    axes[1, 2].set_title('Error Map (FP: Red, FN: Blue)')
    axes[1, 2].axis('off')

    plt.suptitle(f'Image: {fname}', fontsize=16)
    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(save_dir, f"comparison_{fname}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def validate_model(model, args):
    """改进的验证函数"""
    model.eval()
    device = next(model.parameters()).device

    # 获取所有测试图像路径
    val_paths = sorted(glob.glob(f"{args.val_path}/*.png") +
                       glob.glob(f"{args.val_path}/*.jpg"))

    # 限制验证集大小
    val_paths = val_paths[:int(len(val_paths) * 0.4)]

    # 初始化指标收集器
    metrics_calculator = EnhancedMetricsCalculator()

    all_metrics = {
        'precision': [], 'recall': [], 'f1': [], 'auroc': [], 'iou': [],
        'specificity': [], 'accuracy': [], 'fpr': [], 'fnr': [],
        'small_target_detection_rate': [], 'small_target_precision': [],
        'small_target_recall': [], 'small_target_f1': [],
        'boundary_iou': [], 'boundary_precision': [], 'boundary_recall': []
    }

    # 创建可视化目录
    if args.save_visualization:
        vis_dir = os.path.join(args.output_dir, "validation_visualization")
        os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        for img_path in tqdm(val_paths, desc="Validating"):
            # 1. 加载并预处理图像
            image = cv2.imread(img_path)

            # 应用直方图均衡化
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(img_yuv)
            y_eq = cv2.equalizeHist(y)
            img_yuv_eq = cv2.merge((y_eq, cr, cb))
            image = cv2.cvtColor(img_yuv_eq, cv2.COLOR_YCrCb2BGR)

            # 调整尺寸
            image = cv2.resize(image, (args.height, args.height))
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0

            # 2. 生成灰度图像
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0

            # 3. 加载真实标签
            fname = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = f"{args.anomaly_source_path}/{fname}.png"

            if os.path.exists(gt_path):
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                gt = cv2.resize(gt, (args.height, args.height))
                gt = (gt > 0).astype(np.uint8)
            else:
                gt = np.zeros((args.height, args.height), dtype=np.uint8)

            # 4. 模型推理
            _, out_mask = model(image_tensor.to(device), gray_tensor.to(device))

            # 5. 处理模型输出
            if out_mask.shape[1] > 1:
                pred_prob = out_mask[:, 1, :, :].squeeze().cpu().numpy()
            else:
                pred_prob = out_mask.squeeze().cpu().numpy()

            # 应用sigmoid激活
            pred_prob = 1 / (1 + np.exp(-pred_prob))

            # 6. 计算详细指标
            basic_metrics = metrics_calculator.calculate_detailed_metrics(gt, pred_prob, img_path)

            # 计算小目标指标
            small_target_metrics = metrics_calculator.calculate_small_target_metrics(gt, pred_prob)

            # 计算边界指标
            boundary_metrics = metrics_calculator.calculate_boundary_metrics(gt, pred_prob)

            # 收集所有指标
            for key in basic_metrics:
                all_metrics[key].append(basic_metrics[key])

            for key in small_target_metrics:
                all_metrics[key].append(small_target_metrics[key])

            for key in boundary_metrics:
                all_metrics[key].append(boundary_metrics[key])

            # 7. 可视化保存
            if args.save_visualization:
                # 创建对比图
                create_comparison_figure(image, pred_prob, gt, fname, vis_dir)

                # 保存原始预测概率图
                pred_prob_scaled = (pred_prob * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(vis_dir, f"prob_{fname}.png"), pred_prob_scaled)

    # 计算平均指标
    avg_metrics = {}
    for key, values in all_metrics.items():
        if values:  # 确保列表不为空
            avg_metrics[key] = np.mean(values)
        else:
            avg_metrics[key] = 0.0

    # 打印主要指标
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Basic Metrics:")
    print(f"  Precision: {avg_metrics['precision']:.4f}")
    print(f"  Recall:    {avg_metrics['recall']:.4f}")
    print(f"  F1 Score:  {avg_metrics['f1']:.4f}")
    print(f"  IoU:       {avg_metrics['iou']:.4f}")
    print(f"  AUROC:     {avg_metrics['auroc']:.4f}")
    print(f"\nSmall Target Metrics:")
    print(f"  Detection Rate: {avg_metrics['small_target_detection_rate']:.4f}")
    print(f"  Precision:      {avg_metrics['small_target_precision']:.4f}")
    print(f"  Recall:         {avg_metrics['small_target_recall']:.4f}")
    print(f"  F1:             {avg_metrics['small_target_f1']:.4f}")
    print(f"\nBoundary Metrics:")
    print(f"  Boundary IoU:   {avg_metrics['boundary_iou']:.4f}")
    print(f"  Precision:      {avg_metrics['boundary_precision']:.4f}")
    print(f"  Recall:         {avg_metrics['boundary_recall']:.4f}")
    print("=" * 60)

    # 保存指标到文件
    metrics_file = os.path.join(args.output_dir, "validation_metrics.json")
    import json
    with open(metrics_file, 'w') as f:
        json.dump(avg_metrics, f, indent=4)

    # 返回主要指标
    return (
        avg_metrics['precision'],
        avg_metrics['recall'],
        avg_metrics['f1'],
        avg_metrics['auroc'],
        avg_metrics['iou']
    )
