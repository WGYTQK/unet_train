import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import random
from scipy import ndimage


class AdaptiveAugmentation:
    """自适应增强类"""
    
    @staticmethod
    def adaptive_color_jitter(image, target_area_ratio):
        """自适应颜色抖动"""
        # 小目标：增加对比度
        if target_area_ratio < 0.01:
            contrast = random.uniform(1.2, 1.5)
            brightness = random.uniform(0.9, 1.1)
        # 大目标：轻微调整
        elif target_area_ratio > 0.3:
            contrast = random.uniform(0.9, 1.1)
            brightness = random.uniform(0.95, 1.05)
        # 中等目标：正常调整
        else:
            contrast = random.uniform(0.8, 1.2)
            brightness = random.uniform(0.9, 1.1)
        
        # 应用调整
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
        image = cv2.convertScaleAbs(image, alpha=1.0, beta=(brightness - 1.0) * 50)
        
        return image
    
    @staticmethod
    def small_target_copy_paste(image, mask, max_copies=3):
        """小目标复制粘贴增强"""
        # 找到小目标
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        small_targets = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # 小目标阈值
                x, y, w, h = cv2.boundingRect(contour)
                small_targets.append({
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'area': area
                })
        
        if not small_targets or len(small_targets) < 2:
            return image, mask
        
        # 随机选择几个目标进行复制
        num_copies = min(len(small_targets), random.randint(1, max_copies))
        selected_targets = random.sample(small_targets, num_copies)
        
        image_copy = image.copy()
        mask_copy = mask.copy()
        h, w = image.shape[:2]
        
        for target in selected_targets:
            x, y, w_box, h_box = target['bbox']
            
            # 生成随机位置
            new_x = random.randint(0, w - w_box - 1)
            new_y = random.randint(0, h - h_box - 1)
            
            # 确保新位置不与原始位置重叠
            if abs(new_x - x) < w_box and abs(new_y - y) < h_box:
                continue
            
            # 复制图像和掩码区域
            image_roi = image[y:y+h_box, x:x+w_box]
            mask_roi = mask[y:y+h_box, x:x+w_box]
            
            # 粘贴到新位置
            image_copy[new_y:new_y+h_box, new_x:new_x+w_box] = image_roi
            mask_copy[new_y:new_y+h_box, new_x:new_x+w_box] = mask_roi
        
        return image_copy, mask_copy
    
    @staticmethod
    def continuous_target_deformation(image, mask, alpha_range=(80, 120), sigma_range=(8, 12)):
        """连续目标弹性形变"""
        h, w = image.shape[:2]
        
        # 生成随机位移场
        alpha = random.uniform(alpha_range[0], alpha_range[1])
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        
        # 生成随机位移
        dx = np.random.randn(h, w) * alpha
        dy = np.random.randn(h, w) * alpha
        
        # 应用高斯滤波使位移平滑
        from scipy.ndimage import gaussian_filter
        dx = gaussian_filter(dx, sigma)
        dy = gaussian_filter(dy, sigma)
        
        # 创建网格
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # 应用位移
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # 重映射图像和掩码
        deformed_image = cv2.remap(image, map_x, map_y, 
                                  interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
        deformed_mask = cv2.remap(mask, map_x, map_y,
                                 interpolation=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
        
        return deformed_image, deformed_mask


class EnhancedMVTecDRAEMTrainDataset(Dataset):
    """增强的MVTec训练数据集"""
    
    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        self.root_dir = root_dir
        self.mask_dir = anomaly_source_path
        self.resize_shape = resize_shape
        self.beilv = resize_shape[0] if resize_shape else 256
        
        # 自适应增强标志
        self.use_adaptive_aug = True
        self.small_target_threshold = 0.05  # 小目标面积阈值
        self.continuous_target_threshold = 0.1  # 连续目标面积阈值
        
        # 获取图像路径
        self.image_paths = sorted(glob.glob(root_dir + "/*.jpg") + 
                                  glob.glob(root_dir + "/*.png"))
        self.mask_paths = sorted(glob.glob(anomaly_source_path + "/*.jpg") + 
                                 glob.glob(anomaly_source_path + "/*.png"))
        
        # 统计目标特性
        self.target_stats = self.analyze_target_statistics()
    
    def analyze_target_statistics(self):
        """分析目标统计信息"""
        stats = {
            'total_images': len(self.image_paths),
            'small_targets': 0,
            'continuous_targets': 0,
            'medium_targets': 0
        }
        
        for img_path in self.image_paths[:100]:  # 采样分析
            mask_path = img_path.replace("image", "mask").replace(".jpg", ".png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = cv2.resize(mask, (self.beilv, self.beilv))
                    target_area = np.sum(mask > 0)
                    total_area = mask.shape[0] * mask.shape[1]
                    area_ratio = target_area / total_area
                    
                    if area_ratio < self.small_target_threshold:
                        stats['small_targets'] += 1
                    elif area_ratio > self.continuous_target_threshold:
                        stats['continuous_targets'] += 1
                    else:
                        stats['medium_targets'] += 1
        
        return stats
    
    def __len__(self):
        return len(self.image_paths)
    
    def adaptive_augmentation(self, image, mask):
        """自适应数据增强"""
        # 计算目标面积比例
        target_area = np.sum(mask > 0)
        total_area = mask.shape[0] * mask.shape[1]
        area_ratio = target_area / total_area
        
        # 根据目标类型应用不同的增强
        if area_ratio < self.small_target_threshold:
            # 小目标：应用复制粘贴增强
            if random.random() < 0.3:  # 30%概率应用
                image, mask = AdaptiveAugmentation.small_target_copy_paste(image, mask)
            
            # 增强对比度
            image = AdaptiveAugmentation.adaptive_color_jitter(image, area_ratio)
            
        elif area_ratio > self.continuous_target_threshold:
            # 连续目标：应用弹性形变
            if random.random() < 0.4:  # 40%概率应用
                image, mask = AdaptiveAugmentation.continuous_target_deformation(image, mask)
            
            # 轻微颜色调整
            image = AdaptiveAugmentation.adaptive_color_jitter(image, area_ratio)
        
        else:
            # 中等目标：应用常规增强
            if random.random() < 0.5:
                # 随机翻转
                if random.random() < 0.5:
                    image = cv2.flip(image, 1)
                    mask = cv2.flip(mask, 1)
                
                # 随机旋转
                angle = random.uniform(-15, 15)
                rows, cols = image.shape[:2]
                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                image = cv2.warpAffine(image, M, (cols, rows), 
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (cols, rows),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
        
        return image, mask
    
    def __getitem__(self, idx):
        # 随机选择图像增加数据多样性
        if random.random() < 0.3:  # 30%概率随机选择
            idx = random.randint(0, len(self.image_paths) - 1)
        
        # 加载图像
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        # 应用直方图均衡化
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(img_yuv)
        y_eq = cv2.equalizeHist(y)
        img_yuv_eq = cv2.merge((y_eq, cr, cb))
        image = cv2.cvtColor(img_yuv_eq, cv2.COLOR_YCrCb2BGR)
        
        # 加载掩码
        mask_path = img_path.replace("image", "mask").replace(".jpg", ".png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 调整尺寸
        image = cv2.resize(image, (self.beilv, self.beilv), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.beilv, self.beilv), interpolation=cv2.INTER_NEAREST)
        
        # 自适应增强
        if self.use_adaptive_aug:
            image, mask = self.adaptive_augmentation(image, mask)
        
        # 转换为张量
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        
        # 生成灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        
        return {
            'augment_image': image_tensor,
            'mask': mask_tensor,
            'auggray': gray_tensor,
            'image_path': img_path
        }


# 保持原有类名以兼容旧代码
MVTecDRAEMTrainDataset = EnhancedMVTecDRAEMTrainDataset
