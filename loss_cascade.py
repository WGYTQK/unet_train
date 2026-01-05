import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HierarchicalSegmentationLoss(nn.Module):
    """层次化分割损失函数，用于级联分割网络"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}

        # 各阶段损失权重
        self.primary_weight = self.config.get('primary_weight', 0.3)
        self.refine_weight = self.config.get('refine_weight', 0.5)
        self.consistency_weight = self.config.get('consistency_weight', 0.2)

        # 基础损失函数
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

        # 边界感知损失（用于精细分割）
        self.boundary_loss = BoundaryAwareLoss(
            alpha=self.config.get('boundary_alpha', 0.5)
        )

        # 小目标增强损失
        self.small_target_loss = SmallTargetAwareLoss(
            threshold=self.config.get('small_target_threshold', 0.05)
        )

    def forward(self, primary_pred, refine_pred, target):
        """
        计算层次化损失
        Args:
            primary_pred: 初级分割输出 [B, 2, H, W]
            refine_pred: 精细分割输出 [B, 2, H, W]
            target: 真实标签 [B, 1, H, W]
        Returns:
            损失字典
        """
        losses = {}

        # 1. 初级分割损失（相对宽松）
        primary_bce = self.bce_loss(primary_pred[:, 1:2, :, :], target)
        primary_dice = self.dice_loss(primary_pred[:, 1:2, :, :], target)
        losses['primary'] = 0.7 * primary_bce + 0.3 * primary_dice

        # 2. 精细分割损失（更严格）
        refine_bce = self.bce_loss(refine_pred[:, 1:2, :, :], target)
        refine_dice = self.dice_loss(refine_pred[:, 1:2, :, :], target)
        refine_boundary = self.boundary_loss(refine_pred[:, 1:2, :, :], target)
        refine_small = self.small_target_loss(refine_pred[:, 1:2, :, :], target)

        losses['refine'] = (
                0.4 * refine_bce +
                0.3 * refine_dice +
                0.2 * refine_boundary +
                0.1 * refine_small
        )

        # 3. 一致性损失（确保两个阶段结果一致）
        consistency_loss = self.compute_consistency_loss(
            primary_pred[:, 1:2, :, :],
            refine_pred[:, 1:2, :, :]
        )
        losses['consistency'] = consistency_loss

        # 4. 总损失
        losses['total'] = (
                self.primary_weight * losses['primary'] +
                self.refine_weight * losses['refine'] +
                self.consistency_weight * losses['consistency']
        )

        return losses

    def compute_consistency_loss(self, primary_pred, refine_pred):
        """计算一致性损失"""
        primary_prob = torch.sigmoid(primary_pred)
        refine_prob = torch.sigmoid(refine_pred)

        # 使用Jensen-Shannon散度（比KL散度更稳定）
        m = 0.5 * (primary_prob + refine_prob)

        js_loss = 0.5 * (
                F.kl_div(torch.log(primary_prob + 1e-8), m, reduction='batchmean') +
                F.kl_div(torch.log(refine_prob + 1e-8), m, reduction='batchmean')
        )

        return js_loss


class DiceLoss(nn.Module):
    """Dice损失函数"""

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_prob = torch.sigmoid(pred)

        intersection = torch.sum(pred_prob * target)
        union = torch.sum(pred_prob) + torch.sum(target)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class BoundaryAwareLoss(nn.Module):
    """边界感知损失"""

    def __init__(self, alpha=0.5, dilation_kernel=3):
        super().__init__()
        self.alpha = alpha
        self.dilation_kernel = dilation_kernel
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def get_boundary_mask(self, mask):
        """生成边界掩码"""
        device = mask.device
        kernel = torch.ones(1, 1, self.dilation_kernel, self.dilation_kernel).to(device)

        # 膨胀和腐蚀操作
        dilated = F.conv2d(mask.float(), kernel, padding=self.dilation_kernel // 2) > 0
        eroded = F.conv2d(mask.float(), kernel, padding=self.dilation_kernel // 2) < (self.dilation_kernel ** 2)

        # 边界 = 膨胀 - 腐蚀
        boundary = (dilated.float() - eroded.float()).clamp(0, 1)
        return boundary

    def forward(self, pred, target):
        # 计算边界掩码
        boundary_mask = self.get_boundary_mask(target)

        # 边界区域权重更高
        boundary_weight = 1.0 + self.alpha * boundary_mask

        # 计算加权损失
        bce_loss = self.bce(pred, target)
        weighted_loss = (bce_loss * boundary_weight).mean()

        return weighted_loss


class SmallTargetAwareLoss(nn.Module):
    """小目标感知损失"""

    def __init__(self, threshold=0.05):
        super().__init__()
        self.threshold = threshold
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dice = DiceLoss()

    def forward(self, pred, target):
        batch_size = pred.shape[0]
        device = pred.device

        total_loss = torch.tensor(0.0).to(device)
        valid_samples = 0

        for i in range(batch_size):
            target_i = target[i:i + 1]
            pred_i = pred[i:i + 1]

            # 计算目标面积比例
            target_area = torch.sum(target_i > 0.5).float()
            total_pixels = target_i.numel()
            area_ratio = target_area / total_pixels

            # 如果是小目标
            if area_ratio < self.threshold:
                # 使用更高的正样本权重
                pos_weight = torch.clamp(1.0 + 10.0 * target_i, max=20.0)

                # 计算加权BCE损失
                bce_loss = self.bce(pred_i, target_i)
                weighted_bce = (bce_loss * pos_weight).mean()

                # 计算Dice损失
                dice_loss = self.dice(pred_i, target_i)

                # 组合损失
                sample_loss = 0.6 * weighted_bce + 0.4 * dice_loss

                total_loss += sample_loss
                valid_samples += 1

        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(0.0).to(device)


class ProgressiveHierarchicalLoss(nn.Module):
    """渐进式层次化损失，根据训练阶段调整权重"""

    def __init__(self, total_epochs=1000, config=None):
        super().__init__()
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.config = config or {}

        # 基础损失函数
        self.base_loss = HierarchicalSegmentationLoss(config)

    def set_epoch(self, epoch):
        """设置当前epoch，用于调整损失权重"""
        self.current_epoch = epoch

    def get_progressive_weights(self):
        """根据训练进度获取权重"""
        progress = self.current_epoch / self.total_epochs

        if progress < 0.3:  # 早期训练阶段
            return {'primary': 0.4, 'refine': 0.4, 'consistency': 0.2}
        elif progress < 0.7:  # 中期训练阶段
            return {'primary': 0.3, 'refine': 0.5, 'consistency': 0.2}
        else:  # 后期训练阶段
            return {'primary': 0.2, 'refine': 0.6, 'consistency': 0.2}

    def forward(self, primary_pred, refine_pred, target):
        # 获取当前阶段权重
        progressive_weights = self.get_progressive_weights()

        # 计算基础损失
        losses = self.base_loss(primary_pred, refine_pred, target)

        # 应用渐进权重
        weighted_total = (
                progressive_weights['primary'] * losses['primary'] +
                progressive_weights['refine'] * losses['refine'] +
                progressive_weights['consistency'] * losses['consistency']
        )

        losses['total'] = weighted_total
        losses['progressive_weights'] = progressive_weights

        return losses


# 为了保持兼容性，也导出原有损失函数
class FocalLoss(nn.Module):
    """保持原有FocalLoss实现"""

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        # ... 保持原有实现

    def forward(self, logit, target):
        # ... 保持原有实现
        pass


class SoftDiceLoss(DiceLoss):
    """保持原有SoftDiceLoss名称"""
    pass


class SSIM(nn.Module):
    """保持原有SSIM实现"""
    pass


class MultiClassFocalLoss(nn.Module):
    """保持原有MultiClassFocalLoss实现"""
    pass
