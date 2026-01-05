import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HierarchicalLoss(nn.Module):
    """分层损失函数，用于级联分割网络"""

    def __init__(self, primary_weight=0.3, refine_weight=0.7):
        super().__init__()
        self.primary_weight = primary_weight
        self.refine_weight = refine_weight

        # 初级损失组件
        self.primary_loss = nn.BCEWithLogitsLoss()

        # 精细损失组件（更严格）
        self.refine_bce = nn.BCEWithLogitsLoss()
        self.refine_dice = DiceLoss()
        self.refine_boundary = BoundaryAwareLoss()

    def forward(self, primary_pred, refine_pred, target):
        # 初级分割损失
        primary_loss = self.primary_loss(primary_pred[:, 1:2, :, :], target)

        # 精细分割损失（组合多个损失）
        refine_bce_loss = self.refine_bce(refine_pred[:, 1:2, :, :], target)
        refine_dice_loss = self.refine_dice(refine_pred[:, 1:2, :, :], target)
        refine_boundary_loss = self.refine_boundary(refine_pred[:, 1:2, :, :], target)

        refine_loss = refine_bce_loss + refine_dice_loss + 0.5 * refine_boundary_loss

        # 总损失
        total_loss = self.primary_weight * primary_loss + self.refine_weight * refine_loss

        return {
            'total': total_loss,
            'primary': primary_loss,
            'refine_bce': refine_bce_loss,
            'refine_dice': refine_dice_loss,
            'refine_boundary': refine_boundary_loss
        }


class AdaptiveSizeAwareLoss(nn.Module):
    """自适应大小感知损失"""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dice = DiceLoss()

    def compute_target_statistics(self, target):
        """计算目标统计信息"""
        batch_size = target.shape[0]
        stats = []

        for i in range(batch_size):
            target_i = target[i:i + 1]
            area = torch.sum(target_i > 0.5).float()
            total_pixels = target_i.numel()
            area_ratio = area / total_pixels

            # 计算紧凑度（周长/面积）
            contours = self.get_contours(target_i[0, 0].detach().cpu().numpy())
            if contours:
                perimeter = sum([cv2.arcLength(cnt, True) for cnt in contours])
                compactness = perimeter / (torch.sqrt(area) + 1e-8)
            else:
                compactness = torch.tensor(0.0)

            stats.append({
                'area_ratio': area_ratio.item(),
                'compactness': compactness.item(),
                'is_small': area_ratio < 0.01,
                'is_large': area_ratio > 0.3,
                'is_scattered': len(contours) > 10
            })

        return stats

    def forward(self, pred, target):
        target_stats = self.compute_target_statistics(target)
        batch_size = pred.shape[0]

        total_loss = 0
        for i in range(batch_size):
            pred_i = pred[i:i + 1]
            target_i = target[i:i + 1]
            stats = target_stats[i]

            # 根据目标特性调整损失
            if stats['is_small']:
                # 小目标：使用Dice损失，对类别不平衡更友好
                loss = self.dice(pred_i, target_i) * 1.5
            elif stats['is_large']:
                # 大目标：使用BCE + 边界约束
                bce_loss = self.bce(pred_i, target_i).mean()
                loss = bce_loss * 0.7 + self.dice(pred_i, target_i) * 0.3
            elif stats['is_scattered']:
                # 分散目标：强调每个单独目标
                loss = self.dice(pred_i, target_i) * 2.0
            else:
                # 正常目标：平衡损失
                loss = self.bce(pred_i, target_i).mean() + self.dice(pred_i, target_i)

            total_loss += loss

        return total_loss / batch_size


class ConsistencyLoss(nn.Module):
    """一致性损失，确保两个分割网络的结果一致"""

    def __init__(self, mode='kl'):
        super().__init__()
        self.mode = mode

    def forward(self, primary_pred, refine_pred):
        primary_prob = torch.softmax(primary_pred, dim=1)[:, 1:2, :, :]
        refine_prob = torch.softmax(refine_pred, dim=1)[:, 1:2, :, :]

        if self.mode == 'kl':
            # KL散度
            loss = F.kl_div(
                torch.log(primary_prob + 1e-8),
                refine_prob,
                reduction='batchmean'
            )
        elif self.mode == 'mse':
            # MSE
            loss = F.mse_loss(primary_prob, refine_prob)
        elif self.mode == 'correlation':
            # 相关性损失
            batch_size = primary_prob.shape[0]
            corr_loss = 0
            for i in range(batch_size):
                p = primary_prob[i].flatten()
                r = refine_prob[i].flatten()
                correlation = torch.corrcoef(torch.stack([p, r]))[0, 1]
                corr_loss += 1 - correlation
            loss = corr_loss / batch_size
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return loss


class ProgressiveLearningLoss(nn.Module):
    """渐进学习损失，随着训练进程调整损失权重"""

    def __init__(self, total_epochs=1000):
        super().__init__()
        self.total_epochs = total_epochs
        self.current_epoch = 0

        # 不同阶段的损失权重
        self.stage1_weights = {'primary': 0.5, 'refine': 0.3, 'consistency': 0.2}
        self.stage2_weights = {'primary': 0.3, 'refine': 0.5, 'consistency': 0.2}
        self.stage3_weights = {'primary': 0.2, 'refine': 0.6, 'consistency': 0.2}

        # 损失组件
        self.primary_loss = AdaptiveSizeAwareLoss()
        self.refine_loss = AdaptiveSizeAwareLoss()
        self.consistency_loss = ConsistencyLoss(mode='kl')

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_current_weights(self):
        """根据训练阶段调整权重"""
        progress = self.current_epoch / self.total_epochs

        if progress < 0.3:
            return self.stage1_weights  # 早期：关注初级分割
        elif progress < 0.7:
            return self.stage2_weights  # 中期：平衡关注
        else:
            return self.stage3_weights  # 后期：关注精细分割

    def forward(self, primary_pred, refine_pred, target):
        weights = self.get_current_weights()

        # 计算各个损失
        loss_primary = self.primary_loss(primary_pred[:, 1:2, :, :], target)
        loss_refine = self.refine_loss(refine_pred[:, 1:2, :, :], target)
        loss_consistency = self.consistency_loss(primary_pred, refine_pred)

        # 加权总损失
        total_loss = (
                weights['primary'] * loss_primary +
                weights['refine'] * loss_refine +
                weights['consistency'] * loss_consistency
        )

        return {
            'total': total_loss,
            'primary': loss_primary,
            'refine': loss_refine,
            'consistency': loss_consistency,
            'weights': weights
        }


class FocalDiceLoss(nn.Module):
    """Focal Dice Loss，结合了Focal Loss和Dice Loss的优点"""

    def __init__(self, gamma=2.0, alpha=0.25, smooth=1e-5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, pred, target):
        # 计算概率
        pred_prob = torch.sigmoid(pred)

        # Dice损失部分
        intersection = torch.sum(pred_prob * target)
        union = torch.sum(pred_prob) + torch.sum(target)
        dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)

        # Focal损失部分
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss.mean()

        # 组合损失
        combined_loss = dice_loss + focal_loss

        return combined_loss


class BoundaryAwareLoss(nn.Module):
    """改进的边界感知损失"""

    def __init__(self, alpha=0.5, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def get_boundary_mask(self, mask, kernel_sizes=[3, 5]):
        """生成多尺度边界掩码"""
        device = mask.device
        boundary_mask = torch.zeros_like(mask)

        for k_size in kernel_sizes:
            padding = k_size // 2
            kernel = torch.ones(1, 1, k_size, k_size).to(device)

            dilated = F.conv2d(mask.float(), kernel, padding=padding) > 0
            eroded = F.conv2d(mask.float(), kernel, padding=padding) < (k_size * k_size)
            boundary = (dilated.float() - eroded.float()).clamp(0, 1)

            # 根据边界宽度调整权重
            weight = 1.0 + self.alpha * (k_size / 3.0) * boundary
            boundary_mask = torch.max(boundary_mask, weight)

        return boundary_mask

    def forward(self, pred, target):
        # 边界权重
        boundary_weight = self.get_boundary_mask(target)

        # 加权BCE损失
        bce_loss = self.bce(pred, target)
        weighted_bce = (bce_loss * boundary_weight).mean()

        # 连续性约束
        pred_prob = torch.sigmoid(pred)

        # 水平连续性
        diff_h = torch.abs(pred_prob[:, :, 1:, :] - pred_prob[:, :, :-1, :])
        # 垂直连续性
        diff_v = torch.abs(pred_prob[:, :, :, 1:] - pred_prob[:, :, :, :-1])

        continuity_loss = diff_h.mean() + diff_v.mean()

        # 总损失
        total_loss = weighted_bce + self.beta * continuity_loss

        return total_loss


class DiceLoss(nn.Module):
    """Dice损失"""

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_prob = torch.sigmoid(pred)

        intersection = torch.sum(pred_prob * target)
        union = torch.sum(pred_prob) + torch.sum(target)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice

class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, reduction='mean'):
        """
        alpha: 类别权重张量 (形状 [num_classes])
        gamma: 难易样本调节因子
        reduction: 损失聚合方式 ('mean', 'sum' 或 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 输入校验
        assert inputs.dim() == 4, "Input must be 4D: (batch, channels, H, W)"
        assert targets.dim() == 4, "Target must be 4D: (batch, channels, H, W)"

        # 转换为分类任务标准格式
        batch_size, num_classes, H, W = inputs.shape
        inputs = inputs.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        inputs = inputs.view(-1, num_classes)  # [B*H*W, C]
        targets = targets.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        targets = targets.view(-1, num_classes)  # [B*H*W, C]

        # 计算 softmax 概率
        probs = F.softmax(inputs, dim=1)  # 概率归一化

        # 提取真实类别对应的概率
        true_class_probs = torch.sum(probs * targets, dim=1)  # [B*H*W]

        # 避免数值不稳定（log(0)）
        epsilon = 1e-7
        true_class_probs = torch.clamp(true_class_probs, min=epsilon, max=1 - epsilon)

        # 计算交叉熵基础项
        log_probs = -torch.log(true_class_probs)  # 标准交叉熵

        # Focal Loss 调节因子
        focal_weights = (1 - true_class_probs) ** self.gamma

        # 应用类别权重
        if self.alpha is not None:
            alpha_weights = torch.sum(targets * self.alpha, dim=1)  # 按类别提取权重
            focal_weights = alpha_weights * focal_weights

        # 计算最终损失
        loss = focal_weights * log_probs

        # 聚合损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        l = max_val - min_val
    else:
        l = val_range

    padd = window_size // 2
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    c1 = (0.01 * l) ** 2
    c2 = (0.03 * l) ** 2

    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret, ssim_map


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size).cuda()

    def forward(self, img1, img2, asloss=True):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        s_score, ssim_map = ssim(img1, img2, window=window, window_size=self.window_size,
                                 size_average=self.size_average)
        if asloss:
            return 1.0 - s_score
        else:
            return 1.0 - ssim_map


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, y):
        euclidean_distance = F.pairwise_distance(x1, x2)
        loss_contrastive = torch.mean((1 - y) * torch.pow(euclidean_distance, 2) +
                                      y * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


# ssim.py
import numpy as np
import scipy.signal


def ssim_index_new(img1, img2, K, win):
    M, N = img1.shape

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    C1 = (K[0] * 255) ** 2
    C2 = (K[1] * 255) ** 2
    win = win / np.sum(win)

    mu1 = scipy.signal.convolve2d(img1, win, mode='valid')
    mu2 = scipy.signal.convolve2d(img2, win, mode='valid')
    mu1_sq = np.multiply(mu1, mu1)
    mu2_sq = np.multiply(mu2, mu2)
    mu1_mu2 = np.multiply(mu1, mu2)
    sigma1_sq = scipy.signal.convolve2d(np.multiply(img1, img1), win, mode='valid') - mu1_sq
    sigma2_sq = scipy.signal.convolve2d(np.multiply(img2, img2), win, mode='valid') - mu2_sq
    img12 = np.multiply(img1, img2)
    sigma12 = scipy.signal.convolve2d(np.multiply(img1, img2), win, mode='valid') - mu1_mu2

    if (C1 > 0 and C2 > 0):
        ssim1 = 2 * sigma12 + C2
        ssim_map = np.divide(np.multiply((2 * mu1_mu2 + C1), (2 * sigma12 + C2)),
                             np.multiply((mu1_sq + mu2_sq + C1), (sigma1_sq + sigma2_sq + C2)))
        cs_map = np.divide((2 * sigma12 + C2), (sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_map = np.ones(mu1.shape)
        index = np.multiply(denominator1, denominator2)
        # 如果index是真，就赋值，是假就原值
        n, m = mu1.shape
        for i in range(n):
            for j in range(m):
                if (index[i][j] > 0):
                    ssim_map[i][j] = numerator1[i][j] * numerator2[i][j] / denominator1[i][j] * denominator2[i][j]
                else:
                    ssim_map[i][j] = ssim_map[i][j]
        for i in range(n):
            for j in range(m):
                if ((denominator1[i][j] != 0) and (denominator2[i][j] == 0)):
                    ssim_map[i][j] = numerator1[i][j] / denominator1[i][j]
                else:
                    ssim_map[i][j] = ssim_map[i][j]

        cs_map = np.ones(mu1.shape)
        for i in range(n):
            for j in range(m):
                if (denominator2[i][j] > 0):
                    cs_map[i][j] = numerator2[i][j] / denominator2[i][j]
                else:
                    cs_map[i][j] = cs_map[i][j]

    mssim = np.mean(ssim_map)
    mcs = np.mean(cs_map)

    return mssim, mcs


# msssim.py
import numpy as np
import cv2


def msssim(img1, img2):
    K = [0.01, 0.03]
    win = np.multiply(cv2.getGaussianKernel(11, 1.5), (cv2.getGaussianKernel(11, 1.5)).T)  # H.shape == (r, c)
    level = 5
    weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    method = 'product'

    # M,N = img1.shape
    # H,W = win.shape

    downsample_filter = np.ones((2, 2)) / 4
    # img1 = img1.astype(np.float32)
    # img2 = img2.astype(np.float32)
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()

    mssim_array = []
    mcs_array = []

    for i in range(0, level):
        mssim, mcs = ssim_index_new(img1, img2, K, win)
        mssim_array.append(mssim)
        mcs_array.append(mcs)
        filtered_im1 = cv2.filter2D(img1, -1, downsample_filter, anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
        filtered_im2 = cv2.filter2D(img2, -1, downsample_filter, anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
        img1 = filtered_im1[::2, ::2]
        img2 = filtered_im2[::2, ::2]

    print(np.power(mcs_array[:level - 1], weight[:level - 1]))
    print(mssim_array[level - 1] ** weight[level - 1])
    overall_mssim = np.prod(np.power(mcs_array[:level - 1], weight[:level - 1])) * (
                mssim_array[level - 1] ** weight[level - 1])

    return overall_mssim


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)

        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score
