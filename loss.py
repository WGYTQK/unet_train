import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


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
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

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

    padd = window_size//2
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

    def forward(self, img1, img2,asloss=True):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        s_score, ssim_map = ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
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

#ssim.py
import numpy as np
import scipy.signal


def ssim_index_new(img1,img2,K,win):

    M,N = img1.shape

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    C1 = (K[0]*255)**2
    C2 = (K[1]*255) ** 2
    win = win/np.sum(win)

    mu1 = scipy.signal.convolve2d(img1,win,mode='valid')
    mu2 = scipy.signal.convolve2d(img2,win,mode='valid')
    mu1_sq = np.multiply(mu1,mu1)
    mu2_sq = np.multiply(mu2,mu2)
    mu1_mu2 = np.multiply(mu1,mu2)
    sigma1_sq = scipy.signal.convolve2d(np.multiply(img1,img1),win,mode='valid') - mu1_sq
    sigma2_sq = scipy.signal.convolve2d(np.multiply(img2, img2), win, mode='valid') - mu2_sq
    img12 = np.multiply(img1, img2)
    sigma12 = scipy.signal.convolve2d(np.multiply(img1, img2), win, mode='valid') - mu1_mu2

    if(C1 > 0 and C2>0):
        ssim1 =2*sigma12 + C2
        ssim_map = np.divide(np.multiply((2*mu1_mu2 + C1),(2*sigma12 + C2)),np.multiply((mu1_sq+mu2_sq+C1),(sigma1_sq+sigma2_sq+C2)))
        cs_map = np.divide((2*sigma12 + C2),(sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2*mu1_mu2 + C1
        numerator2 = 2*sigma12 + C2
        denominator1 = mu1_sq + mu2_sq +C1
        denominator2 = sigma1_sq + sigma2_sq +C2

        ssim_map = np.ones(mu1.shape)
        index = np.multiply(denominator1,denominator2)
        #如果index是真，就赋值，是假就原值
        n,m = mu1.shape
        for i in range(n):
            for j in range(m):
                if(index[i][j] > 0):
                    ssim_map[i][j] = numerator1[i][j]*numerator2[i][j]/denominator1[i][j]*denominator2[i][j]
                else:
                    ssim_map[i][j] = ssim_map[i][j]
        for i in range(n):
            for j in range(m):
                if((denominator1[i][j] != 0)and(denominator2[i][j] == 0)):
                    ssim_map[i][j] = numerator1[i][j]/denominator1[i][j]
                else:
                    ssim_map[i][j] = ssim_map[i][j]

        cs_map = np.ones(mu1.shape)
        for i in range(n):
            for j in range(m):
                if(denominator2[i][j] > 0):
                    cs_map[i][j] = numerator2[i][j]/denominator2[i][j]
                else:
                    cs_map[i][j] = cs_map[i][j]


    mssim = np.mean(ssim_map)
    mcs = np.mean(cs_map)

    return  mssim,mcs


#msssim.py
import numpy as np
import cv2


def msssim(img1,img2):

    K = [0.01,0.03]
    win  = np.multiply(cv2.getGaussianKernel(11, 1.5), (cv2.getGaussianKernel(11, 1.5)).T)  # H.shape == (r, c)
    level = 5
    weight = [0.0448,0.2856,0.3001,0.2363,0.1333]
    method = 'product'

    #M,N = img1.shape
    #H,W = win.shape

    downsample_filter = np.ones((2,2))/4
    #img1 = img1.astype(np.float32)
    #img2 = img2.astype(np.float32)
    img1= img1.cpu().numpy()
    img2 = img2.cpu().numpy()

    mssim_array = []
    mcs_array = []

    for i in range(0,level):
        mssim,mcs = ssim_index_new(img1,img2,K,win)
        mssim_array.append(mssim)
        mcs_array.append(mcs)
        filtered_im1 = cv2.filter2D(img1,-1,downsample_filter,anchor = (0,0),borderType=cv2.BORDER_REFLECT)
        filtered_im2 = cv2.filter2D(img2,-1,downsample_filter,anchor = (0,0),borderType=cv2.BORDER_REFLECT)
        img1 = filtered_im1[::2,::2]
        img2 = filtered_im2[::2,::2]

    print(np.power(mcs_array[:level-1],weight[:level-1]))
    print(mssim_array[level-1]**weight[level-1])
    overall_mssim = np.prod(np.power(mcs_array[:level-1],weight[:level-1]))*(mssim_array[level-1]**weight[level-1])

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
