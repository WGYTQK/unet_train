import torch
from data_loader import MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from model_unetskip import CascadeAnomalySegmentationNetwork as CombinedNetwork
# from loss import SSIM, FocalLoss, SoftDiceLoss, MultiClassFocalLoss,ProgressiveLearningLoss
import os
from loss_cascade import ProgressiveHierarchicalLoss
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from torchvision.transforms import ToPILImage
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import argparse
import json
from vail import validate_model
import logging
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')


def setup_logger(log_path):
    """增强的日志记录器设置"""
    log_file = os.path.join(log_path, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除之前的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 文件handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_lr(optimizer):
    """获取当前学习率"""
    return optimizer.param_groups[0]['lr']


def weights_init(m):
    """优化的权重初始化"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            # 使用Xavier初始化，对relu和leaky_relu都友好
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class AdaptiveBoundaryLoss(nn.Module):
    """自适应边界感知损失函数"""

    def __init__(self, base_alpha=0.5, adaptive_weight=True):
        super().__init__()
        self.base_alpha = base_alpha
        self.adaptive_weight = adaptive_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def compute_boundary_weights(self, mask, kernel_sizes=[3, 5, 7]):
        """计算多尺度边界权重"""
        device = mask.device
        boundary_weight = torch.zeros_like(mask)

        for k_size in kernel_sizes:
            padding = k_size // 2
            kernel = torch.ones(1, 1, k_size, k_size).to(device)

            # 计算边界
            dilated = F.conv2d(mask.float(), kernel, padding=padding) > 0
            eroded = F.conv2d(mask.float(), kernel, padding=padding) < (k_size * k_size)
            boundary = (dilated.float() - eroded.float()).clamp(0, 1)

            # 根据kernel大小给予不同权重
            weight = 1.0 + self.base_alpha * (k_size / 3.0) * boundary
            boundary_weight = torch.max(boundary_weight, weight)

        return boundary_weight

    def compute_area_adaptive_alpha(self, mask):
        """根据目标面积调整alpha值"""
        # 计算目标面积比例
        target_area = torch.sum(mask > 0.5).float()
        total_area = mask.numel()
        area_ratio = target_area / total_area

        # 小目标：增强边界权重
        # 大目标：适当降低边界权重
        if area_ratio < 0.01:  # 极小目标
            return self.base_alpha * 2.0
        elif area_ratio < 0.05:  # 小目标
            return self.base_alpha * 1.5
        elif area_ratio > 0.3:  # 大目标
            return self.base_alpha * 0.5
        else:
            return self.base_alpha

    def forward(self, pred, target):
        # 自适应调整alpha
        if self.adaptive_weight:
            alpha = self.compute_area_adaptive_alpha(target)
        else:
            alpha = self.base_alpha

        # 计算多尺度边界权重
        boundary_weight = self.compute_boundary_weights(target)

        # 计算基础损失
        base_loss = self.bce(pred, target)

        # 应用边界权重
        weighted_loss = (base_loss * boundary_weight).mean()

        # 连续性约束（改进版）
        pred_sigmoid = torch.sigmoid(pred)

        # 水平连续性
        diff_h = torch.abs(pred_sigmoid[:, :, 1:, :] - pred_sigmoid[:, :, :-1, :])
        # 垂直连续性
        diff_v = torch.abs(pred_sigmoid[:, :, :, 1:] - pred_sigmoid[:, :, :, :-1])

        # 只在边界区域加强连续性约束
        boundary_mask_h = boundary_weight[:, :, :-1, :]  # 水平边界
        boundary_mask_v = boundary_weight[:, :, :, :-1]  # 垂直边界

        continuity_loss = (diff_h * boundary_mask_h).mean() + \
                          (diff_v * boundary_mask_v).mean()

        # 形状保持损失（防止过分割）
        pred_binary = (pred_sigmoid > 0.5).float()
        target_binary = (target > 0.5).float()

        # 计算紧凑度损失（周长/面积）
        pred_boundary = self.compute_boundary_weights(pred_binary)
        target_boundary = self.compute_boundary_weights(target_binary)

        pred_perimeter = torch.sum(pred_boundary > 1.0)
        target_perimeter = torch.sum(target_boundary > 1.0)

        pred_area = torch.sum(pred_binary)
        target_area = torch.sum(target_binary)

        # 避免除零
        pred_compactness = pred_perimeter / (torch.sqrt(pred_area) + 1e-8)
        target_compactness = target_perimeter / (torch.sqrt(target_area) + 1e-8)

        compactness_loss = torch.abs(pred_compactness - target_compactness)

        total_loss = weighted_loss + 0.2 * continuity_loss + 0.1 * compactness_loss

        return total_loss


# class EnhancedSmallTargetLoss(nn.Module):
#     """增强的小目标感知损失"""
#
#     def __init__(self, thresholds=[0.01, 0.05, 0.1], alphas=[0.8, 0.6, 0.4]):
#         super().__init__()
#         self.thresholds = thresholds  # 面积阈值列表
#         self.alphas = alphas  # 对应权重
#         assert len(thresholds) == len(alphas), "阈值和权重数量必须相同"
#
#         self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
#         self.dice_loss = SoftDiceLoss()
#
#     def forward(self, pred, target):
#         batch_size = pred.shape[0]
#         device = pred.device
#
#         # 计算每个样本的目标面积比例
#         target_areas = torch.sum(target > 0, dim=[1, 2, 3]).float()
#         total_area = target.shape[2] * target.shape[3]
#         area_ratios = target_areas / total_area
#
#         total_loss = torch.tensor(0.0).to(device)
#         num_valid_samples = 0
#
#         for i in range(batch_size):
#             area_ratio = area_ratios[i].item()
#
#             # 确定适用的阈值和权重
#             applicable_alpha = 1.0  # 默认权重
#             for threshold, alpha in zip(self.thresholds, self.alphas):
#                 if area_ratio < threshold:
#                     applicable_alpha = alpha
#                     break
#
#             # 只对小目标样本应用增强损失
#             if area_ratio < self.thresholds[-1]:
#                 pred_i = pred[i:i + 1]
#                 target_i = target[i:i + 1]
#
#                 # 动态调整权重：目标越小，正样本权重越高
#                 pos_weight = torch.clamp(1.0 + 10.0 * target_i, max=20.0)
#
#                 # 计算加权BCE损失
#                 bce = self.bce_loss(pred_i, target_i)
#                 weighted_bce = (bce * pos_weight).mean()
#
#                 # 计算Dice损失
#                 dice = self.dice_loss(pred_i, target_i)
#
#                 # 组合损失
#                 sample_loss = applicable_alpha * weighted_bce + (1 - applicable_alpha) * dice
#
#                 # 添加梯度增强（对小目标特别重要）
#                 if area_ratio < 0.01:  # 极小的目标
#                     # 计算梯度匹配损失
#                     pred_grad_x = torch.abs(pred_i[:, :, :, 1:] - pred_i[:, :, :, :-1])
#                     pred_grad_y = torch.abs(pred_i[:, :, 1:, :] - pred_i[:, :, :-1, :])
#
#                     target_grad_x = torch.abs(target_i[:, :, :, 1:] - target_i[:, :, :, :-1])
#                     target_grad_y = torch.abs(target_i[:, :, 1:, :] - target_i[:, :, :-1, :])
#
#                     grad_loss = F.mse_loss(pred_grad_x, target_grad_x) + \
#                                 F.mse_loss(pred_grad_y, target_grad_y)
#
#                     sample_loss = sample_loss + 0.5 * grad_loss
#
#                 total_loss += sample_loss
#                 num_valid_samples += 1
#
#         if num_valid_samples > 0:
#             return total_loss / num_valid_samples
#         else:
#             return torch.tensor(0.0).to(device)
#
#
# class ContinuousTargetLoss(nn.Module):
#     """连续目标损失函数"""
#
#     def __init__(self, area_threshold=0.1, connectivity_weight=0.3):
#         super().__init__()
#         self.area_threshold = area_threshold
#         self.connectivity_weight = connectivity_weight
#         self.bce_loss = nn.BCEWithLogitsLoss()
#
#     def compute_connectivity_loss(self, pred, target):
#         """计算连通性损失"""
#         pred_binary = (torch.sigmoid(pred) > 0.5).float()
#
#         # 计算连通域数量差异（简化实现）
#         # 在实际应用中，可能需要使用CPU计算连通域
#         batch_size = pred.shape[0]
#         connectivity_loss = torch.tensor(0.0).to(pred.device)
#
#         for i in range(batch_size):
#             pred_i = pred_binary[i, 0].detach().cpu().numpy()
#             target_i = target[i, 0].detach().cpu().numpy()
#
#             # 这里使用形态学操作近似估计连通性
#             # 实际应用中可以使用cv2.connectedComponents
#             import cv2
#
#             # 计算孔洞数量（连通性的一个指标）
#             pred_inv = 1 - pred_i
#             target_inv = 1 - target_i
#
#             # 通过膨胀和腐蚀估计连通性
#             kernel = np.ones((3, 3), np.uint8)
#
#             pred_eroded = cv2.erode(pred_i.astype(np.uint8), kernel)
#             pred_dilated = cv2.dilate(pred_i.astype(np.uint8), kernel)
#
#             target_eroded = cv2.erode(target_i.astype(np.uint8), kernel)
#             target_dilated = cv2.dilate(target_i.astype(np.uint8), kernel)
#
#             # 计算形态学差异
#             pred_diff = np.sum(pred_dilated - pred_eroded)
#             target_diff = np.sum(target_dilated - target_eroded)
#
#             connectivity_loss += torch.abs(torch.tensor(pred_diff - target_diff).float())
#
#         return connectivity_loss / batch_size
#
#     def compute_area_constraint(self, pred, target):
#         """面积约束损失"""
#         pred_area = torch.sum(torch.sigmoid(pred))
#         target_area = torch.sum(target)
#
#         return torch.abs(pred_area - target_area) / target.numel()
#
#     def forward(self, pred, target):
#         # 计算基础损失
#         base_loss = self.bce_loss(pred, target)
#
#         # 计算连通性损失
#         connectivity_loss = self.compute_connectivity_loss(pred, target)
#
#         # 计算面积约束损失
#         area_loss = self.compute_area_constraint(pred, target)
#
#         # 组合损失
#         total_loss = base_loss + \
#                      self.connectivity_weight * connectivity_loss + \
#                      0.1 * area_loss
#
#         return total_loss
#
#
# class AdaptiveCompositeLoss(nn.Module):
#     """自适应复合损失函数"""
#
#     def __init__(self, config=None):
#         super().__init__()
#         self.config = config or {}
#
#         # 初始化各个损失组件
#         self.boundary_loss = AdaptiveBoundaryLoss(
#             base_alpha=self.config.get('boundary_alpha', 0.5),
#             adaptive_weight=True
#         )
#
#         self.small_target_loss = EnhancedSmallTargetLoss(
#             thresholds=[0.01, 0.05, 0.1],
#             alphas=[0.8, 0.6, 0.4]
#         )
#
#         self.continuous_target_loss = ContinuousTargetLoss(
#             area_threshold=0.1,
#             connectivity_weight=0.3
#         )
#
#         self.dice_loss = SoftDiceLoss()
#         self.focal_loss = FocalLoss(
#             apply_nonlin=torch.sigmoid,
#             alpha=[0.25, 0.75],  # 增加正样本权重
#             gamma=2,
#             smooth=1e-5,
#         )
#
#         # 动态权重参数
#         self.small_target_weight = self.config.get('small_target_weight', 0.3)
#         self.continuous_target_weight = self.config.get('continuous_target_weight', 0.2)
#         self.boundary_weight = self.config.get('boundary_weight', 0.3)
#         self.base_weight = self.config.get('base_weight', 0.2)
#
#     def analyze_target_type(self, target):
#         """分析目标类型"""
#         batch_size = target.shape[0]
#         target_types = []
#
#         for i in range(batch_size):
#             target_i = target[i:i + 1]
#
#             # 计算目标面积
#             target_area = torch.sum(target_i > 0.5).float()
#             total_area = target_i.numel()
#             area_ratio = target_area / total_area
#
#             # 判断目标类型
#             if area_ratio < 0.01:  # 极小目标
#                 target_types.append('tiny')
#             elif area_ratio < 0.05:  # 小目标
#                 target_types.append('small')
#             elif area_ratio > 0.3:  # 连续大目标
#                 target_types.append('continuous')
#             else:  # 中等目标
#                 target_types.append('medium')
#
#         return target_types
#
#     def forward(self, pred, target):
#         # 分析目标类型
#         target_types = self.analyze_target_type(target)
#
#         losses = {}
#
#         # 计算基础损失（所有样本）
#         losses['base'] = self.focal_loss(pred, target) + self.dice_loss(pred, target)
#
#         # 根据目标类型计算特定损失
#         batch_size = pred.shape[0]
#         specific_losses = []
#
#         for i in range(batch_size):
#             pred_i = pred[i:i + 1]
#             target_i = target[i:i + 1]
#             target_type = target_types[i]
#
#             if target_type in ['tiny', 'small']:
#                 # 小目标：应用增强的小目标损失
#                 loss = self.small_target_loss(pred_i, target_i)
#                 specific_losses.append(loss * self.small_target_weight)
#
#             elif target_type == 'continuous':
#                 # 连续目标：应用连续目标损失
#                 loss = self.continuous_target_loss(pred_i, target_i)
#                 specific_losses.append(loss * self.continuous_target_weight)
#
#             else:
#                 # 中等目标：应用边界损失
#                 loss = self.boundary_loss(pred_i, target_i)
#                 specific_losses.append(loss * self.boundary_weight)
#
#         # 计算特定损失的平均值
#         if specific_losses:
#             losses['specific'] = torch.stack(specific_losses).mean()
#         else:
#             losses['specific'] = torch.tensor(0.0).to(pred.device)
#
#         # 计算总损失
#         total_loss = losses['base'] * self.base_weight + losses['specific']
#
#         return total_loss


def create_output_dir(base_path):
    """创建带有时间戳的输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"train_{timestamp}"
    dir_path = os.path.join(base_path, dir_name)
    os.makedirs(dir_path, exist_ok=True)

    # 创建子目录
    os.makedirs(os.path.join(dir_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(dir_path, "visualization"), exist_ok=True)
    os.makedirs(os.path.join(dir_path, "logs"), exist_ok=True)

    return dir_path


class AdaptiveLearningRateScheduler:
    """自适应学习率调度器"""

    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config
        self.current_epoch = 0

        # 不同的调度策略
        self.scheduler_warmup = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: min(1.0, epoch / 5)
        )

        self.scheduler_cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get('T_0', 10),
            T_mult=1,
            eta_min=config.get('min_lr', 1e-6)
        )

        self.scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=config.get('min_lr', 1e-6)
        )

        self.current_scheduler = self.scheduler_warmup

    def step(self, metric=None):
        """执行一步调度"""
        self.current_epoch += 1

        # 前5个epoch使用warmup
        if self.current_epoch <= 5:
            self.current_scheduler = self.scheduler_warmup
        # 然后切换到plateau或cosine
        elif self.current_epoch > 5 and metric is not None:
            self.current_scheduler = self.scheduler_plateau
            self.current_scheduler.step(metric)
            return
        else:
            self.current_scheduler = self.scheduler_cosine

        self.current_scheduler.step()

    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


class GradientAccumulator:
    """梯度累积器，支持大批量训练"""

    def __init__(self, accumulation_steps=4):
        self.accumulation_steps = accumulation_steps
        self.counter = 0

    def should_update(self):
        """判断是否应该更新参数"""
        self.counter += 1
        return self.counter % self.accumulation_steps == 0

    def reset(self):
        """重置计数器"""
        self.counter = 0


class ModelEMA:
    """模型指数移动平均，提高模型稳定性"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 注册影子参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新影子参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """应用影子参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def train_on_device(args):
    # 1. 创建输出目录
    args.output_dir = create_output_dir(args.checkpoint_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. 初始化日志记录器和配置
    logger = setup_logger(os.path.join(args.output_dir, "logs"))
    logger.info(f"Training configuration:\n{json.dumps(vars(args), indent=4)}")

    # 3. 数据加载优化
    def create_data_loader():
        """创建数据加载器（支持动态数据加载）"""
        train_dataset = MVTecDRAEMTrainDataset(
            args.data_path + "/",
            args.anomaly_source_path,
            resize_shape=[args.height, args.height]
        )

        dataloader = DataLoader(
            train_dataset,
            batch_size=args.bs,
            shuffle=True,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
        )
        return dataloader

    dataloader = create_data_loader()

    # 4. 模型初始化
    model = CombinedNetwork().cuda()
    model.apply(weights_init)

    # 5. 优化器和学习率调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler = AdaptiveLearningRateScheduler(optimizer, {
        'T_0': args.T_0,
        'min_lr': args.lr * 0.01
    })

    # 6. 损失函数配置
    loss_config = {
        'small_target_weight': 0.3,
        'continuous_target_weight': 0.2,
        'boundary_weight': 0.3,
        'base_weight': 0.2,
        'boundary_alpha': 0.5
    }

    # composite_loss = AdaptiveCompositeLoss(loss_config)
    # progressive_loss = ProgressiveLearningLoss(total_epochs=args.epochs)
    # loss_config = {
    #     'primary_weight': 0.3,
    #     'refine_weight': 0.5,
    #     'consistency_weight': 0.2,
    #     'boundary_alpha': 0.5,
    #     'small_target_threshold': 0.05
    # }

    # 创建渐进式层次化损失
    hierarchical_loss = ProgressiveHierarchicalLoss(
        total_epochs=args.epochs,
        config=loss_config
    )
    # 7. 混合精度训练和梯度累积
    scaler = GradScaler()
    gradient_accumulator = GradientAccumulator(accumulation_steps=2)
    model_ema = ModelEMA(model, decay=0.999)
    to_pil = ToPILImage()

    # 8. 模型加载和检查
    start_epoch = 0
    if args.resume:
        try:
            checkpoint = torch.load(os.path.join(args.checkpoint_path, 'la1test_model.pckl'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {str(e)}")

    # 9. 训练状态跟踪
    metrics_history = defaultdict(list)
    best_metrics = {
        'f1': 0,
        'iou': 0,
        'auroc': 0,
        'epoch': 0,
        'loss': float('inf')
    }

    # 改进的早停机制
    class EnhancedEarlyStopping:
        def __init__(self, patience=50, min_delta=0, mode='max'):
            self.patience = patience
            self.min_delta = min_delta
            self.mode = mode
            self.counter = 0
            self.best_score = None
            self.early_stop = False

        def __call__(self, current_score):
            if self.best_score is None:
                self.best_score = current_score
                return False

            if self.mode == 'max':
                improvement = current_score - self.best_score
            else:
                improvement = self.best_score - current_score

            if improvement > self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

            return self.early_stop

    early_stopping = EnhancedEarlyStopping(patience=args.patience, mode='max')

    # 10. 增强的训练循环
    for epoch in range(start_epoch, args.epochs):
        # 每20个epoch重新加载数据，增加数据多样性
        if epoch % 20 == 0 and epoch > 0:
            logger.info("Reloading dataset for diversity...")
            dataloader = create_data_loader()

        model.train()
        epoch_loss = 0.0
        epoch_metrics = defaultdict(float)

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch_idx, sample_batched in enumerate(progress_bar):
            # 数据加载
            augment_image = sample_batched["augment_image"].cuda(non_blocking=True)
            mask = sample_batched["mask"].cuda(non_blocking=True)
            auggray = sample_batched["auggray"].cuda(non_blocking=True)

            # 混合精度训练
            with autocast():

                primary_output, refined_output = model(augment_image, auggray)
                # loss_dict = progressive_loss(primary_output, refined_output, mask)
                loss_dict = hierarchical_loss(primary_output, refined_output, mask)
                loss = loss_dict['total']


                # 使用自适应复合损失
                # loss = composite_loss(out_mask[:, 1:2, :, :], mask)
                epoch_metrics['primary_loss'] += loss_dict['primary'].item()
                epoch_metrics['refine_loss'] += loss_dict['refine'].item()
                epoch_metrics['consistency_loss'] += loss_dict['consistency'].item()

                # 梯度累积
                loss = loss / gradient_accumulator.accumulation_steps

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度累积：达到指定步数时更新参数
            if gradient_accumulator.should_update():
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # 更新EMA
                model_ema.update()

                # 更新学习率
                scheduler.step()

                # 重置梯度累积器
                gradient_accumulator.reset()

            # 记录统计信息
            epoch_loss += loss.item() * gradient_accumulator.accumulation_steps

            current_lr = scheduler.get_lr()
            progress_bar.set_postfix({
                'lr': f'{current_lr:.2e}',
                'loss': f'{loss.item() * gradient_accumulator.accumulation_steps:.4f}',
                'grad_step': gradient_accumulator.counter
            })

            # 可视化采样
            if batch_idx % 20 == 0 and args.save_visualization:
                vis_dir = os.path.join(args.output_dir, "visualization")
                os.makedirs(vis_dir, exist_ok=True)

                # 保存原始图像、掩码和预测
                to_pil(augment_image[0].cpu()).save(
                    os.path.join(vis_dir, f'epoch_{epoch}_image.jpg')
                )
                to_pil(mask[0].cpu()).save(
                    os.path.join(vis_dir, f'epoch_{epoch}_mask.jpg')
                )

                with torch.no_grad():
                    pred_mask = torch.sigmoid(refined_output[:, 1:2, :, :])
                    to_pil(pred_mask[0].cpu()).save(
                        os.path.join(vis_dir, f'epoch_{epoch}_pred.jpg')
                    )

        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        metrics_history['train_loss'].append(avg_loss)
        metrics_history['lr'].append(current_lr)

        logger.info(
            f"Epoch {epoch} - "
            f"Primary Loss: {loss_dict['primary'].item():.4f}, "
            f"Refine Loss: {loss_dict['refine'].item():.4f}, "
            f"Consistency Loss: {loss_dict['consistency'].item():.4f}, "
            f"Total Loss: {loss_dict['total'].item():.4f}"
            # f"Epoch {epoch} - LR: {current_lr:.2e}, Loss: {avg_loss:.4f}"
        )

        # 验证和模型保存
        if (epoch % args.val_interval == 0 and epoch >= 0) or epoch > args.epochs // 2:
            # 使用EMA模型进行验证
            model_ema.apply_shadow()

            val_metrics = validate_model(model, args)
            precisions, recalls, f1s, aurocs, iou = val_metrics

            # 恢复原始模型
            model_ema.restore()

            # 记录验证指标
            metrics_history['val_precision'].append(precisions)
            metrics_history['val_recall'].append(recalls)
            metrics_history['val_f1'].append(f1s)
            metrics_history['val_auroc'].append(aurocs)
            metrics_history['val_iou'].append(iou)

            logger.info(
                f"Validation - Epoch {epoch}: "
                f"Precision: {precisions:.4f}, "
                f"Recall: {recalls:.4f}, "
                f"F1: {f1s:.4f}, "
                f"AUROC: {aurocs:.4f}, "
                f"IoU: {iou:.4f}"
            )

            # 综合评估指标（更关注小目标和边界精度）
            composite_score = 0.4 * f1s + 0.3 * iou + 0.2 * aurocs + 0.1 * precisions

            # 保存最佳模型
            best_composite_score = (
                    0.4 * best_metrics['f1'] +
                    0.3 * best_metrics['iou'] +
                    0.2 * best_metrics['auroc'] +
                    0.1 * precisions
            )

            if composite_score > best_composite_score:
                best_metrics.update({
                    'f1': f1s,
                    'iou': iou,
                    'auroc': aurocs,
                    'precision': precisions,
                    'epoch': epoch,
                    'loss': avg_loss
                })

                # 保存最佳模型
                checkpoint_path = os.path.join(args.output_dir, "checkpoints", 'best_model.pckl')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.current_scheduler.state_dict(),
                    'loss': avg_loss,
                    'metrics': val_metrics,
                    'config': vars(args)
                }, checkpoint_path)

                # 导出ONNX模型
                dummy_input1 = torch.randn(1, 3, args.height, args.height).to('cuda')
                dummy_input2 = torch.randn(1, 1, args.height, args.height).to('cuda')

                torch.onnx.export(
                    model,
                    (dummy_input1, dummy_input2),
                    os.path.join(args.output_dir, "checkpoints", "best_model.onnx"),
                    opset_version=11,
                    input_names=['image', 'gray'],
                    output_names=['rec_output', 'seg_output']
                )

                logger.info(f"New best model saved at epoch {epoch} with score {composite_score:.4f}")

            # 早停检查
            if early_stopping(composite_score):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # 定期保存检查点
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(
                args.output_dir,
                "checkpoints",
                f"model_epoch_{epoch}.pckl"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'metrics': metrics_history
            }, checkpoint_path)

            logger.info(f"Checkpoint saved at epoch {epoch}")

        # 保存最新模型
        latest_path = os.path.join(args.output_dir, "checkpoints", "latest_model.pckl")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'metrics': metrics_history
        }, latest_path)

    # 保存最终模型和训练结果
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'metrics': metrics_history,
        'best_metrics': best_metrics,
        'config': vars(args)
    }

    torch.save(final_checkpoint, os.path.join(args.output_dir, 'final_model.pckl'))

    # 保存训练结果总结
    results_summary = {
        'best_metrics': best_metrics,
        'config': vars(args),
        'metrics_history': dict(metrics_history),
        'training_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(args.output_dir, "training_results.json"), 'w') as f:
        json.dump(results_summary, f, indent=4)

    logger.info(
        f"Training completed. Best model at epoch {best_metrics['epoch']} "
        f"with F1: {best_metrics['f1']:.4f}, IoU: {best_metrics['iou']:.4f}"
    )
    logger.info(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Anomaly Segmentation Training')

    # 训练参数
    parser.add_argument('--obj_id', type=int, default=-1)
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--T_0', type=int, default=10,
                        help='Number of epochs for first restart in cosine scheduler')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--val_interval', type=int, default=5, help='Validation interval in epochs')
    parser.add_argument('--checkpoint_interval', type=int, default=15,
                        help='Checkpoint saving interval')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume training from checkpoint')

    # 设备参数
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help='GPU ID (0-7: different loss combinations, 8: OneCycleLR)')

    # 数据路径
    parser.add_argument('--data_path', type=str, default='G:/AITEX/Defect_images/',
                        help='Training data path')
    parser.add_argument('--val_path', type=str, default='G:/AITEX/Defect_images/',
                        help='Validation data path')
    parser.add_argument('--anomaly_source_path', type=str, default='G:/AITEX/Mask_images/',
                        help='Anomaly source path')

    # 输出路径
    parser.add_argument('--checkpoint_path', type=str, default='./Mod_seg/Mod/',
                        help='Checkpoint save path')
    parser.add_argument('--log_path', type=str, default='./Mod_seg/',
                        help='Log save path')

    # 其他选项
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Enable visualization')
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Enable WandB logging')
    parser.add_argument("--height", type=int, default=256, help="Input image height")
    parser.add_argument("--save_visualization", type=int, default=True,
                        help="Save visualization images during training")

    args = parser.parse_args()

    # 设置CUDA设备
    torch.backends.cudnn.benchmark = True

    train_on_device(args)
