import torch
from data_loader import MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from model_unetskip import AnomalySegmentationNetwork as CombinedNetwork, DepthwiseSeparableConv
from loss import SSIM, FocalLoss, SoftDiceLoss, MultiClassFocalLoss
import os
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
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # 针对leaky_relu调整a值
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)  # 匹配leaky_relu的负斜率
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, DepthwiseSeparableConv):
        # 深度可分离卷积特殊初始化
        nn.init.kaiming_normal_(m.depthwise.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
        nn.init.kaiming_normal_(m.pointwise.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)


class BoundaryAwareLoss(nn.Module):
    """边界感知损失函数"""

    def __init__(self, alpha=0.5, margin=2):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.bce = nn.BCEWithLogitsLoss()

    def get_boundary_mask(self, mask):
        """生成边界区域mask"""
        kernel = torch.ones(1, 1, 3, 3).to(mask.device)
        eroded = F.conv2d(mask.float(), kernel, padding=1) < 9.0
        dilated = F.conv2d(mask.float(), kernel, padding=1) > 0.0
        boundary = (dilated.float() - eroded.float()).clamp(0, 1)
        return boundary

    def forward(self, pred, target):
        # 基础分割损失
        base_loss = self.bce(pred, target)

        # 边界增强损失
        boundary_mask = self.get_boundary_mask(target)
        boundary_loss = F.binary_cross_entropy_with_logits(
            pred, target,
            weight=1.0 + self.alpha * boundary_mask
        )

        # 连续性约束（防止孔洞）
        pred_sigmoid = torch.sigmoid(pred)
        continuity_loss = torch.mean(torch.abs(pred_sigmoid[:, :, 1:, :] - pred_sigmoid[:, :, :-1, :])) + \
                          torch.mean(torch.abs(pred_sigmoid[:, :, :, 1:] - pred_sigmoid[:, :, :, :-1]))

        return boundary_loss + 0.3 * continuity_loss + 0.2 * base_loss

class SmallTargetLoss(nn.Module):
    """增强的小目标感知损失"""

    def __init__(self, threshold=0.05, alpha=0.5):
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.dice_loss = SoftDiceLoss()

    def forward(self, pred, target):
        target_areas = (target > 0).float().sum(dim=(1, 2, 3))
        image_area = target.shape[2] * target.shape[3]
        small_target_mask = (target_areas / image_area) < self.threshold

        if small_target_mask.any():
            small_pred = pred[small_target_mask]
            small_target = target[small_target_mask]

            # 组合BCE和Dice损失
            bce_loss = self.bce_loss(small_pred, small_target)
            weights = torch.clamp(1.0 + 5.0 * small_target, max=10.0)
            weighted_bce = (bce_loss * weights).mean()

            dice_loss = self.dice_loss(small_pred, small_target)

            return self.alpha * weighted_bce + (1 - self.alpha) * dice_loss

        return torch.tensor(0.0).to(pred.device)


def create_output_dir(base_path):
    """创建带有时间戳的输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"train_{timestamp}"
    dir_path = os.path.join(base_path, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def check_model_parameters(model):
    """检查模型参数是否有效"""
    issues = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            issues.append(f"Parameter {name} contains NaN")
        if torch.isinf(param).any():
            issues.append(f"Parameter {name} contains Inf")
    return issues


def check_optimizer_state(optimizer):
    """检查优化器状态是否有效"""
    issues = []
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    issues.append("Gradient contains NaN")
                if torch.isinf(param.grad).any():
                    issues.append("Gradient contains Inf")
    return issues


def configure_optimizer(model, args):
    """可配置的优化器设置"""
    params = []
    for name, param in model.named_parameters():
        if 'bias' in name:
            params.append({'params': param, 'weight_decay': 0.0})
        else:
            params.append({'params': param, 'weight_decay': args.weight_decay})

    optimizer = optim.AdamW(
        params,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    return optimizer


def configure_scheduler(optimizer, args, dataloader):
    """可配置的学习率调度器"""
    if args.gpu_id == 8:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=args.epochs * len(dataloader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.T_0 * len(dataloader),  # 周期长度
            T_mult=1,  # 周期倍增因子
            eta_min=args.lr * 0.01  # 最小学习率
        )
    return scheduler


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=50, delta=0, mode='max'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif ((self.mode == 'max' and score < self.best_score + self.delta) or
              (self.mode == 'min' and score > self.best_score - self.delta)):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


def train_on_device(args):
    # 1. 创建输出目录
    args.output_dir = create_output_dir(args.checkpoint_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. 初始化日志记录器和配置
    logger = setup_logger(args.output_dir)
    logger.info(f"Training configuration:\n{json.dumps(vars(args), indent=4)}")

    # 3. 数据加载优化
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

    # 4. 模型初始化
    model = CombinedNetwork().cuda()
    model.apply(weights_init)

    # 5. 优化器和学习率调度器
    optimizer = configure_optimizer(model, args)
    scheduler = configure_scheduler(optimizer, args, dataloader)

    # 6. 损失函数配置
    focal_loss = FocalLoss(
        apply_nonlin=torch.sigmoid,
        alpha=[0.5, 0.5],
        gamma=2,
        smooth=1e-5,
    )
    dice_loss = SoftDiceLoss()
    iou_loss = nn.BCEWithLogitsLoss()
    small_target_loss = SmallTargetLoss(threshold=0.05, alpha=0.7)

    # 7. 混合精度训练
    scaler = GradScaler()
    to_pil = ToPILImage()

    # 8. 模型加载和检查
    start_epoch = 0
    if args.resume:
        try:
            checkpoint = torch.load(os.path.join(args.checkpoint_path, 'la1test_model.pckl'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            start_epoch = 0
            logger.info(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {str(e)}")

    # 检查模型和优化器状态
    model_issues = check_model_parameters(model)
    optimizer_issues = check_optimizer_state(optimizer)

    if model_issues:
        logger.warning("Model parameter issues detected:\n" + "\n".join(model_issues))
    if optimizer_issues:
        logger.warning("Optimizer state issues detected:\n" + "\n".join(optimizer_issues))
        optimizer = configure_optimizer(model, args)

    # 9. 训练状态跟踪
    metrics = defaultdict(list)
    best_metrics = {
        'f1s': 0,
        'iou': 0,
        'auroc': 0,
        'epoch': 0,
        'loss': float('inf')
    }
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    boundary_loss = BoundaryAwareLoss(alpha=0.7)
    dice_loss = SoftDiceLoss()
    tv_loss = nn.L1Loss()  # 用于平滑性约束

    # 10. 增强的训练循环
    for epoch in range(start_epoch, args.epochs):
        if epoch%20==0:
            # 3. 数据加载优化
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
        model.train()
        epoch_loss = 0.0
        epoch_segment_loss1 = 0.0
        epoch_segment_loss2 = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, sample_batched in enumerate(progress_bar):
            # 数据加载
            augment_image = sample_batched["augment_image"].cuda(non_blocking=True)
            mask = sample_batched["mask"].cuda(non_blocking=True)
            auggray = sample_batched["auggray"].cuda(non_blocking=True)

            # 混合精度训练
            with autocast():
                rec_output, out_mask = model(augment_image, auggray)

                # 损失计算
                gray_rec_sm = torch.softmax(rec_output, dim=1)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                # seg1 = focal_loss(gray_rec_sm, mask)  + F.smooth_l1_loss(gray_rec_sm[:, 1:2, :, :], mask)
                # seg2 = focal_loss(out_mask_sm, mask)  + F.smooth_l1_loss(out_mask_sm[:, 1:2, :, :], mask)
                # seg1 += dice_loss(out_mask_sm[:, 1:2, :, :], mask) + iou_loss(out_mask_sm[:, 1:2, :, :], mask)
                # seg2 += dice_loss(out_mask_sm[:, 1:2, :, :], mask) + iou_loss(out_mask_sm[:, 1:2, :, :], mask)
                # # small_loss = small_target_loss(out_mask[:, 1:2, :, :], mask)
                # loss = 0.7*(seg1 + seg2) #+ 0.3*small_loss
                main_loss = boundary_loss(out_mask[:, 1:2, :, :], mask)
                # 辅助约束
                smooth_loss = tv_loss(out_mask[:, 1:2, :, :], mask)
                area_loss = torch.abs(out_mask[:, 1:2, :, :].sum() - mask.sum()) / (args.height ** 2)

                loss = main_loss + 0.1 * smooth_loss + 0.05 * area_loss

            # 反向传播和优化
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # 记录统计信息
            epoch_loss += loss.item()
            epoch_segment_loss1 += smooth_loss.item()
            epoch_segment_loss2 += area_loss.item()

            current_lr = get_lr(optimizer)
            progress_bar.set_postfix({
                'lr': f'{current_lr:.2e}',
                'loss': f'{loss.item():.4f}',
                'seg1': f'{smooth_loss.item():.4f}',
                'seg2': f'{area_loss.item():.4f}'
            })

            # 可视化采样
            if batch_idx % 15 == 0 and args.save_visualization:
                vis_dir = os.path.join(args.output_dir, "visualization")
                os.makedirs(vis_dir, exist_ok=True)
                to_pil(augment_image[0].cpu()).save(os.path.join(vis_dir, f'epoch_image.jpg'))
                to_pil(mask[0].cpu()).save(os.path.join(vis_dir, f'epoch_mask.jpg'))
                to_pil(out_mask_sm[0, 1, :, :].cpu()).save(os.path.join(vis_dir, f'epoch_out_mask.jpg'))

        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        avg_seg1 = epoch_segment_loss1 / len(dataloader)
        avg_seg2 = epoch_segment_loss2 / len(dataloader)

        # 记录到日志和metrics
        metrics['train_loss'].append(avg_loss)
        metrics['train_seg1'].append(avg_seg1)
        metrics['train_seg2'].append(avg_seg2)
        metrics['lr'].append(current_lr)

        logger.info(
            f"Epoch {epoch} - "
            f"LR: {current_lr:.2e}, "
            f"Loss: {avg_loss:.4f}, "
            f"Seg1: {avg_seg1:.4f}, "
            f"Seg2: {avg_seg2:.4f}"
        )
        dummy_input2 = torch.randn(1, 1, args.height, args.height).to('cuda')
        dummy_input1 = torch.randn(1, 3, args.height, args.height).to('cuda')

        # 验证和模型保存
        if (epoch % args.val_interval == 0 and epoch >= 0) or epoch > args.epochs // 2:
            val_metrics = validate_model(model, args)
            precisions, recalls, f1s, aurocs, iou = val_metrics

            # 记录验证指标
            metrics['val_precision'].append(precisions)
            metrics['val_recall'].append(recalls)
            metrics['val_f1'].append(f1s)
            metrics['val_auroc'].append(aurocs)
            metrics['val_iou'].append(iou)

            logger.info(
                f"Validation - Epoch {epoch}: "
                f"Precision: {precisions:.4f}, "
                f"Recall: {recalls:.4f}, "
                f"F1: {f1s:.4f}, "
                f"AUROC: {aurocs:.4f}, "
                f"IoU: {iou:.4f}"
            )

            # 综合评估指标
            composite_score = 0.5 * f1s + 0.4 * iou + 0.1 * aurocs

            # 保存最佳模型
            if composite_score > (0.5 * best_metrics['f1s'] + 0.4 * best_metrics['iou'] + 0.1 * best_metrics['auroc']):
                best_metrics.update({
                    'f1s': f1s,
                    'iou': iou,
                    'auroc': aurocs,
                    'epoch': epoch,
                    'loss': avg_loss
                })

                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'metrics': val_metrics
                }, os.path.join(args.output_dir, 'best_model.pckl'))

                # 导出ONNX模型

                torch.onnx.export(
                    model,(dummy_input1, dummy_input2),
                    os.path.join(args.output_dir, "best_model.onnx"),
                    opset_version=11)

                logger.info(f"New best model saved at epoch {epoch} with score {composite_score:.4f}")

            # 早停检查
            if early_stopping(composite_score):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # 定期保存检查点
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch}.pckl")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'metrics': metrics
            }, checkpoint_path)

            logger.info(f"Checkpoint saved at epoch {epoch}")
            torch.onnx.export(
                model,
                (dummy_input1, dummy_input2),
                os.path.join(args.output_dir, f"model_epoch_{epoch}.onnx"),
                opset_version=11)
        checkpoint_path = os.path.join(args.output_dir, "latest_model.pckl")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'metrics': metrics
        }, checkpoint_path)
        torch.onnx.export(
            model,
            (dummy_input1, dummy_input2),
            os.path.join(args.output_dir, "latest_model.onnx"),
            opset_version=11
        )

    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'metrics': metrics
    }, os.path.join(args.output_dir, 'final_model.pckl'))


    # 保存训练结果和配置
    results = {
        'best_metrics': best_metrics,
        'config': vars(args),
        'metrics': dict(metrics)
    }

    with open(os.path.join(args.output_dir, "training_results.json"), 'w') as f:
        json.dump(results, f, indent=4)

    logger.info(
        f"Training completed. Best model at epoch {best_metrics['epoch']} with F1: {best_metrics['f1s']:.4f}, IoU: {best_metrics['iou']:.4f}")
    logger.info(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Anomaly Segmentation Training')

    # 训练参数
    parser.add_argument('--obj_id', type=int, default=-1)
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--T_0', type=int, default=10, help='Number of epochs for first restart in cosine scheduler')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--val_interval', type=int, default=5, help='Validation interval in epochs')
    parser.add_argument('--checkpoint_interval', type=int, default=15, help='Checkpoint saving interval')
    parser.add_argument('--resume', action='store_true', default=True,help='Resume training from checkpoint')

    # 设备参数
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help='GPU ID (0-7: different loss combinations, 8: OneCycleLR)')

    # 数据路径
    parser.add_argument('--data_path', type=str, default='D:/datasets/PI/image/', help='Training data path')
    parser.add_argument('--val_path', type=str, default='D:/datasets/PI/image/', help='Validation data path')
    parser.add_argument('--anomaly_source_path', type=str, default='D:/datasets/PI/mask/', help='Anomaly source path')

    # 输出路径
    parser.add_argument('--checkpoint_path', type=str, default='./Mod_seg/Mod/', help='Checkpoint save path')
    parser.add_argument('--log_path', type=str, default='./Mod_seg/', help='Log save path')

    # 其他选项
    parser.add_argument('--visualize', action='store_true', default=True, help='Enable visualization')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Enable WandB logging')
    parser.add_argument("--height", type=int, default=256, help="Input image height")
    parser.add_argument("--save_visualization", type=int, default=True,
                        help="Save visualization images during training")

    args = parser.parse_args()

    # 设置CUDA设备
    torch.backends.cudnn.benchmark = True

    train_on_device(args)
