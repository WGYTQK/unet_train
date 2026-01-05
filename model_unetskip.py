import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积（保持与原代码兼容）"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class LightCBAM(nn.Module):
    """轻量级CBAM注意力模块"""

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_att(x) * x
        # 空间注意力
        sa_input = torch.cat([torch.max(ca, dim=1, keepdim=True)[0],
                              torch.mean(ca, dim=1, keepdim=True)], dim=1)
        sa = self.spatial_att(sa_input) * ca
        return sa


class MultiScaleFeatureBlock(nn.Module):
    """多尺度特征块 - 针对不同大小的目标"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 不同尺度的卷积
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv3x3 = DepthwiseSeparableConv(in_channels, out_channels // 4, kernel_size=3)
        self.conv5x5 = DepthwiseSeparableConv(in_channels, out_channels // 4, kernel_size=5, padding=2)

        # 池化分支
        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 提取不同尺度特征
        feat1 = self.conv1x1(x)
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)

        # 全局特征
        pool_feat = self.pool_branch(x)
        pool_feat = F.interpolate(pool_feat, size=x.shape[2:], mode='bilinear', align_corners=True)

        # 特征融合
        combined = torch.cat([feat1, feat3, feat5, pool_feat], dim=1)
        out = self.fusion(combined)
        out = self.bn(out)
        out = self.relu(out)

        return out


class EncoderBlock(nn.Module):
    """编码器块"""

    def __init__(self, in_channels, out_channels, use_cbam=True):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = LightCBAM(out_channels) if use_cbam else nn.Identity()
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.cbam(x)
        return self.downsample(x), x


class DecoderBlock(nn.Module):
    """解码器块"""

    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DepthwiseSeparableConv(in_channels + skip_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # 调整skip连接尺寸
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SmallTargetEnhancement(nn.Module):
    """小目标增强模块"""

    def __init__(self, in_channels):
        super().__init__()
        # 高频特征提取（小目标通常有高频特征）
        self.high_freq_extract = nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels // 2),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion = nn.Conv2d(in_channels + in_channels // 4, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # 提取高频特征
        high_freq = self.high_freq_extract(x)

        # 生成小目标注意力图
        attention = self.attention(high_freq)

        # 增强小目标区域
        enhanced_high_freq = high_freq * attention

        # 特征融合
        combined = torch.cat([x, enhanced_high_freq], dim=1)
        out = self.fusion(combined)
        out = self.bn(out)
        out = self.relu(out)

        return out + identity


class BoundaryAwareRefinement(nn.Module):
    """边界感知细化模块"""

    def __init__(self, in_channels):
        super().__init__()
        self.boundary_conv = DepthwiseSeparableConv(in_channels, in_channels)
        self.boundary_bn = nn.BatchNorm2d(in_channels)

        # 边界注意力
        self.boundary_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def get_boundary_features(self, x):
        """提取边界特征"""
        # 使用梯度的最大值和平均值作为边界特征
        grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

        # 填充以保持尺寸
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))

        return torch.cat([
            torch.max(grad_x, dim=1, keepdim=True)[0],
            torch.mean(grad_x, dim=1, keepdim=True)
        ], dim=1)

    def forward(self, x):
        identity = x

        # 边界特征提取
        boundary_feat = self.get_boundary_features(x)
        boundary_attention = self.boundary_attention(boundary_feat)

        # 边界增强
        x = self.boundary_conv(x)
        x = self.boundary_bn(x)
        x = x * (1 + boundary_attention)  # 加强边界区域

        return self.relu(x + identity)


class PrimarySegmentationNetwork(nn.Module):
    """初级分割网络"""

    def __init__(self, in_channels=3, out_channels=2, base_width=64):
        super().__init__()

        # 编码器
        self.encoder1 = EncoderBlock(in_channels, base_width)
        self.encoder2 = EncoderBlock(base_width, base_width * 2)
        self.encoder3 = EncoderBlock(base_width * 2, base_width * 4)
        self.encoder4 = EncoderBlock(base_width * 4, base_width * 8, use_cbam=False)

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(base_width * 8, base_width * 8),
            LightCBAM(base_width * 8)
        )

        # 解码器
        self.decoder1 = DecoderBlock(base_width * 8, base_width * 4, skip_channels=base_width * 8)
        self.decoder2 = DecoderBlock(base_width * 4, base_width * 2, skip_channels=base_width * 4)
        self.decoder3 = DecoderBlock(base_width * 2, base_width, skip_channels=base_width * 2)

        # 输出层
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # 编码
        x1, e1 = self.encoder1(x)  # e1: [B, 64, H/2, W/2]
        x2, e2 = self.encoder2(x1)  # e2: [B, 128, H/4, W/4]
        x3, e3 = self.encoder3(x2)  # e3: [B, 256, H/8, W/8]
        x4, e4 = self.encoder4(x3)  # e4: [B, 512, H/16, W/16]

        # 瓶颈
        bottleneck = self.bottleneck(x4)  # [B, 512, H/16, W/16]

        # 解码
        d1 = self.decoder1(bottleneck, e4)  # [B, 256, H/8, W/8]
        d2 = self.decoder2(d1, e3)  # [B, 128, H/4, W/4]
        d3 = self.decoder3(d2, e2)  # [B, 64, H/2, W/2]

        # 输出
        output = self.final_conv(d3)  # [B, 2, H, W]

        return output


class RefinedSegmentationNetwork(nn.Module):
    """精细分割网络"""

    def __init__(self, in_channels=4, out_channels=2, base_channels=48):
        super().__init__()

        # 多尺度特征提取
        self.multiscale = MultiScaleFeatureBlock(in_channels, base_channels * 2)

        # 编码器
        self.encoder1 = EncoderBlock(base_channels * 2, base_channels)
        self.encoder2 = EncoderBlock(base_channels, base_channels * 2)
        self.encoder3 = EncoderBlock(base_channels * 2, base_channels * 4)

        # 小目标增强模块
        self.small_target_enhance = SmallTargetEnhancement(base_channels * 4)

        # 边界细化模块
        self.boundary_refine = BoundaryAwareRefinement(base_channels * 4)

        # 解码器
        self.decoder1 = DecoderBlock(base_channels * 4, base_channels * 2, skip_channels=base_channels * 4)
        self.decoder2 = DecoderBlock(base_channels * 2, base_channels, skip_channels=base_channels * 2)

        # 输出层
        self.output_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DepthwiseSeparableConv(base_channels, base_channels),
            nn.Conv2d(base_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # 多尺度特征提取
        x = self.multiscale(x)  # [B, 96, H, W]

        # 编码
        x1, e1 = self.encoder1(x)  # [B, 48, H/2, W/2]
        x2, e2 = self.encoder2(x1)  # [B, 96, H/4, W/4]
        x3, e3 = self.encoder3(x2)  # [B, 192, H/8, W/8]

        # 小目标增强
        x3 = self.small_target_enhance(x3)

        # 边界细化
        x3 = self.boundary_refine(x3)

        # 解码
        d1 = self.decoder1(x3, e3)  # [B, 96, H/4, W/4]
        d2 = self.decoder2(d1, e2)  # [B, 48, H/2, W/2]

        # 输出
        output = self.output_conv(d2)  # [B, 2, H, W]

        return output


class CascadeAnomalySegmentationNetwork(nn.Module):
    """级联异常分割网络 - 主要类"""

    def __init__(self, primary_base_width=64, refine_base_channels=48):
        super().__init__()

        # 第一阶段：初级分割
        self.primary_net = PrimarySegmentationNetwork(
            in_channels=3,  # 输入RGB图像
            out_channels=2,  # 输出2通道（背景+前景）
            base_width=primary_base_width
        )

        # 第二阶段：精细分割
        self.refine_net = RefinedSegmentationNetwork(
            in_channels=4,  # 输入：RGB(3) + 初级分割结果(1)
            out_channels=2,
            base_channels=refine_base_channels
        )

    def forward(self, x1, x2=None):
        """
        兼容原有训练代码的forward方法
        Args:
            x1: 通常是augment_image [B, 3, H, W] 或 [B, 1, H, W]
            x2: 通常是auggray [B, 1, H, W] 或 None
        """
        # 确定哪个是彩色图像，哪个是灰度图像
        if x1.shape[1] == 3:  # x1是彩色图像
            image = x1
            if x2 is not None and x2.shape[1] == 1:  # x2是灰度图像
                gray = x2
            else:
                # 从彩色图像计算灰度
                gray = torch.mean(image, dim=1, keepdim=True)
        elif x1.shape[1] == 1:  # x1是灰度图像
            gray = x1
            if x2 is not None and x2.shape[1] == 3:  # x2是彩色图像
                image = x2
            else:
                # 将灰度图像复制成3通道作为彩色图像
                image = gray.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"Unexpected input shape: x1={x1.shape}, x2={x2.shape if x2 is not None else None}")

        # 确保所有输入在正确设备上
        if image.device != self.primary_net.encoder1.conv1.depthwise.weight.device:
            image = image.to(self.primary_net.encoder1.conv1.depthwise.weight.device)
        if gray.device != image.device:
            gray = gray.to(image.device)

        # 第一阶段：初级分割
        primary_output = self.primary_net(image)  # [B, 2, H, W]

        # 提取初级分割的前景概率
        primary_prob = torch.softmax(primary_output, dim=1)
        primary_foreground = primary_prob[:, 1:2, :, :]  # 取正类（前景）

        # 准备第二阶段输入：RGB + 初级分割 + 灰度
        refine_input = torch.cat([image, primary_foreground], dim=1)  # [B, 5, H, W]

        # 第二阶段：精细分割
        refined_output = self.refine_net(refine_input)  # [B, 2, H, W]

        return primary_output, refined_output


# 为了保持与原有代码的兼容性
AnomalySegmentationNetwork = CascadeAnomalySegmentationNetwork
