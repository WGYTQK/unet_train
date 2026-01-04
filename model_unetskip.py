import torch
import torch.nn as nn
import torch.nn.functional as F


class LightCBAM(nn.Module):
    """ 轻量化CBAM模块（参数量减少约40%） """

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        # 通道注意力（增大压缩比）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )
        # 空间注意力（减小卷积核）
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=5, padding=2),  # 原为7x7
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


class DepthwiseSeparableConv(nn.Module):
    """ 深度可分离卷积（保持原实现） """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class EncoderBlock(nn.Module):
    """ 优化后的编码块（通道数调整） """

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
    """ 修复通道数不匹配的解码块 """

    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        # 上采样层
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 修正点1：输入通道应该是 in_channels (不是out_channels)
        self.conv1 = DepthwiseSeparableConv(in_channels + skip_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    # def forward(self, x, skip=None):
    #     x = self.up(x)
    #     if skip is not None:
    #         # 尺寸对齐
    #         if x.shape[-2:] != skip.shape[-2:]:
    #             skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=True)
    #         # 通道维度拼接
    #         x = torch.cat([x, skip], dim=1)
    #
    #     # 深度可分离卷积
    #     x = self.relu(self.bn1(self.conv1(x)))
    #     x = self.relu(self.bn2(self.conv2(x)))
    #     return x

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # 直接对skip进行插值，使其空间尺寸与x相同
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)

        # 深度可分离卷积
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ReconstructiveSubNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_width=96):
        super().__init__()
        # Encoder
        self.enc1 = EncoderBlock(in_channels, base_width)
        self.enc2 = EncoderBlock(base_width, base_width * 2)
        self.enc3 = EncoderBlock(base_width * 2, base_width * 4)
        self.enc4 = EncoderBlock(base_width * 4, base_width * 8, use_cbam=False)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(base_width * 8, base_width * 8),
            LightCBAM(base_width * 8)
        )

        # Decoder
        self.dec1 = DecoderBlock(base_width * 8, base_width * 4, skip_channels=base_width * 8)
        self.dec2 = DecoderBlock(base_width * 4, base_width * 2, skip_channels=base_width * 4)
        self.dec3 = DecoderBlock(base_width * 2, base_width, skip_channels=base_width * 2)

        # Final upsampling to match input size
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Encoder
        x, e1 = self.enc1(x)  # [B, 96, H/2, W/2]
        x, e2 = self.enc2(x)  # [B, 192, H/4, W/4]
        x, e3 = self.enc3(x)  # [B, 384, H/8, W/8]
        x, e4 = self.enc4(x)  # [B, 768, H/16, W/16]
        x = self.bottleneck(x)  # [B, 768, H/16, W/16]

        # Decoder with skip connections
        x = self.dec1(x, e4)  # [B, 384, H/8, W/8]
        x = self.dec2(x, e3)  # [B, 192, H/4, W/4]
        x = self.dec3(x, e2)  # [B, 96, H/2, W/2]

        # Final upsampling to original size
        return self.final_upsample(x)  # [B, 3, H, W]


def forward(self, x):
        # Encoder
        x, e1 = self.enc1(x)  # e1: [B, 96, H/2, W/2]
        x, e2 = self.enc2(x)  # e2: [B, 192, H/4, W/4]
        x, e3 = self.enc3(x)  # e3: [B, 384, H/8, W/8]
        x, e4 = self.enc4(x)  # e4: [B, 768, H/16, W/16]
        x = self.bottleneck(x)  # [B, 768, H/16, W/16]

        # Decoder
        x = self.dec1(x, e4)  # [B, 384, H/8, W/8]
        x = self.dec2(x, e3)  # [B, 192, H/4, W/4]
        x = self.dec3(x, e2)  # [B, 96, H/2, W/2]
        return self.final_upsample(x)  # [B, 3, H, W] 与输入同尺寸


class DiscriminativeSubNetwork(nn.Module):
    """ 轻量化判别子网络（基础通道从64降至48） """

    def __init__(self, in_channels=6, out_channels=2, base_channels=48):  # 原为64
        super().__init__()
        # Encoder
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8)

        # 简化FPN结构
        self.fpn_layers = nn.ModuleList([
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=1),  # P5→P4
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1),  # P4→P3
        ])

        # 分割头
        self.seg_head = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 2, base_channels),
            nn.Conv2d(base_channels, out_channels, kernel_size=1)  # 1x1卷积更轻量
        )

    def forward(self, x):
        # Encoder
        x, e1 = self.enc1(x)  # [B, 48, H/2, W/2]
        x, e2 = self.enc2(x)  # [B, 96, H/4, W/4]
        x, e3 = self.enc3(x)  # [B, 192, H/8, W/8]
        x, e4 = self.enc4(x)  # [B, 384, H/16, W/16]

        # 简化FPN
        p4 = F.interpolate(self.fpn_layers[0](x), size=e3.shape[2:], mode='bilinear') + e3
        p3 = F.interpolate(self.fpn_layers[1](p4), size=e2.shape[2:], mode='bilinear') + e2

        # 最终上采样
        out = F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=True)
        return self.seg_head(out)


class AnomalySegmentationNetwork(nn.Module):
    """ 最终轻量级网络 """

    def __init__(self):
        super().__init__()
        self.reconstructive_net = ReconstructiveSubNetwork(in_channels=3, out_channels=2, base_width=96)
        self.discriminative_net = DiscriminativeSubNetwork(in_channels=2, out_channels=2, base_channels=48)
        self.boundary_refine = nn.Sequential(
            DepthwiseSeparableConv(256, 256),
            nn.ReLU(),
            DepthwiseSeparableConv(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        # 添加全局上下文模块
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 1),
            nn.Sigmoid()
        )

    def forward(self, gray_rec, aug_gray_batch):
        rec_output = self.reconstructive_net(gray_rec)
        joined_in = torch.cat((rec_output[:, 1:2, :, :], aug_gray_batch), dim=1)
        segment_output = self.discriminative_net(joined_in)

        return rec_output, segment_output
