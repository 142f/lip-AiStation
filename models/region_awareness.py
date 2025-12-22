import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional, cast
from torch.nn.functional import softmax

from torch.hub import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wideget_backbone50_2': 'https://download.pytorch.org/models/wideget_backbone50_2-95faca4d.pth',
    'wideget_backbone101_2': 'https://download.pytorch.org/models/wideget_backbone101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ---------------------------
# 新增：针对展平特征的通道注意力（向量版 SE）
# 说明：
#  - 原始 SE 是对 4D feature map 做全局池化再两个 FC；这里我们对 (N, C) 向量做同样思想的通道重标定，
#    便于在不修改 backbone 的情况下把 SE 思想加到 feat_cat（拼接后向量）上。
# ---------------------------
class SELayerVec(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super(SELayerVec, self).__init__()
        hidden = max(1, channel // reduction)
        # 中文注释：通道压缩 -> ReLU -> 通道恢复 -> Sigmoid，输出每个通道的缩放系数（每个样本独立）
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C)
        # 输出与 x 相同形状，按通道放缩
        scale = self.fc(x)  # (N, C)
        return x * scale


# ---------------------------
# 新增：尺度敏感的频率增强模块
# ---------------------------
class ScaleAwareFrequencyEnhancer(nn.Module):
    def __init__(self, in_channels=3):
        super(ScaleAwareFrequencyEnhancer, self).__init__()
        
        # 1. 为 Scale 2 准备的小波核 (DWT)
        h = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2.0
        lh = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 2.0
        hl = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 2.0
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2.0
        kernels = torch.stack([h, lh, hl, hh], dim=0).unsqueeze(1).repeat(in_channels, 1, 1, 1)
        self.register_buffer('dwt_kernels', kernels)
        
        # 2. 为 Scale 0/1 准备的简单高通滤波器 (拉普拉斯算子)
        laplacian = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('hpf_kernels', laplacian.repeat(in_channels, 1, 1, 1))

        # 3. 频率特征映射层 (将 DWT 后的 12 通道映射回 3 通道，保持 Backbone 兼容)
        self.dwt_mapping = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1)

    def forward(self, x_scales):
        """
        x_scales: List[Tensor], 每个元素形状为 (B*R, C, H, W)，对应 S=0, 1, 2
        """
        # 显式转换类型以消除 Pylance 报错
        hpf_kernels = cast(Tensor, self.hpf_kernels)
        dwt_kernels = cast(Tensor, self.dwt_kernels)

        # --- 处理 Scale 0 & 1 (头部与面部): 应用简单高通滤波增强边缘 ---
        s0_enhanced = x_scales[0] + F.conv2d(x_scales[0], hpf_kernels, padding=1, groups=3) * 0.1
        s1_enhanced = x_scales[1] + F.conv2d(x_scales[1], hpf_kernels, padding=1, groups=3) * 0.1

        # --- 处理 Scale 2 (嘴部特写): 应用 DWT 深度提取频率指纹 ---
        # 形状变换: (B*R, 12, H/2, W/2)
        dwt_out = F.conv2d(x_scales[2], dwt_kernels, stride=2, groups=3)
        # 还原到原始分辨率并映射通道 (B*R, 3, H, W)
        s2_freq = self.dwt_mapping(F.interpolate(dwt_out, size=x_scales[2].shape[-2:], mode='bilinear', align_corners=False))
        s2_enhanced = x_scales[2] + s2_freq

        # 中文打印：确认增强完成
        # print(f"--- [调试] 尺度敏感频率增强完成: S0={s0_enhanced.shape}, S2={s2_enhanced.shape} ---")
        
        return [s0_enhanced, s1_enhanced, s2_enhanced]


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        ## 融合了局部细节特征与全局语义特征
        self.get_weight = nn.Sequential(
            nn.Linear(512 * block.expansion + 768, 1),  # TODO: 768 is the length of global feature
            nn.Sigmoid()
        )
        self.fc = nn.Linear(512 * block.expansion + 768, num_classes)

        # [新增] 初始化我们的尺度敏感增强器
        self.freq_enhancer = ScaleAwareFrequencyEnhancer(in_channels=3)

        # ---------------------------
        # 新增：位置编码与 SE 向量模块的初始化
        # 说明：
        #  - 默认假设 num_scales = 3, num_regions = 5（与数据预处理一致）
        #  - positional_encoding 的维度与 feat_cat 的维度一致（512*expansion + 768）
        #  - se_layer 在 feat_flat 上做通道重标定
        # ---------------------------
        self.num_scales = 3
        self.num_regions = 5
        feat_dim = 512 * block.expansion + 768
        # 可学习的位置编码，形状为 (S*R, feat_dim)
        self.positional_encoding = nn.Parameter(
            torch.randn(self.num_scales * self.num_regions, feat_dim)
        )
        # 向量版 SE
        self.se_layer = SELayerVec(feat_dim, reduction=16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def _forward_impl(self, x, feature, chunk_size: int = 65536):
        # x: (B, S, R, C, H, W)
        B, S, R, C, H, W = x.shape
        
        # 1. 拆分 Scale (S 维度)
        # x_list 包含 3 个形状为 (B, R, C, H, W) 的 Tensor
        x_list = [x[:, i].contiguous().view(B * R, C, H, W) for i in range(S)]

        # 2. 执行尺度敏感的频率增强 [创新点核心]
        x_list_enhanced = self.freq_enhancer(x_list)

        # 3. 合并回扁平化的 Batch 维度进入 Backbone
        x_flat = torch.cat(x_list_enhanced, dim=0) # (B*S*R, C, H, W)

        # 4. 进入卷积 Backbone
        f = self.conv1(x_flat)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.avgpool(f)
        f = torch.flatten(f, 1)

        # 5. 恢复维度并对接后续融合逻辑
        f = f.view(S, B, R, -1).permute(0, 2, 1, 3) # (S, R, B, feat_dim)

        # ---------------------------
        # 下面的逻辑保持不变 (复制过来即可)
        # ---------------------------
        
        # 5. 拼接全局特征（使用 expand 减少显存复制）
        feat_g = feature.unsqueeze(0).unsqueeze(0).expand(S, R, B, feature.shape[1])
        feat_cat = torch.cat([f, feat_g], dim=3)  # (S, R, B, feat_cat)

        # 6. 添加位置编码 (需确保 self.positional_encoding 已在 __init__ 中定义)
        # 注意：此处假设位置编码长度足够覆盖 S*R
        feat_dim = feat_cat.shape[-1]
        pos_enc = self.positional_encoding[:S * R]
        pos_enc = pos_enc.view(S, R, 1, feat_dim).expand(-1, -1, B, -1)
        feat_cat = feat_cat + pos_enc

        # 7. 向量版 SE 注意力 (需确保 self.se_layer 已定义)
        feat_cat_flat = feat_cat.contiguous().view(-1, feat_dim)
        feat_cat_flat = self.se_layer(feat_cat_flat)
        feat_cat = feat_cat_flat.view(S, R, B, feat_dim)

        # 8. 计算权重
        feat_cat_flat = feat_cat.contiguous().view(-1, feat_cat.shape[-1])
        
        # 支持分块计算防止 OOM
        if chunk_size and feat_cat_flat.shape[0] > chunk_size:
            weights_list = []
            for chunk in torch.split(feat_cat_flat, chunk_size, dim=0):
                weights_list.append(self.get_weight(chunk))
            weights_all = torch.cat(weights_list, dim=0)
        else:
            weights_all = self.get_weight(feat_cat_flat)

        weights = weights_all.view(S, R, B, -1).squeeze(-1) # (S, R, B)

        # 9. 融合与分类
        weights_soft = torch.softmax(weights, dim=0) # Scale 维度归一化
        
        # 加权求和
        fused_parts = (feat_cat * weights_soft.unsqueeze(-1)).sum(dim=0) # (R, B, C)
        weights_sum = weights_soft.sum(dim=0) + 1e-8
        fused_parts = fused_parts / weights_sum.unsqueeze(-1)

        # 区域聚合
        out = fused_parts.sum(dim=0).div(fused_parts.shape[0]) # (B, C)

        pred_score = self.fc(out)
        weights_max = weights_soft.max(dim=0)[0]
        weights_org = weights_soft[0]

        return pred_score, weights_max, weights_org

    def forward(self, x, feature):
        return self._forward_impl(x, feature)


def _get_backbone(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, num_classes=2, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def get_backbone(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # 使用 GroupNorm 替代 BatchNorm，解决小 Batch Size 问题
    norm_layer = lambda channels: nn.GroupNorm(32, channels)
    return _get_backbone('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, norm_layer=norm_layer, **kwargs)


if __name__ == '__main__':
    model = get_backbone()
    data = [[] for i in range(3)]
    for i in range(3):
        for j in range(5):
            data[i].append(torch.rand((10, 3, 224, 224)))
    feature = torch.rand((10, 768))
    pred_score, weights_max, weights_org = model(data, feature)
    pass
