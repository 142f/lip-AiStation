import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
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
# SELayerVec: 针对展平特征的通道注意力
# ---------------------------
class SELayerVec(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super(SELayerVec, self).__init__()
        hidden = max(1, channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        scale = self.fc(x)
        return x * scale


# ---------------------------
# [修改] 模块: DWT 标准小波变换
# ---------------------------
class DWT(nn.Module):
    """
    基于卷积实现的离散小波变换 (Discrete Wavelet Transform)
    使用 Haar 小波核，将输入 (B, 3, H, W) 转换为 (B, 12, H/2, W/2)
    """
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 小波变换本身不需要训练
        self.colors = 3             # RGB通道

        # Haar 小波滤波器定义
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).float()
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]]).float()
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]]).float()
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).float()

        # 组合成 (4, 1, 2, 2)
        filters = torch.cat([
            ll.unsqueeze(0), 
            lh.unsqueeze(0), 
            hl.unsqueeze(0), 
            hh.unsqueeze(0)
        ], dim=0).unsqueeze(1)

        # 扩展到 3个通道: (12, 1, 2, 2)
        # 这种写法是 depthwise convolution 的形式，每个通道独立计算
        filters = filters.repeat(self.colors, 1, 1, 1)

        self.register_buffer('filters', filters)

    def forward(self, x):
        # x: (B, 3, H, W)
        b, c, h, w = x.shape
        # padding 保证尺寸匹配 (如果输入可能是奇数尺寸)
        if h % 2 != 0 or w % 2 != 0:
            x = nn.functional.pad(x, (0, 1, 0, 1), mode='reflect')

        # 使用 group convolution 进行通道独立变换
        # 输出: (B, 12, H/2, W/2)
        out = nn.functional.conv2d(x, self.filters, stride=2, groups=c)
        return out


# ---------------------------
# [修改] 模块: 改进版小波特征提取器
# ---------------------------
class WaveletExtractor(nn.Module):
    def __init__(self, out_dim=256):
        super(WaveletExtractor, self).__init__()
        
        self.dwt = DWT()
        
        # 输入维度: 3 colors * 4 bands (LL, LH, HL, HH) = 12
        in_channels = 12
        
        # 特征提取网络: Stem -> ResBlock -> ResBlock
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = self._make_basic_block(64, 128)
        self.layer2 = self._make_basic_block(128, 256)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 最终投影
        self.fc = nn.Linear(256, out_dim)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # [关键] 零初始化最后一层，保证初始阶段不影响主干梯度
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)

    def _make_basic_block(self, in_planes, out_planes):
        """创建一个带下采样的基础卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, stride=2, padding=1, bias=False), # Downsample
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1. DWT 变换 -> (B, 12, H/2, W/2)
        freq_map = self.dwt(x)
        
        # 2. [关键] Log 对数变换
        # 频域幅值差异很大，取 log 类似于同态滤波，有助于 CNN 学习
        freq_map = torch.log(torch.abs(freq_map) + 1e-6)
        
        # 3. 特征提取
        x = self.conv_stem(freq_map)
        x = self.layer1(x)
        x = self.layer2(x)
        
        # 4. 聚合输出
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


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

        # ---------------------------
        # 修改 1: 初始化小波提取模块 (替换了原 FrequencyExtractor)
        # ---------------------------
        self.freq_dim = 256  # 设定频域特征维度
        self.freq_extractor = WaveletExtractor(out_dim=self.freq_dim)

        # ---------------------------
        # 修改 2: 重新计算总特征维度
        # feat_dim = RGB局部特征 + CLIP全局特征 + 频域特征
        # ---------------------------
        feat_dim = 512 * block.expansion + 768 + self.freq_dim

        ## 融合层
        self.get_weight = nn.Sequential(
            nn.Linear(feat_dim, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(feat_dim, 1)

        # ---------------------------
        # 修改 3: 更新位置编码和 SE 模块的维度
        # ---------------------------
        self.num_scales = 3
        self.num_regions = 5
        
        # 可学习的位置编码
        self.positional_encoding = nn.Parameter(
            torch.randn(self.num_scales * self.num_regions, feat_dim)
        )
        # 向量版 SE，处理融合后的三模态特征
        self.se_layer = SELayerVec(feat_dim, reduction=16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        """
        前向传播逻辑
        """
        # ---------------------------
        # 基础维度信息
        # ---------------------------
        num_scales = len(x)
        num_regions = len(x[0])
        batch_size = x[0][0].shape[0]

        # ---------------------------
        # Step 1: 拼接所有区域为一个大批次
        # ---------------------------
        all_images = [x[s][r] for s in range(num_scales) for r in range(num_regions)]
        all_images = torch.cat(all_images, dim=0)  # (S*R*B, 3, H, W)

        # ---------------------------
        # Step 2: 提取 RGB 局部特征 (ResNet)
        # ---------------------------
        f = self.conv1(all_images)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.avgpool(f)
        f = torch.flatten(f, 1)  # (S*R*B, feat_local_rgb)

        # Reshape 回结构
        f = f.view(num_scales, num_regions, batch_size, -1)  # (S, R, B, feat_rgb)

        # ---------------------------
        # Step 2.5: 提取 频域/小波 局部特征 - [修改]
        # ---------------------------
        # 使用新的 WaveletExtractor
        f_freq_flat = self.freq_extractor(all_images) # (S*R*B, freq_dim)
        
        # Reshape 回结构
        f_freq = f_freq_flat.view(num_scales, num_regions, batch_size, -1) # (S, R, B, freq_dim)

        # ---------------------------
        # Step 3: 准备 CLIP 全局特征
        # ---------------------------
        # 广播 CLIP 特征以匹配局部特征的结构
        feat_g = feature.unsqueeze(0).unsqueeze(0).expand(num_scales, num_regions, batch_size, feature.shape[1])

        # ---------------------------
        # Step 4: 多模态特征融合 (RGB + CLIP + Wavelet)
        # ---------------------------
        # 拼接策略：将三种特征在通道维度拼接
        # feat_cat shape: (S, R, B, rgb_dim + clip_dim + freq_dim)
        feat_cat = torch.cat([f, feat_g, f_freq], dim=3)

        # ---------------------------
        # Step 4.5: 添加位置编码
        # ---------------------------
        feat_dim = feat_cat.shape[-1]
        pos_enc = self.positional_encoding[:num_scales * num_regions]
        pos_enc = pos_enc.view(num_scales, num_regions, 1, feat_dim).expand(-1, -1, batch_size, -1)
        feat_cat = feat_cat + pos_enc

        # ---------------------------
        # Step 4.6: 使用 SE 模块进行融合权重分配
        # ---------------------------
        # SE 模块根据内容自适应调整 RGB、CLIP 和 Freq 三部分的权重
        feat_cat_flat = feat_cat.contiguous().view(-1, feat_dim)
        feat_cat_flat = self.se_layer(feat_cat_flat)  # 通道注意力重标定
        feat_cat = feat_cat_flat.view(num_scales, num_regions, batch_size, feat_dim)

        # ---------------------------
        # Step 5: 批量计算区域权重
        # ---------------------------
        feat_cat_flat = feat_cat.contiguous().view(-1, feat_cat.shape[-1])
        weights_list = []
        if chunk_size and feat_cat_flat.shape[0] > chunk_size:
            for chunk in torch.split(feat_cat_flat, chunk_size, dim=0):
                w_chunk = self.get_weight(chunk)
                weights_list.append(w_chunk)
            weights_all = torch.cat(weights_list, dim=0)
        else:
            weights_all = self.get_weight(feat_cat_flat)

        # ---------------------------
        # Step 6: 恢复权重结构
        # ---------------------------
        weights = weights_all.view(num_scales, num_regions, batch_size, -1)
        weights_scalar = weights.squeeze(-1)

        # ---------------------------
        # Step 7: 尺度注意力 Softmax
        # ---------------------------
        weights_soft = torch.softmax(weights_scalar, dim=0)

        # ---------------------------
        # Step 8: 加权融合
        # ---------------------------
        fused_parts = (feat_cat * weights_soft.unsqueeze(-1)).sum(dim=0)
        weights_sum = weights_soft.sum(dim=0) + 1e-8
        fused_parts = fused_parts / weights_sum.unsqueeze(-1)

        # ---------------------------
        # Step 9: 区域聚合
        # ---------------------------
        out = fused_parts.sum(dim=0).div(fused_parts.shape[0])

        # ---------------------------
        # Step 10: 分类
        # ---------------------------
        pred_score = self.fc(out)

        # ---------------------------
        # Step 11: 辅助输出
        # ---------------------------
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
    model = ResNet(block, layers, num_classes=1, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def get_backbone(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _get_backbone('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


if __name__ == '__main__':
    # 简单的测试代码，确保维度正确
    model = get_backbone()
    # 模拟输入: 3个尺度，每个尺度5个区域
    data = [[] for i in range(3)]
    for i in range(3):
        for j in range(5):
            data[i].append(torch.rand((2, 3, 224, 224))) # batch=2
    feature = torch.rand((2, 768)) # CLIP feature
    
    pred_score, weights_max, weights_org = model(data, feature)
    print("Output shape:", pred_score.shape)
    print("Freq Dim included in pipeline check:", model.freq_dim)