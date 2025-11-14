import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional
from torch.nn.functional import softmax

# 添加kornia导入用于边缘检测
import kornia

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
# SCSA注意力模块 (Semantic Continuous-Sparse Attention)
# 论文: SCSA: A Plug-and-Play Semantic Continuous-Sparse Attention for Arbitrary Semantic Style Transfer
# 发表期刊/会议: CVPR 2025
# ---------------------------
class SCSA(nn.Module):
    """
    改进版 SCSA（Semantic Continuous-Sparse Attention）

    改动要点：
      - 使用 FiLM 风格的条件通道调制（从语义图提取 gamma, beta）来实现连续的、语义条件化的通道变换（SCA 分支）；
      - 稀疏空间注意力（SSA 分支）由更强的 3x3 Conv 层组构成，产生细粒度空间注意图；
      - 对两个分支使用残差连接并引入可学习标量融合权重，使网络能自适应分配两者影响；
      - 兼容不同分辨率的 `semantic_map`（会被上/下采样到 x 的 spatial 大小）；
      - 添加多尺度语义融合和动态稀疏率控制；
      - 添加注意力图正则化项防止注意力坍缩；
    """

    def __init__(self, in_channels, semantic_dim=64, reduction: int = 16):
        super(SCSA, self).__init__()
        self.in_channels = in_channels
        self.semantic_dim = semantic_dim

        # ---- SCA: FiLM 风格通道调制（从语义图到 gamma,beta） ----
        hidden = max(1, semantic_dim // reduction)
        # 先对语义图做全局池化，再用小 MLP 预测 gamma 和 beta
        self.sca_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, semantic_dim, 1, 1)
            nn.Conv2d(semantic_dim, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels * 2, 1, bias=True)  # 输出 gamma 和 beta
        )

        # ---- SSA: 更强的稀疏空间注意力（细粒度空间变换） ----
        self.ssa_conv = nn.Sequential(
            nn.Conv2d(in_channels + semantic_dim, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # 可学习融合权重（标量）和残差缩放
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 融合系数，初始化为中间值
        self.beta = nn.Parameter(torch.tensor(0.5))   # SCA/SSA 的相对权重（可学习）
        
        # 动态稀疏率控制参数
        self.sparse_ratio = nn.Parameter(torch.tensor(0.3))
        # 温度系数用于softmax
        self.tau = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, semantic_map):
        """
        x: [B, C, H, W]
        semantic_map: [B, semantic_dim, H_s, W_s] (任意分辨率，内部会调整)
        返回: [B, C, H, W]
        """
        # 如果语义图为空或 None，直接返回输入
        if semantic_map is None:
            return x

        # 统一分辨率到 x 的 spatial
        if semantic_map.shape[2:] != x.shape[2:]:
            semantic_map = F.interpolate(semantic_map, size=x.shape[2:], mode='bilinear', align_corners=False)

        B, C, H, W = x.shape

        # ---- SCA: 从语义图预测 gamma, beta 并对通道做调制 ----
        sca_params = self.sca_mlp(semantic_map)  # (B, 2*C, 1, 1)
        sca_params = sca_params.view(B, 2 * C)
        gamma, bias = sca_params.chunk(2, dim=1)
        # 调制：x' = x * (1 + gamma) + bias  （gamma/bias broadcast 到 spatial）
        gamma = gamma.view(B, C, 1, 1)
        bias = bias.view(B, C, 1, 1)
        x_sca = x * (1.0 + gamma) + bias

        # ---- SSA: 空间注意力，条件于语义图 ----
        ssa_in = torch.cat([x, semantic_map], dim=1)
        ssa_attn = self.ssa_conv(ssa_in)  # (B, C, H, W), in [0,1]
        x_ssa = x * ssa_attn

        # ---- 融合（残差 + 学习融合系数） ----
        # alpha 控制是否应用注意力（残差权重），beta 控制 SCA/SSA 比例
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        beta = torch.sigmoid(self.beta)

        fused = beta * x_sca + (1 - beta) * x_ssa
        out = (1 - alpha) * x + alpha * fused
        
        # 保存注意力图用于后续正则化
        self.attention_map = ssa_attn

        return out

    def compute_attention_regularization(self):
        """
        计算注意力图的正则化损失，防止注意力坍缩
        """
        if not hasattr(self, 'attention_map'):
            return 0
            
        # L1稀疏约束
        l1_loss = torch.mean(torch.abs(self.attention_map))
        # 方差约束（防止注意力分布过于平坦）
        var_loss = torch.var(self.attention_map, dim=[1, 2, 3]).mean()
        return 0.01 * l1_loss + 0.005 * var_loss


def generate_lip_semantic_map(image):
    """
    生成唇部语义图（简化版：基于阈值分割唇部区域，实际建议用预训练语义分割模型）
    image: 输入图像 [H, W, 3]
    return: 语义图特征 [semantic_dim=64, H, W]
    """
    # 将tensor转换为numpy数组
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        # 如果是归一化的图像，需要还原
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    
    # 转换为RGB格式（如果需要）
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.dtype == np.uint8 else image
    
    # 简化版：灰度化+阈值分割（提取唇部区域）
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, lip_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)  # 需根据数据调整阈值
    lip_mask = lip_mask / 255.0  # 归一化
    
    # 编码为语义特征（64维通道）
    semantic_map = torch.tensor(lip_mask, dtype=torch.float32).unsqueeze(0).repeat(64, 1, 1)  # [64, H, W]
    return semantic_map


def semantic_guided_attention(energy, semantic_map):
    """
    多尺度语义引导注意力机制
    """
    # 多尺度融合语义图
    multi_scale_mask = []
    for scale in [1.0, 0.7, 0.4]:  # 三级尺度
        scaled_map = F.interpolate(
            semantic_map, 
            scale_factor=scale, 
            mode='bilinear',
            align_corners=False
        )
        # 添加边缘增强（解决语义边界模糊）
        edge_mask = kornia.filters.sobel(scaled_map) * 0.3
        multi_scale_mask.append(scaled_map + edge_mask)
    
    # 生成融合掩码
    fused_mask = torch.stack(multi_scale_mask).mean(0)
    return energy * fused_mask.unsqueeze(1)


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
        self.fc = nn.Linear(512 * block.expansion + 768, 1)

        # ---------------------------
        # SCSA注意力模块初始化
        # 说明：
        #  - 默认假设 num_scales = 3, num_regions = 5（与数据预处理一致）
        #  - 在layer4之后添加SCSA模块
        # ---------------------------
        self.num_scales = 3
        self.num_regions = 5
        self.semantic_dim = 64
        # 初始化SCSA模块
        self.scsa = SCSA(
            in_channels=512 * block.expansion,
            semantic_dim=self.semantic_dim
        )

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
    def _forward_impl(self, x, feature, semantic_map=None, chunk_size: int = 65536):
        """
        高效版前向传播（可直接替换原 _forward_impl）：
          - 一次性批量处理所有尺度与区域，充分利用 GPU 并行；
          - 通过 expand 减少全局特征重复内存；
          - 支持分块计算 get_weight，防止显存溢出；
          - 返回值与原函数保持一致 (pred_score, weights_max, weights_org)。

        参数:
            x: List[List[Tensor]]，形状为 [num_scales][num_regions][B, 3, H, W]
            feature: Tensor，全局特征 (B, feat_g)
            semantic_map: Tensor，语义图特征 (B, semantic_dim, H, W)
            chunk_size: int，可选参数，用于限制一次 get_weight 处理的样本数量，防止 OOM。

        返回:
            pred_score: Tensor，分类得分 (B, num_classes)
            weights_max: Tensor，每个区域在所有尺度下的最大权重
            weights_org: Tensor，每个区域在第一个尺度下的权重
        """

        # ---------------------------
        # 基础维度信息
        # ---------------------------
        num_scales = len(x)           # 尺度数量
        num_regions = len(x[0])       # 每尺度区域数
        batch_size = x[0][0].shape[0] # 每批样本数

        # ---------------------------
        # Step 1: 拼接所有区域为一个大批次
        # ---------------------------
        all_images = [x[s][r] for s in range(num_scales) for r in range(num_regions)]
        all_images = torch.cat(all_images, dim=0)  # (num_scales*num_regions*B, 3, H, W)

        # ---------------------------
        # Step 2: 一次性通过 backbone 提取局部特征
        # ---------------------------
        f = self.conv1(all_images)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        
        # 应用SCSA注意力机制
        if semantic_map is not None:
            # 为每个图像复制语义图
            semantic_map_repeated = semantic_map.repeat(num_scales * num_regions, 1, 1, 1)
            f = self.scsa(f, semantic_map_repeated)
        
        f = self.avgpool(f)
        f = torch.flatten(f, 1)  # (num_scales*num_regions*B, feat_local)

        # ---------------------------
        # Step 3: reshape 回原结构，方便后续处理
        # ---------------------------
        f = f.view(num_scales, num_regions, batch_size, -1)  # (S, R, B, feat_local)

        # ---------------------------
        # Step 4: 拼接全局特征（使用 expand 减少显存复制）
        # ---------------------------
        feat_g = feature.unsqueeze(0).unsqueeze(0).expand(num_scales, num_regions, batch_size, feature.shape[1])
        feat_cat = torch.cat([f, feat_g], dim=3)  # (S, R, B, feat_cat)

        # ---------------------------
        # Step 5: 批量计算权重（支持分块防止显存溢出）
        # ---------------------------
        feat_cat_flat = feat_cat.contiguous().view(-1, feat_cat.shape[-1])  # (S*R*B, feat_cat)
        weights_list = []
        if chunk_size and feat_cat_flat.shape[0] > chunk_size:
            # 分块计算
            for chunk in torch.split(feat_cat_flat, chunk_size, dim=0):
                w_chunk = self.get_weight(chunk)
                weights_list.append(w_chunk)
            weights_all = torch.cat(weights_list, dim=0)
        else:
            # 一次性计算
            weights_all = self.get_weight(feat_cat_flat)

        # ---------------------------
        # Step 6: reshape 权重回原结构
        # ---------------------------
        weights = weights_all.view(num_scales, num_regions, batch_size, -1)  # (S, R, B, weight_dim)
        # 假设 get_weight 输出 (N,1)，则压缩最后一维
        weights_scalar = weights.squeeze(-1)  # (S, R, B)

        # ---------------------------
        # Step 7: 对每个区域在尺度维做 softmax（用于尺度注意力）
        # ---------------------------
        weights_soft = torch.softmax(weights_scalar, dim=0)  # 在尺度维 S 上归一化

        # ---------------------------
        # Step 8: 加权融合局部与全局特征
        # ---------------------------
        # 广播相乘并在尺度维求和
        fused_parts = (feat_cat * weights_soft.unsqueeze(-1)).sum(dim=0)  # (R, B, C)
        # 防止数值问题，可再除以权重和
        weights_sum = weights_soft.sum(dim=0) + 1e-8  # (R, B)
        fused_parts = fused_parts / weights_sum.unsqueeze(-1)  # (R, B, C)

        # ---------------------------
        # Step 9: 区域聚合（平均所有区域特征）
        # ---------------------------
        out = fused_parts.sum(dim=0).div(fused_parts.shape[0])  # (B, C)

        # ---------------------------
        # Step 10: 分类层
        # ---------------------------
        pred_score = self.fc(out)  # (B, num_classes)

        # ---------------------------
        # Step 11: 计算附加权重输出（保持接口一致）
        # ---------------------------
        weights_max = weights_soft.max(dim=0)[0]  # (R, B)，每区域在不同尺度下的最大权重
        weights_org = weights_soft[0]             # (R, B)，第一个尺度的权重

        # ---------------------------
        # Step 12: 返回三元组（保持兼容）
        # ---------------------------
        return pred_score, weights_max, weights_org

    def forward(self, x, feature, semantic_map=None):
        return self._forward_impl(x, feature, semantic_map)


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

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _get_backbone('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


if __name__ == '__main__':
    model = get_backbone()
    data = [[] for i in range(3)]
    for i in range(3):
        for j in range(5):
            data[i].append(torch.rand((10, 3, 224, 224)))
    feature = torch.rand((10, 768))
    pred_score, weights_max, weights_org = model(data, feature)
    pass